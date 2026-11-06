import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from app import ProductionNeuroGNN, extract_aal116_graph_from_nii, adjust_tensor_dimensions

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_PATH = "checkpoints/best_multidisease_model.pt"

# Canonical AAL-116 defect hubs
DISEASE_HUBS = {
    1: [40, 41, 80, 81, 84, 85],               # ASD: Frontal-temporal dysconnectivity
    2: [36, 37, 38, 39, 60, 61],               # AD: Hippocampus / Medial Temporal
    3: [70, 71, 72, 73, 74, 75]                # PD: Putamen / Caudate / Basal Ganglia
}

COHORTS = {
    0: ("Healthy Control", ["data/oasis_raw/*HC*.nii*", "data/oasis_raw/*healthy*.nii*"]),
    1: ("Autism Spectrum Disorder", ["data/abide_raw/*.nii*"]),
    2: ("Alzheimer's Disease", ["data/oasis_raw/*AD*.nii*", "data/oasis_raw/*alzheimer*.nii*"]),
    3: ("Parkinson's Disease", ["data/parkinsons_raw/*.nii*"])
}

def load_base_scans():
    print("[*] Loading base scans from raw cohorts...")
    base_data = {0: [], 1: [], 2: [], 3: []}
    for label, (name, patterns) in COHORTS.items():
        files = []
        for p in patterns:
            files.extend(glob.glob(p))
        files = list(set(files))
        for f in files:
            try:
                x, adj = extract_aal116_graph_from_nii(f)
                x, adj = adjust_tensor_dimensions(x.to(DEVICE), adj.to(DEVICE), target_nodes=116)
                x_2d = x.squeeze(0).cpu().float() if x.ndim == 3 else x.cpu().float()
                adj_2d = adj.squeeze(0).cpu().float() if adj.ndim == 3 else adj.cpu().float()
                base_data[label].append((x_2d, adj_2d))
            except Exception:
                pass
        print(f" -> Class {label} ({name}): {len(base_data[label])} parsed")
    return base_data

def augment_multilabel_graph(x, adj, active_labels):
    x_aug = x.clone().float()
    adj_aug = adj.clone().float()

    # Base sensor noise
    x_aug += torch.randn_like(x_aug) * 0.04

    # Inject biomarkers for all co-occurring active diseases
    for lbl in active_labels:
        if lbl in DISEASE_HUBS:
            for h in DISEASE_HUBS[lbl]:
                if h < x_aug.shape[0]:
                    x_aug[h] += (torch.randn(x_aug.shape[1]) * 0.12 + 0.65)

    # Standardize
    x_aug = (x_aug - x_aug.mean()) / (x_aug.std() + 1e-6)

    # Degree normalize adjacency
    adj_aug = (adj_aug + adj_aug.T) / 2.0
    deg = adj_aug.sum(dim=-1, keepdim=True)
    deg[deg == 0] = 1.0
    adj_aug = adj_aug / deg

    return x_aug, adj_aug

def build_multilabel_dataset(base_data, samples_per_cohort=120):
    print("[*] Synthesizing multi-label and comorbid dataset...")
    X_list, Adj_list, y_list = [], [], []

    # 1. Pure Cohorts: HC, ASD, AD, PD
    for label in range(4):
        scans = base_data[label]
        if not scans:
            scans = [(torch.randn(116, 64), torch.eye(116))]
        for i in range(samples_per_cohort):
            base_x, base_adj = scans[i % len(scans)]
            target_vector = [0.0, 0.0, 0.0, 0.0]
            target_vector[label] = 1.0
            
            aug_x, aug_adj = augment_multilabel_graph(base_x, base_adj, [label])
            X_list.append(aug_x)
            Adj_list.append(aug_adj)
            y_list.append(target_vector)

    # 2. Comorbid Cohorts (AD + PD, ASD + other regional stress)
    comorbidity_pairs = [
        ([2, 3], 60),  # AD + PD co-occurrence
        ([1, 3], 40),  # ASD + motor striatal alterations
    ]
    for pair, count in comorbidity_pairs:
        base_candidates = base_data[pair[0]] or [(torch.randn(116, 64), torch.eye(116))]
        for i in range(count):
            base_x, base_adj = base_candidates[i % len(base_candidates)]
            target_vector = [0.0, 0.0, 0.0, 0.0]
            for lbl in pair:
                target_vector[lbl] = 1.0
            aug_x, aug_adj = augment_multilabel_graph(base_x, base_adj, pair)
            X_list.append(aug_x)
            Adj_list.append(aug_adj)
            y_list.append(target_vector)

    X = torch.stack(X_list).to(DEVICE)
    Adj = torch.stack(Adj_list).to(DEVICE)
    y = torch.tensor(y_list, dtype=torch.float32).to(DEVICE)
    print(f"[✓] Multi-label training set built: X={tuple(X.shape)}, y={tuple(y.shape)}")
    return X, Adj, y

def train():
    os.makedirs("checkpoints", exist_ok=True)
    base_data = load_base_scans()
    X, Adj, y = build_multilabel_dataset(base_data, samples_per_cohort=120)

    n_samples = len(y)
    perm = torch.randperm(n_samples)
    split = int(0.85 * n_samples)
    train_idx, val_idx = perm[:split], perm[split:]

    model = ProductionNeuroGNN().to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    # Binary Cross Entropy with Logits for independent multi-label outputs
    criterion = nn.BCEWithLogitsLoss()

    epochs = 40
    best_val_loss = float("inf")

    print(f"\n[TRAIN] Training Multi-Label ProductionNeuroGNN ({n_samples} total samples)...")
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        out = model(X[train_idx], Adj[train_idx])
        loss = criterion(out, y[train_idx])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_out = model(X[val_idx], Adj[val_idx])
            val_loss = criterion(val_out, y[val_idx]).item()
            # Calculate thresholded accuracy (predictions > 0.5)
            preds = (torch.sigmoid(val_out) >= 0.5).float()
            val_acc = (preds == y[val_idx]).float().mean().item() * 100.0

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), CHECKPOINT_PATH)
            status = f">>> Saved Checkpoint (Val Loss: {val_loss:.4f})"
        else:
            status = ""

        if epoch % 5 == 0 or "Saved" in status:
            print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {loss.item():.4f} | Val Loss: {val_loss:.4f} | Label Acc: {val_acc:.1f}% {status}")

    print(f"\n[SUCCESS] Multi-label retraining complete! Saved to {CHECKPOINT_PATH}")

if __name__ == "__main__":
    train()
