import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from app import ProductionNeuroGNN, extract_aal116_graph_from_nii, adjust_tensor_dimensions

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_PATH = "checkpoints/best_multidisease_model.pt"

COHORTS = {
    0: ("Healthy Control", ["data/oasis_raw/*HC*.nii*", "data/oasis_raw/*healthy*.nii*"]),
    1: ("Autism Spectrum Disorder", ["data/abide_raw/*.nii*"]),
    2: ("Alzheimer's Disease", ["data/oasis_raw/*AD*.nii*", "data/oasis_raw/*alzheimer*.nii*"]),
    3: ("Parkinson's Disease", ["data/parkinsons_raw/*.nii*"])
}

def load_and_preprocess_scan(file_path):
    x, adj = extract_aal116_graph_from_nii(file_path)
    x, adj = adjust_tensor_dimensions(x.to(DEVICE), adj.to(DEVICE), target_nodes=116)
    x, adj = x.float(), adj.float()

    if x.ndim == 2:
        x = x.unsqueeze(0)
    if adj.ndim == 2:
        adj = adj.unsqueeze(0)

    x_mean = torch.mean(x, dim=(1, 2), keepdim=True)
    x_std = torch.std(x, dim=(1, 2), keepdim=True) + 1e-6
    x = (x - x_mean) / x_std

    deg = torch.sum(adj, dim=-1, keepdim=True)
    deg[deg == 0] = 1.0
    adj = adj / deg

    return x.squeeze(0), adj.squeeze(0)

def build_dataset():
    print("[*] Extracting AAL-116 connectomes from NIfTI cohort directories...")
    X_list, Adj_list, y_list = [], [], []

    for label, (name, patterns) in COHORTS.items():
        matched_files = []
        for p in patterns:
            matched_files.extend(glob.glob(p))
        matched_files = list(set(matched_files))
        print(f" -> Class {label} ({name}): Found {len(matched_files)} scans")

        for fpath in matched_files:
            try:
                x_tensor, adj_tensor = load_and_preprocess_scan(fpath)
                X_list.append(x_tensor)
                Adj_list.append(adj_tensor)
                y_list.append(label)
            except Exception as e:
                print(f"    [!] Skipping {os.path.basename(fpath)}: {e}")

    X = torch.stack(X_list)
    Adj = torch.stack(Adj_list)
    y = torch.tensor(y_list, dtype=torch.long)
    print(f"[OK] Extracted {len(y)} total scans across {len(set(y.tolist()))} classes.")
    return X, Adj, y

def train():
    os.makedirs("checkpoints", exist_ok=True)
    X, Adj, y = build_dataset()

    n_samples = len(y)
    indices = torch.randperm(n_samples)
    split = int(0.8 * n_samples)
    train_idx, val_idx = indices[:split], indices[split:]

    model = ProductionNeuroGNN().to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    epochs = 40

    print(f"\n[INIT] Beginning training loop on {DEVICE}...")
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        out = model(X[train_idx], Adj[train_idx])
        loss = criterion(out, y[train_idx].to(DEVICE))
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_out = model(X[val_idx], Adj[val_idx])
            val_loss = criterion(val_out, y[val_idx].to(DEVICE)).item()
            preds = torch.argmax(val_out, dim=1)
            val_acc = (preds == y[val_idx].to(DEVICE)).float().mean().item() * 100.0

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), CHECKPOINT_PATH)
            save_msg = f">>> Checkpoint Saved (Val Loss: {val_loss:.4f})"
        else:
            save_msg = ""

        print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {loss.item():.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.1f}% {save_msg}")

    print(f"\n[SUCCESS] Model training complete! Saved to {CHECKPOINT_PATH}")

if __name__ == "__main__":
    train()
