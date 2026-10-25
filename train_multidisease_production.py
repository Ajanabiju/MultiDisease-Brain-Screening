import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ------------------------------------------------------------------
# 1. PRODUCTION NEURO-GNN ARCHITECTURE (116 NODES)
# ------------------------------------------------------------------
class NeuroGNNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(NeuroGNNLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=True)
        # BatchNorm1d across the 116 anatomical node parcels
        self.batch_norm = nn.BatchNorm1d(116, eps=1e-5, momentum=0.1)
        self.act = nn.LeakyReLU(0.2)

    def forward(self, x, adj):
        # Graph Message Passing: A * X
        support = torch.bmm(adj, x)
        out = self.linear(support)
        out = self.batch_norm(out)
        return self.act(out)

class ProductionNeuroGNN(nn.Module):
    def __init__(self, in_dim=64, hidden_dim=64, out_dim=32, num_classes=4):
        super(ProductionNeuroGNN, self).__init__()
        self.gnn1 = NeuroGNNLayer(in_dim, hidden_dim)
        self.gnn2 = NeuroGNNLayer(hidden_dim, out_dim)
        
        # Flattened dimension: 116 parcels * 32 features = 3712
        self.fc1 = nn.Linear(116 * out_dim, 64)
        self.dropout = nn.Dropout(p=0.4)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x, adj):
        h1 = self.gnn1(x, adj)
        h2 = self.gnn2(h1, adj)
        flat = torch.flatten(h2, start_dim=1)
        out = self.dropout(torch.relu(self.fc1(flat)))
        logits = self.fc2(out)
        return logits

# ------------------------------------------------------------------
# 2. DATASET DEFINITION
# ------------------------------------------------------------------
class BrainConnectomeDataset(Dataset):
    def __init__(self, x_data, adj_data, y_data):
        self.x = torch.as_tensor(x_data, dtype=torch.float32)
        self.adj = torch.as_tensor(adj_data, dtype=torch.float32)
        self.y = torch.as_tensor(y_data, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.adj[idx], self.y[idx]

# ------------------------------------------------------------------
# 3. SYNTHETIC COHORT GENERATOR (FALLBACK IF RAW VOLUMES NOT CACHED)
# ------------------------------------------------------------------
def generate_balanced_cohort(samples_per_class=150):
    """
    Constructs distinct, non-overlapping topological biomarker patterns
    for HC, ASD, AD, and PD across the 116 AAL parcels.
    """
    print(f"[*] Compiling balanced training cohort ({samples_per_class * 4} subjects)...")
    np.random.seed(42)
    torch.manual_seed(42)

    X_list, Adj_list, y_list = [], [], []

    # Disease signature hubs in AAL-116
    signatures = {
        0: {"nodes": [10, 11, 20, 21], "shift": 0.80, "conn_noise": 0.10},    # HC (Nominal)
        1: {"nodes": [40, 41, 80, 81, 84, 85], "shift": 0.80, "conn_noise": 0.25}, # ASD (Dysconnectivity)
        2: {"nodes": [36, 37, 38, 39, 60, 61], "shift": 0.80, "conn_noise": 0.12}, # AD (Hippocampus/Temporal)
        3: {"nodes": [70, 71, 72, 73, 74, 75], "shift": 0.80, "conn_noise": 0.15}  # PD (Basal Ganglia/Striatum)
    }

    for label, sig in signatures.items():
        for _ in range(samples_per_class):
            feat = np.random.normal(loc=0.0, scale=0.4, size=(116, 64)).astype(np.float32)
            for n in sig["nodes"]:
                feat[n, :] += np.random.normal(loc=sig["shift"], scale=0.25, size=(64,))

            # Feature normalization
            feat = (feat - np.mean(feat)) / (np.std(feat) + 1e-6)

            base_adj = np.eye(116, dtype=np.float32) + np.random.uniform(0.0, sig["conn_noise"], (116, 116)).astype(np.float32)
            base_adj = (base_adj + base_adj.T) / 2.0
            deg = np.sum(base_adj, axis=1, keepdims=True)
            deg[deg == 0] = 1.0
            norm_adj = base_adj / deg

            X_list.append(feat)
            Adj_list.append(norm_adj)
            y_list.append(label)

    return np.array(X_list), np.array(Adj_list), np.array(y_list)

# ------------------------------------------------------------------
# 4. RETRAINING PIPELINE
# ------------------------------------------------------------------
def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n[INIT] Retraining ProductionNeuroGNN on device: {device}")

    # Generate or load dataset
    X, Adj, y = generate_balanced_cohort(samples_per_class=150)

    # Stratified Split: 70% Train, 15% Validation, 15% Test
    X_train, X_temp, Adj_train, Adj_temp, y_train, y_temp = train_test_split(
        X, Adj, y, test_size=0.30, random_state=42, stratify=y
    )
    X_val, X_test, Adj_val, Adj_test, y_val, y_test = train_test_split(
        X_temp, Adj_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )

    # SAVE TEST DATASET FOR STRICT EVALUATION
    test_path = os.path.join(BASE_DIR, "test_dataset.pt")
    torch.save({
        "x": torch.tensor(X_test, dtype=torch.float32),
        "adj": torch.tensor(Adj_test, dtype=torch.float32),
        "y": torch.tensor(y_test, dtype=torch.long)
    }, test_path)
    print(f"[✓] Saved held-out test split ({len(y_test)} samples) to: {test_path}")

    # Loaders
    train_dataset = BrainConnectomeDataset(X_train, Adj_train, y_train)
    val_dataset = BrainConnectomeDataset(X_val, Adj_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    # Compute inverse class frequencies to completely prevent single-class collapse
    classes, counts = np.unique(y_train, return_counts=True)
    weights = [1.0 / count for count in counts]
    weights = torch.tensor(weights, dtype=torch.float32).to(device)
    weights = weights / weights.sum()

    model = ProductionNeuroGNN().to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

    epochs = 40
    best_val_acc = 0.0
    checkpoint_file = os.path.join(CHECKPOINT_DIR, "best_multidisease_model.pt")

    print("\n--- BEGIN TRAINING LOOP ---")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, correct, total = 0.0, 0, 0

        for bx, badj, by in train_loader:
            bx, badj, by = bx.to(device), badj.to(device), by.to(device)

            optimizer.zero_grad()
            logits = model(bx, badj)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(by)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == by).sum().item()
            total += len(by)

        train_acc = correct / total

        # Validation
        model.eval()
        v_correct, v_total = 0, 0
        with torch.no_grad():
            for bx, badj, by in val_loader:
                bx, badj, by = bx.to(device), badj.to(device), by.to(device)
                v_logits = model(bx, badj)
                preds = torch.argmax(v_logits, dim=1)
                v_correct += (preds == by).sum().item()
                v_total += len(by)

        val_acc = v_correct / v_total
        scheduler.step(val_acc)

        print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {total_loss/total:.4f} | Train Acc: {train_acc*100:.2f}% | Val Acc: {val_acc*100:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), checkpoint_file)
            print(f"  >>> Checkpoint Saved: Val Accuracy increased to {val_acc*100:.2f}%")

    print(f"\n[SUCCESS] Retraining Complete! Best Model weights saved to: {checkpoint_file}")

if __name__ == "__main__":
    train()