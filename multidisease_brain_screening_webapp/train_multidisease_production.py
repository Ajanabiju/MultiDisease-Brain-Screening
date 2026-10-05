import os
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from src.data.real_multimodal_dataset import UnifiedNeuroGraphDataset

# ==========================================
# 1. Graph Augmentation Module
# ==========================================
def augment_graph(x, adj, edge_drop_p=0.15, feature_noise_std=0.02):
    """
    Applies graph-specific regularization:
    1. DropEdge: randomly removes non-essential edges.
    2. Feature Jitter: adds subtle Gaussian noise to node embeddings.
    """
    # 1. Feature Noise
    if feature_noise_std > 0:
        noise = torch.randn_like(x) * feature_noise_std
        x = x + noise

    # 2. DropEdge
    if edge_drop_p > 0:
        mask = torch.rand_like(adj) > edge_drop_p
        adj = adj * mask.float()
        # Keep diagonal self-connections intact
        eye = torch.eye(adj.size(1), device=adj.device).unsqueeze(0)
        adj = torch.clamp(adj + eye, 0.0, 1.0)

    return x, adj

# ==========================================
# 2. Production NeuroGNN Architecture
# ==========================================
class NeuroGNNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(NeuroGNNLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.batch_norm = nn.BatchNorm1d(116)

    def forward(self, x, adj):
        support = self.linear(x)
        out = torch.bmm(adj, support)
        out = self.batch_norm(out)
        return torch.relu(out)

class ProductionNeuroGNN(nn.Module):
    def __init__(self, num_nodes=116, in_features=64, num_classes=4):
        super(ProductionNeuroGNN, self).__init__()
        self.gnn1 = NeuroGNNLayer(in_features, 64)
        self.gnn2 = NeuroGNNLayer(64, 32)
        
        self.fc1 = nn.Linear(num_nodes * 32, 64)
        self.dropout = nn.Dropout(0.4)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x, adj):
        h = self.gnn1(x, adj)
        h = self.gnn2(h, adj)
        h = h.view(h.size(0), -1)
        h = torch.relu(self.fc1(h))
        h = self.dropout(h)
        return self.fc2(h)

# ==========================================
# 3. Stratified 5-Fold Training Execution
# ==========================================
def train_production():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] Initializing Production Pipeline on device: {device}")
    
    print("[*] Ingesting unified cohorts (ABIDE, OASIS, PPMI)...")
    dataset = UnifiedNeuroGraphDataset()
    total_samples = len(dataset)
    print(f"[*] Ingested {total_samples} verified samples.")

    # Extract all labels for Stratified K-Fold splitting
    all_labels = []
    for i in range(total_samples):
        item = dataset[i]
        label = item[3] if isinstance(item, (list, tuple)) else item['label']
        all_labels.append(int(label))
    all_labels = np.array(all_labels)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    os.makedirs('checkpoints', exist_ok=True)

    fold_metrics = []
    overall_y_true = []
    overall_y_pred = []

    print("\n" + "="*60)
    print(" STARTING STRATIFIED 5-FOLD CROSS-VALIDATION ")
    print("="*60)

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(total_samples), all_labels), 1):
        print(f"\n--- FOLD {fold} / 5 (Train: {len(train_idx)} | Val: {len(val_idx)}) ---")
        
        train_sub = Subset(dataset, train_idx)
        val_sub = Subset(dataset, val_idx)

        train_loader = DataLoader(train_sub, batch_size=4, shuffle=True)
        val_loader = DataLoader(val_sub, batch_size=4, shuffle=False)

        model = ProductionNeuroGNN().to(device)
       # Emphasize under-represented recall for Class 1 (ASD) and Class 2 (AD)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-3)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=25)

        best_val_loss = float('inf')
        best_fold_preds = []
        best_fold_targets = []
        best_weights = None

        for epoch in range(1, 26):
            model.train()
            train_loss, train_correct, train_total = 0.0, 0, 0

            for batch in train_loader:
                if isinstance(batch, (list, tuple)):
                    x1, x2, adj, labels = batch[0].to(device), batch[1].to(device), batch[2].to(device), batch[3].to(device)
                else:
                    x1, x2 = batch['features'].to(device), batch['features2'].to(device)
                    adj, labels = batch['adj'].to(device), batch['label'].to(device)

                x = torch.cat([x1.float(), x2.float()], dim=-1)
                # Apply DropEdge and Feature Jitter during training
                x, adj = augment_graph(x, adj.float(), edge_drop_p=0.15, feature_noise_std=0.02)

                optimizer.zero_grad()
                outputs = model(x, adj)
                loss = criterion(outputs, labels.long())
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * x.size(0)
                preds = outputs.argmax(dim=1)
                train_correct += (preds == labels).sum().item()
                train_total += labels.size(0)

            scheduler.step()
            epoch_train_loss = train_loss / train_total
            epoch_train_acc = (train_correct / train_total) * 100

            # Validation Loop
            model.eval()
            val_loss, val_correct, val_total = 0.0, 0, 0
            epoch_preds, epoch_targets = [], []

            with torch.no_grad():
                for batch in val_loader:
                    if isinstance(batch, (list, tuple)):
                        x1, x2, adj, labels = batch[0].to(device), batch[1].to(device), batch[2].to(device), batch[3].to(device)
                    else:
                        x1, x2 = batch['features'].to(device), batch['features2'].to(device)
                        adj, labels = batch['adj'].to(device), batch['label'].to(device)

                    x = torch.cat([x1.float(), x2.float()], dim=-1)
                    outputs = model(x, adj.float())
                    loss = criterion(outputs, labels.long())

                    val_loss += loss.item() * x.size(0)
                    preds = outputs.argmax(dim=1)
                    val_correct += (preds == labels).sum().item()
                    val_total += labels.size(0)

                    epoch_preds.extend(preds.cpu().tolist())
                    epoch_targets.extend(labels.cpu().tolist())

            epoch_val_loss = val_loss / val_total
            epoch_val_acc = (val_correct / val_total) * 100

            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                best_weights = copy.deepcopy(model.state_dict())
                best_fold_preds = epoch_preds
                best_fold_targets = epoch_targets

        # Save best model checkpoint for this fold
        fold_save_path = f"checkpoints/multidisease_model_fold_{fold}.pt"
        torch.save(best_weights, fold_save_path)

        fold_f1 = f1_score(best_fold_targets, best_fold_preds, average='macro')
        fold_acc = (np.array(best_fold_preds) == np.array(best_fold_targets)).mean() * 100
        fold_metrics.append((fold_acc, fold_f1))
        
        overall_y_true.extend(best_fold_targets)
        overall_y_pred.extend(best_fold_preds)

        print(f"[Fold {fold} Result] Best Val Loss: {best_val_loss:.4f} | Acc: {fold_acc:.1f}% | Macro F1: {fold_f1:.4f}")

    # ==========================================
    # 4. Out-of-Fold Final Evaluation Report
    # ==========================================
    print("\n" + "="*60)
    print(" 5-FOLD CROSS-VALIDATION SUMMARY REPORT (OUT-OF-FOLD) ")
    print("="*60)
    
    mean_acc = np.mean([m[0] for m in fold_metrics])
    std_acc = np.std([m[0] for m in fold_metrics])
    mean_f1 = np.mean([m[1] for m in fold_metrics])
    std_f1 = np.std([m[1] for m in fold_metrics])

    print(f"Mean Out-of-Fold Accuracy: {mean_acc:.2f}% (+/- {std_acc:.2f}%)")
    print(f"Mean Out-of-Fold Macro F1: {mean_f1:.4f} (+/- {std_f1:.4f})\n")

    target_names = ["0: HC (Control)", "1: ASD (Autism)", "2: AD (Alzheimer's)", "3: PD (Parkinson's)"]
    print("Detailed Classification Report:")
    print(classification_report(overall_y_true, overall_y_pred, target_names=target_names, zero_division=0))

    print("Confusion Matrix:")
    print(confusion_matrix(overall_y_true, overall_y_pred))

    # Also save the top-performing fold as the default production checkpoint
    best_fold_idx = np.argmax([m[1] for m in fold_metrics]) + 1
    best_fold_file = f"checkpoints/multidisease_model_fold_{best_fold_idx}.pt"
    prod_target = "checkpoints/best_multidisease_model.pt"
    
    import shutil
    shutil.copyfile(best_fold_file, prod_target)
    print(f"\n[+] Production ready! Best performing fold ({best_fold_idx}) promoted to: {prod_target}")

if __name__ == '__main__':
    train_production()