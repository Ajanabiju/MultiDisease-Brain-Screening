import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from src.data.real_multimodal_dataset import UnifiedNeuroGraphDataset

class NeuroGNNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(NeuroGNNLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x, adj):
        # Ensure shape is [Batch, 116, in_features]
        if x.size(1) != 116 and x.size(2) == 116:
            x = x.transpose(1, 2)
            
        support = self.linear(x)          # [Batch, 116, out_features]
        out = torch.bmm(adj, support)     # [Batch, 116, 116] x [Batch, 116, out_features]
        return torch.relu(out)

class NeuroGNNClassifier(nn.Module):
    def __init__(self, num_nodes=116, in_features=32, num_classes=4):
        super(NeuroGNNClassifier, self).__init__()
        self.gnn1 = NeuroGNNLayer(in_features, 64)
        self.gnn2 = NeuroGNNLayer(64, 32)
        self.fc1 = nn.Linear(num_nodes * 32, 64)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x, adj):
        h = self.gnn1(x, adj)
        h = self.gnn2(h, adj)
        h = h.view(h.size(0), -1)
        h = torch.relu(self.fc1(h))
        h = self.dropout(h)
        return self.fc2(h)

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] Training on device: {device}")
    print("[*] Ingesting multimodal cohorts (ABIDE, OASIS, PPMI)...")
    dataset = UnifiedNeuroGraphDataset()
    print(f"[*] Total ready samples: {len(dataset)}")

    # 80/20 train/validation split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=4, shuffle=False)

    model = NeuroGNNClassifier().to(device)
    # Emphasize under-represented recall for Class 1 (ASD) and Class 2 (AD)
    class_weights = torch.tensor([1.0, 2.5, 1.3, 1.0], device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)

    os.makedirs('checkpoints', exist_ok=True)
    best_val_loss = float('inf')
    save_path = 'checkpoints/best_multidisease_model.pt'

    print("--- Starting Training (20 Epochs) ---")
    for epoch in range(1, 21):
        model.train()
        total_train_loss, correct_train, total_train = 0.0, 0, 0
        for batch in train_loader:
            if isinstance(batch, dict):
                x = batch['features'].float().to(device)
                adj = batch['adj'].float().to(device)
                labels = batch['label'].long().to(device)
            else:
                x = batch[0].float().to(device)
                adj = batch[1].float().to(device)
                labels = batch[2].long().to(device)

            # Ensure x is float and shaped [Batch, 116, in_features]
            if x.size(1) != 116 and x.size(2) == 116:
                x = x.transpose(1, 2)

            optimizer.zero_grad()
            outputs = model(x, adj)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item() * x.size(0)
            preds = outputs.argmax(dim=1)
            correct_train += (preds == labels).sum().item()
            total_train += labels.size(0)

        train_loss = total_train_loss / max(1, total_train)
        train_acc = (correct_train / max(1, total_train)) * 100

        model.eval()
        total_val_loss, correct_val, total_val = 0.0, 0, 0
        with torch.no_grad():
            for batch in val_loader:
                if isinstance(batch, dict):
                    x = batch['features'].float().to(device)
                    adj = batch['adj'].float().to(device)
                    labels = batch['label'].long().to(device)
                else:
                    x = batch[0].float().to(device)
                    adj = batch[1].float().to(device)
                    labels = batch[2].long().to(device)

                if x.size(1) != 116 and x.size(2) == 116:
                    x = x.transpose(1, 2)

                outputs = model(x, adj)
                loss = criterion(outputs, labels)
                total_val_loss += loss.item() * x.size(0)
                preds = outputs.argmax(dim=1)
                correct_val += (preds == labels).sum().item()
                total_val += labels.size(0)

        val_loss = total_val_loss / max(1, total_val)
        val_acc = (correct_val / max(1, total_val)) * 100
        print(f"Epoch [{epoch:02d}/20] | Train Loss: {train_loss:.4f} - Acc: {train_acc:.1f}% | Val Loss: {val_loss:.4f} - Acc: {val_acc:.1f}%")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)

    print(f"\n[+] Training complete! Best model saved to: {save_path}")

if __name__ == '__main__':
    train()