import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from pytorch_metric_learning.losses import SupConLoss
from copy import deepcopy

# -------------------------
# Modello MLP -> embedding 32-dim
# -------------------------
class EmbeddingNet(nn.Module):
    def __init__(self, input_dim=86, embed_dim=32,dropout_rate=0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(128, embed_dim)
        )

    def forward(self, x):
        x = self.net(x)
        return F.normalize(x, p=2, dim=1)   # normalizza (cosine)

# -------------------------
# Dataset
# -------------------------
class EmbeddingDataset(Dataset):
    def __init__(self, X, y,device="cpu"):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.device = device

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    def train_embeddings(self, embed_dim=32, epochs=20, batch_size=128, lr=1e-3,weight_decay=1e-4,dropout_rate=0.3,verbose= True):

        loader = DataLoader(self, batch_size=batch_size, shuffle=True)

        model = EmbeddingNet(input_dim=self.X.shape[1], embed_dim=embed_dim,dropout_rate=dropout_rate).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        loss_fn = SupConLoss()  # supervised contrastive loss

        best_loss = float('inf')  # <-- aggiunto
        best_model_state = None  # <-- aggiunto

        for epoch in range(epochs):
            model.train()
            total_loss = 0
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                emb = model(xb)  # (batch, embed_dim)
                loss = loss_fn(emb, yb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(loader)
            if verbose:
                print(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f}")

            # --- SALVA IL MODELLO MIGLIORE ---
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_model_state = model.state_dict().copy()

        # Carica il modello con la loss migliore
        model.load_state_dict(best_model_state)

        return model

    def extract_embeddings(self,model):
        model.eval()
        with torch.no_grad():
            X_new = model(self.X.detach().clone().to(self.device)).cpu().numpy()
        return pd.DataFrame(X_new)


    def train_and_evaluate(self,model, batch_size, lr, epochs, patience=10, weight_decay=1e-4,verbose=True):
        # Split dataset in train e validation
        val_size = int(0.2 * len(self))
        train_size = len(self) - val_size
        train_dataset, val_dataset = random_split(self, [train_size, val_size])

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        loss_fn = torch.nn.CrossEntropyLoss()  # o SupConLoss se vuoi supervised contrastive

        best_model_state = deepcopy(model.state_dict())
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(epochs):
            model.train()
            total_loss = 0
            for xb, yb in train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                emb = model(xb)
                loss = loss_fn(emb, yb)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            avg_train_loss = total_loss / len(train_loader)

            # Validation
            model.eval()
            val_loss_total = 0
            correct = 0
            total = 0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(self.device), yb.to(self.device)
                    emb = model(xb)
                    loss = loss_fn(emb, yb)
                    val_loss_total += loss.item()
                    preds = torch.argmax(emb, dim=1)
                    correct += (preds == yb).sum().item()
                    total += yb.size(0)
            avg_val_loss = val_loss_total / len(val_loader)
            val_acc = correct / total

            # Early stopping check
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_state = deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch + 1}")
                    break
            if verbose:
                print(
                    f"Epoch {epoch + 1}/{epochs} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f} - Val Acc: {val_acc:.4f}")

        # Carica il miglior modello
        model.load_state_dict(best_model_state)
        return model, val_acc






