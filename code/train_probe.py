"""
Train MLP probe on extracted hidden state features.

Trains a 3-layer MLP classifier to distinguish factual vs hallucinated
hidden states. Uses BCELoss with a final Sigmoid activation.

Usage:
    python code/train_probe.py
"""

import os
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

DATA_DIR = "data"
MODEL_DIR = "models"


class HallucinationProbe(nn.Module):
    """3-layer MLP probe."""

    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def load_features():
    """Load extracted hidden state features and create labels."""
    print("Loading features...")
    factual_feat1 = np.load(os.path.join(DATA_DIR, "factual_feat1_natural.npy"))
    factual_feat2 = np.load(os.path.join(DATA_DIR, "factual_feat2_natural.npy"))
    halluc_feat1 = np.load(os.path.join(DATA_DIR, "halluc_feat1_natural.npy"))
    halluc_feat2 = np.load(os.path.join(DATA_DIR, "halluc_feat2_natural.npy"))

    factual_features = np.concatenate([factual_feat1, factual_feat2], axis=1)
    halluc_features = np.concatenate([halluc_feat1, halluc_feat2], axis=1)

    # Labels: 0 = factual, 1 = hallucinated
    X = np.concatenate([factual_features, halluc_features], axis=0)
    y = np.concatenate(
        [np.zeros(len(factual_features)), np.ones(len(halluc_features))]
    )

    print(f"Total samples: {len(X)}, Feature dim: {X.shape[1]}")
    return X, y


def train_and_evaluate(X, y, epochs=50, batch_size=32, lr=1e-3):
    """Train MLP probe and evaluate on a validation split."""
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.FloatTensor(y_train)
    X_val_t = torch.FloatTensor(X_val)
    y_val_t = torch.FloatTensor(y_val)

    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HallucinationProbe(X.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    best_auc = 0.0

    print(f"\nTraining on {device}...")
    print(f"Train: {len(X_train)} | Val: {len(X_val)}")
    print("-" * 50)

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            pred = model(batch_X)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t.to(device)).cpu().numpy()
            val_auc = roc_auc_score(y_val, val_pred)
            val_acc = accuracy_score(y_val, (val_pred > 0.5).astype(int))

        if val_auc > best_auc:
            best_auc = val_auc
            os.makedirs(MODEL_DIR, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, "probe_natural.pt"))

        if (epoch + 1) % 10 == 0 or epoch == 0:
            avg_loss = train_loss / len(train_loader)
            print(f"Epoch {epoch+1:3d} | Loss: {avg_loss:.4f} | Val AUC: {val_auc:.4f} | Val Acc: {val_acc:.4f}")

    print("-" * 50)
    print(f"Best Val AUC: {best_auc:.4f}")
    return best_auc


def main():
    X, y = load_features()
    best_auc = train_and_evaluate(X, y)
    print(f"\nFinal Best AUC: {best_auc:.4f}")


if __name__ == "__main__":
    main()
