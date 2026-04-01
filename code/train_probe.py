"""
Train MLP probe on extracted hidden state features.

Trains a 3-layer MLP classifier to distinguish factual vs hallucinated
hidden states, following MIND's architecture.

Usage:
    python code/train_probe.py --tag _semantic_labeled --epochs 100
    python code/train_probe.py --tag _adversarial_layer14
"""

import argparse
import os

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

DATA_DIR = "data"
MODEL_DIR = "models"


class HallucinationProbe(nn.Module):
    """3-layer MLP probe following MIND's architecture."""

    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 1),
            # Sigmoid removed — BCEWithLogitsLoss applies it internally
            # for numerical stability (log-sum-exp trick).
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def load_features(tag="_adversarial"):
    """Load extracted hidden state features and create labels.

    Args:
        tag: File suffix matching extract_hidden_states.py output.
             Examples: '_adversarial', '_semantic_labeled'
    """
    print(f"Loading features with tag '{tag}'...")
    factual_feat1 = np.load(os.path.join(DATA_DIR, f"factual_feat1{tag}.npy"))
    factual_feat2 = np.load(os.path.join(DATA_DIR, f"factual_feat2{tag}.npy"))
    halluc_feat1 = np.load(os.path.join(DATA_DIR, f"halluc_feat1{tag}.npy"))
    halluc_feat2 = np.load(os.path.join(DATA_DIR, f"halluc_feat2{tag}.npy"))

    factual_features = np.concatenate([factual_feat1, factual_feat2], axis=1)
    halluc_features = np.concatenate([halluc_feat1, halluc_feat2], axis=1)

    # Labels: 0 = factual, 1 = hallucinated
    X = np.concatenate([factual_features, halluc_features], axis=0)
    y = np.concatenate(
        [np.zeros(len(factual_features)), np.ones(len(halluc_features))]
    )

    print(f"Total samples: {len(X)}")
    print(f"  Factual: {len(factual_features)}")
    print(f"  Hallucinated: {len(halluc_features)}")
    print(f"  Feature dimension: {X.shape[1]}")

    return X, y


def train_and_evaluate(
    X, y, tag="_adversarial", epochs=50, batch_size=32, lr=1e-3, test_size=0.15
):
    """Train the MLP probe and evaluate with a 3-way split (Train/Val/Test)."""
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    # ~15% of total for validation as well
    val_size_relative = test_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=val_size_relative,
        random_state=42,
        stratify=y_train_val,
    )

    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.FloatTensor(y_train)
    X_val_t = torch.FloatTensor(X_val)
    y_val_t = torch.FloatTensor(y_val)
    X_test_t = torch.FloatTensor(X_test)
    y_test_t = torch.FloatTensor(y_test)

    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HallucinationProbe(X.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    # pos_weight=2.0 penalises missed hallucinations twice as heavily,
    # resolving the class asymmetry problem (factual recall >> halluc recall).
    pos_weight = torch.tensor([2.0]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_auc = 0.0
    best_epoch = 0
    patience_counter = 0
    patience = 15

    print(f"\nTraining on {device}...")
    print(f"Split: Train={len(X_train)} | Val={len(X_val)} | Test={len(X_test)}")
    print("-" * 60)

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
            val_logits = model(X_val_t.to(device))
            val_pred = torch.sigmoid(val_logits).cpu().numpy()
            val_auc = roc_auc_score(y_val, val_pred)
            val_acc = accuracy_score(y_val, (val_pred > 0.5).astype(int))

        if val_auc > best_auc:
            best_auc = val_auc
            best_epoch = epoch + 1
            patience_counter = 0
            os.makedirs(MODEL_DIR, exist_ok=True)
            model_path = os.path.join(MODEL_DIR, f"probe{tag}.pt")
            torch.save(model.state_dict(), model_path)
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0 or epoch == 0:
            avg_loss = train_loss / len(train_loader)
            print(
                f"Epoch {epoch+1:3d} | Loss: {avg_loss:.4f} | "
                f"Val AUC: {val_auc:.4f} | Early Stop: {patience_counter}/{patience}"
            )

        if patience_counter >= patience:
            print(f"\nEarly stopping triggered at epoch {epoch+1}!")
            break

    print("-" * 60)
    print(f"Best Val AUC: {best_auc:.4f} (Epoch {best_epoch})")

    # Final evaluation on held-out test set
    model_path = os.path.join(MODEL_DIR, f"probe{tag}.pt")
    model.load_state_dict(torch.load(model_path))
    model.eval()
    with torch.no_grad():
        test_logits = model(X_test_t.to(device))
        test_pred = torch.sigmoid(test_logits).cpu().numpy()
        test_auc = roc_auc_score(y_test, test_pred)
        test_acc = accuracy_score(y_test, (test_pred > 0.5).astype(int))

    print(f"\n{'*'*20} FINAL TEST SET RESULTS {'*'*20}")
    print(f"Test AUC: {test_auc:.4f}")
    print(f"Test Acc: {test_acc:.4f}")
    print(f"{'*'*60}\n")

    print("Classification Report (Test Set):")
    print(
        classification_report(
            y_test,
            (test_pred > 0.5).astype(int),
            target_names=["Factual", "Hallucinated"],
        )
    )

    return test_auc


def main():
    parser = argparse.ArgumentParser(
        description="Train MLP probe on hidden state features"
    )
    parser.add_argument("--tag", type=str, default="_adversarial")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--test_size", type=float, default=0.15)
    args = parser.parse_args()

    print(
        f"{'='*60}\nTag: {args.tag} | Epochs: {args.epochs} | LR: {args.lr}\n{'='*60}\n"
    )

    X, y = load_features(tag=args.tag)
    final_auc = train_and_evaluate(
        X, y, tag=args.tag, epochs=args.epochs, lr=args.lr, test_size=args.test_size
    )
    print(f"\nFinal Verified AUC (Test Set): {final_auc:.4f}")


if __name__ == "__main__":
    main()
