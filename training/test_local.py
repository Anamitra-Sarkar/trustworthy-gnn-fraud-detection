#!/usr/bin/env python3
"""
Local CPU smoke test for Trustworthy GNN Fraud Detection.

This script avoids PyG-native kernels so it can run on a 4 GB RAM laptop.
It still exercises graph feature engineering, threshold tuning, and early stopping.
Run: python3 test_local.py
"""

import functools
import sys
import time

print = functools.partial(print, flush=True)

def check_deps() -> None:
    missing = []
    for pkg in ["numpy", "torch", "sklearn"]:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"Missing deps: {missing}")
        print("Install: pip install " + " ".join(missing))
        sys.exit(1)


check_deps()

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score


HIDDEN = 64
EPOCHS = 40
PATIENCE = 8
LR = 3e-4
WEIGHT_DECAY = 1e-4


def make_synthetic_graph(n=2000, feat_dim=10, fraud_ratio=0.12, edge_prob=0.005):
    np.random.seed(42)
    torch.manual_seed(42)

    n_fraud = int(n * fraud_ratio)
    labels = np.zeros(n, dtype=np.int64)
    labels[:n_fraud] = 1
    np.random.shuffle(labels)

    features = np.random.randn(n, feat_dim).astype(np.float32)
    features[labels == 1] += 1.0
    features[labels == 0, :4] -= 0.25

    src, dst = [], []
    for i in range(n):
        for j in range(i + 1, n):
            if np.random.random() >= edge_prob:
                continue
            if labels[i] == labels[j]:
                if np.random.random() < 0.85:
                    src.extend([i, j])
                    dst.extend([j, i])
            else:
                if np.random.random() < 0.15:
                    src.extend([i, j])
                    dst.extend([j, i])

    fraud_idx = np.where(labels == 1)[0]
    licit_idx = np.where(labels == 0)[0]
    np.random.shuffle(fraud_idx)
    np.random.shuffle(licit_idx)

    def split_ids(idx, n_train, n_val):
        return idx[:n_train], idx[n_train:n_train + n_val], idx[n_train + n_val :]

    n_f = len(fraud_idx)
    n_l = len(licit_idx)
    ft, fv, fte = split_ids(fraud_idx, int(n_f * 0.5), int(n_f * 0.2))
    lt, lv, lte = split_ids(licit_idx, int(n_l * 0.5), int(n_l * 0.2))
    train_idx = np.concatenate([ft, lt])
    val_idx = np.concatenate([fv, lv])
    test_idx = np.concatenate([fte, lte])

    train_mask = np.zeros(n, dtype=bool)
    val_mask = np.zeros(n, dtype=bool)
    test_mask = np.zeros(n, dtype=bool)
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True

    edge_index = np.asarray([src, dst], dtype=np.int64)
    return {
        "x": features,
        "edge_index": edge_index,
        "y": labels,
        "train_mask": train_mask,
        "val_mask": val_mask,
        "test_mask": test_mask,
    }


def add_degree_features(edge_index, num_nodes):
    src, dst = edge_index
    deg_out = np.bincount(src, minlength=num_nodes).astype(np.float32)
    deg_in = np.bincount(dst, minlength=num_nodes).astype(np.float32)
    deg = deg_in + deg_out
    return np.stack([np.log1p(deg), np.log1p(deg_in), np.log1p(deg_out)], axis=1)


def add_pagerank(edge_index, num_nodes, alpha=0.85, max_iter=30):
    src, dst = edge_index
    deg_out = np.bincount(src, minlength=num_nodes).astype(np.float32)
    deg_out[deg_out == 0] = 1.0
    pr = np.full(num_nodes, 1.0 / num_nodes, dtype=np.float32)
    for _ in range(max_iter):
        pr_new = np.zeros(num_nodes, dtype=np.float32)
        np.add.at(pr_new, dst, pr[src] / deg_out[src])
        pr = (1 - alpha) / num_nodes + alpha * pr_new
    return np.log1p(pr)[:, None].astype(np.float32)


def add_neighbor_stats(x, edge_index):
    src, dst = edge_index
    n, d = x.shape
    deg = np.bincount(dst, minlength=n).astype(np.float32)
    deg[deg == 0] = 1.0

    neighbor_mean = np.zeros((n, d), dtype=np.float32)
    np.add.at(neighbor_mean, dst, x[src])
    neighbor_mean /= deg[:, None]

    diff = x[src] - neighbor_mean[dst]
    var = np.zeros((n, d), dtype=np.float32)
    np.add.at(var, dst, diff * diff)
    var /= deg[:, None]
    neighbor_std = np.sqrt(np.maximum(var, 1e-8))
    return np.concatenate([neighbor_mean, neighbor_std], axis=1)


def engineer_features(sample):
    x = sample["x"]
    edge_index = sample["edge_index"]
    feats = [
        x,
        add_degree_features(edge_index, x.shape[0]),
        add_pagerank(edge_index, x.shape[0]),
        add_neighbor_stats(x, edge_index),
    ]
    sample["x"] = np.concatenate(feats, axis=1).astype(np.float32)
    return sample


class MLP(nn.Module):
    def __init__(self, in_dim, hidden=HIDDEN):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, 2),
        )

    def forward(self, x):
        return self.net(x)


def train_quick(model, x, y, train_mask, val_mask):
    device = torch.device("cpu")
    model = model.to(device)
    x = torch.from_numpy(x).to(device)
    y = torch.from_numpy(y).to(device)
    train_mask = torch.from_numpy(train_mask).to(device)
    val_mask = torch.from_numpy(val_mask).to(device)

    num_pos = y[train_mask].sum().item()
    num_neg = int(train_mask.sum().item() - num_pos)
    class_weight = torch.tensor([1.0, max(num_neg / max(num_pos, 1), 1.0)], device=device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_f1 = 0.0
    best_state = None
    wait = 0

    for epoch in range(EPOCHS):
        model.train()
        optimizer.zero_grad()
        logits = model(x)
        loss = F.cross_entropy(logits[train_mask], y[train_mask], weight=class_weight)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
        optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            probs = F.softmax(model(x), dim=-1)[val_mask, 1].cpu().numpy()
            val_true = y[val_mask].cpu().numpy()
            cur_f1 = 0.0
            for thresh in np.arange(0.05, 0.95, 0.02):
                pred = (probs > thresh).astype(int)
                f1v = f1_score(val_true, pred, average="binary", pos_label=1, zero_division=0)
                if f1v > cur_f1:
                    cur_f1 = f1v

        if cur_f1 > best_f1:
            best_f1 = cur_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1

        if epoch % 5 == 0:
            print(f"   Epoch {epoch:02d}: loss={loss.item():.4f} val_f1={cur_f1:.4f} best={best_f1:.4f}")

        if wait >= PATIENCE:
            print(f"   Early stopping at epoch {epoch}")
            break

    model.load_state_dict(best_state)
    return model, best_f1


def evaluate(model, x, y, val_mask, test_mask):
    model.eval()
    with torch.no_grad():
        probs = F.softmax(model(torch.from_numpy(x)), dim=-1).cpu().numpy()[:, 1]

    val_probs = probs[val_mask]
    val_true = y[val_mask]
    best_thresh, best_val_f1 = 0.5, 0.0
    for thresh in np.arange(0.05, 0.95, 0.02):
        pred = (val_probs > thresh).astype(int)
        f1v = f1_score(val_true, pred, average="binary", pos_label=1, zero_division=0)
        if f1v > best_val_f1:
            best_val_f1 = f1v
            best_thresh = thresh

    test_probs = probs[test_mask]
    test_true = y[test_mask]
    test_pred = (test_probs > best_thresh).astype(int)
    return {
        "f1_macro": float(f1_score(test_true, test_pred, average="macro", zero_division=0)),
        "f1_fraud": float(f1_score(test_true, test_pred, average="binary", pos_label=1, zero_division=0)),
        "best_threshold": float(best_thresh),
    }


def main():
    print("=" * 60)
    print("Local Test: Trustworthy GNN Fraud Detection")
    print("Tests feature engineering + CPU smoke training on a synthetic graph")
    print("=" * 60)

    print("\n1. Creating synthetic graph...")
    sample = make_synthetic_graph()
    print(f"   Nodes: {sample['x'].shape[0]}, Edges: {sample['edge_index'].shape[1]}")
    print(f"   Train: {sample['train_mask'].sum()}, Val: {sample['val_mask'].sum()}, Test: {sample['test_mask'].sum()}")
    print(f"   Fraud ratio: {sample['y'][sample['train_mask']].mean():.4f}")

    print("\n2. Running feature engineering...")
    t0 = time.time()
    sample = engineer_features(sample)
    elapsed = time.time() - t0
    feat_mb = sample["x"].nbytes / 1024 / 1024
    print(f"   Feature dims: {sample['x'].shape[1]}")
    print(f"   Feature matrix size: {feat_mb:.2f} MB")
    print(f"   Took: {elapsed:.2f}s")

    print("\n3. Training CPU model...")
    model = MLP(sample["x"].shape[1], hidden=HIDDEN)
    t0 = time.time()
    model, val_f1 = train_quick(
        model,
        sample["x"],
        sample["y"],
        sample["train_mask"],
        sample["val_mask"],
    )
    train_time = time.time() - t0
    metrics = evaluate(model, sample["x"], sample["y"], sample["val_mask"], sample["test_mask"])
    params = sum(p.numel() for p in model.parameters())

    print(f"   Best val F1: {val_f1:.4f}")
    print(f"   Test F1 macro: {metrics['f1_macro']:.4f}")
    print(f"   Test fraud F1: {metrics['f1_fraud']:.4f}")
    print(f"   Best threshold: {metrics['best_threshold']:.2f}")
    print(f"   Parameters: {params:,}")
    print(f"   Train time: {train_time:.1f}s")

    print("\nRESULTS")
    print("=" * 60)
    print(f"  CPU smoke model: val F1={val_f1:.4f}, test fraud F1={metrics['f1_fraud']:.4f}")
    print("\n✓ Local test complete. CPU path is healthy.")


if __name__ == "__main__":
    main()
