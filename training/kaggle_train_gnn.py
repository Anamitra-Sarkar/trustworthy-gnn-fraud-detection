"""
Kaggle Training Script: Trustworthy GNN Financial Fraud Detection
Run on Kaggle GPU with: kaggle kernels push -p training/

Trains GraphSAGE, GAT, GCN across 4 graph topologies on Elliptic Bitcoin dataset.
Uploads best checkpoints to HuggingFace Hub.
"""

import os
import json
import subprocess
import sys

def install_deps():
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
        "torch-geometric", "torch-scatter", "torch-sparse",
        "safetensors", "huggingface_hub", "scikit-learn"])

install_deps()

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv, GATConv, GCNConv, GraphNorm, InstanceNorm
from safetensors.torch import save_file
from huggingface_hub import HfApi, login

# ── Config ──
HF_TOKEN = os.environ.get("HF_TOKEN", "")
HF_REPO = "Arko007/trustworthy-gnn-fraud-models"
DATA_DIR = "/kaggle/input/elliptic-data-set/elliptic_bitcoin_dataset"
if not os.path.exists(DATA_DIR):
    DATA_DIR = "/kaggle/input/elliptic-data-set"
EPOCHS = 200
PATIENCE = 15
LR = 1e-3
HIDDEN_DIM = 128
NUM_LAYERS = 3
DROPOUT = 0.3

login(token=HF_TOKEN)
api = HfApi()
try:
    api.create_repo(HF_REPO, exist_ok=True)
except Exception:
    pass


# ── Data Loading ──
def load_elliptic():
    features_df = pd.read_csv(os.path.join(DATA_DIR, "elliptic_txs_features.csv"), header=None)
    edges_df = pd.read_csv(os.path.join(DATA_DIR, "elliptic_txs_edgelist.csv"))
    classes_df = pd.read_csv(os.path.join(DATA_DIR, "elliptic_txs_classes.csv"))

    node_ids = features_df[0].values
    timesteps = features_df[1].values.astype(int)
    features = features_df.iloc[:, 2:].values.astype(np.float32)

    node_id_map = {nid: idx for idx, nid in enumerate(node_ids)}
    num_nodes = len(node_ids)

    classes_df.columns = ["txId", "class"]
    class_map = dict(zip(classes_df["txId"].values, classes_df["class"].values))

    labels = np.full(num_nodes, -1, dtype=np.int64)
    for idx, nid in enumerate(node_ids):
        cls = class_map.get(nid, "unknown")
        if cls == "1":
            labels[idx] = 1
        elif cls == "2":
            labels[idx] = 0

    labeled_mask = labels >= 0
    train_mask = np.zeros(num_nodes, dtype=bool)
    val_mask = np.zeros(num_nodes, dtype=bool)
    test_mask = np.zeros(num_nodes, dtype=bool)

    for idx in range(num_nodes):
        t = timesteps[idx]
        if labeled_mask[idx]:
            if t <= 34:
                train_mask[idx] = True
            elif t <= 42:
                val_mask[idx] = True
            else:
                test_mask[idx] = True

    scaler = StandardScaler()
    scaler.fit(features[train_mask])
    features = scaler.transform(features)

    src_col, dst_col = edges_df.columns[0], edges_df.columns[1]
    valid_edges = np.array([
        (node_id_map[s], node_id_map[d])
        for s, d in zip(edges_df[src_col].values, edges_df[dst_col].values)
        if s in node_id_map and d in node_id_map
    ])
    edge_index = torch.tensor(valid_edges.T, dtype=torch.long)

    data = Data(
        x=torch.tensor(features, dtype=torch.float32),
        edge_index=edge_index,
        y=torch.tensor(labels, dtype=torch.long),
        train_mask=torch.tensor(train_mask),
        val_mask=torch.tensor(val_mask),
        test_mask=torch.tensor(test_mask),
    )
    data.timesteps = timesteps
    return data


# ── Graph Topology Builders ──
def build_similarity_graph(x, threshold=0.92, batch_size=5000):
    num_nodes = x.shape[0]
    src, dst = [], []
    for start in range(0, num_nodes, batch_size):
        end = min(start + batch_size, num_nodes)
        sim = cosine_similarity(x[start:end], x)
        rows, cols = np.where(sim > threshold)
        rows += start
        mask = rows != cols
        src.extend(rows[mask].tolist())
        dst.extend(cols[mask].tolist())
    if not src:
        return torch.zeros((2, 0), dtype=torch.long)
    return torch.tensor([src, dst], dtype=torch.long)

def build_knn_graph(x, k=5):
    nn_model = NearestNeighbors(n_neighbors=k + 1, metric='euclidean')
    nn_model.fit(x)
    _, indices = nn_model.kneighbors(x)
    src, dst = [], []
    for i in range(len(x)):
        for j in indices[i, 1:]:
            src.extend([i, j])
            dst.extend([j, i])
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    return torch.unique(edge_index, dim=1)

def build_temporal_snap_graph(x, timesteps, k=3):
    unique_times = np.sort(np.unique(timesteps))
    src, dst = [], []
    for i in range(len(unique_times) - 1):
        idx_curr = np.where(timesteps == unique_times[i])[0]
        idx_next = np.where(timesteps == unique_times[i + 1])[0]
        if len(idx_next) == 0 or len(idx_curr) == 0:
            continue
        actual_k = min(k, len(idx_next))
        nn_model = NearestNeighbors(n_neighbors=actual_k, metric='euclidean')
        nn_model.fit(x[idx_next])
        _, indices = nn_model.kneighbors(x[idx_curr])
        for li, gi in enumerate(idx_curr):
            for lj in indices[li]:
                gj = idx_next[lj]
                src.extend([gi, gj])
                dst.extend([gj, gi])
    if not src:
        return torch.zeros((2, 0), dtype=torch.long)
    return torch.unique(torch.tensor([src, dst], dtype=torch.long), dim=1)

def build_augmented_graph(original_ei, x, threshold=0.92):
    sim_ei = build_similarity_graph(x, threshold)
    return torch.unique(torch.cat([original_ei, sim_ei], dim=1), dim=1)


# ── GNN Backbones ──
class GraphSAGEModel(nn.Module):
    def __init__(self, in_ch, hid=128, out=2, layers=3, drop=0.3):
        super().__init__()
        self.drop = drop
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in_ch, hid))
        for _ in range(layers - 2):
            self.convs.append(SAGEConv(hid, hid))
        self.convs.append(SAGEConv(hid, out))
        for c in self.convs:
            if hasattr(c, 'lin_l'): nn.init.xavier_uniform_(c.lin_l.weight)
            if hasattr(c, 'lin_r'): nn.init.xavier_uniform_(c.lin_r.weight)

    def forward(self, x, ei):
        for c in self.convs[:-1]:
            x = F.dropout(F.relu(c(x, ei)), p=self.drop, training=self.training)
        return self.convs[-1](x, ei)

class GATModel(nn.Module):
    def __init__(self, in_ch, hid=128, out=2, layers=3, drop=0.3, heads=4):
        super().__init__()
        self.drop = drop
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.convs.append(GATConv(in_ch, hid, heads=heads, concat=True))
        self.norms.append(GraphNorm(hid * heads))
        for _ in range(layers - 2):
            self.convs.append(GATConv(hid * heads, hid, heads=heads, concat=True))
            self.norms.append(GraphNorm(hid * heads))
        self.convs.append(GATConv(hid * heads, out, heads=1, concat=False))
        for c in self.convs:
            if hasattr(c, 'lin_src'): nn.init.xavier_uniform_(c.lin_src.weight)

    def forward(self, x, ei):
        for i, c in enumerate(self.convs[:-1]):
            x = F.dropout(F.elu(self.norms[i](c(x, ei))), p=self.drop, training=self.training)
        return self.convs[-1](x, ei)

class GCNModel(nn.Module):
    def __init__(self, in_ch, hid=128, out=2, layers=3, drop=0.3):
        super().__init__()
        self.drop = drop
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.convs.append(GCNConv(in_ch, hid))
        self.norms.append(InstanceNorm(hid))
        for _ in range(layers - 2):
            self.convs.append(GCNConv(hid, hid))
            self.norms.append(InstanceNorm(hid))
        self.convs.append(GCNConv(hid, out))
        for c in self.convs:
            if hasattr(c, 'lin'): nn.init.uniform_(c.lin.weight, -0.1, 0.1)

    def forward(self, x, ei):
        for i, c in enumerate(self.convs[:-1]):
            x = F.dropout(F.relu(self.norms[i](c(x, ei))), p=self.drop, training=self.training)
        return self.convs[-1](x, ei)


BACKBONES = {
    "graphsage": GraphSAGEModel,
    "gat": GATModel,
    "gcn": GCNModel,
}


# ── Training Loop ──
def train_model(model, data, device, epochs=EPOCHS, patience=PATIENCE, lr=LR):
    model = model.to(device)
    data = data.to(device)

    num_pos = data.y[data.train_mask].sum().item()
    num_neg = data.train_mask.sum().item() - num_pos
    pos_weight = torch.tensor([num_neg / max(num_pos, 1)], device=device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=7, factor=0.5)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, pos_weight.item()], device=device))

    best_f1 = 0
    best_state = None
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_out = model(data.x, data.edge_index)
            val_pred = val_out[data.val_mask].argmax(dim=1).cpu().numpy()
            val_true = data.y[data.val_mask].cpu().numpy()
            val_f1 = f1_score(val_true, val_pred, average='macro', zero_division=0)

        scheduler.step(val_f1)

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 20 == 0:
            print(f"  Epoch {epoch}: loss={loss.item():.4f} val_f1={val_f1:.4f} best={best_f1:.4f}")

        if patience_counter >= patience:
            print(f"  Early stopping at epoch {epoch}")
            break

    model.load_state_dict(best_state)
    return model, best_f1


def evaluate_model(model, data, device):
    model.eval()
    data = data.to(device)
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        probs = F.softmax(out, dim=-1)
        pred = probs[data.test_mask].argmax(dim=1).cpu().numpy()
        true = data.y[data.test_mask].cpu().numpy()
        prob_np = probs[data.test_mask].cpu().numpy()

    metrics = {
        "f1_macro": float(f1_score(true, pred, average='macro', zero_division=0)),
        "f1_fraud": float(f1_score(true, pred, average='binary', pos_label=1, zero_division=0)),
        "precision_fraud": float(precision_score(true, pred, pos_label=1, zero_division=0)),
        "recall_fraud": float(recall_score(true, pred, pos_label=1, zero_division=0)),
    }
    try:
        metrics["auc_roc"] = float(roc_auc_score(true, prob_np[:, 1]))
    except ValueError:
        metrics["auc_roc"] = 0.0

    return metrics


# ── Main ──
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading Elliptic Bitcoin dataset...")
    data = load_elliptic()
    print(f"Nodes: {data.x.shape[0]}, Edges: {data.edge_index.shape[1]}")
    print(f"Train: {data.train_mask.sum()}, Val: {data.val_mask.sum()}, Test: {data.test_mask.sum()}")
    print(f"Fraud ratio (train): {data.y[data.train_mask].float().mean():.4f}")

    x_np = data.x.numpy()
    timesteps = data.timesteps

    print("\nBuilding graph topologies...")
    topologies = {"original": data.edge_index}

    print("  Building k-NN graph (k=5)...")
    topologies["knn"] = build_knn_graph(x_np, k=5)
    print(f"  k-NN edges: {topologies['knn'].shape[1]}")

    print("  Building temporal SNAP graph...")
    topologies["temporal"] = build_temporal_snap_graph(x_np, timesteps, k=3)
    print(f"  Temporal edges: {topologies['temporal'].shape[1]}")

    print("  Building similarity graph (threshold=0.92)...")
    topologies["similarity"] = build_similarity_graph(x_np, threshold=0.92)
    print(f"  Similarity edges: {topologies['similarity'].shape[1]}")

    print("  Building augmented graph...")
    topologies["augmented"] = build_augmented_graph(data.edge_index, x_np)
    print(f"  Augmented edges: {topologies['augmented'].shape[1]}")

    in_channels = data.x.shape[1]
    model_config = {}
    all_metrics = {}

    for topo_name, edge_index in topologies.items():
        topo_data = Data(
            x=data.x, edge_index=edge_index, y=data.y,
            train_mask=data.train_mask, val_mask=data.val_mask, test_mask=data.test_mask,
        )

        for backbone_name, BackboneCls in BACKBONES.items():
            model_name = f"{backbone_name}_{topo_name}_elliptic"
            print(f"\n{'='*60}")
            print(f"Training: {model_name}")
            print(f"{'='*60}")

            model = BackboneCls(in_channels, HIDDEN_DIM, 2, NUM_LAYERS, DROPOUT)
            model, best_f1 = train_model(model, topo_data, device)

            metrics = evaluate_model(model, topo_data, device)
            all_metrics[model_name] = metrics
            print(f"Test metrics: {json.dumps(metrics, indent=2)}")

            state_dict = {k: v for k, v in model.state_dict().items()}
            filename = f"{model_name}.safetensors"
            save_file(state_dict, filename)

            api.upload_file(
                path_or_fileobj=filename,
                path_in_repo=filename,
                repo_id=HF_REPO,
            )
            print(f"Uploaded {filename} to {HF_REPO}")

            model_config[model_name] = {
                "backbone": backbone_name,
                "dataset": "elliptic",
                "topology": topo_name,
                "edl": False,
                "metrics": metrics,
            }

    config_path = "model_config.json"
    with open(config_path, "w") as f:
        json.dump(model_config, f, indent=2)
    api.upload_file(path_or_fileobj=config_path, path_in_repo=config_path, repo_id=HF_REPO)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE - Results Summary")
    print("=" * 60)
    for name, m in sorted(all_metrics.items(), key=lambda x: -x[1]["f1_macro"]):
        print(f"{name}: F1={m['f1_macro']:.4f} AUC={m['auc_roc']:.4f} P={m['precision_fraud']:.4f} R={m['recall_fraud']:.4f}")


if __name__ == "__main__":
    main()
