#!/usr/bin/env python3
"""
Kaggle Training Script: Trustworthy GNN Financial Fraud Detection (v4)
CPU-only training with feature engineering + residual connections + DropEdge + ensemble.
Trains 3 backbones x 2 topologies = 6 models.
"""

import os, json, subprocess, sys, functools, traceback, gc
print = functools.partial(print, flush=True)

def install_deps():
    print("Setting up dependencies...")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "-q",
        "torch-geometric", "safetensors", "huggingface_hub", "scikit-learn",
        "networkx",
    ])
    print("Dependencies ready!")

install_deps()

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv, GATConv, GCNConv
from safetensors.torch import save_file
from huggingface_hub import HfApi, login

HF_TOKEN = os.environ.get("HF_TOKEN", "PLACEHOLDER_HF_TOKEN")
if "PLACEHOLDER" in HF_TOKEN:
    HF_TOKEN = os.environ.get("HF_TOKEN", "")
if not HF_TOKEN:
    try:
        from kaggle_secrets import UserSecretsClient
        HF_TOKEN = UserSecretsClient().get_secret("HF_TOKEN")
    except Exception:
        pass
if not HF_TOKEN:
    raise ValueError("HF_TOKEN not set!")

HF_REPO = "Arko007/trustworthy-gnn-fraud-models"
DATA_DIR = "/kaggle/input/elliptic-data-set/elliptic_bitcoin_dataset"
if not os.path.exists(os.path.join(DATA_DIR, "elliptic_txs_features.csv")):
    DATA_DIR = "/kaggle/input/elliptic-data-set"
if not os.path.exists(os.path.join(DATA_DIR, "elliptic_txs_features.csv")):
    DATA_DIR = "/kaggle/input/ellipticco/elliptic-data-set/elliptic_bitcoin_dataset"
if not os.path.exists(os.path.join(DATA_DIR, "elliptic_txs_features.csv")):
    DATA_DIR = "/kaggle/input/datasets/organizations/ellipticco/elliptic-data-set/elliptic_bitcoin_dataset"

HIDDEN = 128
NUM_LAYERS = 4
DROPOUT = 0.25
EPOCHS = 120
PATIENCE = 18
LR = 5e-4
WEIGHT_DECAY = 1e-4
EDGE_DROP_RATE = 0.1  # DropEdge regularization
LABEL_SMOOTHING = 0.05

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, label_smoothing=0.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing
    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction='none',
                             weight=self.alpha, label_smoothing=self.label_smoothing)
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()

class LayerNorm(nn.Module):
    """Simple LayerNorm that works on (N, F) inputs."""
    def __init__(self, dim):
        super().__init__()
        self.ln = nn.LayerNorm(dim)
    def forward(self, x):
        return self.ln(x)

class GraphSAGE(nn.Module):
    def __init__(self, in_ch, hid=HIDDEN, out=2, layers=NUM_LAYERS, drop=DROPOUT):
        super().__init__()
        self.drop = drop
        self.layers = layers
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.residuals = nn.ModuleList()
        self.convs.append(SAGEConv(in_ch, hid))
        self.norms.append(LayerNorm(hid))
        self.residuals.append(nn.Linear(in_ch, hid) if in_ch != hid else nn.Identity())
        for _ in range(layers - 2):
            self.convs.append(SAGEConv(hid, hid))
            self.norms.append(LayerNorm(hid))
            self.residuals.append(nn.Identity())
        self.convs.append(SAGEConv(hid, out))
    def forward(self, x, ei):
        for i, c in enumerate(self.convs[:-1]):
            h = c(x, ei)
            h = self.norms[i](h)
            h = F.dropout(F.relu(h), p=self.drop, training=self.training)
            x = self.residuals[i](x) + h
        return self.convs[-1](x, ei)

class GAT(nn.Module):
    def __init__(self, in_ch, hid=HIDDEN, out=2, layers=NUM_LAYERS, drop=DROPOUT, heads=4):
        super().__init__()
        self.drop = drop
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.residuals = nn.ModuleList()
        self.convs.append(GATConv(in_ch, hid // heads, heads=heads, concat=True))
        self.norms.append(LayerNorm(hid))
        self.residuals.append(nn.Linear(in_ch, hid) if in_ch != hid else nn.Identity())
        for _ in range(layers - 2):
            self.convs.append(GATConv(hid, hid // heads, heads=heads, concat=True))
            self.norms.append(LayerNorm(hid))
            self.residuals.append(nn.Identity())
        self.convs.append(GATConv(hid, out, heads=1, concat=False))
    def forward(self, x, ei):
        for i, c in enumerate(self.convs[:-1]):
            h = c(x, ei)
            h = self.norms[i](h)
            h = F.dropout(F.elu(h), p=self.drop, training=self.training)
            x = self.residuals[i](x) + h
        return self.convs[-1](x, ei)

class GCN(nn.Module):
    def __init__(self, in_ch, hid=HIDDEN, out=2, layers=NUM_LAYERS, drop=DROPOUT):
        super().__init__()
        self.drop = drop
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.residuals = nn.ModuleList()
        self.convs.append(GCNConv(in_ch, hid))
        self.norms.append(LayerNorm(hid))
        self.residuals.append(nn.Linear(in_ch, hid) if in_ch != hid else nn.Identity())
        for _ in range(layers - 2):
            self.convs.append(GCNConv(hid, hid))
            self.norms.append(LayerNorm(hid))
            self.residuals.append(nn.Identity())
        self.convs.append(GCNConv(hid, out))
    def forward(self, x, ei):
        for i, c in enumerate(self.convs[:-1]):
            h = c(x, ei)
            h = self.norms[i](h)
            h = F.dropout(F.relu(h), p=self.drop, training=self.training)
            x = self.residuals[i](x) + h
        return self.convs[-1](x, ei)

BACKBONES = {"graphsage": GraphSAGE, "gat": GAT, "gcn": GCN}

login(token=HF_TOKEN)
api = HfApi()
api.create_repo(HF_REPO, exist_ok=True)

# ── Feature Engineering ──────────────────────────────────────────────

def add_degree_features(edge_index, num_nodes):
    """Compute in-degree, out-degree, and total degree for each node."""
    deg_in = torch.zeros(num_nodes, dtype=torch.float32)
    deg_out = torch.zeros(num_nodes, dtype=torch.float32)
    deg_out.index_add_(0, edge_index[0], torch.ones(edge_index.shape[1]))
    deg_in.index_add_(0, edge_index[1], torch.ones(edge_index.shape[1]))
    deg = deg_in + deg_out
    # Log scale to handle heavy-tailed degree distributions
    log_deg = torch.log1p(deg)
    log_deg_in = torch.log1p(deg_in)
    log_deg_out = torch.log1p(deg_out)
    return torch.stack([log_deg, log_deg_in, log_deg_out], dim=1)

def add_pagerank(edge_index, num_nodes, alpha=0.85, max_iter=50):
    """Simple PageRank computation using power iteration."""
    n = num_nodes
    deg_out = torch.zeros(n, dtype=torch.float32)
    deg_out.index_add_(0, edge_index[0], torch.ones(edge_index.shape[1]))
    deg_out = deg_out.clamp(min=1)
    pr = torch.ones(n, dtype=torch.float32) / n
    for _ in range(max_iter):
        pr_new = torch.zeros(n, dtype=torch.float32)
        src, dst = edge_index
        pr_new.index_add_(0, dst, pr[src] / deg_out[src])
        pr = (1 - alpha) / n + alpha * pr_new
    return torch.log1p(pr).unsqueeze(1)

def add_neighbor_aggregation(node_features, edge_index, num_nodes):
    """Compute mean neighbor features for each node without unsupported ops."""
    src, dst = edge_index
    n = node_features.shape[0]
    ones = torch.ones(src.shape[0], dtype=torch.float32)
    deg = torch.zeros(n, dtype=torch.float32)
    deg.index_add_(0, dst, ones).clamp_(min=1)
    neighbor_mean = torch.zeros_like(node_features)
    neighbor_mean.index_add_(0, dst, node_features[src])
    neighbor_mean = neighbor_mean / deg.unsqueeze(1)
    diff = node_features[src] - neighbor_mean[dst]
    diff_sq = diff ** 2
    var = torch.zeros_like(node_features)
    var.index_add_(0, dst, diff_sq)
    var = var / deg.unsqueeze(1)
    neighbor_std = torch.sqrt(var.clamp(min=1e-8))

    return torch.cat([neighbor_mean, neighbor_std], dim=1)

def engineer_features(data, timesteps, x_np):
    """Add engineered features to the graph."""
    num_nodes = data.x.shape[0]
    edge_index = data.edge_index
    feat_list = [data.x]

    deg_feat = add_degree_features(edge_index, num_nodes)
    feat_list.append(deg_feat)
    print(f"  Added degree features: {deg_feat.shape[1]} dims")

    pagerank = add_pagerank(edge_index, num_nodes)
    feat_list.append(pagerank)
    print(f"  Added PageRank: {pagerank.shape[1]} dims")

    neigh = add_neighbor_aggregation(data.x, edge_index, num_nodes)
    feat_list.append(neigh)
    print(f"  Added neighbor stats: {neigh.shape[1]} dims")

    new_x = torch.cat(feat_list, dim=1)
    print(f"  Total features: {data.x.shape[1]} -> {new_x.shape[1]}")
    data.x = new_x
    return data

# ── Data Loading ─────────────────────────────────────────────────────

def load_elliptic():
    features_df = pd.read_csv(os.path.join(DATA_DIR, "elliptic_txs_features.csv"), header=None)
    edges_df = pd.read_csv(os.path.join(DATA_DIR, "elliptic_txs_edgelist.csv"))
    classes_df = pd.read_csv(os.path.join(DATA_DIR, "elliptic_txs_classes.csv"))
    node_ids = features_df[0].values
    timesteps = features_df[1].values.astype(int)
    features = features_df.iloc[:, 2:].values.astype(np.float32)
    node_id_map = {nid: idx for idx, nid in enumerate(node_ids)}
    classes_df.columns = ["txId", "class"]
    class_map = dict(zip(classes_df["txId"].values, classes_df["class"].values))
    labels = np.full(len(node_ids), -1, dtype=np.int64)
    for idx, nid in enumerate(node_ids):
        cls = class_map.get(nid, "unknown")
        if cls == "1": labels[idx] = 1
        elif cls == "2": labels[idx] = 0
    labeled = labels >= 0
    train_mask = np.zeros(len(node_ids), dtype=bool)
    val_mask = np.zeros(len(node_ids), dtype=bool)
    test_mask = np.zeros(len(node_ids), dtype=bool)
    for idx in range(len(node_ids)):
        t = timesteps[idx]
        if labeled[idx]:
            if t <= 34: train_mask[idx] = True
            elif t <= 42: val_mask[idx] = True
            else: test_mask[idx] = True
    # Use RobustScaler for outlier robustness
    scaler = RobustScaler()
    scaler.fit(features[train_mask])
    features = scaler.transform(features)
    src_col, dst_col = edges_df.columns[0], edges_df.columns[1]
    valid = np.array([(node_id_map[s], node_id_map[d])
        for s, d in zip(edges_df[src_col].values, edges_df[dst_col].values)
        if s in node_id_map and d in node_id_map])
    return Data(
        x=torch.tensor(features, dtype=torch.float32),
        edge_index=torch.tensor(valid.T, dtype=torch.long),
        y=torch.tensor(labels, dtype=torch.long),
        train_mask=torch.tensor(train_mask),
        val_mask=torch.tensor(val_mask),
        test_mask=torch.tensor(test_mask),
    ), timesteps, features

def build_temporal_snap_graph(x, timesteps, k=5):
    """Multi-snapshot temporal graph: link nodes across consecutive timesteps."""
    unique = np.sort(np.unique(timesteps))
    src_list, dst_list = [], []
    for i in range(len(unique) - 1):
        curr = np.where(timesteps == unique[i])[0]
        next_ = np.where(timesteps == unique[i + 1])[0]
        if len(next_) == 0: continue
        actual_k = min(k, len(next_))
        nn_model = NearestNeighbors(n_neighbors=actual_k, metric='euclidean', n_jobs=1)
        nn_model.fit(x[next_])
        _, indices = nn_model.kneighbors(x[curr])
        gi, gj = np.repeat(curr, actual_k), next_[indices.flatten()]
        src_list.extend(np.concatenate([gi, gj]))
        dst_list.extend(np.concatenate([gj, gi]))
    if not src_list: return torch.zeros((2, 0), dtype=torch.long)
    return torch.unique(torch.tensor([src_list, dst_list], dtype=torch.long), dim=1)

# ── DropEdge ─────────────────────────────────────────────────────────

def drop_edge(edge_index, drop_rate=EDGE_DROP_RATE, training=True):
    """Randomly drop edges during training (DropEdge regularization)."""
    if not training or drop_rate <= 0:
        return edge_index
    n = edge_index.shape[1]
    keep = torch.rand(n) > drop_rate
    return edge_index[:, keep]

# ── Training ─────────────────────────────────────────────────────────

def train_model(model, data, device):
    model = model.to(device)
    data = data.to(device)
    num_pos = data.y[data.train_mask].sum().item()
    num_neg = data.train_mask.sum().item() - num_pos
    pos_weight = num_neg / max(num_pos, 1)
    alpha = torch.tensor([1.0, pos_weight], device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # Warmup + Cosine annealing
    warmup = LinearLR(optimizer, start_factor=0.1, total_iters=10)
    cosine = CosineAnnealingLR(optimizer, T_max=EPOCHS - 10)
    scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[10])

    criterion = FocalLoss(alpha=alpha, gamma=2.0, label_smoothing=LABEL_SMOOTHING)
    best_f1 = 0.0
    best_state = None
    cnt = 0
    for epoch in range(EPOCHS):
        model.train()
        optimizer.zero_grad()
        ei_dropped = drop_edge(data.edge_index, EDGE_DROP_RATE, training=True).to(device)
        loss = criterion(model(data.x, ei_dropped)[data.train_mask], data.y[data.train_mask])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
        optimizer.step()
        scheduler.step()
        model.eval()
        with torch.no_grad():
            val_probs = F.softmax(model(data.x, data.edge_index), dim=-1)[data.val_mask, 1].cpu().numpy()
            val_true = data.y[data.val_mask].cpu().numpy()
            best_val_f1 = 0.0
            for thresh in np.arange(0.01, 0.99, 0.01):
                f1v = f1_score(val_true, (val_probs > thresh).astype(int), average='binary', pos_label=1, zero_division=0)
                if f1v > best_val_f1: best_val_f1 = f1v
        if best_val_f1 > best_f1:
            best_f1 = best_val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            cnt = 0
        else: cnt += 1
        if epoch % 10 == 0: print(f"  Epoch {epoch:3d}: loss={loss.item():.4f} val_f1={best_val_f1:.4f} best={best_f1:.4f}")
        if cnt >= PATIENCE:
            print(f"  Early stopping at epoch {epoch}")
            break
    model.load_state_dict(best_state)
    return model, best_f1

def evaluate_model(model, data, device):
    model.eval()
    data = data.to(device)
    with torch.no_grad():
        probs = F.softmax(model(data.x, data.edge_index), dim=-1).cpu().numpy()
    val_probs, val_true = probs[data.val_mask.numpy(), 1], data.y[data.val_mask].numpy()
    best_thresh, best_val_f1 = 0.5, 0.0
    for thresh in np.arange(0.01, 0.99, 0.01):
        f1v = f1_score(val_true, (val_probs > thresh).astype(int), average='binary', pos_label=1, zero_division=0)
        if f1v > best_val_f1: best_val_f1, best_thresh = f1v, thresh
    test_probs, test_true = probs[data.test_mask.numpy(), 1], data.y[data.test_mask].numpy()
    test_pred = (test_probs > best_thresh).astype(int)
    metrics = {"f1_macro": float(f1_score(test_true, test_pred, average='macro', zero_division=0)),
               "f1_fraud": float(f1_score(test_true, test_pred, average='binary', pos_label=1, zero_division=0)),
               "f1_licit": float(f1_score(test_true, test_pred, average='binary', pos_label=0, zero_division=0)),
               "precision_fraud": float(precision_score(test_true, test_pred, pos_label=1, zero_division=0)),
               "recall_fraud": float(recall_score(test_true, test_pred, pos_label=1, zero_division=0)),
               "auc_roc": 0.0}
    try: metrics["auc_roc"] = float(roc_auc_score(test_true, test_probs))
    except: pass
    metrics["best_threshold"] = float(best_thresh)
    return metrics

def compute_aps(probs, true_label):
    sp = np.sort(probs)[::-1]
    xi = np.random.uniform(0, 1)
    return float(np.sum(sp[sp > probs[true_label]]) + xi * probs[true_label])

def calibrate(model, data, device, name):
    model.eval()
    data_dev = data.to(device)
    with torch.no_grad():
        probs = F.softmax(model(data_dev.x, data_dev.edge_index), dim=-1).cpu().numpy()
    vp, vl = probs[data.val_mask.numpy()], data.y[data.val_mask].numpy()
    n = len(vl)
    scores = np.array([compute_aps(vp[i], vl[i]) for i in range(n)])
    ALPHA = 0.1
    ql = min(np.ceil((n + 1) * (1 - ALPHA)) / n, 1.0)
    qt = float(np.quantile(scores, ql))
    mondrian = {}
    for c in range(2):
        mask = vl == c
        if mask.sum() > 0:
            cs = scores[mask]
            nc = len(cs)
            qc = min(np.ceil((nc + 1) * (1 - ALPHA)) / nc, 1.0)
            mondrian[str(c)] = float(np.quantile(cs, qc))
    tp, tl = probs[data.test_mask.numpy()], data.y[data.test_mask].numpy()
    covered = 0
    for i in range(len(tl)):
        p, xi = tp[i], np.random.uniform(0, 1)
        ps = []
        for k in range(2):
            sp = np.sort(p)[::-1]
            sk = float(np.sum(sp[sp > p[k]]) + xi * p[k])
            if sk <= qt: ps.append(k)
        if not ps: ps = [int(np.argmax(p))]
        if tl[i] in ps: covered += 1
    return {"quantile_threshold": qt, "mondrian_thresholds": mondrian, "alpha": ALPHA,
            "coverage_rate": float(covered / len(tl)), "calibration_size": n}

# ── Ensemble ─────────────────────────────────────────────────────────

def ensemble_predict(models, data, device):
    """Average predictions from all models."""
    all_probs = []
    for model in models:
        model.eval()
        model = model.to(device)
        with torch.no_grad():
            probs = F.softmax(model(data.x.to(device), data.edge_index.to(device)), dim=-1).cpu().numpy()
        all_probs.append(probs[:, 1])
        model.cpu()
        gc.collect()
    avg_probs = np.mean(all_probs, axis=0)
    return avg_probs

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        try:
            major, _ = torch.cuda.get_device_capability(0)
            if major < 7:
                print(f"WARNING: GPU sm_{major} < 7.0 incompatible. Falling back to CPU.")
                device = torch.device("cpu")
            else:
                print(f"GPU: {torch.cuda.get_device_name(0)} (sm_{major})")
        except Exception as e:
            print(f"GPU check failed: {e}. Falling back to CPU.")
            device = torch.device("cpu")
    print(f"Device: {device}")

    print("Loading Elliptic dataset...")
    data, timesteps, x_np = load_elliptic()
    print(f"Nodes: {data.x.shape[0]}, Edges: {data.edge_index.shape[1]}")
    print(f"Train: {data.train_mask.sum()}, Val: {data.val_mask.sum()}, Test: {data.test_mask.sum()}")
    print(f"Fraud ratio: {data.y[data.train_mask].float().mean():.4f}")

    print("\nFeature engineering...")
    data = engineer_features(data, timesteps, x_np)

    print("\nBuilding topologies...")
    topologies = {"original": data.edge_index}
    print("  temporal...")
    topologies["temporal"] = build_temporal_snap_graph(x_np, timesteps, k=3)
    print(f"  Edges: {topologies['temporal'].shape[1]}")

    in_channels = data.x.shape[1]
    model_config, all_metrics, cal_data = {}, {}, {}
    trained_models = []

    for topo_name, edge_index in topologies.items():
        topo_data = Data(x=data.x.clone(), edge_index=edge_index, y=data.y,
            train_mask=data.train_mask, val_mask=data.val_mask, test_mask=data.test_mask)
        for backbone_name, BackboneCls in BACKBONES.items():
            model_name = f"{backbone_name}_{topo_name}_elliptic"
            print(f"\n{'='*60}\nTraining: {model_name}\n{'='*60}")
            try:
                model = BackboneCls(in_channels).to(device)
                model, val_f1 = train_model(model, topo_data, device)
                metrics = evaluate_model(model, topo_data, device)
                all_metrics[model_name] = metrics
                print(f"  Test: F1_macro={metrics['f1_macro']:.4f} AUC={metrics['auc_roc']:.4f} P={metrics['precision_fraud']:.4f} R={metrics['recall_fraud']:.4f} thresh={metrics['best_threshold']:.2f}")
                fn = f"{model_name}.safetensors"
                save_file({k: v.cpu() for k, v in model.state_dict().items()}, fn)
                api.upload_file(path_or_fileobj=fn, path_in_repo=fn, repo_id=HF_REPO)
                cal = calibrate(model, topo_data, device, model_name)
                cal_data[model_name] = cal
                print(f"  Coverage: {cal['coverage_rate']:.4f}")
                model_config[model_name] = {"backbone": backbone_name, "dataset": "elliptic", "topology": topo_name,
                    "edl": False, "hidden_dim": HIDDEN, "num_layers": NUM_LAYERS, "dropout": DROPOUT,
                    "edge_drop": EDGE_DROP_RATE, "loss": "focal_smoothed", "gamma": 2.0,
                    "feature_dim": int(in_channels), "feat_eng": "deg+pagerank+neighbor_mean+neighbor_std",
                    "metrics": metrics}
                trained_models.append((model_name, model, topo_data))
                model.cpu()
                gc.collect()
            except Exception as e:
                print(f"  FAILED: {e}")
                traceback.print_exc()

    # Ensemble evaluation
    if len(trained_models) >= 2:
        print(f"\n{'='*60}\nEnsemble Evaluation ({len(trained_models)} models)\n{'='*60}")
        ref_data = trained_models[0][2]
        avg_probs = ensemble_predict([m for _, m, _ in trained_models], ref_data, device)
        test_true = ref_data.y[ref_data.test_mask].numpy()
        val_true = ref_data.y[ref_data.val_mask].numpy()
        val_probs = avg_probs[ref_data.val_mask.numpy()]
        best_thresh, best_f1 = 0.5, 0.0
        for thresh in np.arange(0.01, 0.99, 0.01):
            f1v = f1_score(val_true, (val_probs > thresh).astype(int), average='binary', pos_label=1, zero_division=0)
            if f1v > best_f1: best_f1, best_thresh = f1v, thresh
        test_pred = (avg_probs[ref_data.test_mask.numpy()] > best_thresh).astype(int)
        ens_metrics = {
            "f1_macro": float(f1_score(test_true, test_pred, average='macro', zero_division=0)),
            "f1_fraud": float(f1_score(test_true, test_pred, average='binary', pos_label=1, zero_division=0)),
            "f1_licit": float(f1_score(test_true, test_pred, average='binary', pos_label=0, zero_division=0)),
            "precision_fraud": float(precision_score(test_true, test_pred, pos_label=1, zero_division=0)),
            "recall_fraud": float(recall_score(test_true, test_pred, pos_label=1, zero_division=0)),
            "best_threshold": float(best_thresh),
        }
        try: ens_metrics["auc_roc"] = float(roc_auc_score(test_true, avg_probs[ref_data.test_mask.numpy()]))
        except: ens_metrics["auc_roc"] = 0.0
        all_metrics["ensemble"] = ens_metrics
        model_config["ensemble"] = {"backbone": "ensemble", "dataset": "elliptic", "topology": "all",
            "edl": False, "n_models": len(trained_models), "feature_dim": int(in_channels), "metrics": ens_metrics}
        print(f"  Ensemble: F1_macro={ens_metrics['f1_macro']:.4f} AUC={ens_metrics['auc_roc']:.4f} P={ens_metrics['precision_fraud']:.4f} R={ens_metrics['recall_fraud']:.4f}")

    with open("model_config.json", "w") as f: json.dump(model_config, f, indent=2)
    api.upload_file(path_or_fileobj="model_config.json", path_in_repo="model_config.json", repo_id=HF_REPO)
    with open("conformal_calibration.json", "w") as f: json.dump(cal_data, f, indent=2)
    api.upload_file(path_or_fileobj="conformal_calibration.json", path_in_repo="conformal_calibration.json", repo_id=HF_REPO)

    print("\n" + "="*60 + "\nRESULTS\n" + "="*60)
    for name, m in sorted(all_metrics.items(), key=lambda x: -x[1]["f1_macro"]):
        print(f"  {name}: F1={m['f1_macro']:.4f} AUC={m.get('auc_roc',0):.4f} P={m.get('precision_fraud',0):.4f} R={m.get('recall_fraud',0):.4f}")

if __name__ == "__main__":
    main()
