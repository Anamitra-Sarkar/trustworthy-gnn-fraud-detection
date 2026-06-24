"""
Kaggle Training Script: EDL (Evidential Deep Learning) variant.
Trains GNN backbones with Dirichlet-based evidential loss.
"""

import os
import json
import subprocess
import sys

def install_deps():
    print("Installing dependencies...", flush=True)
    try:
        import torch
        if torch.cuda.is_available():
            major, _ = torch.cuda.get_device_capability(0)
            if major < 7:
                print(f"CUDA capability is {major}.x (< 7.0). Reinstalling official PyTorch from PyPI with CUDA 12.1...", flush=True)
                subprocess.check_call([sys.executable, "-m", "pip", "install", "--force-reinstall", "torch", "torchvision", "torchaudio", "--index-url", "https://download.pytorch.org/whl/cu121"])
    except Exception as e:
        print(f"PyTorch check/reinstall failed: {e}", flush=True)
    
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
        "torch-geometric", "safetensors", "huggingface_hub", "scikit-learn"])
    print("Dependencies installed successfully!", flush=True)

install_deps()

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, roc_auc_score
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv, GATConv, GCNConv, GraphNorm, InstanceNorm
from safetensors.torch import save_file
from huggingface_hub import HfApi, login

HF_TOKEN = "PLACEHOLDER_HF_TOKEN"
if HF_TOKEN == "PLACEHOLDER_HF_TOKEN":
    HF_TOKEN = os.environ.get("HF_TOKEN", "")
if not HF_TOKEN:
    try:
        from kaggle_secrets import UserSecretsClient
        HF_TOKEN = UserSecretsClient().get_secret("HF_TOKEN")
    except Exception:
        pass

if not HF_TOKEN:
    local_path = "/home/anamitra/Downloads/API_Keys_and_Secrets/hf_token"
    if os.path.exists(local_path):
        try:
            with open(local_path) as f:
                HF_TOKEN = f.read().strip()
        except Exception:
            pass

if not HF_TOKEN:
    raise ValueError("HF_TOKEN is not set in environment or Kaggle User Secrets!")

HF_REPO = "Arko007/trustworthy-gnn-fraud-models"
DATA_DIR = "/kaggle/input/elliptic-data-set/elliptic_bitcoin_dataset"
if not os.path.exists(os.path.join(DATA_DIR, "elliptic_txs_features.csv")):
    DATA_DIR = "/kaggle/input/elliptic-data-set"
if not os.path.exists(os.path.join(DATA_DIR, "elliptic_txs_features.csv")):
    for fallback in [".", "data", "../data", "/home/anamitra/Data_and_Logs"]:
        if os.path.exists(os.path.join(fallback, "elliptic_txs_features.csv")):
            DATA_DIR = fallback
            break

EPOCHS = 200
PATIENCE = 15
LR = 1e-3
HIDDEN_DIM = 128
ANNEALING_EPOCHS = 10

login(token=HF_TOKEN)
api = HfApi()


# ── Reuse data loading from main script ──
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
        if cls == "1": labels[idx] = 1
        elif cls == "2": labels[idx] = 0

    labeled_mask = labels >= 0
    train_mask = np.zeros(num_nodes, dtype=bool)
    val_mask = np.zeros(num_nodes, dtype=bool)
    test_mask = np.zeros(num_nodes, dtype=bool)
    for idx in range(num_nodes):
        t = timesteps[idx]
        if labeled_mask[idx]:
            if t <= 34: train_mask[idx] = True
            elif t <= 42: val_mask[idx] = True
            else: test_mask[idx] = True

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
    return data


# ── EDL Backbone (outputs evidence via Softplus) ──
class EDLBackbone(nn.Module):
    def __init__(self, base_cls, in_ch, hid=128, num_classes=2, layers=3, drop=0.3, **kwargs):
        super().__init__()
        self.base = base_cls(in_ch, hid, hid, layers - 1, drop, **kwargs)
        self.evidence_head = nn.Linear(hid, num_classes)
        self.num_classes = num_classes

    def forward(self, x, ei):
        h = self.base(x, ei)
        evidence = F.softplus(self.evidence_head(h))
        alpha = evidence + 1.0
        return alpha


# ── EDL Loss ──
def edl_loss(alpha, targets, epoch, num_classes=2, annealing_epochs=10):
    S = torch.sum(alpha, dim=1, keepdim=True)
    p_hat = alpha / S
    one_hot = F.one_hot(targets, num_classes).float()

    mse = torch.sum((one_hot - p_hat) ** 2, dim=1)
    variance = torch.sum(p_hat * (1 - p_hat) / (S + 1), dim=1)
    bayes_risk = mse + variance

    lambda_t = min(1.0, epoch / annealing_epochs)
    alpha_tilde = one_hot + (1 - one_hot) * alpha
    alpha_0 = torch.ones_like(alpha)

    S_tilde = torch.sum(alpha_tilde, dim=1, keepdim=True)
    S_0 = torch.sum(alpha_0, dim=1, keepdim=True)
    kl = (
        torch.lgamma(S_tilde) - torch.lgamma(S_0)
        - torch.sum(torch.lgamma(alpha_tilde), dim=1, keepdim=True)
        + torch.sum(torch.lgamma(alpha_0), dim=1, keepdim=True)
        + torch.sum((alpha_tilde - alpha_0) * (torch.digamma(alpha_tilde) - torch.digamma(S_tilde)), dim=1, keepdim=True)
    ).squeeze(1)

    return torch.mean(bayes_risk + lambda_t * kl)


# ── Base backbones (output hidden dim, not classes) ──
class SAGEBase(nn.Module):
    def __init__(self, in_ch, hid=128, out=128, layers=2, drop=0.3):
        super().__init__()
        self.drop = drop
        self.convs = nn.ModuleList([SAGEConv(in_ch, hid)])
        for _ in range(layers - 1):
            self.convs.append(SAGEConv(hid, out))
        for c in self.convs:
            if hasattr(c, 'lin_l'): nn.init.xavier_uniform_(c.lin_l.weight)

    def forward(self, x, ei):
        for c in self.convs:
            x = F.dropout(F.relu(c(x, ei)), p=self.drop, training=self.training)
        return x

class GATBase(nn.Module):
    def __init__(self, in_ch, hid=128, out=128, layers=2, drop=0.3, heads=4):
        super().__init__()
        self.drop = drop
        self.convs = nn.ModuleList([GATConv(in_ch, hid, heads=heads, concat=True)])
        self.norms = nn.ModuleList([GraphNorm(hid * heads)])
        for _ in range(layers - 1):
            self.convs.append(GATConv(hid * heads, hid, heads=heads, concat=True))
            self.norms.append(GraphNorm(hid * heads))
        self.proj = nn.Linear(hid * heads, out)

    def forward(self, x, ei):
        for i, c in enumerate(self.convs):
            x = F.dropout(F.elu(self.norms[i](c(x, ei))), p=self.drop, training=self.training)
        return self.proj(x)

class GCNBase(nn.Module):
    def __init__(self, in_ch, hid=128, out=128, layers=2, drop=0.3):
        super().__init__()
        self.drop = drop
        self.convs = nn.ModuleList([GCNConv(in_ch, hid)])
        self.norms = nn.ModuleList([InstanceNorm(hid)])
        for _ in range(layers - 1):
            self.convs.append(GCNConv(hid, out))
            self.norms.append(InstanceNorm(out))

    def forward(self, x, ei):
        for i, c in enumerate(self.convs):
            x = F.dropout(F.relu(self.norms[i](c(x, ei))), p=self.drop, training=self.training)
        return x


BASES = {"graphsage": SAGEBase, "gat": GATBase, "gcn": GCNBase}


def train_edl(model, data, device, epochs=EPOCHS, patience=PATIENCE):
    model = model.to(device)
    data = data.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=5e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=7, factor=0.5)

    best_f1, best_state, patience_counter = 0, None, 0

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        alpha = model(data.x, data.edge_index)
        loss = edl_loss(alpha[data.train_mask], data.y[data.train_mask], epoch,
                       annealing_epochs=ANNEALING_EPOCHS)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_alpha = model(data.x, data.edge_index)[data.val_mask]
            val_pred = (val_alpha / val_alpha.sum(dim=1, keepdim=True)).argmax(dim=1).cpu().numpy()
            val_true = data.y[data.val_mask].cpu().numpy()
            val_f1 = f1_score(val_true, val_pred, average='macro', zero_division=0)

            S = val_alpha.sum(dim=1)
            vacuity = (2.0 / S).mean().item()

        scheduler.step(val_f1)

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 20 == 0:
            print(f"  Epoch {epoch}: loss={loss.item():.4f} f1={val_f1:.4f} vacuity={vacuity:.4f}")

        if patience_counter >= patience:
            print(f"  Early stopping at epoch {epoch}")
            break

    model.load_state_dict(best_state)
    return model, best_f1


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    data = load_elliptic()
    in_ch = data.x.shape[1]
    model_config_updates = {}

    for backbone_name, BaseCls in BASES.items():
        model_name = f"{backbone_name}_original_elliptic_edl"
        print(f"\n{'='*60}")
        print(f"EDL Training: {model_name}")
        print(f"{'='*60}")

        kwargs = {"heads": 4} if backbone_name == "gat" else {}
        model = EDLBackbone(BaseCls, in_ch, HIDDEN_DIM, 2, 3, 0.3, **kwargs)
        model, best_f1 = train_edl(model, data, device)

        model.eval()
        with torch.no_grad():
            alpha = model(data.x.to(device), data.edge_index.to(device))
            test_alpha = alpha[data.test_mask].cpu()
            S = test_alpha.sum(dim=1)
            probs = test_alpha / S.unsqueeze(1)
            pred = probs.argmax(dim=1).numpy()
            true = data.y[data.test_mask].numpy()

        metrics = {
            "f1_macro": float(f1_score(true, pred, average='macro', zero_division=0)),
            "mean_vacuity": float((2.0 / S).mean().item()),
        }
        print(f"Test: {json.dumps(metrics, indent=2)}")

        state_dict = {k: v for k, v in model.state_dict().items()}
        filename = f"{model_name}.safetensors"
        save_file(state_dict, filename)
        api.upload_file(path_or_fileobj=filename, path_in_repo=filename, repo_id=HF_REPO)

        model_config_updates[model_name] = {
            "backbone": backbone_name,
            "dataset": "elliptic",
            "topology": "original",
            "edl": True,
            "metrics": metrics,
        }

    existing.update(model_config_updates)
    with open("model_config.json", "w") as f:
        json.dump(existing, f, indent=2)
    api.upload_file(path_or_fileobj="model_config.json", path_in_repo="model_config.json", repo_id=HF_REPO)
    print("\nEDL training complete!")

    # ── Conformal Calibration for All Models ──
    print("\n" + "=" * 60)
    print("CONFORMAL CALIBRATION")
    print("=" * 60)

    # GNN backbone definitions for loading weights
    class GraphSAGEModel(nn.Module):
        def __init__(self, in_ch, hid=128, out=2, layers=3, drop=0.3):
            super().__init__()
            self.drop = drop
            self.convs = nn.ModuleList([SAGEConv(in_ch, hid)])
            for _ in range(layers - 2):
                self.convs.append(SAGEConv(hid, hid))
            self.convs.append(SAGEConv(hid, out))
            for c in self.convs:
                if hasattr(c, 'lin_l') and getattr(c, 'lin_l', None) is not None:
                    nn.init.xavier_uniform_(c.lin_l.weight)
                if hasattr(c, 'lin_r') and getattr(c, 'lin_r', None) is not None:
                    nn.init.xavier_uniform_(c.lin_r.weight)
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
                if hasattr(c, 'lin_src') and getattr(c, 'lin_src', None) is not None:
                    nn.init.xavier_uniform_(c.lin_src.weight)
        def forward(self, x, ei):
            for i, c in enumerate(self.convs[:-1]):
                x = F.dropout(F.elu(self.norms[i](c(x, ei))), p=self.drop, training=self.training)
            return self.convs[-1](x, ei)

    class GCNModel(nn.Module):
        def __init__(self, in_ch, hid=128, out=2, layers=3, drop=0.3):
            super().__init__()
            self.drop = drop
            self.convs = nn.ModuleList([GCNConv(in_ch, hid)])
            self.norms = nn.ModuleList([InstanceNorm(hid)])
            for _ in range(layers - 2):
                self.convs.append(GCNConv(hid, hid))
                self.norms.append(InstanceNorm(hid))
            self.convs.append(GCNConv(hid, out))
        def forward(self, x, ei):
            for i, c in enumerate(self.convs[:-1]):
                x = F.dropout(F.relu(self.norms[i](c(x, ei))), p=self.drop, training=self.training)
            return self.convs[-1](x, ei)

    GNN_BACKBONES = {
        "graphsage": GraphSAGEModel,
        "gat": GATModel,
        "gcn": GCNModel,
    }

    # Pre-build graph topologies for evaluation
    x_np = data.x.numpy()
    timesteps = data.timesteps
    print("Pre-building graph topologies for calibration...")
    topologies = {
        "original": data.edge_index,
        "knn": build_knn_graph(x_np, k=5),
        "temporal": build_temporal_snap_graph(x_np, timesteps, k=3),
        "similarity": build_similarity_graph(x_np, threshold=0.92),
        "augmented": build_augmented_graph(data.edge_index, x_np),
    }

    calibration_data = {}
    ALPHA = 0.1

    # Reload the full unified model config from HF
    try:
        from huggingface_hub import hf_hub_download
        cfg_path = hf_hub_download(repo_id=HF_REPO, filename="model_config.json", token=HF_TOKEN)
        with open(cfg_path) as f:
            full_config = json.load(f)
    except Exception as e:
        print(f"Failed to load final config: {e}")
        full_config = existing

    for model_name, info in full_config.items():
        print(f"\nCalibrating {model_name}...")
        is_edl = info.get("edl", False)
        backbone_name = info["backbone"]
        topo_name = info["topology"]
        
        # Load topology
        topo_ei = topologies[topo_name]

        # Instantiate model
        if is_edl:
            BaseCls = BASES[backbone_name]
            kwargs = {"heads": 4} if backbone_name == "gat" else {}
            model = EDLBackbone(BaseCls, in_ch, HIDDEN_DIM, 2, 3, 0.3, **kwargs)
        else:
            model = GNN_BACKBONES[backbone_name](in_ch, HIDDEN_DIM, 2, 3, 0.3)

        # Download weights
        try:
            weights_path = hf_hub_download(repo_id=HF_REPO, filename=f"{model_name}.safetensors", token=HF_TOKEN)
            from safetensors.torch import load_file
            state_dict = load_file(weights_path)
            model.load_state_dict(state_dict)
        except Exception as e:
            print(f"  Skipping calibration of {model_name} (weights not found): {e}")
            continue

        model = model.to(device).eval()
        data_dev = data.to(device)

        with torch.no_grad():
            if is_edl:
                alpha = model(data_dev.x, topo_ei.to(device))
                val_alpha = alpha[data.val_mask].cpu()
                val_S = val_alpha.sum(dim=1)
                val_probs = (val_alpha / val_S.unsqueeze(1)).numpy()
                
                test_alpha = alpha[data.test_mask].cpu()
                test_S = test_alpha.sum(dim=1)
                test_probs = (test_alpha / test_S.unsqueeze(1)).numpy()
            else:
                logits = model(data_dev.x, topo_ei.to(device))
                probs = F.softmax(logits, dim=-1).cpu().numpy()
                val_probs = probs[data.val_mask.numpy()]
                test_probs = probs[data.test_mask.numpy()]

        val_labels = data.y[data.val_mask].numpy()
        n = len(val_labels)

        # APS scores
        scores = np.array([compute_aps_score(val_probs[i], val_labels[i]) for i in range(n)])
        q_level = min(np.ceil((n + 1) * (1 - ALPHA)) / n, 1.0)
        quantile_threshold = float(np.quantile(scores, q_level))

        # Mondrian APS
        mondrian = {}
        for c in range(2):
            mask = val_labels == c
            if mask.sum() > 0:
                c_scores = scores[mask]
                n_c = len(c_scores)
                q_c = min(np.ceil((n_c + 1) * (1 - ALPHA)) / n_c, 1.0)
                mondrian[str(c)] = float(np.quantile(c_scores, q_c))

        # Test coverage
        test_labels = data.y[data.test_mask].numpy()
        covered = 0
        for i in range(len(test_labels)):
            p = test_probs[i]
            xi = np.random.uniform(0, 1)
            pred_set = []
            for k in range(2):
                sorted_p = np.sort(p)[::-1]
                score_k = np.sum(sorted_p[sorted_p > p[k]]) + xi * p[k]
                if score_k <= quantile_threshold:
                    pred_set.append(k)
            if not pred_set:
                pred_set = [int(np.argmax(p))]
            if test_labels[i] in pred_set:
                covered += 1
        coverage = covered / len(test_labels)

        calibration_data[model_name] = {
            "quantile_threshold": quantile_threshold,
            "mondrian_thresholds": mondrian,
            "alpha": ALPHA,
            "coverage_rate": coverage,
            "calibration_size": n,
        }
        print(f"  Calibrated. Threshold: {quantile_threshold:.4f}, Mondrian: {mondrian}, Coverage: {coverage:.4f}")

    # Upload conformal calibration
    cal_path = "conformal_calibration.json"
    with open(cal_path, "w") as f:
        json.dump(calibration_data, f, indent=2)
    api.upload_file(path_or_fileobj=cal_path, path_in_repo=cal_path, repo_id=HF_REPO)
    print(f"\nCalibration data successfully uploaded to {HF_REPO}!")


# ── Graph Topology Builders for Conformal Calibration ──
def build_similarity_graph(x, threshold=0.92, batch_size=1000):
    from sklearn.metrics.pairwise import cosine_similarity
    num_nodes = x.shape[0]
    src, dst = [], []
    x_f32 = x.astype(np.float32)
    for start in range(0, num_nodes, batch_size):
        end = min(start + batch_size, num_nodes)
        sim = cosine_similarity(x_f32[start:end], x_f32)
        rows, cols = np.where(sim > threshold)
        rows += start
        mask = rows != cols
        src.extend(rows[mask].tolist())
        dst.extend(cols[mask].tolist())
    if not src:
        return torch.zeros((2, 0), dtype=torch.long)
    return torch.tensor([src, dst], dtype=torch.long)

def build_knn_graph(x, k=5):
    from sklearn.neighbors import NearestNeighbors
    nn_model = NearestNeighbors(n_neighbors=k + 1, metric='euclidean', n_jobs=-1)
    nn_model.fit(x)
    _, indices = nn_model.kneighbors(x)
    num_nodes = len(x)
    row_indices = np.repeat(np.arange(num_nodes), k)
    col_indices = indices[:, 1:].flatten()
    src = np.concatenate([row_indices, col_indices])
    dst = np.concatenate([col_indices, row_indices])
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    return torch.unique(edge_index, dim=1)

def build_temporal_snap_graph(x, timesteps, k=3):
    from sklearn.neighbors import NearestNeighbors
    unique_times = np.sort(np.unique(timesteps))
    src_list, dst_list = [], []
    for i in range(len(unique_times) - 1):
        idx_curr = np.where(timesteps == unique_times[i])[0]
        idx_next = np.where(timesteps == unique_times[i + 1])[0]
        if len(idx_next) == 0 or len(idx_curr) == 0:
            continue
        actual_k = min(k, len(idx_next))
        nn_model = NearestNeighbors(n_neighbors=actual_k, metric='euclidean', n_jobs=-1)
        nn_model.fit(x[idx_next])
        _, indices = nn_model.kneighbors(x[idx_curr])
        gi = np.repeat(idx_curr, actual_k)
        gj = idx_next[indices.flatten()]
        src_list.extend(np.concatenate([gi, gj]))
        dst_list.extend(np.concatenate([gj, gi]))
    if not src_list:
        return torch.zeros((2, 0), dtype=torch.long)
    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    return torch.unique(edge_index, dim=1)

def build_augmented_graph(original_ei, x, threshold=0.92):
    sim_ei = build_similarity_graph(x, threshold)
    return torch.unique(torch.cat([original_ei, sim_ei], dim=1), dim=1)

def compute_aps_score(probs, true_label):
    sorted_probs = np.sort(probs)[::-1]
    true_prob = probs[true_label]
    xi = np.random.uniform(0, 1)
    return np.sum(sorted_probs[sorted_probs > true_prob]) + xi * true_prob


if __name__ == "__main__":
    main()
