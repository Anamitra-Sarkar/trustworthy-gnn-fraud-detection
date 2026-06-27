#!/usr/bin/env python3
"""
Kaggle Training Script: EDL (Evidential Deep Learning) variant. (v3)
CPU-only. Uses default Kaggle PyTorch.
Trains 3 EDL models + runs conformal calibration for all HF models.
"""

import os, json, subprocess, sys, functools, traceback
print = functools.partial(print, flush=True)

def install_deps():
    print("Setting up dependencies...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
        "torch-geometric", "safetensors", "huggingface_hub", "scikit-learn"])
    print("Dependencies ready!")

install_deps()

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv, GATConv, GCNConv, LayerNorm
from safetensors.torch import save_file
from huggingface_hub import HfApi, hf_hub_download, login

HF_TOKEN = os.environ.get("HF_TOKEN", "PLACEHOLDER_HF_TOKEN")
if "PLACEHOLDER" in HF_TOKEN:
    HF_TOKEN = os.environ.get("HF_TOKEN", "")
if not HF_TOKEN:
    try:
        from kaggle_secrets import UserSecretsClient
        HF_TOKEN = UserSecretsClient().get_secret("HF_TOKEN")
    except Exception: pass
if not HF_TOKEN: raise ValueError("HF_TOKEN not set!")

HF_REPO = "Arko007/trustworthy-gnn-fraud-models"
DATA_DIR = "/kaggle/input/elliptic-data-set/elliptic_bitcoin_dataset"
if not os.path.exists(os.path.join(DATA_DIR, "elliptic_txs_features.csv")):
    DATA_DIR = "/kaggle/input/elliptic-data-set"
if not os.path.exists(os.path.join(DATA_DIR, "elliptic_txs_features.csv")):
    DATA_DIR = "/kaggle/input/ellipticco/elliptic-data-set/elliptic_bitcoin_dataset"
if not os.path.exists(os.path.join(DATA_DIR, "elliptic_txs_features.csv")):
    DATA_DIR = "/kaggle/input/datasets/organizations/ellipticco/elliptic-data-set/elliptic_bitcoin_dataset"

HIDDEN = 128
NUM_LAYERS = 3
DROPOUT = 0.3
EPOCHS = 200
PATIENCE = 20
LR = 5e-4
WEIGHT_DECAY = 5e-4
ANNEALING_EPOCHS = 10

login(token=HF_TOKEN)
api = HfApi()
api.create_repo(HF_REPO, exist_ok=True)

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
    scaler = StandardScaler()
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
    )

class SAGEBase(nn.Module):
    def __init__(self, in_ch, hid=HIDDEN, out=128, layers=2, drop=DROPOUT):
        super().__init__()
        self.drop = drop
        self.convs = nn.ModuleList([SAGEConv(in_ch, hid)])
        self.norms = nn.ModuleList([LayerNorm(hid)])
        for _ in range(layers - 1):
            self.convs.append(SAGEConv(hid, out))
            self.norms.append(LayerNorm(out))
    def forward(self, x, ei):
        for i, c in enumerate(self.convs):
            x = F.dropout(F.relu(self.norms[i](c(x, ei))), p=self.drop, training=self.training)
        return x

class GATBase(nn.Module):
    def __init__(self, in_ch, hid=HIDDEN, out=128, layers=2, drop=DROPOUT, heads=4):
        super().__init__()
        self.drop = drop
        self.convs = nn.ModuleList([GATConv(in_ch, hid // heads, heads=heads, concat=True)])
        self.norms = nn.ModuleList([LayerNorm(hid)])
        for _ in range(layers - 1):
            self.convs.append(GATConv(hid, hid // heads, heads=heads, concat=True))
            self.norms.append(LayerNorm(hid))
        self.proj = nn.Linear(hid, out)
    def forward(self, x, ei):
        for i, c in enumerate(self.convs):
            x = F.dropout(F.elu(self.norms[i](c(x, ei))), p=self.drop, training=self.training)
        return self.proj(x)

class GCNBase(nn.Module):
    def __init__(self, in_ch, hid=HIDDEN, out=128, layers=2, drop=DROPOUT):
        super().__init__()
        self.drop = drop
        self.convs = nn.ModuleList([GCNConv(in_ch, hid)])
        self.norms = nn.ModuleList([LayerNorm(hid)])
        for _ in range(layers - 1):
            self.convs.append(GCNConv(hid, out))
            self.norms.append(LayerNorm(out))
    def forward(self, x, ei):
        for i, c in enumerate(self.convs):
            x = F.dropout(F.relu(self.norms[i](c(x, ei))), p=self.drop, training=self.training)
        return x

BASES = {"graphsage": SAGEBase, "gat": GATBase, "gcn": GCNBase}

class EDLBackbone(nn.Module):
    def __init__(self, base_cls, in_ch, hid=HIDDEN, num_classes=2, layers=NUM_LAYERS, drop=DROPOUT, **kwargs):
        super().__init__()
        self.base = base_cls(in_ch, hid, hid, layers - 1, drop, **kwargs)
        self.evidence_head = nn.Linear(hid, num_classes)
    def forward(self, x, ei):
        return F.softplus(self.evidence_head(self.base(x, ei))) + 1.0

class GraphSAGE(nn.Module):
    def __init__(self, in_ch, hid=HIDDEN, out=2, layers=NUM_LAYERS, drop=DROPOUT):
        super().__init__()
        self.drop = drop
        self.convs = nn.ModuleList([SAGEConv(in_ch, hid)])
        self.norms = nn.ModuleList([LayerNorm(hid)])
        for _ in range(layers - 2):
            self.convs.append(SAGEConv(hid, hid))
            self.norms.append(LayerNorm(hid))
        self.convs.append(SAGEConv(hid, out))
    def forward(self, x, ei):
        for i, c in enumerate(self.convs[:-1]):
            x = F.dropout(F.relu(self.norms[i](c(x, ei))), p=self.drop, training=self.training)
        return self.convs[-1](x, ei)

class GAT(nn.Module):
    def __init__(self, in_ch, hid=HIDDEN, out=2, layers=NUM_LAYERS, drop=DROPOUT, heads=4):
        super().__init__()
        self.drop = drop
        self.convs = nn.ModuleList([GATConv(in_ch, hid // heads, heads=heads, concat=True)])
        self.norms = nn.ModuleList([LayerNorm(hid)])
        for _ in range(layers - 2):
            self.convs.append(GATConv(hid, hid // heads, heads=heads, concat=True))
            self.norms.append(LayerNorm(hid))
        self.convs.append(GATConv(hid, out, heads=1, concat=False))
    def forward(self, x, ei):
        for i, c in enumerate(self.convs[:-1]):
            x = F.dropout(F.elu(self.norms[i](c(x, ei))), p=self.drop, training=self.training)
        return self.convs[-1](x, ei)

class GCN(nn.Module):
    def __init__(self, in_ch, hid=HIDDEN, out=2, layers=NUM_LAYERS, drop=DROPOUT):
        super().__init__()
        self.drop = drop
        self.convs = nn.ModuleList([GCNConv(in_ch, hid)])
        self.norms = nn.ModuleList([LayerNorm(hid)])
        for _ in range(layers - 2):
            self.convs.append(GCNConv(hid, hid))
            self.norms.append(LayerNorm(hid))
        self.convs.append(GCNConv(hid, out))
    def forward(self, x, ei):
        for i, c in enumerate(self.convs[:-1]):
            x = F.dropout(F.relu(self.norms[i](c(x, ei))), p=self.drop, training=self.training)
        return self.convs[-1](x, ei)

GNN_BACKBONES = {"graphsage": GraphSAGE, "gat": GAT, "gcn": GCN}

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
    kl = (torch.lgamma(S_tilde) - torch.lgamma(S_0)
        - torch.sum(torch.lgamma(alpha_tilde), dim=1, keepdim=True)
        + torch.sum(torch.lgamma(alpha_0), dim=1, keepdim=True)
        + torch.sum((alpha_tilde - alpha_0) * (torch.digamma(alpha_tilde) - torch.digamma(S_tilde)), dim=1, keepdim=True)
    ).squeeze(1)
    return torch.mean(bayes_risk + lambda_t * kl)

def train_edl(model, data, device):
    model = model.to(device)
    data = data.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)
    best_f1, best_state, cnt = -1.0, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
    for epoch in range(EPOCHS):
        model.train(); optimizer.zero_grad()
        alpha = model(data.x, data.edge_index)
        loss = edl_loss(alpha[data.train_mask], data.y[data.train_mask], epoch, annealing_epochs=ANNEALING_EPOCHS)
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step(); scheduler.step()
        model.eval()
        with torch.no_grad():
            val_alpha = model(data.x, data.edge_index)[data.val_mask]
            S = val_alpha.sum(dim=1)
            val_pred = (val_alpha / S.unsqueeze(1)).argmax(dim=1).cpu().numpy()
            val_true = data.y[data.val_mask].cpu().numpy()
            val_f1 = f1_score(val_true, val_pred, average='macro', zero_division=0)
            vacuity = float((2.0 / S).mean().item())
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            cnt = 0
        else: cnt += 1
        if epoch % 20 == 0: print(f"  Epoch {epoch:3d}: loss={loss.item():.4f} f1={val_f1:.4f} vacuity={vacuity:.4f}")
        if cnt >= PATIENCE: print(f"  Early stopping at epoch {epoch}"); break
    model.load_state_dict(best_state)
    return model, best_f1

def compute_aps(probs, tl):
    sp = np.sort(probs)[::-1]
    xi = np.random.uniform(0, 1)
    return float(np.sum(sp[sp > probs[tl]]) + xi * probs[tl])

def calibrate(probs, data):
    val_mask = data.val_mask.detach().cpu().numpy()
    test_mask = data.test_mask.detach().cpu().numpy()
    labels = data.y.detach().cpu().numpy()
    vp, vl = probs[val_mask], labels[val_mask]
    n = len(vl)
    if n == 0:
        return {"quantile_threshold": 1.0, "mondrian_thresholds": {}, "alpha": 0.1,
                "coverage_rate": 0.0, "calibration_size": 0}
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
    tp, tl = probs[test_mask], labels[test_mask]
    if len(tl) == 0:
        return {"quantile_threshold": 1.0, "mondrian_thresholds": mondrian, "alpha": ALPHA,
                "coverage_rate": 0.0, "calibration_size": n}
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

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        try:
            major, minor = torch.cuda.get_device_capability(0)
            print(f"GPU: {torch.cuda.get_device_name(0)} (sm_{major}.{minor})")
        except Exception as e:
            print(f"GPU check failed: {e}. Continuing on CUDA because it is available.")
    print(f"Device: {device}")
    data = load_elliptic()
    in_ch = data.x.shape[1]
    print(f"Nodes: {data.x.shape[0]}, Edges: {data.edge_index.shape[1]}, Fraud: {data.y[data.train_mask].float().mean():.4f}")

    updates, cal_updates = {}, {}
    for backbone_name, BaseCls in BASES.items():
        model_name = f"{backbone_name}_original_elliptic_edl"
        print(f"\n{'='*60}\nEDL: {model_name}\n{'='*60}")
        try:
            kwargs = {"heads": 4} if backbone_name == "gat" else {}
            model = EDLBackbone(BaseCls, in_ch, HIDDEN, 2, NUM_LAYERS, DROPOUT, **kwargs)
            model, best_f1 = train_edl(model, data, device)
            model.eval()
            with torch.no_grad():
                alpha = model(data.x.to(device), data.edge_index.to(device))
                test_a = alpha[data.test_mask].cpu()
                S = test_a.sum(dim=1)
                pred = (test_a / S.unsqueeze(1)).argmax(dim=1).cpu().numpy()
                true = data.y[data.test_mask].detach().cpu().numpy()
            metrics = {"f1_macro": float(f1_score(true, pred, average='macro', zero_division=0)),
                       "mean_vacuity": float((2.0 / S).mean().item())}
            print(f"  F1_macro={metrics['f1_macro']:.4f} Vacuity={metrics['mean_vacuity']:.4f}")
            fn = f"{model_name}.safetensors"
            save_file({k: v.cpu() for k, v in model.state_dict().items()}, fn)
            api.upload_file(path_or_fileobj=fn, path_in_repo=fn, repo_id=HF_REPO)
            with torch.no_grad():
                all_alpha = model(data.x.to(device), data.edge_index.to(device))
                all_probs = (all_alpha / all_alpha.sum(dim=1).unsqueeze(1)).cpu().numpy()
            cal = calibrate(all_probs, data)
            cal_updates[model_name] = cal
            print(f"  Coverage: {cal['coverage_rate']:.4f}")
            updates[model_name] = {
                "backbone": backbone_name,
                "dataset": "elliptic",
                "topology": "original",
                "edl": True,
                "hidden_dim": HIDDEN,
                "num_layers": NUM_LAYERS,
                "dropout": DROPOUT,
                "feature_dim": int(in_ch),
                "metrics": metrics,
            }
        except Exception as e:
            print(f"  FAILED: {e}")
            traceback.print_exc()

    existing = {}
    try:
        cfg = hf_hub_download(repo_id=HF_REPO, filename="model_config.json", token=HF_TOKEN)
        with open(cfg) as f: existing = json.load(f)
    except: pass
    existing.update(updates)
    with open("model_config.json", "w") as f: json.dump(existing, f, indent=2)
    api.upload_file(path_or_fileobj="model_config.json", path_in_repo="model_config.json", repo_id=HF_REPO)

    existing_cal = {}
    try:
        calp = hf_hub_download(repo_id=HF_REPO, filename="conformal_calibration.json", token=HF_TOKEN)
        with open(calp) as f: existing_cal = json.load(f)
    except: pass
    existing_cal.update(cal_updates)
    with open("conformal_calibration.json", "w") as f: json.dump(existing_cal, f, indent=2)
    api.upload_file(path_or_fileobj="conformal_calibration.json", path_in_repo="conformal_calibration.json", repo_id=HF_REPO)
    print("\nEDL training complete!")

if __name__ == "__main__":
    main()
