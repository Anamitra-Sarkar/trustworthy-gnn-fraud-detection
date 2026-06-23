"""
Post-training conformal calibration script.
Computes APS scores and Mondrian quantiles on validation set.
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
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv, GATConv, GCNConv, GraphNorm, InstanceNorm
from safetensors.torch import load_file
from huggingface_hub import HfApi, hf_hub_download, login
import torch.nn as nn

HF_TOKEN = os.environ.get("HF_TOKEN", "")
HF_REPO = "Arko007/trustworthy-gnn-fraud-models"
DATA_DIR = "/kaggle/input/elliptic-data-set/elliptic_bitcoin_dataset"
ALPHA = 0.1

login(token=HF_TOKEN)
api = HfApi()


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

    data = Data(
        x=torch.tensor(features, dtype=torch.float32),
        edge_index=torch.tensor(valid_edges.T, dtype=torch.long),
        y=torch.tensor(labels, dtype=torch.long),
        train_mask=torch.tensor(train_mask),
        val_mask=torch.tensor(val_mask),
        test_mask=torch.tensor(test_mask),
    )
    return data


class GraphSAGEModel(nn.Module):
    def __init__(self, in_ch, hid=128, out=2, layers=3, drop=0.3):
        super().__init__()
        self.drop = drop
        self.convs = nn.ModuleList([SAGEConv(in_ch, hid)])
        for _ in range(layers - 2): self.convs.append(SAGEConv(hid, hid))
        self.convs.append(SAGEConv(hid, out))
    def forward(self, x, ei):
        for c in self.convs[:-1]:
            x = F.dropout(F.relu(c(x, ei)), p=self.drop, training=self.training)
        return self.convs[-1](x, ei)

class GATModel(nn.Module):
    def __init__(self, in_ch, hid=128, out=2, layers=3, drop=0.3, heads=4):
        super().__init__()
        self.drop = drop
        self.convs = nn.ModuleList([GATConv(in_ch, hid, heads=heads, concat=True)])
        self.norms = nn.ModuleList([GraphNorm(hid * heads)])
        for _ in range(layers - 2):
            self.convs.append(GATConv(hid * heads, hid, heads=heads, concat=True))
            self.norms.append(GraphNorm(hid * heads))
        self.convs.append(GATConv(hid * heads, out, heads=1, concat=False))
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

BACKBONES = {"graphsage": GraphSAGEModel, "gat": GATModel, "gcn": GCNModel}


def compute_aps_score(probs, true_label):
    sorted_probs = np.sort(probs)[::-1]
    true_prob = probs[true_label]
    xi = np.random.uniform(0, 1)
    return np.sum(sorted_probs[sorted_probs > true_prob]) + xi * true_prob


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_elliptic()
    in_ch = data.x.shape[1]

    cfg_path = hf_hub_download(repo_id=HF_REPO, filename="model_config.json", token=HF_TOKEN)
    with open(cfg_path) as f:
        model_config = json.load(f)

    calibration_data = {}

    for model_name, info in model_config.items():
        if info.get("edl"):
            continue

        backbone_name = info["backbone"]
        print(f"\nCalibrating: {model_name}")

        model = BACKBONES[backbone_name](in_ch, 128, 2, 3, 0.3)
        try:
            weights_path = hf_hub_download(
                repo_id=HF_REPO, filename=f"{model_name}.safetensors", token=HF_TOKEN
            )
            state_dict = load_file(weights_path)
            model.load_state_dict(state_dict)
        except Exception as e:
            print(f"  Skipping {model_name}: {e}")
            continue

        model = model.to(device).eval()
        data_dev = data.to(device)

        with torch.no_grad():
            logits = model(data_dev.x, data_dev.edge_index)
            probs = F.softmax(logits, dim=-1).cpu().numpy()

        val_probs = probs[data.val_mask.numpy()]
        val_labels = data.y[data.val_mask].numpy()
        n = len(val_labels)

        scores = np.array([compute_aps_score(val_probs[i], val_labels[i]) for i in range(n)])

        q_level = np.ceil((n + 1) * (1 - ALPHA)) / n
        q_level = min(q_level, 1.0)
        quantile_threshold = float(np.quantile(scores, q_level))

        # Mondrian: per-class thresholds
        mondrian = {}
        for c in range(2):
            mask = val_labels == c
            if mask.sum() > 0:
                c_scores = scores[mask]
                n_c = len(c_scores)
                q_c = min(np.ceil((n_c + 1) * (1 - ALPHA)) / n_c, 1.0)
                mondrian[str(c)] = float(np.quantile(c_scores, q_c))

        # Test coverage
        test_probs = probs[data.test_mask.numpy()]
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
        print(f"  Threshold: {quantile_threshold:.4f}, Coverage: {coverage:.4f}")

    cal_path = "conformal_calibration.json"
    with open(cal_path, "w") as f:
        json.dump(calibration_data, f, indent=2)
    api.upload_file(path_or_fileobj=cal_path, path_in_repo=cal_path, repo_id=HF_REPO)
    print(f"\nCalibration data uploaded to {HF_REPO}")


if __name__ == "__main__":
    main()
