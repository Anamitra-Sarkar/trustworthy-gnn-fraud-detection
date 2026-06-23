import os
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data


def load_elliptic(data_dir: str = "data/elliptic_bitcoin_dataset") -> Data:
    """Load Elliptic Bitcoin dataset with temporal split.

    Temporal split: timesteps 1-34 train, 35-42 val, 43-49 test.
    Labels: 0=licit, 1=illicit (fraud). Unknown labels are masked out.
    """
    features_df = pd.read_csv(os.path.join(data_dir, "elliptic_txs_features.csv"), header=None)
    edges_df = pd.read_csv(os.path.join(data_dir, "elliptic_txs_edgelist.csv"))
    classes_df = pd.read_csv(os.path.join(data_dir, "elliptic_txs_classes.csv"))

    node_ids = features_df[0].values
    timesteps = features_df[1].values.astype(int)
    features = features_df.iloc[:, 2:].values.astype(np.float32)  # 166 features (col 0=id, col 1=timestep)

    node_id_map = {nid: idx for idx, nid in enumerate(node_ids)}
    num_nodes = len(node_ids)

    classes_df.columns = ["txId", "class"]
    class_map = dict(zip(classes_df["txId"].values, classes_df["class"].values))

    labels = np.full(num_nodes, -1, dtype=np.int64)
    for idx, nid in enumerate(node_ids):
        cls = class_map.get(nid, "unknown")
        if cls == "1":
            labels[idx] = 1  # illicit
        elif cls == "2":
            labels[idx] = 0  # licit

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

    src_col = edges_df.columns[0]
    dst_col = edges_df.columns[1]
    src = edges_df[src_col].values
    dst = edges_df[dst_col].values

    valid_edges = np.array([
        (node_id_map[s], node_id_map[d])
        for s, d in zip(src, dst)
        if s in node_id_map and d in node_id_map
    ])

    edge_index = torch.tensor(valid_edges.T, dtype=torch.long) if len(valid_edges) > 0 else torch.zeros((2, 0), dtype=torch.long)

    data = Data(
        x=torch.tensor(features, dtype=torch.float32),
        edge_index=edge_index,
        y=torch.tensor(labels, dtype=torch.long),
        train_mask=torch.tensor(train_mask),
        val_mask=torch.tensor(val_mask),
        test_mask=torch.tensor(test_mask),
    )
    data.timesteps = torch.tensor(timesteps, dtype=torch.int32)
    data.node_ids = node_ids
    data.labeled_mask = torch.tensor(labeled_mask)

    return data
