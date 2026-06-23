import os
import numpy as np
import torch
from torch_geometric.data import Data


def load_amazon(data_dir: str = "data/amazon_fraud") -> Data:
    """Load Amazon-Fraud dataset.

    11,944 nodes, 348,464 edges. E-commerce review network.
    Multi-relational undirected. Cross-validation with class balancing.
    """
    x = np.load(os.path.join(data_dir, "amazon_node_features.npy")).astype(np.float32)
    y = np.load(os.path.join(data_dir, "amazon_node_labels.npy")).astype(np.int64)
    edges = np.load(os.path.join(data_dir, "amazon_edges.npy")).astype(np.int64)

    num_nodes = len(x)

    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import StratifiedShuffleSplit

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
    train_idx, temp_idx = next(sss.split(x, y))

    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
    val_idx, test_idx = next(sss2.split(x[temp_idx], y[temp_idx]))
    val_idx = temp_idx[val_idx]
    test_idx = temp_idx[test_idx]

    train_mask = np.zeros(num_nodes, dtype=bool)
    val_mask = np.zeros(num_nodes, dtype=bool)
    test_mask = np.zeros(num_nodes, dtype=bool)
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True

    scaler = StandardScaler()
    scaler.fit(x[train_mask])
    x = scaler.transform(x)

    edge_index = torch.tensor(edges.T, dtype=torch.long)

    data = Data(
        x=torch.tensor(x, dtype=torch.float32),
        edge_index=edge_index,
        y=torch.tensor(y, dtype=torch.long),
        train_mask=torch.tensor(train_mask),
        val_mask=torch.tensor(val_mask),
        test_mask=torch.tensor(test_mask),
    )
    return data
