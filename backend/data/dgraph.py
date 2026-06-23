import os
import numpy as np
import torch
from torch_geometric.data import Data


def load_dgraph(data_dir: str = "data/dgraph_fin") -> Data:
    """Load DGraph-Fin dataset.

    3.7M nodes, 4.3M edges. Directed consumer credit social contacts.
    Random split 70:15:15.
    """
    x = np.load(os.path.join(data_dir, "dgraph_node_features.npy")).astype(np.float32)
    y = np.load(os.path.join(data_dir, "dgraph_node_labels.npy")).astype(np.int64)
    edges = np.load(os.path.join(data_dir, "dgraph_edges.npy")).astype(np.int64)

    num_nodes = len(x)
    indices = np.random.RandomState(42).permutation(num_nodes)
    train_end = int(0.7 * num_nodes)
    val_end = int(0.85 * num_nodes)

    train_mask = np.zeros(num_nodes, dtype=bool)
    val_mask = np.zeros(num_nodes, dtype=bool)
    test_mask = np.zeros(num_nodes, dtype=bool)
    train_mask[indices[:train_end]] = True
    val_mask[indices[train_end:val_end]] = True
    test_mask[indices[val_end:]] = True

    from sklearn.preprocessing import StandardScaler
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
