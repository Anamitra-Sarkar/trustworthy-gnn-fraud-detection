import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity


def build_similarity_graph(x: np.ndarray, threshold: float = 0.92,
                           batch_size: int = 5000) -> torch.LongTensor:
    """Build edges between nodes with cosine similarity > threshold.
    Processes in batches to avoid OOM on large graphs."""
    num_nodes = x.shape[0]
    edges_src, edges_dst = [], []

    for start in range(0, num_nodes, batch_size):
        end = min(start + batch_size, num_nodes)
        sim_block = cosine_similarity(x[start:end], x)
        rows, cols = np.where(sim_block > threshold)
        rows += start
        mask = rows != cols
        edges_src.extend(rows[mask].tolist())
        edges_dst.extend(cols[mask].tolist())

    if not edges_src:
        return torch.zeros((2, 0), dtype=torch.long)

    edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long)
    return edge_index


def build_knn_graph(x: np.ndarray, k: int = 5) -> torch.LongTensor:
    """Build symmetric k-NN graph using L2 distance."""
    nn = NearestNeighbors(n_neighbors=k + 1, metric='euclidean', algorithm='auto')
    nn.fit(x)
    distances, indices = nn.kneighbors(x)

    edges_src, edges_dst = [], []
    for i in range(len(x)):
        for j in indices[i, 1:]:  # skip self
            edges_src.append(i)
            edges_dst.append(j)
            edges_src.append(j)
            edges_dst.append(i)

    edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long)
    # Deduplicate
    edge_index = torch.unique(edge_index, dim=1)
    return edge_index


def build_temporal_snap_graph(x: np.ndarray, timesteps: np.ndarray,
                              k: int = 3) -> torch.LongTensor:
    """Build cross-timestep k-NN graph connecting nodes between t and t+1."""
    unique_times = np.sort(np.unique(timesteps))
    edges_src, edges_dst = [], []

    for i in range(len(unique_times) - 1):
        t_curr = unique_times[i]
        t_next = unique_times[i + 1]

        mask_curr = timesteps == t_curr
        mask_next = timesteps == t_next

        idx_curr = np.where(mask_curr)[0]
        idx_next = np.where(mask_next)[0]

        if len(idx_next) == 0 or len(idx_curr) == 0:
            continue

        actual_k = min(k, len(idx_next))
        nn = NearestNeighbors(n_neighbors=actual_k, metric='euclidean')
        nn.fit(x[idx_next])
        _, indices = nn.kneighbors(x[idx_curr])

        for local_i, global_i in enumerate(idx_curr):
            for local_j in indices[local_i]:
                global_j = idx_next[local_j]
                edges_src.append(global_i)
                edges_dst.append(global_j)
                edges_src.append(global_j)
                edges_dst.append(global_i)

    if not edges_src:
        return torch.zeros((2, 0), dtype=torch.long)

    edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long)
    edge_index = torch.unique(edge_index, dim=1)
    return edge_index


def build_augmented_graph(original_edge_index: torch.LongTensor,
                          x: np.ndarray, threshold: float = 0.92) -> torch.LongTensor:
    """Combine original transaction edges with similarity-based edges."""
    sim_edges = build_similarity_graph(x, threshold=threshold)
    combined = torch.cat([original_edge_index, sim_edges], dim=1)
    combined = torch.unique(combined, dim=1)
    return combined
