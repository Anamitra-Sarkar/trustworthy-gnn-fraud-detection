import torch
import numpy as np
from typing import Tuple


class MCDropoutAnalyzer:
    """Monte Carlo Dropout for Bayesian uncertainty approximation."""

    def __init__(self, num_passes: int = 50):
        self.num_passes = num_passes

    @torch.no_grad()
    def analyze(self, model: torch.nn.Module, x: torch.Tensor,
                edge_index: torch.Tensor, node_indices: np.ndarray = None) -> dict:
        """Run T stochastic forward passes and decompose uncertainty."""
        all_probs = []
        for _ in range(self.num_passes):
            logits = model.forward_with_dropout(x, edge_index)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            if node_indices is not None:
                probs = probs[node_indices]
            all_probs.append(probs)

        all_probs = np.stack(all_probs, axis=0)  # [T, N, K]
        mean_probs = np.mean(all_probs, axis=0)  # [N, K]

        predictive_entropy = -np.sum(
            mean_probs * np.log(mean_probs + 1e-10), axis=-1
        )  # [N]

        per_pass_entropy = -np.sum(
            all_probs * np.log(all_probs + 1e-10), axis=-1
        )  # [T, N]
        expected_entropy = np.mean(per_pass_entropy, axis=0)  # [N]

        epistemic = predictive_entropy - expected_entropy  # [N]
        aleatoric = expected_entropy  # [N]

        predictions = np.argmax(mean_probs, axis=-1)

        return {
            "predictions": predictions.tolist(),
            "mean_probs": mean_probs.tolist(),
            "epistemic_uncertainty": epistemic.tolist(),
            "aleatoric_uncertainty": aleatoric.tolist(),
            "total_uncertainty": predictive_entropy.tolist(),
            "num_passes": self.num_passes,
        }

    def analyze_single(self, model: torch.nn.Module, x: torch.Tensor,
                       edge_index: torch.Tensor, node_idx: int) -> dict:
        """Analyze a single node."""
        result = self.analyze(model, x, edge_index, np.array([node_idx]))
        return {
            "prediction": result["predictions"][0],
            "mean_probs": result["mean_probs"][0],
            "epistemic_uncertainty": result["epistemic_uncertainty"][0],
            "aleatoric_uncertainty": result["aleatoric_uncertainty"][0],
            "total_uncertainty": result["total_uncertainty"][0],
            "num_passes": self.num_passes,
        }
