import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple


class EvidentialAnalyzer:
    """Evidential Deep Learning uncertainty analysis via Dirichlet parameterization."""

    @staticmethod
    def compute_subjective_logic(alpha: np.ndarray) -> dict:
        """Decompose Dirichlet parameters into subjective logic components.

        Args:
            alpha: Dirichlet concentration parameters [K] for a single node.

        Returns:
            Dict with belief masses, vacuity, dissonance, and expected probabilities.
        """
        K = len(alpha)
        S = np.sum(alpha)
        evidence = alpha - 1.0
        beliefs = evidence / S
        vacuity = K / S
        expected_probs = alpha / S

        # Dissonance: measures conflicting evidence between classes
        dissonance = 0.0
        for i in range(K):
            for j in range(K):
                if i != j:
                    bal = 1.0 - abs(evidence[i] - evidence[j]) / (evidence[i] + evidence[j] + 1e-8)
                    dissonance += beliefs[i] * bal * beliefs[j] / (1.0 - beliefs[i] + 1e-8)

        return {
            "alpha": alpha.tolist(),
            "dirichlet_strength": float(S),
            "beliefs": beliefs.tolist(),
            "vacuity": float(vacuity),
            "dissonance": float(dissonance),
            "expected_probs": expected_probs.tolist(),
            "predicted_class": int(np.argmax(expected_probs)),
            "licit_evidence": float(evidence[0]),
            "fraud_evidence": float(evidence[1]) if K > 1 else 0.0,
        }

    @staticmethod
    def batch_analyze(alpha_batch: np.ndarray) -> list:
        return [EvidentialAnalyzer.compute_subjective_logic(alpha_batch[i]) for i in range(len(alpha_batch))]

    @staticmethod
    def edl_loss(alpha: torch.Tensor, targets: torch.Tensor, epoch: int,
                 num_classes: int = 2, annealing_epochs: int = 10) -> torch.Tensor:
        """Compute EDL loss: MSE Bayes risk + annealed KL divergence."""
        S = torch.sum(alpha, dim=1, keepdim=True)
        p_hat = alpha / S

        one_hot = F.one_hot(targets, num_classes).float()

        # MSE Bayes risk
        mse = torch.sum((one_hot - p_hat) ** 2, dim=1)
        variance = torch.sum(p_hat * (1 - p_hat) / (S + 1), dim=1)
        bayes_risk = mse + variance

        # KL divergence penalty (annealed)
        lambda_t = min(1.0, epoch / annealing_epochs)
        alpha_tilde = one_hot + (1 - one_hot) * alpha
        alpha_0 = torch.ones_like(alpha)
        kl = EvidentialAnalyzer._kl_dirichlet(alpha_tilde, alpha_0)

        loss = torch.mean(bayes_risk + lambda_t * kl)
        return loss

    @staticmethod
    def _kl_dirichlet(alpha: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        """KL divergence between two Dirichlet distributions."""
        S_alpha = torch.sum(alpha, dim=1, keepdim=True)
        S_beta = torch.sum(beta, dim=1, keepdim=True)

        kl = (
            torch.lgamma(S_alpha) - torch.lgamma(S_beta)
            - torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
            + torch.sum(torch.lgamma(beta), dim=1, keepdim=True)
            + torch.sum((alpha - beta) * (torch.digamma(alpha) - torch.digamma(S_alpha)), dim=1, keepdim=True)
        )
        return kl.squeeze(1)

    @staticmethod
    def epn_reg_loss(alpha_u: torch.Tensor, alpha_v: torch.Tensor,
                     edge_weights: torch.Tensor) -> torch.Tensor:
        """Topological regularization for Evidential Probing Networks."""
        kl_uv = EvidentialAnalyzer._kl_dirichlet(alpha_u, alpha_v)
        return torch.mean(edge_weights * kl_uv)
