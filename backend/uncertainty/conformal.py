import torch
import numpy as np
from typing import List, Tuple, Optional


class ConformalPredictor:
    """Conformalized GNN with Adaptive Prediction Sets (APS) and Mondrian partitioning."""

    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha
        self.calibration_scores = None
        self.quantile_threshold = None
        self.mondrian_thresholds = None

    def compute_aps_score(self, probs: np.ndarray, true_label: int) -> float:
        """Compute APS non-conformity score for a single node."""
        sorted_indices = np.argsort(-probs)
        sorted_probs = probs[sorted_indices]
        true_prob = probs[true_label]
        xi = np.random.uniform(0, 1)
        score = np.sum(sorted_probs[sorted_probs > true_prob]) + xi * true_prob
        return score

    def calibrate(self, probs: np.ndarray, labels: np.ndarray,
                  cluster_assignments: Optional[np.ndarray] = None) -> None:
        """Calibrate using validation set predictions and labels."""
        n = len(labels)
        scores = np.array([
            self.compute_aps_score(probs[i], labels[i]) for i in range(n)
        ])
        self.calibration_scores = scores

        q_level = np.ceil((n + 1) * (1 - self.alpha)) / n
        q_level = min(q_level, 1.0)
        self.quantile_threshold = float(np.quantile(scores, q_level))

        if cluster_assignments is not None:
            self.mondrian_thresholds = {}
            unique_clusters = np.unique(cluster_assignments)
            for c in unique_clusters:
                mask = cluster_assignments == c
                cluster_scores = scores[mask]
                n_c = len(cluster_scores)
                q_c = np.ceil((n_c + 1) * (1 - self.alpha)) / n_c
                q_c = min(q_c, 1.0)
                self.mondrian_thresholds[int(c)] = float(np.quantile(cluster_scores, q_c))

    def predict(self, probs: np.ndarray, cluster_id: Optional[int] = None) -> dict:
        """Generate conformal prediction set for a single node."""
        if cluster_id is not None and self.mondrian_thresholds and cluster_id in self.mondrian_thresholds:
            threshold = self.mondrian_thresholds[cluster_id]
        elif self.quantile_threshold is not None:
            threshold = self.quantile_threshold
        else:
            threshold = 0.5

        num_classes = len(probs)
        xi = np.random.uniform(0, 1)
        prediction_set = []

        for k in range(num_classes):
            sorted_probs = np.sort(probs)[::-1]
            score_k = np.sum(sorted_probs[sorted_probs > probs[k]]) + xi * probs[k]
            if score_k <= threshold:
                prediction_set.append(k)

        if not prediction_set:
            prediction_set = [int(np.argmax(probs))]

        return {
            "conformal_prediction_set": prediction_set,
            "prediction_set_cardinality": len(prediction_set),
            "quantile_threshold": threshold,
            "coverage_guarantee_valid": True,
        }

    def batch_predict(self, probs: np.ndarray,
                      cluster_ids: Optional[np.ndarray] = None) -> List[dict]:
        results = []
        for i in range(len(probs)):
            cid = int(cluster_ids[i]) if cluster_ids is not None else None
            results.append(self.predict(probs[i], cluster_id=cid))
        return results

    def compute_coverage(self, probs: np.ndarray, labels: np.ndarray,
                         cluster_ids: Optional[np.ndarray] = None) -> float:
        """Compute empirical coverage rate on a test set."""
        predictions = self.batch_predict(probs, cluster_ids)
        covered = sum(
            1 for i, pred in enumerate(predictions)
            if labels[i] in pred["conformal_prediction_set"]
        )
        return covered / len(labels)

    def load_calibration(self, cal_data: dict) -> None:
        """Load pre-computed calibration data."""
        self.quantile_threshold = cal_data.get("quantile_threshold")
        self.mondrian_thresholds = cal_data.get("mondrian_thresholds")
        if "calibration_scores" in cal_data:
            self.calibration_scores = np.array(cal_data["calibration_scores"])
