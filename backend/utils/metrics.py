import numpy as np
from typing import List


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray,
                                num_bins: int = 15) -> float:
    """Compute Expected Calibration Error (ECE)."""
    predictions = np.argmax(probs, axis=1)
    confidences = np.max(probs, axis=1)
    accuracies = (predictions == labels).astype(float)

    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    ece = 0.0
    n = len(labels)

    for i in range(num_bins):
        mask = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        if np.sum(mask) == 0:
            continue
        bin_acc = np.mean(accuracies[mask])
        bin_conf = np.mean(confidences[mask])
        ece += (np.sum(mask) / n) * abs(bin_acc - bin_conf)

    return float(ece)


def brier_score(probs: np.ndarray, labels: np.ndarray, num_classes: int = 2) -> float:
    """Compute Brier Score."""
    one_hot = np.eye(num_classes)[labels]
    return float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))


def empirical_coverage_rate(prediction_sets: List[List[int]], labels: np.ndarray) -> float:
    """Compute empirical coverage rate for conformal prediction."""
    covered = sum(
        1 for i, pset in enumerate(prediction_sets)
        if labels[i] in pset
    )
    return covered / len(labels)
