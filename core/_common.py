"""Common utilities for LDP protocols."""

import numpy as np


def matrix_inversion(count_report: np.ndarray, n: int, p: float, q: float) -> np.ndarray:
    """
    Matrix Inversion (MI) frequency estimator.

    Computes unbiased frequency estimates from perturbed reports using the
    inverse of the LDP channel matrix.

    Args:
        count_report: Array recording how many times each value was reported.
        n: Total number of reports.
        p: Probability of reporting the true value.
        q: Probability of reporting any other value.

    Returns:
        Rounded non-negative frequency estimates.
    """
    est_freq = np.array((count_report - n * q) / (p - q)).clip(0)
    return np.round(est_freq)