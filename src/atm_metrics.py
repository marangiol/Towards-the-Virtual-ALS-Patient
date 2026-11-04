"""Shared ATM metrics for empirical recordings and simulations.

Upper/lower summaries exclude the diagonal. Frobenius norm and SVD entropy
use the full matrix including its diagonal. No symmetrization, scaling,
outlier exclusion, or statistical tests are performed here.
"""
import numpy as np

METRIC_NAMES = (
    "ATM_mean_mat", "ATM_median_mat", "ATM_norm", "Entropy_ATM_SVD",
    "ATM_mean_lower", "ATM_median_lower",
)
ENTROPY_DEFINITION = "Shannon of normalized singular values, natural logarithm (nats)"


def _validated_matrix(matrix):
    raw = np.asarray(matrix)
    if np.iscomplexobj(raw):
        raise ValueError("ATM must be real-valued")
    matrix = np.asarray(raw, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] < 2:
        raise ValueError("ATM must be square with at least two regions")
    if not np.isfinite(matrix).all() or (matrix < 0).any():
        raise ValueError("ATM must contain finite, nonnegative values")
    return matrix


def _entropy(singular_values):
    total = singular_values.sum()
    if not np.isfinite(total):
        raise ValueError("Singular-value sum is nonfinite")
    if total == 0:
        return float("nan")
    # Keep the same normalization as the empirical and simulation SVD notebooks.
    probabilities = singular_values[singular_values > 0] / total
    return float(-np.sum(probabilities * np.log(probabilities)))


def entropy_atm_svd(matrix):
    """Shannon entropy of normalized singular values in nats; zero ATM -> NaN."""
    matrix = _validated_matrix(matrix)
    return _entropy(np.linalg.svd(matrix, compute_uv=False, hermitian=False))


def metrics_from_matrix(matrix):
    """Return exactly the six shared ATM metrics, with no preprocessing.

    Historical names ATM_mean_mat/ATM_median_mat refer to the strict upper
    triangle; *_lower refers to the strict lower triangle. A zero matrix has
    zero means, medians and norm, but undefined (NaN) SVD entropy. Handling
    subjects with no valid avalanches remains the responsibility of the caller.
    """
    matrix = _validated_matrix(matrix)
    upper = matrix[np.triu_indices_from(matrix, k=1)]
    lower = matrix[np.tril_indices_from(matrix, k=-1)]
    singular_values = np.linalg.svd(matrix, compute_uv=False, hermitian=False)
    return dict(
        ATM_mean_mat=float(upper.mean()),
        ATM_median_mat=float(np.median(upper)),
        ATM_norm=float(np.linalg.norm(matrix, ord="fro")),
        Entropy_ATM_SVD=_entropy(singular_values),
        ATM_mean_lower=float(lower.mean()),
        ATM_median_lower=float(np.median(lower)),
    )
