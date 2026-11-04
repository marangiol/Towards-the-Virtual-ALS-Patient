import numpy as np
from numba import njit

@njit(cache=True)
def directional_atm(binary):
    """Stessa media delle transizioni non nulle della pipeline precedente."""
    n, t = binary.shape
    total = np.zeros((n, n))
    nonzero = np.zeros((n, n))
    k = 0
    events = 0
    while k < t:
        if not binary[:, k].any():
            k += 1
            continue
        start = k
        while k < t and binary[:, k].any():
            k += 1
        if k - start < 2:
            continue
        counts = np.zeros((n, n))
        activations = np.zeros(n)
        for a in range(start, k):
            for i in range(n):
                if binary[i, a]:
                    activations[i] += 1
                    if a < k - 1:
                        for j in range(n):
                            if binary[j, a + 1]:
                                counts[i, j] += 1
        for i in range(n):
            if activations[i] > 0:
                for j in range(n):
                    value = counts[i, j] / activations[i]
                    total[i, j] += value
                    nonzero[i, j] += value != 0
        events += 1
    result = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if nonzero[i, j] > 0:
                result[i, j] = total[i, j] / nonzero[i, j]
    return result, events
