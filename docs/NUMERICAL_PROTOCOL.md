# Numerical protocol and reproduction limits

The original simulator is copied byte-for-byte to src/model.py. It uses stochastic Heun integration, dt=1 ms, sigma=0.05, 300 s total and a 60 s transient. Every tenth sample is retained (23,999 samples at 100 Hz). Initial states are uniform on [0,0.1). There is no clipping of gating variables.

The transfer function has a historical branch for abs(denominator)<1e-12 returning a*u*0.5. This is preserved for reproducibility. It differs from the analytic removable-singularity limit 1/d. Any correction must be versioned and assessed separately; it was not silently changed here.

ATM detection uses absolute regional z-score >2.5. One-bin avalanches are excluded. Each transition is divided by the source activations within that avalanche (including the final bin). Each matrix entry is averaged over avalanches where that entry is nonzero, not over all avalanches or all source-active avalanches. Region order affects the upper/lower triangle summaries and must be preserved.

Six features: strict-upper mean and median, full Frobenius norm, full SVD entropy (natural logarithm), strict-lower mean and median. Mean and variance heatmaps use the raw retained gating signal, averaged over regional temporal statistics.

Demo: random symmetric network with a fixed seed, zero diagonal and arbitrary scaling. It is not anatomical. Study: exact original NPZ file SHA256 checked, no re-normalization. Anatomical labels and network provenance still require documentation.

Training: two independent models (W,G) and (WF,WNF,G), G free in each. The fixed G=1.89 grid is visualization only. New validation defaults to training-pool-only minmax scaling. Historical scaling used the full bank. Exact historical splits were inherited across growing banks; a fresh split with seed 42 is not guaranteed identical. Supply the original synthetic split explicitly for historical reproduction. Software versions and floating-point platform can also affect reproduction.

Rejection sampling can become slow for poorly supported contexts. This repository does not silently switch the posterior sampler. Investigate warnings before interpreting results.
