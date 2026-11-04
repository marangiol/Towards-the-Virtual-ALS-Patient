# Demo W-G bank: 4,000 simulations

Generated with the seeded synthetic 80-region demo connectome, not the original study connectome. Contains no empirical recordings, clinical tables or subject posterior samples.

- Parameters: W in [2,4], G in [1.5,2.5], sampled independently and uniformly.
- Six directional ATM features, raw signal mean/variance, avalanche count and retained sample count.
- simulation_id and noise_seed identify each run.
- design.csv and manifest.json document the design, physics, network configuration and generating source hashes.
- simulations.csv is the complete bank. Manifest hashes refer to the generating code; later documentation or progress-display edits may differ.

The first notebook loads this bank by default in demo mode. It can be used immediately for synthetic training and validation. It does not reproduce the paper results.
