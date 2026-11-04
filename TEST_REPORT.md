# Verification report

Checked on 3 November 2026 with Python 3.9.0 on Windows.

- Four core tests passed: demo network reproducibility and partition, directional ATM example, metric definitions, and parameter designs.
- Both notebooks passed nbformat validation and compilation of every code cell.
- Three full-duration simulations ran with the demo network and original numerical protocol (300 s, first 60 s discarded). Each returned 23,999 samples and finite metrics. Tested (W,G): (2,1.5), (3,2), (4,2.5).
- Both two- and three-parameter inference interfaces ran on 100 artificial feature vectors, with 20 held-out cases, short training and 32 posterior draws per case. These are software checks, not scientific validation of the model.
- Heatmaps and all four validation figure types were generated. A posterior figure was visually inspected.

No full 4,000/20,000-case training, cluster execution, multiprocessing benchmark or private-connectome reproduction was run as part of these checks. Temporary test outputs and artificial feature vectors are excluded from this package.

## Installed versions used for checks

Python 3.9.0; NumPy 1.26.4; SciPy 1.11.4; Numba 0.58.1; pandas 2.3.3; Matplotlib 3.5.1; VBI 0.4.1; sbi 0.21.0; PyTorch 2.2.1; nbformat 5.10.4.

These document an existing working environment, not a tested fresh-install lockfile. The cluster environment should be frozen separately for paper reproduction.
