# Towards the Virtual ALS Patient

Amyotrophic lateral sclerosis (ALS) is a multisystem neurodegenerative disorder in which motor-neuron degeneration is accompanied by widespread changes in cortical dynamics. Cortical hyperexcitability is a consistent neurophysiological finding, particularly in motor and premotor areas. How these local circuit alterations shape distributed whole-brain activity remains poorly understood.

This study combines source-reconstructed resting-state magnetoencephalography (MEG), tractography-informed whole-brain modeling, and simulation-based inference (SBI) to investigate whether changes in recurrent excitation can account for ALS-related alterations in large-scale brain dynamics. Neuronal avalanche transition matrices (ATMs) characterize how activity propagates across brain regions and provide the features used for model inversion.

We use a stochastic reduced Wong-Wang model of interacting excitatory populations coupled through a structural connectome. Local recurrent excitation serves as a model-based descriptor of excitability-related circuit changes; inhibition is not represented by a separately inferred parameter. Two complementary configurations are fitted independently:

- **Global configuration:** a shared recurrent-excitation parameter, W, across all 80 regions, jointly inferred with global coupling, G.
- **Regional configuration:** separate recurrent-excitation parameters for a 34-region motor-frontal partition (WF) and the complementary 46-region partition (WNF), jointly inferred with G. The motor-frontal partition includes motor-related and frontal association areas and is not an exclusively motor network.

In both configurations, G is freely estimated. Six directional ATM features are used: the mean and median of the upper and lower triangles, the matrix norm, and singular-value entropy. Features condition a neural posterior estimator, yielding joint distributions over model parameters. Synthetic cases with known parameters are used to assess recovery accuracy, posterior uncertainty, and credible-interval coverage.

In the empirical study, ALS patients exhibited higher model-derived recurrent excitation than controls. The regional analysis investigated the anatomical distribution of these alterations, and higher motor-frontal recurrent excitation was associated with greater functional impairment. Together, these findings support enhanced recurrent excitation as a candidate mechanism contributing to altered large-scale avalanche propagation in ALS and motivate further investigation of its regional and clinical relevance.

This repository makes the synthetic simulation and inference workflow available through two notebooks, reusable functions, simulation banks, and validation figures. A synthetic-connectome demo supports running the workflow without private inputs; the study configuration uses the original connectome. The distributed banks contain synthetic model outputs, while empirical recordings, clinical data, and subject-level inferred results are not included.

## Study overview

![Study overview: empirical MEG avalanche features, whole-brain model configurations, simulation-based inference and clinical analyses](docs/images/study_overview.png)

*Graphical overview of the study: extraction of empirical avalanche transition matrix features, whole-brain simulations with uniform or regional excitability, and simulation-based inference of model parameters. The figure also summarizes the empirical analyses; the notebooks in this repository provide the synthetic workflow.*

## Notebooks

Two notebooks demonstrate the study pipeline without empirical or clinical data.

1. `notebooks/01_uniform_W_G.ipynb`: explore W and G, show six ATM heatmaps and raw-signal mean/variance, then train and validate on synthetic data.
2. `notebooks/02_regional_WF_WNF_G.ipynb`: explore WF and WNF at G=1.89, then jointly infer WF, WNF and G from a separate three-dimensional bank.

The 100-point demo exploration and the 4,000-case demo simulation bank are complete. The bank is included in `data/demo_uniform4000/`, with its design and generation manifest. Final training and validation results are not yet included.

## Available synthetic banks

| Folder | Network | Parameters | Cases |
| --- | --- | --- | ---: |
| `data/demo_uniform4000/` | Synthetic demo | W, G | 4,000 |
| `data/study_global_uniform4000/` | Original study connectome | W, G | 4,000 |
| `data/study_regional_uniform20000/` | Original study connectome | WF, WNF, G | 20,000 |

Study banks contain raw synthetic ATM features and parameter values. The original connectome and all empirical and clinical tables remain excluded. Read the README and metadata within each bank before use. Training on the saved features does not require the connectome; regenerating study simulations does.

## Synthetic validation figures

- [Global W-G, 4,000 study simulations](figures/study_global_uniform4000/README.md)
- [Regional WF-WNF-G, 20,000 study simulations](figures/study_regional_uniform20000/README.md)

These figures use archived synthetic validation results. They are distinct from the freshly trained demo and from rerunning the notebooks with the default training-only normalization. Each gallery documents this distinction.

## Setup

Use an isolated Python environment (the study cluster uses Python 3.11):

```bash
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell instead:
# .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pytest tests
python -m jupyterlab
```

The requirements file lists dependencies; it is not a version-locked environment. Consult TEST_REPORT.md for what was actually tested. Save `pip freeze` from the study cluster before claiming exact reproduction.

## Choose the network

- `configs/demo.json`: generates a synthetic 80-region network. No private input is needed. This illustrates the pipeline and does not reproduce study results.
- `configs/study.json`: requires the original network at `data/private/connectome.npz`. A checksum verifies its identity. No weights are modified.

The source and redistribution terms of the original network must still be documented. Generic demo region labels do not identify real anatomy.

## Run the notebooks

Read the short explanations above each cell. Set `CONFIG_NAME` to `demo` or `study`. Set `RUN_GRID=True` for the first exploration run. The default grid has 100 points. Each simulation retains about 240 seconds after a 60-second transient, so even 100 runs may take hours.

The inference bank is separate. Set `RUN_BANK=True` or point `BANK_FOLDER` to an existing compatible synthetic bank. The example uses 4,000 cases; this can be changed to 20,000. Do not infer three parameters using a grid where G is fixed.

Outputs are kept in separate demo/study and uniform/regional folders. Completed simulation rows are cached. Changing settings requires a different output folder.

For parallel simulation jobs, run from the repository root:

```bash
python scripts/run_simulations.py --config configs/demo.json --mode uniform --count 100 --grid --workers 4 --output outputs/demo/uniform/exploration
python scripts/run_simulations.py --config configs/demo.json --mode regional --count 4000 --workers 4 --output outputs/demo/regional/inference_bank
```

## Validation

The test cases have known synthetic parameters. Figures show six illustrative posterior distributions, shrinkage versus z-score, true versus inferred parameters and coverage. All parameters use the same illustrative cases, without selecting good fits.

## What is not included

No empirical recordings, empirical features or scalers, clinical tables, subject posterior samples, trained subject-specific outputs or original connectome are included. Synthetic demo and study banks are included as documented above. Additional banks can be generated by the user.
