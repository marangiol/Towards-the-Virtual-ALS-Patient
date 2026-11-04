# Study simulation bank: 4,000 cases

Parameters: w, G. W/WF/WNF range: [2,4]; G range: [1.5,2.5]; b=0.16.

Generated using the original study connectome, identified by checksum in metadata.json. Only synthetic parameters, noise seeds and six raw ATM features are distributed. No empirical observations, clinical records, subject posterior samples or connectome matrix are included.

The global bank uses a single W shared by all regions, with W and G varying across cases. The regional bank jointly varies WF, WNF and G. These are different model configurations.

Use this bank with mode="uniform" in src.inference.fit_and_validate. No connectome is needed to train from these saved features; regenerating the simulations requires the original network. The notebooks default to demo data. To use this bank, set RUN_BANK=False and BANK_FOLDER=ROOT / 'data' / 'study_global_uniform4000' in the bank-loading cell.

The default training-only scaler and newly generated train/test split need not reproduce the historical paper analysis. Reusing the exact historical split and scaler policy requires their documented original settings.
