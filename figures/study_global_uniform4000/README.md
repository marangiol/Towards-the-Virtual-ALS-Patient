# Synthetic validation: study_global_uniform4000

Figures regenerated from archived synthetic test posterior samples. True parameters were checked against the published simulation bank. These are historical results, not a new training run with the notebook default training-only scaler. Exact reproduction requires the original split, scaler policy, fitted network and sampling settings.

Shrinkage panels use x=[0,1] and z-score y=[0,8]. Values outside those limits are not displayed. The six cases are selected at evenly spaced positions in the test array, consistently across parameters.

## Parameter-dependent recovery accuracy and credible-interval coverage

Equal-width bins of the true parameter (10 per parameter). Recovery errors are in parameter units. Their error bars are pointwise percentile bootstrap 95% confidence intervals, based on 2,000 resamples of test cases within each bin (seed 42). Bins with fewer than two cases have no error interval. MAE and RMSE positions are slightly offset for readability. Coverage is the fraction of cases whose true value lies in the inferred 95% credible interval. Bars show pointwise 95% Wilson intervals; n indicates the number of cases per bin. Results are marginal over the other parameter. Historical preprocessing settings apply, as above.

## General synthetic validation

### W


![w six_cases](w_six_cases.png)

![w shrinkage_zscore](w_shrinkage_zscore.png)

![w recovery](w_recovery.png)

![w coverage](w_coverage.png)

### G

![G six_cases](G_six_cases.png)

![G shrinkage_zscore](G_shrinkage_zscore.png)

![G recovery](G_recovery.png)

![G coverage](G_coverage.png)

![W parameter-dependent recovery](w_parameter_dependent_recovery_bootstrap.png)

![G parameter-dependent recovery](G_parameter_dependent_recovery_bootstrap.png)

---
