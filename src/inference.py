"""Train on synthetic data and evaluate on held-out synthetic cases only."""
from pathlib import Path
import random,json,hashlib
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from .atm_metrics import METRIC_NAMES

def fit_and_validate(bank,folder,mode='uniform',seed=91011,max_epochs=400,draws=1000,scaler_scope='train',split_file=None):
    """Fit a neural posterior estimator and validate parameter recovery.

    Scientific workflow
    -------------------
    Each CSV row represents one synthetic simulation: its known parameters
    theta and six raw directional-ATM features f(theta). The task is to learn
    p(theta | x), where x is the normalized six-feature vector.

    Two independent model configurations are available:
      * uniform: theta=(W,G), with W~U(2,4), G~U(1.5,2.5);
      * regional: theta=(WF,WNF,G), with WF,WNF~U(2,4), G~U(1.5,2.5).
    BoxUniform specifies independent uniform priors. G is inferred freely in
    both configurations; it is not fixed using a previous inversion.

    The default split reserves 20% of simulations for external synthetic
    testing (split seed 42). The remaining 80% form the training pool. SBI
    internally reserves 10% of this pool for early stopping, giving roughly
    72% optimization, 8% internal validation, and 20% external testing.
    split_file can supply original train/test row indices instead.

    Each feature is min-max scaled: x=(f-minimum)/(maximum-minimum).
    scaler_scope='train' estimates these limits from the 80% training pool,
    including its internal validation subset. The external test set does not
    contribute. Test values can fall outside [0,1] and are not clipped.
    scaler_scope='full' reproduces the historical policy in which all bank
    rows, including external tests, contribute to the feature limits. This
    policy must be disclosed when reporting validation results.

    SNPE uses a masked autoregressive flow (MAF) as a conditional density
    estimator. Here one fixed simulation bank is used: there is no adaptive
    sequence of simulation rounds. Training maximizes conditional density
    at known simulated parameters, with batch size 128, learning rate 0.0005,
    and early stopping after 30 epochs without validation improvement, subject
    to max_epochs. SBI constructs a posterior with the specified prior support.

    For every external test vector, draw `draws` joint posterior samples.
    Joint sampling preserves dependence between inferred parameters. For each
    parameter compute its posterior mean and standard deviation, central
    equal-tailed credible intervals, shrinkage and absolute posterior z-score.
    Coverage is evaluated across test cases whose true parameters are known.
    These are posterior credible intervals for parameters, not predictive
    intervals for simulated signals or ATM features.

    Reproducibility and outputs
    ---------------------------
    seed controls training RNGs; each test case has a deterministic sampling
    seed. training.json records the bank checksum and settings, while
    scaler_and_split.npz saves normalization limits, feature order and indices.
    estimator.pt and per-case samples allow compatible runs to resume.
    validation.csv contains test-case diagnostics; posterior_samples.npz
    contains joint draws, true parameters and prior bounds for plotting.
    An existing output folder with different recorded settings is rejected.

    This function evaluates synthetic recovery only. Empirical conditioning,
    empirical scaling, clinical analyses and posterior predictive simulations
    are separate analysis steps. A fixed seed alone does not guarantee identical
    results across different software versions or computing environments.
    """
    # 1. Set RNG seeds and define parameters and independent uniform priors.
    import torch
    from sbi.inference import SNPE
    from sbi.utils import BoxUniform
    torch.set_num_threads(1);torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    frame=pd.read_csv(bank)
    names=['w','G'] if mode=='uniform' else ['wf','wnf','G']
    low=np.array([2,1.5] if mode=='uniform' else [2,2,1.5],dtype=np.float32)
    high=np.array([4,2.5] if mode=='uniform' else [4,4,2.5],dtype=np.float32)
    theta=frame[names].to_numpy(np.float32);raw=frame[list(METRIC_NAMES)].to_numpy(float)
    if len(frame)<100:raise ValueError('Use at least 100 cases for an instructional fit; many more for scientific validation.')
    if not np.isfinite(raw).all() or not np.isfinite(theta).all():raise ValueError('Nonfinite bank')
    if np.any(theta<low) or np.any(theta>high) or np.any(np.ptp(theta,axis=0)==0):raise ValueError('All inferred parameters must vary within the prior')
    # 2. Keep external synthetic test cases separate from the training pool.
    if split_file:
        with np.load(split_file) as s:train=s['train'];test=s['test']
    else:
        order=np.random.default_rng(42).permutation(len(frame));cut=int(np.ceil(.2*len(frame)))
        test,train=order[:cut],order[cut:]
    if sorted(np.r_[train,test].tolist())!=list(range(len(frame))):raise ValueError('Invalid train/test partition')
    # 3. Fit one min-max transformation per feature and reuse it on test cases.
    if scaler_scope not in ['train','full']:raise ValueError('scaler_scope must be train or full')
    reference=raw[train] if scaler_scope=='train' else raw
    minimum=reference.min(axis=0);span=np.ptp(reference,axis=0)
    if np.any(span<=0):raise ValueError('Constant feature in scaler reference')
    x=((raw-minimum)/span).astype(np.float32) # No clipping of test values outside [0,1].
    # 4. Record the exact input bank and settings before training or resuming.
    manifest=dict(bank_sha256=hashlib.sha256(Path(bank).read_bytes()).hexdigest(),seed=seed,max_epochs=max_epochs,draws=draws,mode=mode,scaler_scope=scaler_scope,train=train.tolist(),test=test.tolist())
    path=folder/'training.json'
    if path.exists() and json.loads(path.read_text())!=manifest:raise ValueError('Training settings changed; use a new folder')
    path.write_text(json.dumps(manifest,indent=2))
    np.savez(folder/'scaler_and_split.npz',minimum=minimum,span=span,train=train,test=test,features=METRIC_NAMES)
    # 5. Learn the conditional parameter density using SNPE with a MAF.
    prior=BoxUniform(low=torch.tensor(low),high=torch.tensor(high))
    engine=SNPE(prior=prior,density_estimator='maf')
    checkpoint=folder/'estimator.pt'
    if checkpoint.exists():
        # Load only a checkpoint created locally by this script.
        estimator=torch.load(checkpoint,map_location='cpu',weights_only=False)
    else:
        estimator=engine.append_simulations(torch.tensor(theta[train]),torch.tensor(x[train])).train(max_num_epochs=max_epochs,stop_after_epochs=30,training_batch_size=128,learning_rate=.0005,validation_fraction=.1)
        torch.save(estimator,checkpoint)
    # The posterior is conditioned separately on each six-feature test vector.
    posterior=engine.build_posterior(estimator)
    records=[];all_draws=[];cache=folder/'cases';cache.mkdir(exist_ok=True)
    # 6. Sample joint posteriors and compare each marginal with its known truth.
    for k,index in enumerate(tqdm(test, desc='Synthetic validation', unit='case')):
        target=cache/f'{int(index):06d}.npz'
        if target.exists():
            with np.load(target) as s:samples=s['samples']
        else:
            torch.manual_seed(seed+100000+int(index))
            samples=posterior.sample((draws,),x=torch.tensor(x[index]),show_progress_bars=False).detach().cpu().numpy()
            np.savez_compressed(target,samples=samples)
        if samples.shape!=(draws,len(names)) or not np.isfinite(samples).all():raise ValueError('Invalid posterior samples')
        if np.any(samples<low) or np.any(samples>high):raise ValueError('Posterior outside prior')
        all_draws.append(samples)
        for j,name in enumerate(names):
            values=samples[:,j].astype(float);truth=float(theta[index,j]);sd=values.std()
            # Uniform-prior variance is (upper-lower)^2/12.
            # Shrinkage = 1 - posterior variance / prior variance.
            # Absolute z-score = |posterior mean - truth| / posterior SD.
            # Shrinkage can be negative; a small z-score alone can reflect broad uncertainty.
            row=dict(test_position=k,simulation_id=int(frame.iloc[index].simulation_id),parameter=name,truth=truth,mean=values.mean(),sd=sd,shrinkage=1-values.var()/((high[j]-low[j])**2/12),zscore=abs(values.mean()-truth)/sd if sd>0 else np.nan)
            # Central equal-tailed intervals: e.g. 2.5th and 97.5th percentiles at 95%.
            # covered_mass marks whether the known true parameter lies inside.
            for mass in [.5,.8,.9,.95]:
                lo,hi=np.quantile(values,[(1-mass)/2,(1+mass)/2]);row[f'covered_{mass}']=bool(lo<=truth<=hi)
                if mass==.95:row.update(lo95=lo,hi95=hi)
            records.append(row)

    # 7. Save diagnostics and draws; plotting and aggregate error metrics follow separately.
    validation=pd.DataFrame(records);validation.to_csv(folder/'validation.csv',index=False)
    np.savez_compressed(folder/'posterior_samples.npz',samples=np.array(all_draws),parameters=names,test_indices=test,truth=theta[test],low=low,high=high)
    return validation
