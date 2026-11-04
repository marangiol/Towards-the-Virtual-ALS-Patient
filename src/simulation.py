"""Generate reproducible grids or uniform simulation banks."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS']:
    os.environ.setdefault(key,'1')
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm.auto import tqdm
import hashlib,json
import numpy as np
import pandas as pd
from scipy.stats import zscore
from .model import RWW_sde
from .avalanches import directional_atm
from .atm_metrics import METRIC_NAMES,metrics_from_matrix
from .connectome import load_connectome
ROOT=Path(__file__).resolve().parents[1]

def atomic_json(path,value):
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False))
    temporary.replace(path)

def make_design(mode='uniform',count=100,grid=False,fixed_G=1.89,seed=20261028):
    if mode not in ['uniform','regional']:raise ValueError('Unknown model configuration')
    names=['w','G'] if mode=='uniform' else ['wf','wnf','G']
    low=np.array([2,1.5] if mode=='uniform' else [2,2,1.5])
    high=np.array([4,2.5] if mode=='uniform' else [4,4,2.5])
    if grid:
        side=int(np.sqrt(count))
        if side*side!=count:raise ValueError('A grid requires a square count, e.g. 100.')
        first,second=np.meshgrid(np.linspace(2,4,side),np.linspace(1.5 if mode=='uniform' else 2,2.5 if mode=='uniform' else 4,side))
        theta=np.column_stack([first.ravel(),second.ravel()])
        if mode=='regional':theta=np.column_stack([theta,np.full(count,fixed_G)])
    else:theta=np.random.default_rng(seed).uniform(low,high,size=(count,len(names)))
    design=pd.DataFrame(theta,columns=names)
    design.insert(0,'simulation_id',np.arange(count))
    design['noise_seed']=40261028+np.arange(count)
    return design

def run_one(job,network,physics):
    parameters=dict(physics,weights=network['W'],nn=80,G=float(job['G']),seed=int(job['noise_seed']))
    weights=np.empty(80)
    weights[network['frontal_indices']]=job.get('wf',job.get('w'))
    weights[network['no_frontal_indices']]=job.get('wnf',job.get('w'))
    signal=RWW_sde(parameters).run({'w':weights,'G':float(job['G'])})['S'].astype(float)
    if len(signal)<2 or not np.isfinite(signal).all():raise ValueError('Invalid simulated signal')
    # Average temporal mean and temporal variance across regions, BEFORE z-scoring.
    mean=float(signal.mean(axis=0).mean())
    variance=float(signal.var(axis=0,ddof=0).mean())
    normalized=zscore(signal,axis=0,ddof=0)
    if not np.isfinite(normalized).all():raise ValueError('Constant regional signal')
    matrix,events=directional_atm(np.ascontiguousarray((np.abs(normalized)>2.5).T))
    features=metrics_from_matrix(matrix)
    if events==0 or not np.isfinite(list(features.values())).all():raise ValueError('No valid avalanches: do not replace features with zeros')
    return dict(job,**features,signal_mean=mean,signal_variance=variance,n_samples=len(signal),n_avalanches=int(events))

def _cached_job(arguments):
    job,network,physics,folder=arguments
    path=Path(folder)/'rows'/f"{int(job['simulation_id']):06d}.json"
    if path.exists():return json.loads(path.read_text())
    result=run_one(job,network,physics)
    atomic_json(path,result)
    return result

def run_bank(design,config,folder,workers=1,physics=None):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True);(folder/'rows').mkdir(exist_ok=True)
    network=load_connectome(config)
    if physics is None:physics=json.loads((ROOT/'configs/physics.json').read_text())
    manifest=dict(design=design.to_dict('records'),physics=physics,network=network['provenance'],
                  source_hashes={n:hashlib.sha256((ROOT/'src'/n).read_bytes()).hexdigest() for n in ['model.py','simulation.py','avalanches.py','atm_metrics.py','connectome.py']})
    path=folder/'manifest.json'
    if path.exists() and json.loads(path.read_text())!=manifest:raise ValueError('Configuration changed: choose a new output folder')
    atomic_json(path,manifest);design.to_csv(folder/'design.csv',index=False)
    jobs=[(j,network,physics,str(folder)) for j in design.to_dict('records')]
    rows=[]
    pending=[]
    for arguments in jobs:
        saved=folder/'rows'/f"{int(arguments[0]['simulation_id']):06d}.json"
        if saved.exists():
            rows.append(json.loads(saved.read_text()))
        else:
            pending.append(arguments)
    with tqdm(total=len(jobs), initial=len(rows), desc='Simulations', unit='sim') as progress:
        if workers==1:
            for arguments in pending:
                rows.append(_cached_job(arguments))
                progress.update(1)
        elif pending:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                futures=[pool.submit(_cached_job, arguments) for arguments in pending]
                for future in as_completed(futures):
                    rows.append(future.result())
                    progress.update(1)
    frame=pd.DataFrame(rows).sort_values('simulation_id')
    frame.to_csv(folder/'simulations.csv',index=False)
    print('Saved',len(frame),'simulations:',folder)
    return frame
