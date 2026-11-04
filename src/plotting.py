"""Figures for model exploration and synthetic validation."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from .atm_metrics import METRIC_NAMES
COLORS={'w':'#777777','wf':'#A03B78','wnf':'#CFA826','G':'#2E7D32'}

def save(fig,folder,name):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    for ext in ['png','pdf']:fig.savefig(folder/f'{name}.{ext}',dpi=200,bbox_inches='tight')
    if plt.get_backend().lower() != 'agg':
        plt.show()
    plt.close(fig)

def heatmaps(frame,x,y,folder):
    for feature in [*METRIC_NAMES,'signal_mean','signal_variance']:
        grid=frame.pivot(index=y,columns=x,values=feature).sort_index().sort_index(axis=1)
        if grid.isna().any().any():raise ValueError('Heatmaps require a complete rectangular grid; no extrapolation is applied')
        fig,ax=plt.subplots(figsize=(6,5))
        image=ax.pcolormesh(grid.columns.to_numpy(),grid.index.to_numpy(),grid.to_numpy(),shading='nearest',cmap='viridis')
        ax.set(xlabel=x,ylabel=y,title=feature)
        fig.colorbar(image,ax=ax,label=feature)
        save(fig,folder,feature)

def validation_figures(folder):
    folder=Path(folder);table=pd.read_csv(folder/'validation.csv')
    with np.load(folder/'posterior_samples.npz') as s:
        samples=s['samples'];names=s['parameters'].tolist();truth=s['truth'];low=s['low'];high=s['high']
    # Same six cases for every parameter; chosen by truth positions, not good recovery.
    selected=np.linspace(0,len(truth)-1,min(6,len(truth)),dtype=int)
    pd.DataFrame({'case_number':np.arange(1,len(selected)+1),'test_position':selected}).to_csv(folder/'illustrative_cases.csv',index=False)
    summaries=[]
    for j,name in enumerate(names):
        color=COLORS[name];data=table[table.parameter==name].sort_values('test_position')
        fig,ax=plt.subplots(figsize=(7,5));positions=np.arange(1,len(selected)+1)
        violin=ax.violinplot([samples[k,:,j] for k in selected],positions=positions,showmeans=True,showextrema=True)
        for body in violin['bodies']:body.set_facecolor(color);body.set_alpha(.35)
        for key in ['cmeans','cmins','cmaxes','cbars']:violin[key].set_color(color)
        for position,k in zip(positions,selected):ax.hlines(truth[k,j],position-.25,position+.25,color='black',ls='--')
        ax.set(xlabel='Synthetic case',ylabel=name,title='Posterior distributions',ylim=(low[j],high[j]))
        ax.legend(handles=[Line2D([0],[0],color=color,label='Posterior mean'),
                           Line2D([0],[0],color='black',ls='--',label='Ground truth')],
                  loc='upper center',bbox_to_anchor=(.5,-.15),ncol=2,frameon=False)
        save(fig,folder/'figures',name+'_six_cases')
        fig,ax=plt.subplots(figsize=(6,5));ax.scatter(data.shrinkage,data.zscore,s=12,c=color,alpha=.4)
        ax.scatter(data.iloc[selected].shrinkage,data.iloc[selected].zscore,s=60,facecolors='none',edgecolors='black')
        ax.set(xlabel='Posterior shrinkage',ylabel='Absolute posterior z-score',title=name)
        save(fig,folder/'figures',name+'_shrinkage_zscore')
        covered=data['covered_0.95'].to_numpy(bool)
        fig,ax=plt.subplots(figsize=(6,5))
        ax.scatter(data.truth,data['mean'],s=15,c=np.where(covered,color,'#aaaaaa'))
        ax.plot([low[j],high[j]],[low[j],high[j]],'k--');ax.set(xlabel='True '+name,ylabel='Inferred '+name,title=f'95% coverage: {covered.mean():.1%}')
        save(fig,folder/'figures',name+'_recovery')
        fig,ax=plt.subplots(figsize=(5,5));masses=np.array([.5,.8,.9,.95]);rates=[data[f'covered_{m}'].mean() for m in masses]
        ax.plot(masses,rates,'o-',c=color);ax.plot([0,1],[0,1],'k--');ax.set(xlim=(0,1),ylim=(0,1),xlabel='Nominal probability',ylabel='Observed coverage',title=name)
        save(fig,folder/'figures',name+'_coverage')
        error=data['mean']-data.truth
        summaries.append(dict(parameter=name,n_test=len(data),MAE=abs(error).mean(),RMSE=np.sqrt((error**2).mean()),coverage95=covered.mean()))
    summary=pd.DataFrame(summaries);summary.to_csv(folder/'validation_summary.csv',index=False)
    return summary
