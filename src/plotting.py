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
        ax.set(xlabel='Posterior shrinkage',ylabel='Absolute posterior z-score',title=name, xlim=(0,1), ylim=(0,8))
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


def parameter_dependent_recovery(folder, bins=10, bootstrap_repetitions=2000, seed=42):
    """Evaluate marginal recovery by true-parameter bins on synthetic test cases.

    Bins cover the full prior. Empty bins remain missing. Coverage uncertainty
    uses pointwise 95% Wilson intervals; it is not a posterior credible interval.
    """
    folder=Path(folder)
    if not isinstance(bins,int) or bins<2:
        raise ValueError('bins must be an integer of at least 2')
    table=pd.read_csv(folder/'validation.csv')
    with np.load(folder/'posterior_samples.npz') as saved:
        names=saved['parameters'].tolist()
        low=saved['low'];high=saved['high']
    if names!=['w','G']:
        raise ValueError('This analysis expects the global W-G model')
    if bootstrap_repetitions<100:
        raise ValueError('Use at least 100 bootstrap repetitions')
    rng=np.random.default_rng(seed)
    records=[]
    for j,name in enumerate(names):
        data=table.loc[table.parameter==name].copy()
        if not data.truth.between(low[j],high[j]).all():
            raise ValueError('True parameter outside the prior')
        edges=np.linspace(low[j],high[j],bins+1)
        membership=np.clip(np.searchsorted(edges,data.truth,side='right')-1,0,bins-1)
        for k in range(bins):
            part=data.loc[membership==k]
            n=len(part)
            error=part['mean']-part.truth
            covered=part['covered_0.95']
            if covered.dtype==object:
                covered=covered.astype(str).str.lower().map({'true':True,'false':False})
                if covered.isna().any():raise ValueError('Invalid coverage indicator')
            hits=int(covered.sum()) if n else 0
            rate=hits/n if n else np.nan
            # Wilson score interval for the coverage fraction across test cases.
            if n:
                z=1.959963984540054
                den=1+z*z/n
                center=(rate+z*z/(2*n))/den
                half=z*np.sqrt(rate*(1-rate)/n+z*z/(4*n*n))/den
                lo,hi=center-half,center+half
            else:lo,hi=np.nan,np.nan
            # Resample test cases within this bin, preserving each case's error.
            if n>=2:
                draws=rng.choice(error.to_numpy(),size=(bootstrap_repetitions,n),replace=True)
                mae_lo,mae_hi=np.quantile(np.abs(draws).mean(axis=1),[.025,.975])
                rmse_lo,rmse_hi=np.quantile(np.sqrt((draws**2).mean(axis=1)),[.025,.975])
            else:mae_lo,mae_hi,rmse_lo,rmse_hi=[np.nan]*4
            records.append(dict(parameter=name,bin=k,lower=edges[k],upper=edges[k+1],
                center=(edges[k]+edges[k+1])/2,n_test=n,
                MAE=abs(error).mean(),RMSE=np.sqrt((error**2).mean()),
                bias=error.mean(),MAE_lo95=mae_lo,MAE_hi95=mae_hi,RMSE_lo95=rmse_lo,RMSE_hi95=rmse_hi,coverage95=rate,coverage_lo95=lo,coverage_hi95=hi))
    result=pd.DataFrame(records)
    result.to_csv(folder/'parameter_dependent_recovery.csv',index=False)
    for j,name in enumerate(names):
        part=result[result.parameter==name]
        valid=part[part.n_test>0]
        fig,axes=plt.subplots(2,1,figsize=(7,6.5),sharex=True)
        color=COLORS[name]
        for metric,marker,style,offset in [('MAE','o','-',-.035),('RMSE','s','--',.035)]:
            y=part[metric].to_numpy()
            errors=np.vstack([y-part[metric+'_lo95'].to_numpy(),part[metric+'_hi95'].to_numpy()-y])
            axes[0].errorbar(part.center.to_numpy()+offset*(high[j]-low[j])/bins,y,
                yerr=np.maximum(errors,0),fmt=marker+style,color=color,capsize=3,
                alpha=1 if metric=='MAE' else .7,label=metric+'; bootstrap 95% CI')
        axes[0].set(ylabel='Recovery error',title='Parameter-dependent recovery: '+name)
        axes[0].legend(frameon=False,loc='upper center',bbox_to_anchor=(.5,-.12),fontsize=9)
        axes[1].errorbar(valid.center.to_numpy(),valid.coverage95.to_numpy(),
            yerr=np.vstack([valid.coverage95-valid.coverage_lo95,
                            valid.coverage_hi95-valid.coverage95]),
            fmt='o',color=color,capsize=3,label='Observed coverage; 95% Wilson interval')
        axes[1].axhline(.95,color='black',ls='--',label='Nominal 95%')
        for row in part.itertuples():
            axes[1].text(row.center,.035,'n='+str(row.n_test),ha='center',rotation=90,fontsize=8)
        axes[1].set(xlabel='True '+name,ylabel='95% credible-interval coverage',
                    xlim=(low[j],high[j]),ylim=(0,1.04))
        axes[1].legend(frameon=False,fontsize=8,loc='upper center',bbox_to_anchor=(.5,-.22))
        fig.tight_layout(h_pad=3)
        save(fig,folder/'figures',name+'_parameter_dependent_recovery')
    return result
