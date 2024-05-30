import seaborn as sns
import pylab as plt

import pandas as pd
import numpy as np
from scipy import stats

def plot_context():
    sns.set_context("talk", rc={"font.size":18,"axes.titlesize":18,"axes.labelsize":16,"font_scale":0.9})
    
def add_median_labels(ax, values,fmt='.1f',remove=0):
    lines = ax.get_lines()
    boxes = [c for c in ax.get_children() if type(c).__name__ == 'PathPatch']
    lines_per_box = int(len(lines) / len(boxes))
    for i,median in enumerate(lines[4:len(lines)-remove:lines_per_box]):
        x, y = (data.mean() for data in median.get_data())
        value = x if (median.get_xdata()[1] - median.get_xdata()[0]) == 0 else y
        value = values.iloc[i]
        text = ax.text(x, y+0.01, value, ha='center', va='center',
                       fontweight='bold', color='k',fontsize=12,rotation=90,bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
def plot_coefs_onefold(coefs):
    fig = plt.figure(figsize=(10,3))
    plot_context()
    important = pd.concat([coefs.sort_values(by='coef').head(10),coefs.sort_values(by='coef').tail(10)])
    ax = sns.barplot(x='index',y='coef',data=important.reset_index())
    ax.set_xticklabels(ax.get_xticklabels(),rotation=90);
    plt.show()

def plot_coefs(coefs,save=[]):
    fig = plt.figure(figsize=(10,10))
    plot_context()
    mean = coefs.groupby('predictor').mean()
    std = coefs.groupby('predictor').std()
    lc = mean - 1.96 * std
    uc = mean + 1.96 * std
    p_values = pd.DataFrame([stats.ttest_1samp(coefs.loc[(slice(None),c), 'coef'].astype(float),0)[1] for c in mean.index],index=mean.index,
                           columns=['p'])
    alpha = 0.05/mean.shape[0]
    print(p_values)
    significant_coefs = mean[p_values['p'] < alpha]
    if significant_coefs.shape[0]==0:
        print('none reached corrected p-thresh, use 0.05 instead')
        significant_coefs = mean[p_values['p'] < 0.05]
    ax = sns.barplot(y='predictor',x='coef',data=significant_coefs.reset_index().sort_values('coef'),color='gray')
    for i, (feature, mean_coef) in enumerate(significant_coefs.sort_values('coef').T.iteritems()):
        ax.errorbar(x=mean_coef, y=i, xerr=std.loc[feature], fmt='none', ecolor='black')
    if save:
        plt.savefig(f'{save}sign_coefs_acrossfolds.png',dpi=300,bbox_inches='tight')
        plt.savefig(f'{save}sign_coefs_acrossfolds.pdf',dpi=300,bbox_inches='tight')
    plt.show()
    
def plot_performance(results,save=[]):
    plot_context()
    ax = sns.barplot(x='param_logistic__C',y='mean_test_score',data=results)
    x_coords = [p.get_x() + 0.5 * p.get_width() for p in ax.patches]
    y_coords = [p.get_height() for p in ax.patches]
    ax.errorbar(x=x_coords, y=y_coords, yerr=results["std_test_score"], fmt="none", c="k")
    if save:
        plt.savefig(f'{save}performance_C.png',dpi=300,bbox_inches='tight')
        plt.savefig(f'{save}performance_C.pdf',dpi=300,bbox_inches='tight')
    plt.show()
    
def plot_predproba_diag(data,save=[]):
    sns.boxplot(x='diagnosis',y='pred_proba',data=data)
    if save:
        plt.savefig(f'{save}predproba.png',dpi=300,bbox_inches='tight')
        plt.savefig(f'{save}predproba.pdf',dpi=300,bbox_inches='tight')
    plt.show()   