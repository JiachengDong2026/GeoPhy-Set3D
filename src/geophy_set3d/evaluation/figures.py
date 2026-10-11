"""Standalone paper-style PNG/PDF figures; corresponding numeric CSVs are retained."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':120})


def save(fig,path):
    fig.savefig(path,dpi=200,bbox_inches='tight');fig.savefig(Path(path).with_suffix('.pdf'),bbox_inches='tight');plt.close(fig)


def view_count_figures(rows,folder):
    for metric in ('relative_l1','relative_l2','rmse'):
        fig,ax=plt.subplots(figsize=(5.5,3.8))
        for pool,label in (('train','Pool A'),('ood','Pool B (unseen, same family)')):
            points=[r for r in rows if r['pool']==pool]
            ax.errorbar([r['views'] for r in points],[r['mean_'+metric] for r in points],
                        yerr=[r['draw_std_'+metric] for r in points],marker='o',capsize=3,label=label)
        ax.axvspan(2.5,6.5,color='grey',alpha=.08,label='Trained V range (3–6)')
        ax.set(xlabel='Number of stations V',ylabel=metric.replace('_',' '),xticks=[2,3,4,5,6,8]);ax.legend(fontsize=8)
        save(fig,Path(folder)/f'view_count_{metric}.png')


def coverage_figures(rows,individual,folder):
    for v in (4,6):
        points=[r for r in rows if r['views']==v]
        fig,ax=plt.subplots(figsize=(5,3.6));ax.bar([r['protocol'] for r in points],[r['mean_relative_l1'] for r in points],
           yerr=[r['draw_std_relative_l1'] for r in points],capsize=4)
        ax.set(ylabel='Mean per-case relative L1',title=f'Val Pool A, V={v}');save(fig,Path(folder)/f'angular_coverage_v{v}.png')
    fig,ax=plt.subplots(figsize=(6,4))
    for label in ('spread','random','clustered'):
        points=[r for r in individual if r['protocol']==label]
        ax.scatter([r['azimuth_span_deg'] for r in points],[r['relative_l1'] for r in points],s=9,alpha=.35,label=label)
    ax.set(xlabel='Azimuth span (degrees)',ylabel='Per-case relative L1');ax.legend();save(fig,Path(folder)/'error_vs_azimuth_span.png')


def pool_figure(a,b,folder):
    fig,axes=plt.subplots(1,2,figsize=(8,3.5),constrained_layout=True)
    for ax,metric in zip(axes,('relative_l1','relative_l2')):
        ax.bar(['Pool A','Pool B'],[a['mean_per_case'][metric],b['mean_per_case'][metric]])
        ax.set(ylabel=metric.replace('_',' '),title='Reference fixed V=4')
    save(fig,Path(folder)/'pool_a_vs_b_v4.png')


def condition_figures(groups,rows,folder):
    fields=[f for f in ('source_family','wind_speed_m_s','pressure_pa','active_hole_count') if any(r['field']==f for r in groups)]
    fig,axes=plt.subplots(2,2,figsize=(11,7),constrained_layout=True)
    for ax,field in zip(axes.flat,fields):
        points=[r for r in groups if r['field']==field]
        labels=[f"{r['value']} (n={r['case_count']})"+('*' if r['small_n'] else '') for r in points]
        ax.bar(range(len(points)),[r['mean_relative_l1'] for r in points]);ax.set_xticks(range(len(points)),labels,rotation=35,ha='right')
        ax.set(title=field,ylabel='Mean relative L1')
    fig.suptitle('Val-A-V4; * denotes n<3');save(fig,Path(folder)/'condition_stratification.png')
    names=('gt_max_concentration','gt_mean_concentration','active_voxel_fraction','gt_integrated_concentration_proxy','concentration_center_y_m','concentration_center_x_m')
    fig,axes=plt.subplots(2,3,figsize=(12,7),constrained_layout=True)
    for ax,name in zip(axes.flat,names):
        ax.scatter([r[name] for r in rows],[r['relative_l1'] for r in rows],s=18);ax.set(xlabel=name,ylabel='relative L1')
    save(fig,Path(folder)/'error_vs_plume_characteristics.png')


def multiseed_figure(rows,folder):
    fig,ax=plt.subplots(figsize=(6,4));x=np.arange(len(rows));width=.35
    ax.bar(x-width/2,[r['val_a_relative_l1'] for r in rows],width,label='Pool A')
    ax.bar(x+width/2,[r['val_b_relative_l1'] for r in rows],width,label='Pool B')
    ax.set_xticks(x,[str(r['seed']) for r in rows]);ax.set(ylabel='Mean per-case relative L1');ax.legend()
    save(fig,Path(folder)/'multiseed_relative_l1.png')


def horizon_figure(reference_rows,diagnostic_rows,folder):
    fig,axes=plt.subplots(1,2,figsize=(10,3.8),constrained_layout=True)
    for label,rows in (('Reference cosine T_max=150',reference_rows),('Diagnostic cosine T_max=220',diagnostic_rows)):
        axes[0].plot([r['epoch']+1 for r in rows],[r['val']['loss']['total'] for r in rows],label=label)
        axes[1].plot([r['epoch']+1 for r in rows],[r['val']['metrics']['mean_per_case']['relative_l1'] for r in rows],label=label)
    for ax,title in zip(axes,('Validation loss','Validation relative L1')):
        ax.set(title=title,xlabel='Epoch',yscale='log');ax.legend(fontsize=8)
    save(fig,Path(folder)/'horizon_probe.png')
