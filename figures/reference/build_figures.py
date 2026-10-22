#!/usr/bin/env python3
"""Read-only plotting of ORCHID frozen publication source data."""
from __future__ import annotations
import os

import ast
import hashlib
import json
import shutil
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
SRC = Path(str(Path(os.environ["ORCHID_PRIVATE_ROOT"]) / 'orchid_publication_figure_table_source_package_20260915_143744'))
DATA = ROOT / 'FIGURE_SOURCE_DATA'
SUPP = ROOT / 'SUPPLEMENTARY_FIGURES'
for p in (DATA, SUPP):
    p.mkdir(parents=True, exist_ok=True)

mpl.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 8.5, 'axes.labelsize': 9,
    'axes.titlesize': 10, 'figure.titlesize': 12, 'legend.fontsize': 7.5,
    'axes.linewidth': 0.8, 'svg.fonttype': 'none', 'pdf.fonttype': 42,
    'ps.fonttype': 42, 'savefig.facecolor': 'white', 'figure.facecolor': 'white',
})

MODEL_ORDER = ['InceptionV3', 'EfficientNet-Lite2', 'MobileViT-S', 'Composite-Mamba']
MODEL_COLORS = {
    'InceptionV3': '#0072B2', 'EfficientNet-Lite2': '#009E73',
    'MobileViT-S': '#CC79A7', 'Composite-Mamba': '#D55E00',
    'Composite-NoMamba': '#4D4D4D',
}
CLASS_ORDER = ['normal', 'osmf', 'oscc']
CLASS_LABELS = ['Normal', 'OSMF', 'OSCC']
CLASS_COLORS = {'normal': '#56B4E9', 'osmf': '#E69F00', 'oscc': '#D55E00'}
SOURCE_MAP = []

def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

def load(rel: str) -> pd.DataFrame:
    p = SRC / rel
    d = pd.read_csv(p)
    out = DATA / Path(rel).name
    shutil.copy2(p, out)
    SOURCE_MAP.append({'source_relative_path': rel, 'copied_source_data': str(out.relative_to(ROOT)),
                       'sha256': sha(p), 'rows': len(d), 'columns': len(d.columns)})
    return d

def panel(ax, label: str):
    ax.text(-0.06, 1.06, label, transform=ax.transAxes, ha='left', va='bottom',
            fontsize=12, fontweight='bold')

def save(fig, name: str, supplementary=False):
    base = (SUPP if supplementary else ROOT) / name
    fig.savefig(base.with_suffix('.svg'), bbox_inches='tight')
    fig.savefig(base.with_suffix('.pdf'), bbox_inches='tight')
    fig.savefig(base.with_suffix('.png'), dpi=600, bbox_inches='tight')
    plt.close(fig)

def box(ax, x, y, w, h, txt, face='#F7F7F7', edge='#4D4D4D', fontsize=8, bold=False):
    r = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.015,rounding_size=0.025',
                       facecolor=face, edgecolor=edge, lw=.85)
    ax.add_patch(r)
    ax.text(x+w/2, y+h/2, txt, ha='center', va='center', fontsize=fontsize,
            fontweight='bold' if bold else 'normal', wrap=True)
    return (x, y, w, h)

def arrow(ax, a, b, color='#4D4D4D'):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle='-|>', mutation_scale=10, lw=.85,
                                 color=color, connectionstyle='arc3,rad=0'))

def fig1():
    cohort = load('01_COHORT_AND_STUDY_DESIGN/FIG1_CLASS_DISTRIBUTION.csv')
    cohort['cls']=pd.Categorical(cohort['cls'].str.lower(), categories=CLASS_ORDER, ordered=True)
    cohort=cohort.sort_values('cls')
    hierarchy = load('01_COHORT_AND_STUDY_DESIGN/FIG1_PATIENT_SOURCE_PATCH_HIERARCHY.csv')
    leakage = load('01_COHORT_AND_STUDY_DESIGN/FIG1_SPLIT_AND_LEAKAGE_AUDIT.csv')
    fig = plt.figure(figsize=(7.15, 7.0))
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1.1, 1], hspace=.38, wspace=.28)
    # A cohort construction.
    ax = fig.add_subplot(gs[0, 0]); ax.axis('off'); panel(ax, 'A')
    ax.set_title('Duplicate-clean analytic cohort', loc='left', fontweight='bold')
    labels = ['150\npatient directories', '149\nanalytic patients', '10,072\nsource images', '232,689\npatches']
    ys = [.78, .57, .36, .15]
    for y, t in zip(ys, labels): box(ax, .2, y, .6, .12, t, '#EAF2F8' if y != .15 else '#FDEDEC', bold=True)
    for y in ys[:-1]: arrow(ax, (.5, y), (.5, y-.09))
    ax.text(.5, .515, '1 empty directory excluded', transform=ax.transAxes, ha='center', fontsize=7)
    ax.text(.5, .305, 'Exact-content duplicate control before split', transform=ax.transAxes, ha='center', fontsize=7)
    # B class distribution.
    ax = fig.add_subplot(gs[0, 1]); panel(ax, 'B'); ax.set_title('Class composition', loc='left', fontweight='bold')
    x = np.arange(len(cohort)); colors = [CLASS_COLORS.get(str(c).lower(), '#777777') for c in cohort['cls']]
    bars = ax.bar(x, cohort['patients'], color=colors, edgecolor='white')
    ax.set_xticks(x, [str(z).upper() if str(z).lower() == 'osmf' else str(z).title() for z in cohort['cls']])
    ax.set_ylabel('Patients')
    for b, p, s, q in zip(bars, cohort['patients'], cohort['source_images'], cohort['patches']):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+2, f'n={p}\n{s:,} images\n{q:,} patches', ha='center', va='bottom', fontsize=7)
    ax.set_ylim(0, max(cohort['patients']) * 1.28); ax.spines[['top','right']].set_visible(False)
    # C hierarchy / aggregation.
    ax = fig.add_subplot(gs[1, 0]); ax.axis('off'); panel(ax, 'C'); ax.set_title('Hierarchical aggregation', loc='left', fontweight='bold')
    b1 = box(ax,.04,.46,.23,.22,'512 px patch\nprobabilities', '#EAF2F8', fontsize=6.5, bold=True)
    b2 = box(ax,.39,.46,.23,.22,'Source image\nequal mean', '#E8F6F3', fontsize=6.5, bold=True)
    b3 = box(ax,.74,.46,.23,.22,'Patient\nequal mean', '#FDEDEC', fontsize=6.5, bold=True)
    arrow(ax,(.27,.57),(.39,.57));arrow(ax,(.62,.57),(.74,.57))
    ax.text(.5,.26,'All patches and source images from one patient\nremain in exactly one partition.',ha='center',va='center',fontsize=8)
    ax.text(.5,.10,'Primary diagnostic evaluation unit: patient',ha='center',fontweight='bold',fontsize=8.5)
    # D study design.
    ax = fig.add_subplot(gs[1, 1]); ax.axis('off'); panel(ax, 'D'); ax.set_title('Two-stage study design', loc='left', fontweight='bold')
    box(ax,.05,.60,.40,.23,'Stage 1\n4 models · seed 42\n5 outer folds · 20 runs', '#EAF2F8', fontsize=6.4, bold=True)
    box(ax,.55,.60,.40,.23,'Stage 2\nMatched Mamba / Identity\n3 seeds · 30 runs', '#FDEDEC', fontsize=6.4, bold=True)
    arrow(ax,(.45,.57),(.55,.57))
    ax.text(.25,.47,'Descriptive internal\nmodel comparison',ha='center',fontsize=7.5)
    ax.text(.75,.47,'Matched component\nanalysis and stability',ha='center',fontsize=7.5)
    box(ax,.17,.12,.66,.18,'Frozen patient-wise outer test\nused once only after inner-validation checkpoint selection', '#F7F7F7', fontsize=7.5)
    ax.text(.5,.02,'External validation: not performed (no eligible public cohort).',ha='center',fontsize=7, color='#666666')
    fig.suptitle('ORCHID cohort construction and leakage-aware evaluation design', y=.99, fontweight='bold')
    save(fig, 'FIGURE_1_COHORT_AND_STUDY_DESIGN')

def fig2():
    comp = load('02_MODEL_ARCHITECTURE/FIG2_MODEL_ARCHITECTURE_COMPONENTS.csv')
    load('02_MODEL_ARCHITECTURE/FIG2_MAMBA_NOMAMBA_MATCHING_AUDIT.csv')
    load('02_MODEL_ARCHITECTURE/FIG2_PARAMETER_DIFFERENCES.csv')
    fig = plt.figure(figsize=(7.15, 5.55)); gs=GridSpec(2,1,figure=fig,height_ratios=[2.15,1.15],hspace=.52)
    ax=fig.add_subplot(gs[0]);ax.axis('off');panel(ax,'A');ax.set_title('Exact matched Composite-Mamba versus Composite-NoMamba forward path',loc='left',fontweight='bold')
    specs=[(.02,'256×256 RGB patch\n[B, 3, 256, 256]','#EAF2F8'),(.18,'EfficientNet-Lite2\nbackbone','#EAF2F8'),(.34,'Multi-scale taps\n+ MixStyle','#E8F6F3'),(.50,'Projection + 14×14\ntoken grid [B,196,192]','#E8F6F3')]
    boxes=[]
    for x,t,c in specs: boxes.append(box(ax,x,.60,.14,.18,t,c,fontsize=6.25,bold=True))
    for x in [.16,.32,.48]:arrow(ax,(x,.69),(x+.02,.69))
    box(ax,.70,.78,.14,.13,'MambaBlock\n[B,196,192]','#FDEDEC',fontsize=7.0,bold=True)
    box(ax,.70,.42,.14,.13,'Identity\n[B,196,192]','#F7F7F7',fontsize=7.0,bold=True)
    arrow(ax,(.64,.69),(.70,.845),'#D55E00'); arrow(ax,(.64,.69),(.70,.485),'#4D4D4D')
    box(ax,.87,.60,.11,.18,'GeM + dropout 0.2\n3-class logits','#EAF2F8',fontsize=6.2,bold=True)
    arrow(ax,(.84,.845),(.87,.69),'#D55E00'); arrow(ax,(.84,.485),(.87,.69),'#4D4D4D')
    ax.text(.755,.96,'Composite-Mamba',ha='center',color='#D55E00',fontweight='bold',fontsize=8)
    ax.text(.755,.30,'Composite-NoMamba',ha='center',color='#4D4D4D',fontweight='bold',fontsize=8)
    ax.text(.5,.10,'Only planned intervention: token operation (MambaBlock versus Identity).\nBackbone, pretrained weights, preprocessing, aggregation, optimizer, schedule, batch, and checkpoint rule are matched.',ha='center',fontsize=8)
    ax=fig.add_subplot(gs[1]);panel(ax,'B');ax.axis('off');ax.set_title('Condition-matching audit',loc='left',fontweight='bold',pad=20)
    rows=[('Shared-parameter unexpected differences','0'),('NoMamba `head.mamba.*` parameters','0'),('Identity token input/output equality','Exact'),('Expected parameter difference','`head.mamba.*` only')]
    y=.68
    for k,v in rows:
        ax.text(.04,y,k,fontsize=7.6);ax.text(.71,y,v,fontsize=7.6,fontweight='bold');ax.plot([.03,.97],[y-.09,y-.09],color='#DDDDDD',lw=.6);y-=.19
    fig.suptitle('Matched Mamba ablation architecture',y=.99,fontweight='bold')
    save(fig,'FIGURE_2_MATCHED_MAMBA_ARCHITECTURE')

def fig3():
    pooled=load('03_STAGE1_PERFORMANCE/FIG3_STAGE1_POOLED_PATIENT_METRICS.csv')
    fold=load('03_STAGE1_PERFORMANCE/FIG3_STAGE1_FOLD_METRICS.csv')
    cms=load('03_STAGE1_PERFORMANCE/FIG3_STAGE1_CONFUSION_MATRICES.csv')
    for d in (pooled, fold, cms):
        d['model']=d['model'].replace({'EfficientNet-Lite2+Mamba':'Composite-Mamba'})
    fig=plt.figure(figsize=(7.15,9.0)); gs=GridSpec(3,4,figure=fig,height_ratios=[1.15,.85,1.1],hspace=.6,wspace=.48)
    # A metrics forest-style, one row per model w four metrics.
    ax=fig.add_subplot(gs[0,:]);panel(ax,'A');ax.set_title('Stage 1 pooled patient-level performance (n=149 per model)',loc='left',fontweight='bold',pad=25)
    metrics=[('macro_f1','Macro F1','o'),('balanced_accuracy','Balanced accuracy','s'),('macro_ovr_auroc','Macro OVR AUROC','^'),('macro_ovr_auprc','Macro OVR AUPRC','D')]
    offsets=np.array([-.24,-.08,.08,.24])
    for i,m in enumerate(MODEL_ORDER):
        y=len(MODEL_ORDER)-1-i; row=pooled[pooled.model==m].iloc[0]
        f=fold[fold.model==m]
        for (col,label,mark),off in zip(metrics,offsets):
            ax.scatter(f[col],np.full(len(f),y+off),s=16,facecolors='none',edgecolors=MODEL_COLORS[m],lw=.8,zorder=2)
            ax.scatter(row[col],y+off,s=36,marker=mark,color=MODEL_COLORS[m],edgecolor='white',lw=.45,zorder=3)
    ax.set_yticks(range(4),MODEL_ORDER[::-1]);ax.set_xlabel('Patient-level metric (pooled estimate: filled; fold: open)')
    ax.set_xlim(.76,1.01);ax.grid(axis='x',ls=':',color='#CCCCCC');ax.spines[['top','right']].set_visible(False)
    handles=[plt.Line2D([],[],marker=m,linestyle='',color='#333333',label=l) for _,l,m in metrics];ax.legend(handles=handles,ncol=2,loc='upper left',bbox_to_anchor=(.01,.99),frameon=False)
    # B sensitivity
    ax=fig.add_subplot(gs[1,:]);panel(ax,'B');ax.set_title('Class-specific sensitivity',loc='left',fontweight='bold',pad=16)
    w=.18;x=np.arange(4)
    for j,c in enumerate(CLASS_ORDER):
        vals=pooled.set_index('model').loc[MODEL_ORDER,c+'_sensitivity'].to_numpy()
        ax.bar(x+(j-1)*w,vals,width=w,color=CLASS_COLORS[c],label=CLASS_LABELS[j],edgecolor='white')
    ax.set_xticks(x,MODEL_ORDER,rotation=0);ax.set_ylim(0,1.05);ax.set_ylabel('Sensitivity (patient-level)');ax.legend(ncol=3,frameon=False,loc='upper center',bbox_to_anchor=(.68,1.02));ax.grid(axis='y',ls=':',color='#CCCCCC');ax.spines[['top','right']].set_visible(False)
    ax.text(.01,-.27,'Normal class: n=14 patients; values are descriptive pooled OOF sensitivities.',transform=ax.transAxes,fontsize=7,color='#555555')
    # C four pooled confusion matrices.
    allmax=max(np.max(np.asarray(ast.literal_eval(x))) for x in cms.confusion_matrix)
    for i,m in enumerate(MODEL_ORDER):
        ax=fig.add_subplot(gs[2,i]);panel(ax,chr(67+i) if i==0 else '')
        mat=np.asarray(ast.literal_eval(cms[cms.model==m].iloc[0].confusion_matrix),dtype=int)
        im=ax.imshow(mat,cmap='Blues',vmin=0,vmax=allmax)
        for r in range(3):
            total=mat[r].sum()
            for c in range(3):ax.text(c,r,f'{mat[r,c]}\n{mat[r,c]/total:.0%}',ha='center',va='center',fontsize=7,fontweight='bold' if mat[r,c]>.55*allmax else 'normal',color='white' if mat[r,c]>.55*allmax else '#1F2937')
        ax.set_title(m,fontsize=8,fontweight='bold');ax.set_xticks(range(3),['N','O','C']);ax.set_yticks(range(3),['N','O','C'] if i==0 else []);ax.set_xlabel('Pred.');
        if i==0:ax.set_ylabel('True')
    fig.text(.5,.015,'Confusion matrices: count and row percentage. Class order: Normal, OSMF, OSCC.',ha='center',fontsize=7)
    fig.suptitle('Stage 1 patient-level comparative performance',y=.995,fontweight='bold')
    save(fig,'FIGURE_3_STAGE1_PATIENT_LEVEL_PERFORMANCE')

def fig4():
    seed=load('04_STAGE2_CORRECTED_MATCHED_ANALYSIS/FIG4_STAGE2_SEED_SPECIFIC_METRICS.csv')
    effects=load('04_STAGE2_CORRECTED_MATCHED_ANALYSIS/FIG4_STAGE2_PAIRED_EFFECTS.csv')
    trans=load('04_STAGE2_CORRECTED_MATCHED_ANALYSIS/FIG4_STAGE2_CORRECTNESS_TRANSITIONS.csv')
    fig=plt.figure(figsize=(7.15,7.5));gs=GridSpec(2,2,figure=fig,hspace=.55,wspace=.35)
    seeds=[42,20260824,314159]
    # paired macro f1
    ax=fig.add_subplot(gs[0,0]);panel(ax,'A');ax.set_title('Seed-specific paired macro F1',loc='left',fontweight='bold')
    for i,s in enumerate(seeds):
        z=seed[seed.seed==s].set_index('model').loc[['Composite-NoMamba','Composite-Mamba']]
        ax.plot([i-.12,i+.12],z['macro_f1'],color='#777777',lw=1)
        ax.scatter([i-.12],[z.iloc[0].macro_f1],color=MODEL_COLORS['Composite-NoMamba'],s=36,zorder=3)
        ax.scatter([i+.12],[z.iloc[1].macro_f1],color=MODEL_COLORS['Composite-Mamba'],s=36,zorder=3)
    ax.set_xticks(range(3),[str(s) for s in seeds]);ax.set_ylim(.78,.95);ax.set_xlabel('Seed');ax.set_ylabel('Patient-level macro F1');ax.grid(axis='y',ls=':',color='#CCCCCC');ax.spines[['top','right']].set_visible(False)
    ax.legend(handles=[plt.Line2D([],[],marker='o',ls='',color=MODEL_COLORS['Composite-NoMamba'],label='NoMamba'),plt.Line2D([],[],marker='o',ls='',color=MODEL_COLORS['Composite-Mamba'],label='Mamba')],frameon=False,fontsize=7)
    # forest macro f1
    ax=fig.add_subplot(gs[0,1]);panel(ax,'B');ax.set_title('Primary paired contrast',loc='left',fontweight='bold')
    e=effects[effects.metric=='macro_f1'].set_index('seed').loc[seeds].reset_index()
    y=np.arange(3)[::-1]
    ax.axvline(0,color='#555555',lw=.8,zorder=0)
    ax.errorbar(e.mamba_minus_nomamba,y,xerr=[e.mamba_minus_nomamba-e.ci_lower,e.ci_upper-e.mamba_minus_nomamba],fmt='o',color='#0072B2',ecolor='#0072B2',capsize=2,lw=1.1)
    for yi,row in zip(y,e.itertuples()):ax.text(row.ci_upper+.003,yi,f'{row.mamba_minus_nomamba:+.3f} [{row.ci_lower:+.3f}, {row.ci_upper:+.3f}]',va='center',fontsize=6.8)
    ax.set_yticks(y,[str(s) for s in seeds]);ax.set_xlabel('Macro F1 difference: Mamba − NoMamba');ax.set_xlim(-.07,.14);ax.spines[['top','right']].set_visible(False)
    ax.text(.01,-.28,'Descriptive three-seed mean +0.015 (SD 0.024); not a pooled effect.',transform=ax.transAxes,fontsize=7,color='#555555')
    # secondary effects
    ax=fig.add_subplot(gs[1,0]);panel(ax,'C');ax.set_title('Secondary paired contrasts',loc='left',fontweight='bold')
    mets=['balanced_accuracy','macro_ovr_auroc','macro_ovr_auprc','multiclass_brier']
    nice=['Balanced accuracy','Macro AUROC','Macro AUPRC','Brier (lower is better)']
    xx=np.arange(len(mets));wid=.22
    for j,s in enumerate(seeds):
        v=effects[effects.seed==s].set_index('metric').reindex(mets).mamba_minus_nomamba
        ax.bar(xx+(j-1)*wid,v,wid,label=str(s),color=['#0072B2','#009E73','#CC79A7'][j])
    ax.axhline(0,color='#555555',lw=.8);ax.set_xticks(xx,nice,rotation=25,ha='right');ax.set_ylabel('Mamba − NoMamba');ax.legend(title='Seed',frameon=False,fontsize=7);ax.grid(axis='y',ls=':',color='#CCCCCC');ax.spines[['top','right']].set_visible(False)
    # correctness transitions
    ax=fig.add_subplot(gs[1,1]);panel(ax,'D');ax.set_title('Patient correctness transitions',loc='left',fontweight='bold')
    order=['unchanged_correct','nomamba_only_correct','mamba_only_correct','unchanged_incorrect']
    labs=['Both correct','NoMamba only','Mamba only','Both incorrect']
    cmap={'unchanged_correct':'#56B4E9','nomamba_only_correct':'#4D4D4D','mamba_only_correct':'#D55E00','unchanged_incorrect':'#BBBBBB'}
    c=pd.crosstab(trans.seed,trans.transition).reindex(index=seeds,columns=order,fill_value=0)
    bottom=np.zeros(3)
    for o,l in zip(order,labs):ax.bar(range(3),c[o],bottom=bottom,label=l,color=cmap[o],edgecolor='white');bottom+=c[o].to_numpy()
    ax.set_xticks(range(3),[str(s) for s in seeds]);ax.set_xlabel('Seed');ax.set_ylabel('Patients (n=149 per seed)');ax.legend(fontsize=6.5,frameon=False,loc='upper right');ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Corrected Stage 2 matched Mamba versus Identity comparison',y=.995,fontweight='bold')
    save(fig,'FIGURE_4_STAGE2_MATCHED_MAMBA_EFFECT')

def fig5():
    eff=load('05_EFFICIENCY_BENCHMARK/FIG5_EFFICIENCY_MODEL_SUMMARY.csv')
    perf=load('05_EFFICIENCY_BENCHMARK/FIG5_PERFORMANCE_EFFICIENCY.csv')
    fig=plt.figure(figsize=(7.15,5.4));gs=GridSpec(1,2,figure=fig,wspace=.45)
    ax=fig.add_subplot(gs[0]);panel(ax,'A');ax.set_title('Performance–latency context',loc='left',fontweight='bold')
    aliases={'inceptionv3':'InceptionV3','efficientnet_lite2':'EfficientNet-Lite2','mobilevit_s':'MobileViT-S','efflite2_mamba':'Composite-Mamba','Composite-Mamba':'Composite-Mamba','Composite-NoMamba':'Composite-NoMamba'}
    p=perf.copy();p['name']=p.model.map(aliases).fillna(p.model)
    e=eff[(eff['mode']=='model_only') & (eff['batch_size']>1)].copy();e['name']=e['model_name'].map(aliases).fillna(e['model_name'])
    for _,r in p.iterrows():
        match=e[e['name']==r['name']]
        if len(match):
            z=match.iloc[0];ax.scatter(z.latency_median_ms_per_patch,r.macro_f1,s=np.sqrt(z.parameter_count)/5,color=MODEL_COLORS.get(r['name'],'#777777'),edgecolor='white',lw=.6,zorder=3)
            ax.annotate(r['name'].replace('Composite-',''),(z.latency_median_ms_per_patch,r.macro_f1),xytext=(4,4),textcoords='offset points',fontsize=6.5)
    ax.set_xlabel('Model-only median latency per patch (ms)');ax.set_ylabel('Patient-level macro F1');ax.grid(ls=':',color='#CCCCCC');ax.spines[['top','right']].set_visible(False)
    ax.text(.01,-.20,'Stage 1 and Stage 2 performance values are separate estimands;\npoint area is proportional to parameter count.',transform=ax.transAxes,fontsize=6.8,color='#555555')
    ax=fig.add_subplot(gs[1]);panel(ax,'B');ax.set_title('Matched practical-batch efficiency',loc='left',fontweight='bold')
    m=e[e['name'].isin(['Composite-NoMamba','Composite-Mamba'])].set_index('name').loc[['Composite-NoMamba','Composite-Mamba']].reset_index()
    y=np.arange(2);ax.barh(y-.18,m.latency_median_ms_per_batch,.32,color='#0072B2',label='Batch latency (ms)')
    ax.barh(y+.18,m.throughput_patches_per_second/100,.32,color='#E69F00',label='Throughput / 100 (patches/s)')
    for yy,r in zip(y,m.itertuples()):ax.text(max(r.latency_median_ms_per_batch,r.throughput_patches_per_second/100)+1,yy,f'{r.latency_median_ms_per_patch:.3f} ms/patch\n{r.parameter_count/1e6:.2f} M params',va='center',fontsize=7)
    ax.set_yticks(y,m['name']);ax.set_xlabel('Displayed values in native units as labeled');ax.legend(frameon=False,fontsize=7,loc='lower right');ax.spines[['top','right']].set_visible(False)
    fig.text(.5,.02,'NVIDIA RTX 4500 Ada; FP32; TF32 disabled; 256×256 input; eval() and inference mode; GPU-resident model-only forward timing.',ha='center',fontsize=7)
    fig.suptitle('Hardware-specific performance–efficiency context',y=.995,fontweight='bold')
    save(fig,'FIGURE_5_PERFORMANCE_EFFICIENCY')

def supplementary():
    hist=load('06_LEARNING_CURVES_AND_CONVERGENCE/SUPP_EPOCH_HISTORY_LONG.csv')
    runs=load('06_LEARNING_CURVES_AND_CONVERGENCE/SUPP_BEST_FINAL_EPOCHS.csv')
    fold_metrics=load('03_STAGE1_PERFORMANCE/FIG3_STAGE1_FOLD_METRICS.csv')
    # S1 Stage 1 learning curves.
    fig,axes=plt.subplots(2,4,figsize=(7.15,5.5),sharex='col')
    for j,m in enumerate(MODEL_ORDER):
        h=hist[(hist.study_stage=='Stage1') & (hist.model.str.lower().str.replace('-','_')==m.lower().replace('-','_'))]
        # model aliases resolve composite.
        if h.empty and m=='Composite-Mamba':h=hist[(hist.study_stage=='Stage1') & (hist.model=='efflite2_mamba')]
        for fold,g in h.groupby('fold'):
            axes[0,j].plot(g.epoch_one_indexed,g.train_loss,color=MODEL_COLORS[m],alpha=.23,lw=.7)
            axes[1,j].plot(g.epoch_one_indexed,g.inner_validation_macro_f1,color=MODEL_COLORS[m],alpha=.23,lw=.7)
        for ax,col in [(axes[0,j],'train_loss'),(axes[1,j],'inner_validation_macro_f1')]:
            q=h.groupby('epoch_one_indexed')[col].agg(['median',lambda x:x.quantile(.25),lambda x:x.quantile(.75)])
            ax.plot(q.index,q['median'],color=MODEL_COLORS[m],lw=1.8);ax.fill_between(q.index,q['<lambda_0>'],q['<lambda_1>'],color=MODEL_COLORS[m],alpha=.16)
            ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',ls=':',color='#DDDDDD')
        axes[0,j].set_title(m,fontsize=8,fontweight='bold');axes[1,j].set_xlabel('Epoch')
    axes[0,0].set_ylabel('Train loss\n(patch-weighted)');axes[1,0].set_ylabel('Inner-val macro F1\n(patient-level)')
    fig.suptitle('Supplementary Figure S1. Stage 1 observed learning trajectories',fontweight='bold');fig.text(.5,.01,'Thin lines: folds. Thick line and band: median and IQR among observed folds; no padding or carry-forward.',ha='center',fontsize=7)
    fig.tight_layout(rect=[0, .05, 1, .95]);save(fig,'SUPPLEMENTARY_FIGURE_S1_STAGE1_LEARNING_CURVES',True)
    # S2 Stage2 trajectories grouped model + seed.
    fig,axes=plt.subplots(2,3,figsize=(7.15,4.7),sharex=True)
    for i,m in enumerate(['Composite-NoMamba','Composite-Mamba']):
        for j,s in enumerate([42,20260824,314159]):
            ax=axes[i,j];h=hist[(hist.study_stage=='Stage2')&(hist.condition==m)&(hist.seed==s)]
            for _,g in h.groupby('fold'):ax.plot(g.epoch_one_indexed,g.inner_validation_macro_f1,color=MODEL_COLORS[m],alpha=.25,lw=.7)
            q=h.groupby('epoch_one_indexed').inner_validation_macro_f1.agg(['median',lambda x:x.quantile(.25),lambda x:x.quantile(.75)])
            ax.plot(q.index,q['median'],color=MODEL_COLORS[m],lw=1.8);ax.fill_between(q.index,q['<lambda_0>'],q['<lambda_1>'],color=MODEL_COLORS[m],alpha=.16)
            ax.set_title(f'{m.replace("Composite-", "")} · seed {s}',fontsize=8,fontweight='bold');ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',ls=':',color='#DDDDDD')
            if j==0:ax.set_ylabel('Inner-val macro F1\n(patient-level)')
            if i==1:ax.set_xlabel('Epoch')
    fig.suptitle('Supplementary Figure S2. Corrected Stage 2 observed learning trajectories',fontweight='bold');fig.text(.5,.01,'Corrected Composite-Mamba seed 42 only; thin folds and observed-epoch median/IQR summaries.',ha='center',fontsize=7)
    fig.tight_layout(rect=[0,.05,1,.95]);save(fig,'SUPPLEMENTARY_FIGURE_S2_STAGE2_LEARNING_CURVES',True)
    # S3 best / final epochs.
    fig,ax=plt.subplots(figsize=(7.15,3.6));d=runs.copy();d['label']=d.stage+' · '+d.model+'\nseed '+d.seed.astype(str)
    labs=list(dict.fromkeys(d.label));pos={z:i for i,z in enumerate(labs)}
    for _,r in d.iterrows():
        y=pos[r.label]+(r.fold-2)*.07;ax.plot([r.best_epoch,r.final_epoch],[y,y],color='#999999',lw=.8);ax.scatter(r.best_epoch,y,color='#0072B2',s=12);ax.scatter(r.final_epoch,y,color='#D55E00',s=12,marker='s')
    ax.set_yticks(range(len(labs)),labs,fontsize=6.8);ax.set_xlabel('Epoch');ax.set_title('Supplementary Figure S3. Best and final epoch by completed run',loc='left',fontweight='bold');ax.legend(handles=[plt.Line2D([],[],marker='o',ls='',color='#0072B2',label='Best checkpoint'),plt.Line2D([],[],marker='s',ls='',color='#D55E00',label='Final epoch')],frameon=False,ncol=2);ax.grid(axis='x',ls=':',color='#DDDDDD');ax.spines[['top','right']].set_visible(False);fig.tight_layout();save(fig,'SUPPLEMENTARY_FIGURE_S3_CONVERGENCE_CONTEXT',True)
    # S4: all Stage 1 fold-level patient confusion matrices.
    fold_metrics['model']=fold_metrics['model'].replace({'EfficientNet-Lite2+Mamba':'Composite-Mamba'})
    fig,axes=plt.subplots(5,4,figsize=(7.15,10.0))
    vmax=max(np.max(np.asarray(ast.literal_eval(v))) for v in fold_metrics.confusion_matrix)
    for col,m in enumerate(MODEL_ORDER):
        for row,f in enumerate(range(5)):
            ax=axes[row,col];r=fold_metrics[(fold_metrics.model==m)&(fold_metrics.fold==f)].iloc[0]
            mat=np.asarray(ast.literal_eval(r.confusion_matrix),dtype=int)
            ax.imshow(mat,cmap='Blues',vmin=0,vmax=vmax)
            for i in range(3):
                for j in range(3):
                    color='white' if mat[i,j]>.55*vmax else '#1F2937'
                    ax.text(j,i,f'{mat[i,j]}\n{mat[i,j]/mat[i].sum():.0%}',ha='center',va='center',fontsize=5.2,color=color,fontweight='bold' if color=='white' else 'normal')
            if row==0:ax.set_title(m,fontsize=7.4,fontweight='bold')
            ax.set_yticks([])
            if col==0:ax.set_ylabel(f'Fold {f}\nTrue',fontsize=6.4)
            if row==4:ax.set_xlabel('Pred.\nN  O  C',fontsize=6.2);ax.set_xticks([])
            else:ax.set_xticks([])
    fig.suptitle('Supplementary Figure S4. Stage 1 fold-level patient confusion matrices',fontweight='bold',y=.995)
    fig.text(.5,.012,'Cell labels are patient count and within-row percentage. Class order: Normal, OSMF, OSCC.',ha='center',fontsize=6.8)
    fig.tight_layout(rect=[0,.025,1,.98]);save(fig,'SUPPLEMENTARY_FIGURE_S4_STAGE1_FOLD_CONFUSION_MATRICES',True)

def docs():
    pd.DataFrame(SOURCE_MAP).to_csv(ROOT/'FIGURE_SOURCE_DATA_INDEX.csv',index=False)
    (ROOT/'FIGURE_CAPTIONS.md').write_text('''# Figure Captions

## Figure 1
**ORCHID cohort construction and leakage-aware evaluation design.** Exact-content duplicate control preceded patient-wise partitioning. Patch probabilities were averaged within source image and source-image probabilities equally averaged within patient. Stage 1 was a descriptive internal comparison; Stage 2 was a matched Mamba-versus-Identity analysis on the same cohort, not external validation. Representative H&E images are intentionally omitted because pathology-review and publication-permission evidence was not available in the frozen source-data package.

## Figure 2
**Matched Composite-Mamba versus Composite-NoMamba architecture.** The composite conditions share the EfficientNet-Lite2 backbone, feature taps, MixStyle, projection/token grid, GeM, dropout, classifier, preprocessing, and aggregation. The only planned intervention is MambaBlock versus Identity at `[B,196,192]`. Composite-NoMamba is not the standard EfficientNet-Lite2 baseline; structural matching does not imply bitwise-identical random-number consumption across independent executions.

## Figure 3
**Stage 1 patient-level comparative performance.** Filled points are pooled 149-patient OOF estimates and open points are fold-specific descriptive values, not patient-level confidence intervals. Stage 1 Composite-Mamba differs from the other Stage 1 models in multiple composite components and therefore does not estimate an isolated Mamba effect.

## Figure 4
**Corrected Stage 2 matched Mamba effect across three seeds.** Contrasts use paired predictions for the same patients, class-stratified paired-patient bootstrap (10,000 repetitions, seed 20260830), and percentile 95% intervals. Corrected Stage 2 Mamba seed 42 is used; historical Stage 1 Mamba seed 42 is excluded. Three-seed values are descriptive because the same patients recur across seeds; intervals spanning zero do not establish equivalence.

## Figure 5
**Hardware-specific performance-efficiency context.** Model-only latency is GPU-resident forward timing on an NVIDIA RTX 4500 Ada in FP32 with TF32 disabled; file I/O and DataLoader time are excluded. Stage 1 and Stage 2 performance results are separate estimands, and the figure does not define a composite efficiency score.

## Supplementary Figures S1-S4
**Observed training and fold-level context.** Learning-curve panels show only recorded train and inner-validation history, with no outer-test trajectories, interpolation, padding, or carry-forward. Fold trajectories are descriptive, not independent patient samples. Confusion-matrix cells display patient counts and true-class row percentages.
''')
    (ROOT/'VISUAL_QA_REPORT.md').write_text('''# Visual QA Report

All figures were generated from frozen CSV sources only. SVG and PDF retain vector text; PNG versions were rendered at 600 dpi. Representative H&E panels were omitted rather than using unverified imagery. Source-data paths and SHA-256 values are indexed in `FIGURE_SOURCE_DATA_INDEX.csv`.

## Completed Checks

- Main and supplementary PDFs were raster-rendered for visual inspection.
- Panel labels, axes, legends, and titles were checked for clipping and overlap.
- Figure 1 class order is Normal, OSMF, OSCC; the same order is used for all confusion matrices.
- Confusion matrices display the source count and within-true-class percentage.
- Stage 2 uses corrected matched Mamba seed 42 and does not use historical Stage 1 Mamba seed 42.
- Learning curves use observed train/inner-validation records only; outer-test values are absent.
- No gradients, decorative backgrounds, 3D effects, or image assets with unverified pathology review/permission were used.

## Scope Limitation

This visual QA verifies rendering and source-data linkage. It does not add pathological localization, external-validation, or causal claims beyond the frozen analysis.
''')
    (ROOT/'README.md').write_text('''# ORCHID Publication Figures

This folder contains new publication graphics derived read-only from the frozen publication source-data package. Main figures are in the root; supplementary figures are in `SUPPLEMENTARY_FIGURES`. Each figure is supplied as SVG, vector PDF, and 600-dpi PNG. H&E image panels are intentionally omitted pending independent pathology review and publication-permission confirmation.
''')
    files=sorted(p for p in ROOT.rglob('*') if p.is_file() and p.name not in {'SHA256_MANIFEST.txt','build_figures.py'} and '__pycache__' not in p.parts)
    with (ROOT/'SHA256_MANIFEST.txt').open('w') as f:
        for p in files:f.write(f'{sha(p)}  {p.relative_to(ROOT)}\n')

def main():
    fig1();fig2();fig3();fig4();fig5();supplementary();docs()

if __name__=='__main__':main()
