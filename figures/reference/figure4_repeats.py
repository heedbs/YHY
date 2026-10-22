import os
from pathlib import Path
import csv, hashlib, json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import fontManager
import numpy as np

out = Path(str(Path(os.environ["ORCHID_PRIVATE_ROOT"]) / 'orchid_figure4_corrected_matched_repeats_v2_20261009'))
out.mkdir(exist_ok=True)
fontManager.addfont('/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf')
plt.rcParams.update({'font.family':'Liberation Sans','font.size':8.5,'axes.titlesize':10.5,'axes.labelsize':8.5,'axes.linewidth':0.6,'svg.fonttype':'none','pdf.fonttype':42,'xtick.labelsize':8,'ytick.labelsize':8.5})
seeds=['42','20260824','314159']
no=np.array([.8477,.8983,.8712]); ma=np.array([.8873,.9118,.8624])
d=np.array([.0396,.0135,-.0087]); lo=np.array([-.0270,-.0123,-.0468]); hi=np.array([.1187,.0406,.0215])
assert np.all(lo < 0) and np.all(hi > 0)
blue='#28628A'; gray='#7E8A92'
fig=plt.figure(figsize=(170/25.4,89/25.4),facecolor='white')
# Fixed vertical bounds ensure seed rows align across both panels and the numeric column.
a=fig.add_axes([.12,.27,.25,.52]); b=fig.add_axes([.475,.27,.245,.52]); nums=fig.add_axes([.745,.27,.25,.52])
y=np.array([2,1,0])
for ax in [a,b,nums]: ax.set_ylim(-.95,2.55)
for ax in [a,b]:
    ax.spines[['top','right']].set_visible(False)
    ax.spines['left'].set_color('#A1AAB0'); ax.spines['bottom'].set_color('#A1AAB0')
    ax.tick_params(width=.6,length=3,color='#A1AAB0')
    ax.set_yticks(y); ax.set_yticklabels(['Repeat 1','Repeat 2','Repeat 3'])
for yy,n,m in zip(y,no,ma):
    a.plot([n,m],[yy,yy],color='#B2BCC2',lw=.8,zorder=1)
    a.scatter(n,yy,s=25,facecolor='white',edgecolor=gray,lw=1,zorder=3)
    a.scatter(m,yy,s=25,facecolor=blue,edgecolor=blue,lw=.6,zorder=3)
    a.annotate(f'{n:.4f}',(n,yy),xytext=(0,-15),textcoords='offset points',ha='center',va='top',fontsize=8,color='#53636D')
    a.annotate(f'{m:.4f}',(m,yy),xytext=(0,8),textcoords='offset points',ha='center',va='bottom',fontsize=8,color=blue)
a.set_xlim(.82,.94); a.set_xticks(np.arange(.82,.941,.02)); a.set_xlabel('Macro F1',labelpad=7)
a.set_title('Performance across\ntraining repeats',loc='left',pad=12)
b.axvline(0,color='#808B92',lw=.9,ls=(0,(3,3)),zorder=0)
b.errorbar(d,y,xerr=np.array([d-lo,hi-d]),fmt='o',color=blue,markersize=4.5,elinewidth=.9,capsize=3,capthick=.8,zorder=3)
b.set_xlim(-.06,.13); b.set_xticks([-.06,0,.06,.12]); b.set_xticklabels(['−0.06','0','0.06','0.12'])
b.tick_params(axis='y',length=0); b.spines['left'].set_visible(False)
b.set_xlabel('Macro-F1 difference (Mamba − NoMamba)',labelpad=7,fontsize=8)
b.set_title('Paired macro-F1 differences',loc='left',pad=22)
nums.axis('off')
nums.text(0,1.02,'Difference [95% CI]',transform=nums.transAxes,fontsize=8.2,ha='left')
def signed(x): return f'{x:+.4f}'.replace('-', '−')
for yy,dd,ll,hh in zip(y,d,lo,hi): nums.text(0,yy,f'{signed(dd)} [{signed(ll)}, {signed(hh)}]',ha='left',va='center',fontsize=8)
fig.text(.045,.91,'A',weight='bold',fontsize=11)
fig.text(.425,.91,'B',weight='bold',fontsize=11)
fig.text(.5,.982,'Three training repeats with different random seeds, using the same patients and data splits.',ha='center',va='top',fontsize=8.5)
from matplotlib.lines import Line2D
handles=[Line2D([],[],marker='o',ls='',markerfacecolor='white',markeredgecolor=gray,label='Composite-NoMamba',markersize=5),Line2D([],[],marker='o',ls='',color=blue,label='Composite-Mamba',markersize=5)]
fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.035),ncol=2,frameon=False,fontsize=8.5,handletextpad=.5,columnspacing=2)
for ext in ['svg','pdf','png']: fig.savefig(out/f'FIGURE_4_CORRECTED_STAGE2.{ext}',dpi=600,facecolor='white')
rows=[{'seed':s,'composite_nomamba_macro_f1':f'{n:.4f}','composite_mamba_macro_f1':f'{m:.4f}','difference':f'{dd:+.4f}','ci_lower':f'{ll:+.4f}','ci_upper':f'{hh:+.4f}'} for s,n,m,dd,ll,hh in zip(seeds,no,ma,d,lo,hi)]
with (out/'FIGURE_4_SOURCE_DATA.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
(out/'FIGURE_4_CAPTION.md').write_text('Figure 4. Corrected Stage 2 matched comparison of Composite-Mamba and Composite-NoMamba. (A) Patient-level macro F1 across training repeats; lines connect the two conditions within each repeat. Points are displayed to four decimal places without error bars. (B) Frozen paired macro-F1 differences (Composite-Mamba minus Composite-NoMamba) with percentile 95% confidence intervals from 10,000 class-stratified paired-patient bootstrap repetitions (bootstrap seed 20260830). Repeats 1–3 correspond to random seeds 42, 20260824, and 314159, respectively. Random seeds control stochastic aspects of training. Each repeat evaluates the same 149 patients using the same patient-wise data splits and patient-level out-of-fold predictions. All three primary confidence intervals include zero. Repeated seeds are not 447 independent patients. Differences are the supplied frozen estimates and are not recomputed from the rounded values in panel A.\n',encoding='utf-8')
(out/'QA_REPORT.json').write_text(json.dumps({'supplied_values_preserved':True,'seed_order':seeds,'all_ci_cross_zero':True,'no_pooled_effect':True,'no_p_values':True,'font':'Liberation Sans (Arial-compatible substitute)','width_mm':170,'png_dpi':600},indent=2))
(out/'SHA256_MANIFEST.txt').write_text(''.join(f'{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}\n' for p in sorted(out.iterdir()) if p.is_file() and p.name != 'SHA256_MANIFEST.txt'))
print(out)
