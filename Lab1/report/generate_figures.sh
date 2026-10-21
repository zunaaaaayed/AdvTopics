#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
if [[ -z "${PYTHON_BIN:-}" ]]; then
    if [[ -x ../../.venv/bin/python ]]; then
        PYTHON_BIN=../../.venv/bin/python
    else
        PYTHON_BIN=python3
    fi
fi
"$PYTHON_BIN" -B - <<'PY'
import ast
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import torch

report = Path.cwd()
lab = report.parent
artifacts = lab / 'artifacts'
figures = report / 'figures'
data = report / 'data'
figures.mkdir(exist_ok=True)
data.mkdir(exist_ok=True)
font_dir = os.environ.get('TIMES_FONT_DIR')
if font_dir:
    for font in Path(font_dir).glob('*'):
        if font.suffix.lower() == '.ttf':
            font_manager.fontManager.addfont(str(font))
font_manager.findfont('Times New Roman', fallback_to_default=False)
plt.rcParams.update({'font.family': 'Times New Roman', 'font.size': 11,
                     'axes.titlesize': 12, 'axes.labelsize': 11,
                     'legend.fontsize': 9, 'pdf.fonttype': 42,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': True, 'grid.alpha': 0.2,
                     'savefig.bbox': 'tight'})
torch.set_num_threads(2)
sys.path.insert(0, str(lab / 'src'))
from model import ConvNet

read_json = lambda path: json.loads(path.read_text())
baseline = read_json(artifacts / 'baseline/metrics.json')
config = read_json(artifacts / 'pruning/config.json')
history = list(csv.DictReader((artifacts / 'baseline/history.csv').open()))
curves = {stage: read_json(artifacts / 'pruning' / f'{stage}_results.json')
          for stage in ('layerwise', 'global', 'finetuned', 'iterative')}
selections = read_json(artifacts / 'pruning/selections.json')
final = ast.literal_eval((artifacts / 'pruning/final_evaluation.log').read_text())
selected = selections['finetuned']['selected']
base_acc = baseline['best_validation_accuracy'] * 100
threshold = base_acc - 100 * config['accuracy_drop']
colors = ['#24649b', '#d17822', '#42816b', '#9053a0', '#a84c4c', '#666666']

def save(figure, name):
    figure.savefig(figures / f'{name}.pdf')
    figure.savefig(figures / f'{name}.png', dpi=180)
    plt.close(figure)

def rules(axis):
    axis.axhline(base_acc, color='0.3', linestyle='--', linewidth=1,
                 label=f'Baseline ({base_acc:.2f}%)')
    axis.axhline(threshold, color='0.5', linestyle=':', linewidth=1,
                 label=f'Selection threshold ({threshold:.2f}%)')

figure, axes = plt.subplots(1, 2, figsize=(10.2, 3.8))
labels = ['Layerwise L1', 'Global L1', 'Global L1 + fine-tuning', 'Iterative L1 + fine-tuning']
for axis in axes:
    for stage, label, color in zip(curves, labels, colors):
        rows = curves[stage]
        axis.plot([100*r['actual_sparsity'] for r in rows],
                  [100*r['validation_accuracy'] for r in rows], 'o-',
                  color=color, label=label, markersize=4)
    rules(axis)
    axis.scatter([60], [72.77], s=135, facecolors='none', edgecolors='black',
                 linewidths=1.2, zorder=5)
    axis.set(xlabel='Actual zero parameters (%)', ylabel='Validation accuracy (%)', xlim=(-1,92))
axes[0].set(ylim=(0,80), title='All measured models')
axes[1].set(ylim=(68.5,74.3), title='Detail near the selection threshold')
handles, labels = axes[0].get_legend_handles_labels()
figure.legend(handles, labels, loc='lower center', ncol=3, frameon=False,
              bbox_to_anchor=(0.5,-0.06))
figure.tight_layout(rect=(0,0.12,1,1))
save(figure, 'overview')

figure, axes = plt.subplots(1,2, figsize=(10.2,3.5))
epochs = [int(r['epoch']) for r in history]
for axis, fields, label in zip(axes, [('train_loss','val_loss'),('train_accuracy','val_accuracy')],
                              ['Cross-entropy loss','Accuracy (%)']):
    factor = 100 if 'accuracy' in fields[0] else 1
    for field, title, color in zip(fields,['Training','Validation'],colors):
        axis.plot(epochs,[factor*float(r[field]) for r in history],label=title,color=color,linewidth=1.2)
    axis.set(xlabel='Epoch',ylabel=label,xlim=(1,200))
axes[1].axhline(base_acc,color='0.5',linestyle='--',label='Best validation')
axes[0].legend(frameon=False,fontsize=8)
axes[1].legend(frameon=False,fontsize=8)
figure.tight_layout()
save(figure,'baseline_training')

state = torch.load(artifacts/'baseline/best_model_params.pt', map_location='cpu', weights_only=True)
weights=state['features.24.weight'].flatten()
k=round(0.2*weights.numel())
cutoff=torch.kthvalue(weights.abs(),k).values.item()
limit=float(torch.quantile(weights.abs(),0.995))
figure,axis=plt.subplots(figsize=(8.5,3.4))
axis.hist(weights.numpy(),bins=180,range=(-limit,limit),color=colors[0],alpha=0.85)
axis.axvspan(-cutoff,cutoff,color=colors[1],alpha=0.5,
             label=f'20% smallest magnitudes: |w| <= {cutoff:.5f}')
axis.set(xlabel='Baseline weight value',ylabel='Count per bin',
         title='Last convolution: features.24.weight')
axis.legend(frameon=False)
figure.tight_layout()
save(figure,'weight_histogram')

layers40=read_json(artifacts/'pruning/global_0.4_layers.json')
layers60=read_json(artifacts/'pruning/finetuned_0.6_layers.json')
layer_names=list(layers40['layers'])
y=np.arange(len(layer_names))
figure,axis=plt.subplots(figsize=(8.8,5.4))
axis.barh(y-0.18,[100*layers40['layers'][n] for n in layer_names],height=.34,color=colors[0],label='Global 40% (selected, Exercise 3)')
axis.barh(y+0.18,[100*layers60['layers'][n] for n in layer_names],height=.34,color=colors[1],label='Global 60% + fine-tuning (final)')
axis.set(yticks=y,yticklabels=layer_names,xlabel='Actual zero parameters in layer (%)',xlim=(0,100))
axis.invert_yaxis()
axis.legend(loc='lower right',frameon=False)
figure.tight_layout()
save(figure,'layer_sparsity')

figure,axes=plt.subplots(1,2,figsize=(10.2,3.8))
for idx,row in enumerate(curves['global']):
    ratio=row['requested_sparsity']
    h=read_json(artifacts/'pruning'/f'finetuned_{ratio}_history.json')
    values=[100*row['validation_accuracy']]+[100*r['validation']['accuracy'] for r in h]
    for axis in axes:
        axis.plot(range(6),values,'o-',markersize=3.5,color=colors[idx],label=f'{ratio:.0%} sparse')
for axis in axes:
    axis.axhline(threshold,color='0.3',linestyle=':',label='Selection threshold')
    axis.set(xlabel='Fine-tuning epoch (0 = before)',ylabel='Validation accuracy (%)',xticks=range(6))
axes[0].set(ylim=(18,76),title='Recovery at all sparsities')
axes[1].set(ylim=(69,74.2),title='Detail near the baseline')
handles,labels=axes[0].get_legend_handles_labels()
figure.legend(handles,labels,ncol=4,loc='lower center',frameon=False,bbox_to_anchor=(.5,-.06))
figure.tight_layout(rect=(0,.11,1,1))
save(figure,'finetuning_history')

figure,axis=plt.subplots(figsize=(9.3,3.7))
for step,row in enumerate(curves['iterative'][1:],1):
    h=read_json(artifacts/'pruning'/f'iterative_step_{step}_history.json')
    x=[(step-1)*5+r['epoch'] for r in h]
    axis.plot(x,[100*r['validation']['accuracy'] for r in h],'o-',color=colors[step-1],
              label=f'Step {step}: {100*row["actual_sparsity"]:.2f}% sparse',markersize=4)
    axis.scatter([step*5],[100*row['validation_accuracy']],marker='D',s=46,
                 facecolors='none',edgecolors='black',zorder=4,
                 label='Retained checkpoint (may be epoch 0)' if step==1 else None)
    if step<5: axis.axvline(step*5+.5,color='0.75',linestyle=':',linewidth=.8)
axis.axhline(threshold,color='0.35',linestyle='--',linewidth=1)
axis.set(xlabel='Cumulative fine-tuning epochs executed',ylabel='Validation accuracy (%)',xticks=[1,5,10,15,20,25],ylim=(71,74.3))
axis.legend(ncol=2,frameon=False,fontsize=9,loc='lower left')
figure.tight_layout()
save(figure,'iterative_history')

figure,axis=plt.subplots(figsize=(7.3,3.2))
values=[100*baseline['test']['accuracy'],100*final['test']['accuracy']]
bars=axis.bar(['Baseline (0% sparse)','Selected model (60% sparse)'],values,color=colors[:2],width=.55)
for bar,value in zip(bars,values):
    axis.text(bar.get_x()+bar.get_width()/2,value+.6,f'{value:.2f}%',ha='center')
axis.set(ylabel='Test accuracy (%)',ylim=(0,82))
figure.tight_layout()
save(figure,'final_test')

model=ConvNet()
modules=dict(model.named_modules())
params={name:sum(p.numel() for p in layer.parameters(recurse=False))
        for name,layer in modules.items() if any(True for p in layer.parameters(recurse=False))}
assert sum(params.values())==4740516

def table(name,rows):
    (data/name).write_text('\n'.join(' & '.join(map(str,row))+r' \\' for row in rows)+'\n')
pct=lambda value:f'{100*value:.2f}'+r'\%'
number=lambda value:f'{value:,}'
arch=[]
for block,(start,stop,width,size) in enumerate([(0,6,64,16),(7,13,128,8),(14,20,256,4),(21,27,512,4)],1):
    count=sum(params.get(f'features.{i}',0) for i in range(start,stop))
    arch.append([f'Block {block}',f'$2\\times(3\\times3)$ conv + BN + ReLU',width,
                 f'${size}\\times{size}$',number(count)])
arch.append(['Global pooling + classifier','Linear',100,'$1\\times1$',number(params['classifier'])])
table('architecture_rows.tex',arch)
table('layerwise_rows.tex',[[pct(r['actual_sparsity']),pct(r['validation_accuracy']),f'{r["validation_loss"]:.4f}',f'{base_acc-100*r["validation_accuracy"]:.2f}'] for r in curves['layerwise']])
table('global_rows.tex',[[pct(g['actual_sparsity']),pct(l['validation_accuracy']),pct(g['validation_accuracy']),f'{base_acc-100*g["validation_accuracy"]:.2f}'] for l,g in zip(curves['layerwise'],curves['global'])])
ft=[]
for g,f in zip(curves['global'],curves['finetuned']):
    h=read_json(artifacts/'pruning'/f'finetuned_{g["requested_sparsity"]}_history.json')
    ft.append([pct(f['actual_sparsity']),pct(g['validation_accuracy']),pct(h[-1]['validation']['accuracy']),pct(f['validation_accuracy']),f'{100*(f["validation_accuracy"]-g["validation_accuracy"]):+.2f}'])
table('finetuned_rows.tex',ft)
table('iterative_rows.tex',[[i,pct(r['actual_sparsity']),5*i,pct(r['validation_accuracy']),f'{base_acc-100*r["validation_accuracy"]:.2f}'] for i,r in enumerate(curves['iterative'])])
table('layer_rows.tex',[[r'\code{'+n.replace('_',r'\_')+'}',type(modules[n]).__name__,number(params[n]),pct(layers40['layers'][n]),pct(layers60['layers'][n])] for n in layer_names])
macros={'NParameters':number(4740516),'BaselineVal':pct(baseline['best_validation_accuracy']),
        'BaselineTest':pct(baseline['test']['accuracy']),'SelectedVal':pct(selected['validation_accuracy']),
        'SelectedTest':pct(final['test']['accuracy']),'SelectedEvalVal':pct(final['validation']['accuracy']),
        'Threshold':f'{threshold:.2f}'+r'\%', 'SelectedZeros':number(layers60['zeros']),
        'SelectedNonzero':number(layers60['parameters']-layers60['zeros']),
        'HistogramThreshold':f'{cutoff:.6f}',
        'BaselineMB':f'{(artifacts/"baseline/best_model_params.pt").stat().st_size/1e6:.2f}',
        'MaskedMB':f'{(artifacts/"pruning/global_0.4.pt").stat().st_size/1e6:.2f}',
        'DenseMB':f'{(artifacts/"pruning/global_0.4_dense.pt").stat().st_size/1e6:.2f}',
        'FinalMB':f'{(artifacts/"pruning/finetuned_0.6.pt").stat().st_size/1e6:.2f}'}
(data/'metrics.tex').write_text('% Generated from the saved experiment artifacts.\n'+
    '\n'.join('\\newcommand{\\'+key+'}{'+value+'}' for key,value in macros.items())+'\n')
# Figure/table inputs are hashed so the report can be traced to exact raw data.
inputs=[p for p in artifacts.rglob('*') if p.suffix in ('.json','.csv') and p.is_file()]
inputs += [artifacts/'pruning/final_evaluation.log', artifacts/'baseline/best_model_params.pt']
provenance={str(p.relative_to(lab)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(inputs)}
(data/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
for name in ('layerwise','global','finetuned','iterative'):
    import shutil
    shutil.copy2(artifacts/'pruning'/f'{name}_curve.png',figures/f'{name}_curve.png')
print('Generated seven vector PDF/PNG figures, copied four original curves, and wrote data-driven tables.')
print('Metrics:',json.dumps(macros))
PY
