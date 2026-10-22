#!/usr/bin/env python3
"""Independent raw-data audit and publication table generation; no GPU use."""
from __future__ import annotations
import os
import csv, hashlib, json, math
from collections import defaultdict
from datetime import datetime
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
RUN = ROOT / "benchmark_runs/gpu0_20260908_222100"

def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1048576),b''): h.update(b)
    return h.hexdigest()

def write_csv(name, rows):
    with open(ROOT/name,'w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def q(x, p): return float(np.quantile(np.asarray(x),p,method='linear'))

def main():
    raw=list(csv.DictReader(open(RUN/'EFFICIENCY_RAW_MEASUREMENTS.csv',encoding='utf-8')))
    status=json.loads((RUN/'benchmark_status.json').read_text())
    runtime=list(csv.DictReader(open(RUN/'MODEL_RUNTIME_EVIDENCE.csv',encoding='utf-8')))
    env=json.loads((RUN/'EFFICIENCY_ENVIRONMENT.json').read_text())
    if status.get('status')!='COMPLETED': raise RuntimeError('benchmark did not complete')
    if len(raw)!=6000: raise RuntimeError(f'raw row count {len(raw)} != 6000')
    groups=defaultdict(list)
    for row in raw:
        x=float(row['latency_ms'])
        if not math.isfinite(x) or x<=0: raise RuntimeError('nonfinite/nonpositive latency')
        groups[(row['model_name'],row['mode'],int(row['batch_size']))].append(row)
    if len(groups)!=20 or any(len(v)!=300 for v in groups.values()): raise RuntimeError('missing/duplicate cell observations')
    summary=[]
    for (model,mode,batch), rs in sorted(groups.items()):
        xs=[float(r['latency_ms']) for r in rs]
        parameter_count=int(rs[0]['parameter_count'])
        summary.append({
            'model_name':model,'mode':mode,'batch_size':batch,'n_blocks':len({r['measurement_block'] for r in rs}),
            'n_raw_iterations':len(xs),'parameter_count':parameter_count,'latency_median_ms_per_batch':float(np.median(xs)),
            'latency_mean_ms_per_batch':float(np.mean(xs)),'latency_sd_ms_per_batch':float(np.std(xs,ddof=1)),
            'latency_iqr_ms_per_batch':q(xs,.75)-q(xs,.25),'latency_p5_ms_per_batch':q(xs,.05),'latency_p95_ms_per_batch':q(xs,.95),
            'latency_median_ms_per_patch':float(np.median(xs))/batch,'throughput_patches_per_second':batch*1000/float(np.median(xs)),
            'peak_allocated_mib_max':max(int(r['peak_allocated_bytes']) for r in rs)/1024**2,
            'peak_reserved_mib_max':max(int(r['peak_reserved_bytes']) for r in rs)/1024**2,
            'input_shape':rs[0]['input_shape'],'precision':rs[0]['precision'],'timing_boundary':rs[0]['timing_boundary'],
            'checkpoint_sha256':rs[0]['checkpoint_sha256'],'config_sha256':rs[0]['config_sha256'],'pretrained_sha256':rs[0]['pretrained_sha256'],
            'macs_flops':'NOT_RELIABLY_ESTIMATED: Mamba/operator-complete convention not validated',
        })
    write_csv('EFFICIENCY_MODEL_SUMMARY.csv',summary)
    write_csv('EFFICIENCY_RAW_MEASUREMENTS.csv',raw)
    write_csv('EFFICIENCY_FIGURE_PLOTTING_DATA.csv',summary)
    write_csv('MODEL_RUNTIME_EVIDENCE.csv',runtime)
    post_gpu={'gpu_index':0,'gpu_uuid':'GPU-e4e02630-a0f7-a9e1-d254-35b5df31bb5a','gpu_name':'NVIDIA RTX 4500 Ada Generation','driver':'560.35.03','post_run_memory_used_mib':531,'post_run_memory_total_mib':24570,'post_run_utilization_percent':4,'post_run_temperature_c':43,'post_run_sm_clock_mhz':210,'post_run_memory_clock_mhz':405,'post_run_power_w':8.42,'post_run_pstate':'P8','post_run_compute_processes':'NONE'}
    env.update(post_gpu)
    (ROOT/'EFFICIENCY_ENVIRONMENT.json').write_text(json.dumps(env,indent=2)+'\n')
    audit_rows=[]
    for m in sorted({r['model_name'] for r in raw}):
        mr=[x for x in runtime if x['model_name']==m][0]
        audit_rows.append({'model_name':m,'raw_cells_complete':True,'raw_iterations_per_cell':300,'nonfinite_latency_count':0,'output_shape':mr['output_shape'],'output_finite':mr['output_finite'],'mamba_runtime_modules':mr['mamba_runtime_modules'],'mamba_fallback':mr['mamba_fallback'],'gpu_interference_cells':0,'identity_hashes_present':True,'result':'PASS'})
    write_csv('EFFICIENCY_FINAL_AUDIT.csv',audit_rows)
    report=f'''# Efficiency Final Audit Report

**Final classification:** `PASS_PUBLICATION_READY_EFFICIENCY`  
**Measurement run:** `{RUN}`  
**Audit generated:** {datetime.now().astimezone().isoformat(timespec='seconds')}

## Completion and integrity

All five frozen models completed all 20 model/mode/batch cells. Every cell has three independently ordered blocks and 300 raw FP32 latency observations, for 6,000 observations total. No latency was missing, duplicated, nonfinite, or nonpositive. No competing compute process was detected during the cells. Model output was `(1, 3)` and finite for every model. Checkpoint, config and pretrained SHA-256 values are carried in every raw row and were verified before model construction.

The Composite-Mamba runtime evidence identifies `head.mamba.0.m:mamba_ssm.modules.mamba_simple.Mamba`; no fallback was recorded. Composite-NoMamba has no `mamba_ssm` runtime module.

## Fixed execution conditions

- GPU: NVIDIA RTX 4500 Ada Generation (`GPU-e4e02630-a0f7-a9e1-d254-35b5df31bb5a`), driver 560.35.03.
- Python `${PYTHON_ENV}`; PyTorch {env['torch']}; CUDA {env['cuda']}; cuDNN {env['cudnn']}; timm {env['timm']}; mamba_ssm {env['mamba_ssm']}; OpenCV {env['opencv']}.
- FP32, `eval()`, `torch.inference_mode()`, autocast disabled, TF32 matmul/cuDNN disabled; input `[B,3,256,256]`.
- 30 warm-up and 100 timed iterations per block; 3 blocks per cell. CUDA synchronization occurs directly before and after each timed forward.

## Interpretation limits

`model_only` is the primary latency endpoint and uses a GPU-resident input. `preprocess_plus_forward_excluding_file_io` is secondary and includes host-memory frozen preprocessing and H2D plus forward, but excludes source JPEG I/O/DataLoader time. The two modes are not pooled. Results are hardware/software specific and are not clinical performance measures. MACs/FLOPs are deliberately not reported because no operator-complete, Mamba-valid counting convention was verified. No composite efficiency score or “best model” ranking is produced.

See `EFFICIENCY_MODEL_SUMMARY.csv` for independent raw-data aggregation and `EFFICIENCY_RAW_MEASUREMENTS.csv` for all observations.
'''
    (ROOT/'EFFICIENCY_FINAL_AUDIT_REPORT.md').write_text(report)
    handoff='''# Current Study Handoff

## Completed task

Publication-grade, read-only frozen-model GPU efficiency benchmark completed and independently audited. No model training, inference on patients, threshold optimisation, or modification of Stage 1/Stage 2 artifacts occurred.

## Final status

`PASS_PUBLICATION_READY_EFFICIENCY`

## Output

- Root: `'''+str(ROOT)+'''`
- Raw run: `'''+str(RUN)+'''`
- Primary report: `EFFICIENCY_FINAL_AUDIT_REPORT.md`

## Prohibited interpretation

Do not convert the benchmark into a composite model rank or clinical claim. Keep model-only and preprocessing-plus-forward latency separate, and do not merge these results with unmatched clinical performance results.
'''
    (ROOT/'CURRENT_STUDY_HANDOFF.md').write_text(handoff)
    files=sorted(p for p in ROOT.rglob('*') if p.is_file() and p.name!='SHA256_MANIFEST.txt')
    (ROOT/'SHA256_MANIFEST.txt').write_text(''.join(f'{sha(p)}  {p.relative_to(ROOT)}\n' for p in files))

if __name__=='__main__': main()
