#!/usr/bin/env python3
"""Standalone FP32 benchmark rerun that preserves every latency observation."""
from __future__ import annotations
import os
import argparse, csv, hashlib, importlib, json, os, random, subprocess, sys, time, traceback
from pathlib import Path
import numpy as np

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def gpu_snapshot():
 try:
  q='name,driver_version,temperature.gpu,clocks.sm,clocks.mem,pstate,memory.used,memory.total'
  return subprocess.check_output(['nvidia-smi',f'--query-gpu={q}','--format=csv,noheader,nounits'],text=True).strip()
 except Exception as e:return f'not_available:{type(e).__name__}'
def import_code(root):
 sys.path.insert(0,str(root))
 for n in list(sys.modules):
  if n=='orchid' or n.startswith('orchid.'):del sys.modules[n]
 return importlib.import_module('orchid.models'),importlib.import_module('orchid.data')
def validate(row):
 import torch
 if sha(Path(row['config_path']))!=row['config_sha256']:raise RuntimeError('config SHA mismatch')
 if sha(Path(row['checkpoint_path']))!=row['checkpoint_sha256']:raise RuntimeError('checkpoint SHA mismatch')
 c=json.loads(Path(row['config_path']).read_text()); ck=torch.load(row['checkpoint_path'],map_location='cpu',weights_only=False)
 for k in ['model','config_sha256','split_sha256','code_sha256','pretrained_sha256','class_order','model_identity']:
  if k not in ck:raise RuntimeError(f'checkpoint missing {k}')
 if ck['config_sha256']!=row['config_sha256'] or ck['pretrained_sha256']!=c['pretrained_sha256']:raise RuntimeError('checkpoint identity mismatch')
 return c,ck
def preloaded_image(data):
 import cv2
 import pandas as pd
 frame=pd.read_csv(data.PATCH_MANIFEST,nrows=1); x=cv2.imread(str(frame.iloc[0]['cache_path']),cv2.IMREAD_COLOR)
 if x is None:raise RuntimeError('frozen representative cache unreadable')
 return x
def transform(image,pre,batch):
 import cv2, torch
 x=cv2.cvtColor(image,cv2.COLOR_BGR2RGB);x=cv2.resize(x,(256,256),interpolation=cv2.INTER_CUBIC).astype(np.float32)/255
 x=(x-np.asarray(pre.mean,np.float32))/np.asarray(pre.std,np.float32)
 return torch.from_numpy(np.ascontiguousarray(x.transpose(2,0,1))).unsqueeze(0).repeat(batch,1,1,1)
def quantile(a,q):return float(np.quantile(np.asarray(a),q,method='linear'))
def run_cell(row,device,spec,mode,batch,position):
 import torch
 config,ck=validate(row);models,data=import_code(Path(row['code_root']));_,prehash=models.verify_pretrained(row['model_name'])
 if prehash!=config['pretrained_sha256']:raise RuntimeError('runtime pretrained SHA mismatch')
 model=models.create_model(row['model_name'],pretrained=False).to(device).float().eval(); missing,unexpected=model.load_state_dict(ck['model'],strict=True)
 if missing or unexpected or next(model.parameters()).dtype!=torch.float32:raise RuntimeError('state or FP32 mismatch')
 # Explicit Mamba/no-Mamba identity evidence; no fallback is accepted.
 names=[n for n,_ in model.named_parameters()]
 if row['model_name']=='Composite-Mamba' and not any(n.startswith('head.mamba.') for n in names):raise RuntimeError('Mamba parameter path missing')
 if row['model_name']=='Composite-NoMamba' and any(n.startswith('head.mamba.') for n in names):raise RuntimeError('NoMamba unexpectedly has Mamba parameters')
 image=preloaded_image(data); pre=models.preprocessing(row['model_name']); frozen=transform(image,pre,batch).to(device)
 make=lambda: transform(image,pre,batch).to(device)
 source=frozen if mode=='model_only' else None
 for _ in range(spec['warmup_iterations']):
  with torch.inference_mode():model(source if source is not None else make())
 torch.cuda.synchronize(device);torch.cuda.reset_peak_memory_stats(device); raw=[]
 before=gpu_snapshot()
 for i in range(spec['timed_iterations']):
  torch.cuda.synchronize(device);start=time.perf_counter_ns()
  with torch.inference_mode():model(source if source is not None else make())
  torch.cuda.synchronize(device);raw.append((time.perf_counter_ns()-start)/1e6)
 after=gpu_snapshot();alloc=torch.cuda.max_memory_allocated(device);reserved=torch.cuda.max_memory_reserved(device)
 params=sum(p.numel() for p in model.parameters()); del model;torch.cuda.empty_cache()
 common={'model_name':row['model_name'],'mode':mode,'batch_size':batch,'precision':'fp32_tf32_disabled','input_shape':f'{batch}x3x256x256','warmup_iterations':spec['warmup_iterations'],'timed_iterations':spec['timed_iterations'],'execution_position':position,'checkpoint_sha256':row['checkpoint_sha256'],'config_sha256':row['config_sha256'],'pretrained_sha256':prehash,'cuda_synchronize_each_iteration':True,'data_loading_excluded':True,'model_initialization_excluded':True,'preprocessing':'BGR_to_RGB_INTER_CUBIC_/255_model_specific_mean_std','gpu_before':before,'gpu_after':after,'parameter_count':params,'peak_allocated_bytes':alloc,'peak_reserved_bytes':reserved}
 return common,raw
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--spec',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--gpu-id',default='0');ap.add_argument('--preflight',action='store_true');a=ap.parse_args()
 spec=json.loads(a.spec.read_text());
 for r in spec['models']:validate(r)
 if a.preflight:print(json.dumps({'preflight':'PASS','models':[r['model_name'] for r in spec['models']]}));return
 if a.output.exists():raise RuntimeError('output collision')
 os.environ['CUDA_VISIBLE_DEVICES']=str(a.gpu_id);import torch
 if not torch.cuda.is_available() or 'RTX 4500 Ada' not in torch.cuda.get_device_name(0):raise RuntimeError('required GPU unavailable')
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 a.output.mkdir(parents=True);(a.output/'benchmark_status.json').write_text(json.dumps({'status':'RUNNING'})+'\n')
 try:
  cells=[(r,m,b) for r in spec['models'] for m in ['model_only','end_to_end'] for b in [1,int(r['operational_batch_size'])]]
  rng=random.Random(spec['order_seed']);rng.shuffle(cells);rawrows=[];summ=[]
  for pos,(r,m,b) in enumerate(cells,1):
   common,lat=run_cell(r,torch.device('cuda:0'),spec,m,b,pos)
   rawrows += [dict(common,iteration=i+1,latency_ms=x) for i,x in enumerate(lat)]
   summ.append(dict(common,latency_median_ms=float(np.median(lat)),latency_iqr_ms=quantile(lat,.75)-quantile(lat,.25),latency_p5_ms=quantile(lat,.05),latency_p95_ms=quantile(lat,.95),throughput_images_per_second=b*1000/float(np.median(lat))))
  for name,rows in [('EFFICIENCY_RAW_MEASUREMENTS.csv',rawrows),('EFFICIENCY_MODEL_SUMMARY.csv',summ)]:
   with open(a.output/name,'w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
  import cv2,timm
  try:import mamba_ssm;mv=getattr(mamba_ssm,'__version__','unknown')
  except Exception:mv='IMPORT_FAILED'
  env={'gpu':torch.cuda.get_device_name(0),'gpu_snapshot_final':gpu_snapshot(),'torch':torch.__version__,'cuda':torch.version.cuda,'cudnn':torch.backends.cudnn.version(),'timm':timm.__version__,'mamba_ssm':mv,'opencv':cv2.__version__,'tf32_matmul':torch.backends.cuda.matmul.allow_tf32,'tf32_cudnn':torch.backends.cudnn.allow_tf32}
  (a.output/'EFFICIENCY_ENVIRONMENT.json').write_text(json.dumps(env,indent=2)+'\n');(a.output/'benchmark_status.json').write_text(json.dumps({'status':'COMPLETED','models':[r['model_name'] for r in spec['models']]},indent=2)+'\n')
 except Exception:
  (a.output/'benchmark_status.json').write_text(json.dumps({'status':'FAILED','traceback':traceback.format_exc()},indent=2)+'\n');raise
if __name__=='__main__':main()
