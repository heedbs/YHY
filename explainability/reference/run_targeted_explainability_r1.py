#!/usr/bin/env python3
"""GPU-only runner. Uses frozen cases and checkpoints; never trains or alters source artifacts."""
from __future__ import annotations
import os
import argparse, hashlib, importlib, json, os, sys, time, traceback
from pathlib import Path
import cv2, numpy as np, pandas as pd, torch

ROOT=Path(__file__).resolve().parent; PATCH=Path(str(Path(os.environ["ORCHID_PRIVATE_ROOT"]) / 'orchid_comparative_protocol_v2_freeze_20260824_120821/final_unique_patch_manifest.csv'))
CASES=ROOT/'EXPLAINABILITY_CASE_SELECTION_FROZEN.csv'; CLASSES=['normal','osmf','oscc']
S1=Path(str(Path(os.environ["ORCHID_PRIVATE_ROOT"]) / 'orchid_comparative_v3_minimal')); S2=Path(str(Path(os.environ["ORCHID_PRIVATE_ROOT"]) / 'orchid_comparative_stage2_minimal')); CORR_QUEUE=Path(str(Path(os.environ["ORCHID_PRIVATE_ROOT"]) / 'orchid_stage2_seed42_mamba_correction_20260908_224742/seed42_mamba_correction_queue.csv'))
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for c in iter(lambda:f.read(1048576),b''):h.update(c)
 return h.hexdigest()
def status(out,**kw):
 t=out/'status.tmp'; t.write_text(json.dumps(kw,indent=2));os.replace(t,out/'status.json')
def load_model(impl, name, ckpt, device):
 for k in list(sys.modules):
  if k=='orchid' or k.startswith('orchid.'):del sys.modules[k]
 sys.path.insert(0,str(impl)); models=importlib.import_module('orchid.models'); m=models.create_model(name,pretrained=False).to(device).eval(); c=torch.load(ckpt,map_location='cpu',weights_only=False);m.load_state_dict(c['model'],strict=True);return m,models.preprocessing(name),c
def tensor(path,pre,device):
 b=cv2.imread(str(path),cv2.IMREAD_COLOR)
 if b is None:raise RuntimeError(f'undecodable cache {path}')
 rgb=cv2.cvtColor(b,cv2.COLOR_BGR2RGB); rgb=cv2.resize(rgb,(256,256),interpolation=cv2.INTER_CUBIC); a=rgb.astype('float32')/255.;a=(a-np.array(pre.mean,dtype='float32'))/np.array(pre.std,dtype='float32');return torch.from_numpy(a.transpose(2,0,1).copy()).unsqueeze(0).to(device),rgb
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);ap.add_argument('--gpu-id',required=True);a=ap.parse_args();out=Path(a.output);device=torch.device('cuda:0');
 if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.set_grad_enabled(True)
 cases=pd.read_csv(CASES).query("availability == 'SELECTED'");patch=pd.read_csv(PATCH);status(out,status='RUNNING',started=time.strftime('%FT%T%z'),models_completed=[])
 corr=pd.read_csv(CORR_QUEUE); corr_map={int(r.outer_fold):Path(r.output_path)/'checkpoints'/'best.pt' for _,r in corr.iterrows()}
 if set(corr_map) != set(range(5)) or not all(p.is_file() for p in corr_map.values()): raise RuntimeError('corrected Mamba checkpoint queue binding failure')
 specs=[('EfficientNet-Lite2',S1,'EfficientNet-Lite2',lambda f:S1/'stage1_outputs'/'efficientnet_lite2'/f'fold{f}'/'checkpoints'/'best.pt','Stage1_EfficientNet-Lite2_seed42'),('Composite-NoMamba',S2,'Composite-NoMamba',lambda f:S2/'stage2_outputs'/'composite_nomamba'/'seed42'/f'fold{f}'/'checkpoints'/'best.pt','Stage2_corrected_matched_seed42'),('Composite-Mamba',S2,'Composite-Mamba',lambda f:corr_map[int(f)],'Stage2_corrected_matched_seed42')]
 audit=[]
 for tag,impl,name,cp,setname in specs:
  subset=cases[cases.selection_set==setname]
  for fold,cs in subset.groupby('fold'):
   ck=cp(int(fold));m,pre,c=load_model(impl,name,ck,device); layer=(getattr(getattr(m,'backbone',m),'model',getattr(m,'backbone',m)).blocks[-1]); acts=[]
   def hook(_m,_i,o): acts[:] = [o]
   h=layer.register_forward_hook(hook)
   for _,case in cs.iterrows():
    frame=patch[patch.patient_key.astype(str).eq(str(case.patient_id))].sort_values('uid'); scores=[]
    for _,r in frame.iterrows():
     x,_=tensor(r.cache_path,pre,device)
     with torch.inference_mode():pr=torch.softmax(m(x),1)[0].cpu().numpy()
     scores.append((str(r.uid),str(r.cache_path),pr))
    target=int(case.target); scores.sort(key=lambda z:(float(z[2][target]),z[0])); picks=[('median_evidence',scores[(len(scores)-1)//2]),('high_evidence',scores[-1])]
    for kind,(uid,path,probs) in picks:
     x,rgb=tensor(path,pre,device)
     for target_name,t in [('true_class',target),('predicted_class',int(np.argmax(probs)))]:
      m.zero_grad(set_to_none=True);acts.clear();logits=m(x);activation=acts[0];activation.retain_grad();logits[0,t].backward();g=activation.grad;cam=torch.relu((g.mean((2,3),keepdim=True)*activation).sum(1))[0].detach().cpu().numpy();
      if not np.isfinite(cam).all() or cam.max()<=cam.min():raise RuntimeError('invalid Grad-CAM')
      cam=(cam-cam.min())/(cam.max()-cam.min());cam=cv2.resize(cam,(rgb.shape[1],rgb.shape[0]),interpolation=cv2.INTER_LINEAR);hm=cv2.applyColorMap(np.uint8(cam*255),cv2.COLORMAP_JET);hm=cv2.cvtColor(hm,cv2.COLOR_BGR2RGB);ov=cv2.addWeighted(rgb,.55,hm,.45,0);base=out/'images'/tag/f'fold{fold}'/f'{case.patient_id}_{uid}_{kind}_{target_name}';base.parent.mkdir(parents=True,exist_ok=True);cv2.imwrite(str(base)+'_original.png',cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR));cv2.imwrite(str(base)+'_heatmap.png',cv2.cvtColor(hm,cv2.COLOR_RGB2BGR));cv2.imwrite(str(base)+'_overlay.png',cv2.cvtColor(ov,cv2.COLOR_RGB2BGR));audit.append({'model':tag,'fold':fold,'patient_id':case.patient_id,'patch_uid':uid,'patch_kind':kind,'target_kind':target_name,'target_class':t,'patch_predicted_class':int(np.argmax(probs)),'p_normal':probs[0],'p_osmf':probs[1],'p_oscc':probs[2],'checkpoint_sha256':sha(ck),'heatmap_finite':True})
   h.remove()
  status(out,status='RUNNING',started=time.strftime('%FT%T%z'),models_completed=[x[0] for x in specs[:specs.index((tag,impl,name,cp,setname))+1]])
 pd.DataFrame(audit).to_csv(out/'EXPLAINABILITY_RUNTIME_AUDIT.csv',index=False);status(out,status='PASS_INTERNAL_EXPLAINABILITY_REVIEW_READY',finished=time.strftime('%FT%T%z'),models_completed=[x[0] for x in specs])
if __name__=='__main__':
 try:
  main()
 except Exception as exc:
  if '--output' in sys.argv:
   output=Path(sys.argv[sys.argv.index('--output')+1]); output.mkdir(parents=True,exist_ok=True)
   status(output,status='FAILED',finished=time.strftime('%FT%T%z'),error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc())
  raise
