"""CPU-only tests using artificial SYNTH identities. No research training or inference."""
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
import csv

os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['PYTHONDONTWRITEBYTECODE']='1'
ROOT=Path(__file__).resolve().parents[1]
import numpy as np
import pandas as pd

def module(stage,name):
    package='synthetic_'+stage
    parent=types.ModuleType(package)
    parent.__path__=[str(ROOT/stage/'orchid')]
    parent.CLASS_NAMES=['normal','osmf','oscc']
    parent.CLASS_TO_INDEX={c:i for i,c in enumerate(parent.CLASS_NAMES)}
    sys.modules[package]=parent
    spec=importlib.util.spec_from_file_location(package+'.'+name,ROOT/stage/'orchid'/f'{name}.py')
    m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
    return m

def fixture():
    rows=[]
    for i,p in enumerate([[.9,.05,.05]]*3+[[.1,.85,.05]]):
        rows.append(dict(patient_id='SYNTH_A',source_path='SYNTH_IMAGE_A' if i<3 else 'SYNTH_IMAGE_B',target=0,patch_uid='SYNTH_'+str(i),p_normal=p[0],p_osmf=p[1],p_oscc=p[2]))
    for i,p in enumerate([[.1,.8,.1],[.1,.1,.8]],1):
        rows.append(dict(patient_id='SYNTH_'+str(i),source_path='SYNTH_SINGLE_'+str(i),target=i,patch_uid='SYNTH_X'+str(i),p_normal=p[0],p_osmf=p[1],p_oscc=p[2]))
    return pd.DataFrame(rows)

class CPUChecks(unittest.TestCase):
    def test_all_python_syntax(self):
        for p in ROOT.rglob('*.py'):ast.parse(p.read_text(),filename=str(p.relative_to(ROOT)))

    def test_original_scientific_symbol_parity(self):
        for r in json.loads((ROOT/'code_semantic_fingerprints.json').read_text()):
            t=ast.parse((ROOT/r['public_path']).read_text())
            node=next(n for n in ast.walk(t) if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name==r['symbol'])
            # Python 3.12 adds empty type_params to definitions; the source
            # fingerprint was generated with 3.9, before that AST field existed.
            for child in ast.walk(node):
                if hasattr(child,'type_params') and child.type_params==[]:
                    child._fields=tuple(f for f in child._fields if f!='type_params')
            h=hashlib.sha256(ast.dump(node,include_attributes=False).encode()).hexdigest()
            if 'public_binding_ast_sha256' in r:
                self.assertEqual(h,r['public_binding_ast_sha256'],r['public_path']+':'+r['symbol'])
                # Only the explicit pretrained-file constructor binding changed.
                node.body=[n for n in node.body if not isinstance(n,ast.FunctionDef) or n.name!='__init__']
                non_init=hashlib.sha256(ast.dump(node,include_attributes=False).encode()).hexdigest()
                self.assertEqual(non_init,r['historical_non_init_ast_sha256'])
            else:
                self.assertEqual(h,r['historical_ast_sha256'],r['public_path']+':'+r['symbol'])

    def test_image_equal_patient_aggregation(self):
        for stage in ['stage1','stage2']:
            ev=module(stage,'evaluation');p=ev.aggregate_patient_predictions(fixture())
            a=p.set_index('patient_id').loc['SYNTH_A']
            np.testing.assert_allclose([a.p_normal,a.p_osmf,a.p_oscc],[.5,.45,.05],atol=1e-14)
            self.assertEqual(a.n_images,2);self.assertEqual(a.n_patches,4)
            self.assertNotAlmostEqual(a.p_normal,.7)

    def test_patient_target_contradiction_rejected(self):
        for stage in ['stage1','stage2']:
            ev=module(stage,'evaluation');f=fixture();f.loc[0,'target']=1
            with self.assertRaises(RuntimeError):ev.aggregate_patient_predictions(f)

    def test_invalid_patch_probability_rejected(self):
        for stage in ['stage1','stage2']:
            ev=module(stage,'evaluation');f=fixture();f.loc[0,'p_normal']=1.9
            with self.assertRaises(RuntimeError):ev.aggregate_patient_predictions(f)

    def test_patient_metrics_not_patch_metrics(self):
        for stage in ['stage1','stage2']:
            ev=module(stage,'evaluation');p=ev.aggregate_patient_predictions(fixture());m=ev.patient_metrics(p)
            self.assertEqual(m['patient_n'],3);self.assertEqual(m['accuracy'],1.0)
            self.assertEqual(m['macro_f1'],1.0)
            probability=p[ev.PROBABILITY_COLUMNS].to_numpy(float)
            truth=p.target.to_numpy(int)
            self.assertAlmostEqual(m['brier_score'],np.mean(np.sum((probability-np.eye(3)[truth])**2,axis=1)))

    def test_stage_specific_selection_and_tie_break(self):
        a=dict(macro_ovr_auroc=.99,macro_f1=.4,loss=1.,epoch=0)
        b=dict(macro_ovr_auroc=.90,macro_f1=.9,loss=1.,epoch=0)
        self.assertGreater(module('stage1','evaluation').metric_key(a),module('stage1','evaluation').metric_key(b))
        self.assertGreater(module('stage2','evaluation').metric_key(b),module('stage2','evaluation').metric_key(a))
        for stage in ['stage1','stage2']:
            ev=module(stage,'evaluation')
            self.assertGreater(ev.metric_key(a),ev.metric_key(dict(a,epoch=1)))

    def test_synthetic_dataset_preprocessing_deterministic(self):
        import cv2
        with tempfile.TemporaryDirectory() as temp:
            os.environ['ORCHID_MANIFEST_DIR']=temp
            img=np.empty((16,16,3),np.uint8);img[:]=[20,80,180]
            file=Path(temp)/'SYNTH.png';self.assertTrue(cv2.imwrite(str(file),img))
            frame=pd.DataFrame([dict(cache_path=str(file),uid='SYNTH_UID',patient_key='SYNTH_PATIENT',path='SYNTH_SOURCE',source_sha256='SYNTH_NOT_RESEARCH_HASH',cls='normal')])
            for stage in ['stage1','stage2']:
                data=module(stage,'data')
                ds=data.OrchidPatchDataset(frame,data.PreprocessConfig((.5,.5,.5),(.5,.5,.5)),False)
                x,y,meta=ds[0];again=ds[0][0]
                np.testing.assert_array_equal(x.numpy(),again.numpy())
                np.testing.assert_allclose(x[:,0,0].numpy(),(np.array([180,80,20])/255-.5)/.5,atol=1e-6)
                self.assertEqual(tuple(x.shape),(3,256,256));self.assertEqual(int(y),0)

    def test_catalog_50_corrected_conditions(self):
        with (ROOT/'configs/run_catalog.csv').open() as f:rows=list(csv.DictReader(f))
        self.assertEqual(len(rows),50)
        self.assertEqual(sum(r['stage']=='Stage 1' for r in rows),20)
        seen=set()
        for r in rows:
            key=(r['stage'],r['paper_model'],r['seed'],r['outer_fold']);self.assertNotIn(key,seen);seen.add(key)
            c=json.loads((ROOT/r['config_example']).read_text())
            self.assertEqual(c['epochs'],80 if r['stage']=='Stage 1' else 40)
            self.assertEqual(c['early_stopping_patience'],12)
            self.assertFalse(c['amp']);self.assertEqual(c['precision'],'fp32')
            self.assertEqual(c['batch_size']*c['gradient_accumulation_steps'],32)
            if r['stage']=='Corrected Stage 2' and r['paper_model']=='Composite-Mamba' and r['seed']=='42':
                self.assertIn('correction',r['run_id'])

    def test_local_config_binding_collision_and_mismatch(self):
        with tempfile.TemporaryDirectory() as temp:
            d=Path(temp);split=d/'final_outer_inner_split_manifest.csv';split.write_text('SYNTHETIC_FIXTURE_ONLY\n')
            template=d/'fixture.json';config={'run_id':'SYNTH_RUN','historical_code_sha256':'SYNTH_HISTORY','split_sha256':hashlib.sha256(split.read_bytes()).hexdigest(), 'model_name':'InceptionV3', 'epochs':80, 'checkpoint_criterion':'inner_patient_macro_ovr_auroc|inner_patient_macro_f1|validation_loss|earlier_epoch'};template.write_text(json.dumps(config))
            out=d/'bound.json';env=dict(os.environ,ORCHID_MANIFEST_DIR=temp,PYTHONDONTWRITEBYTECODE='1')
            cmd=[sys.executable,str(ROOT/'scripts/configure_local_run.py'),'--stage','stage1','--template',str(template),'--output',str(out)]
            result=subprocess.run(cmd,env=env,capture_output=True,text=True);self.assertEqual(result.returncode,0)
            self.assertEqual(len(json.loads(out.read_text())['code_sha256']),64)
            self.assertNotEqual(subprocess.run(cmd,env=env,capture_output=True).returncode,0)
            split.write_text('SYNTHETIC_CHANGED\n');cmd[-1]=str(d/'other.json')
            self.assertNotEqual(subprocess.run(cmd,env=env,capture_output=True).returncode,0)

    def test_extracted_final_metric_functions_on_synthetic_only(self):
        def load(name):
            spec=importlib.util.spec_from_file_location('synthetic_'+name,ROOT/'analysis'/f'{name}.py')
            m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
        y=np.array([0,1,2]);p=np.array([[.8,.1,.1],[.1,.8,.1],[.1,.1,.8]])
        stage2=load('stage2_statistics');m,cm,pred=stage2.metrics(y,p)
        self.assertEqual(m['accuracy'],1.);self.assertEqual(cm.sum(),3)
        stage1=load('stage1_metrics')
        f=pd.DataFrame(dict(target=y,predicted_class=pred,p_normal=p[:,0],p_osmf=p[:,1],p_oscc=p[:,2]))
        n=stage1.calculate_metrics(f)
        self.assertAlmostEqual(n['brier_score'],m['multiclass_brier'])
        self.assertEqual(n['macro_f1'],m['macro_f1'])
        # bootstrap_metrics is deliberately not invoked in packaging tests.

if __name__=='__main__':unittest.main(verbosity=2)
