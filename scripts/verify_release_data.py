"""Frozen data alignment/integrity only; no metrics, predictions or bootstrap."""
import csv
import io
import json
import math
from pathlib import Path
from release_data import load_policy, read_reviewed
from prepare_local_manifests import validate_rows

ROOT = Path(__file__).resolve().parents[1]


def verify(root=ROOT):
    policy = load_policy(root)
    if len(policy) != 12: raise ValueError('expected two input files and ten OOF sets')
    def rows(path): return list(csv.DictReader(io.StringIO(read_reviewed(root,policy[path]).decode('utf-8'),newline='')))
    patches=rows('data/reproduction/patches.csv.gz')
    split=rows('data/reproduction/final_outer_inner_split_manifest.csv')
    expected=json.loads((root/'data/manifest_bundle_identity.json').read_text())['expected']
    validate_rows(patches,split,expected)
    target_index={'normal':0,'osmf':1,'oscc':2}
    patients={r['patient_id']:(int(r['outer_fold']),target_index[r['class']]) for r in split if r['split']=='outer_test'}
    counts={}
    for r in patches:
        group=counts.setdefault(r['patient_key'],{'images':set(),'patches':0})
        group['images'].add(r['source_sha256']);group['patches']+=1
    with (root/'configs/run_catalog.csv').open(newline='') as handle: catalog=list(csv.DictReader(handle))
    catalog={r['run_id']:r for r in catalog};sets=[]
    for path,entry in policy.items():
        if entry['kind']!='oof':continue
        predictions=rows(path);seen=set();max_error=0.0
        for row in predictions:
            pid=row['patient_id']
            if pid in seen or pid not in patients:raise ValueError('duplicate/extra OOF patient')
            seen.add(pid)
            if (int(row['outer_fold']),int(row['target']))!=patients[pid]:raise ValueError('OOF target/fold mismatch')
            record=catalog[row['run_id']]
            if any(str(row[k])!=str(entry[v]) for k,v in [('model_name','model'),('seed','seed')]):raise ValueError('OOF model/seed mismatch')
            if (record['stage'],record['source_model'],record['seed'],record['outer_fold'])!=(entry['stage'],row['model_name'],row['seed'],row['outer_fold']):raise ValueError('OOF run catalog mismatch')
            if entry['stage']=='Corrected Stage 2' and row['model_name']=='Composite-Mamba' and row['seed']=='42' and not row['run_id'].startswith('stage2_correction_'):raise ValueError('historical Mamba substitution')
            probs=[float(row[k]) for k in ('p_normal','p_osmf','p_oscc')]
            if any(not math.isfinite(p) or not 0<=p<=1 for p in probs):raise ValueError('invalid probability')
            error=abs(sum(probs)-1);max_error=max(max_error,error)
            if error>1e-5 or int(row['predicted_class'])!=max(range(3),key=probs.__getitem__):raise ValueError('probability sum/argmax mismatch')
            if int(row['n_images'])!=len(counts[pid]['images']) or int(row['n_patches'])!=counts[pid]['patches']:raise ValueError('patient hierarchy counts mismatch')
        if seen!=set(patients):raise ValueError('missing OOF patient')
        sets.append(dict(path=path,stage=entry['stage'],model=entry['model'],seed=entry['seed'],patients=len(seen),probability_sum_max_error=max_error))
    if sum(r['stage']=='Stage 1' for r in sets)!=4 or sum(r['stage']=='Corrected Stage 2' for r in sets)!=6:raise ValueError('wrong study family count')
    return dict(status='PASS_FIELD_REVIEWED_FROZEN_DATA_ALIGNMENT',sets=sets,patients=149,patches=len(patches),split_rows=len(split),metrics_recalculated=False,new_inference=False)


if __name__=='__main__':print(json.dumps(verify(),indent=2))
