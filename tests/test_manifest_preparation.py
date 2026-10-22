"""Synthetic metadata fixtures only; no model, image decode or real patient data."""
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('local_manifest_tool',ROOT/'scripts/prepare_local_manifests.py')
tool=importlib.util.module_from_spec(spec);spec.loader.exec_module(tool)


def fixture():
    patches=[];splits=[]
    classes=('normal','osmf','oscc')
    for i in range(6):
        p='SYNTH_'+str(i);target=classes[i%3]
        patches.append(dict(uid='SYNTH_PATCH_'+str(i),x='0',y='0',w='512',h='512',cls=target,
                            patient_key=p,source_sha256=hashlib.sha256(p.encode()).hexdigest(),
                            source_relative_path='SYNTH_SOURCE_'+str(i)+'.png'))
        for f in range(5):
            split='outer_test' if i%5==f else 'inner_val' if i==next(j for j in range(6) if j%5!=f) else 'train'
            splits.append(dict(patient_id=p,outer_fold=str(f),split=split,**{'class':target}))
    expected=dict(patients=6,sources=6,patches=6,class_patients=dict(normal=2,osmf=2,oscc=2))
    return patches,splits,expected


class ManifestChecks(unittest.TestCase):
    def test_valid_synthetic_structure(self):
        tool.validate_rows(*fixture())

    def test_duplicate_uid_and_wrong_target(self):
        p,s,e=fixture();p[1]['uid']=p[0]['uid']
        with self.assertRaises(ValueError):tool.validate_rows(p,s,e)
        p,s,e=fixture();s[0]['class']='oscc'
        with self.assertRaises(ValueError):tool.validate_rows(p,s,e)

    def test_duplicate_or_missing_partition(self):
        p,s,e=fixture();s.append(dict(s[0]))
        with self.assertRaises(ValueError):tool.validate_rows(p,s,e)
        p,s,e=fixture();s=s[1:]
        with self.assertRaises(ValueError):tool.validate_rows(p,s,e)

    def test_path_escape_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            for name in ('../outside.png','/absolute.png','dir/../../outside.png'):
                with self.assertRaises(ValueError):tool.safe_source(root,name)
            self.assertEqual(tool.safe_source(root,'SYNTH.png'),root/'SYNTH.png')

    def test_prepare_and_collision_and_sha_fail_stop(self):
        original=tool.ROOT
        try:
            with tempfile.TemporaryDirectory() as temp:
                root=Path(temp);tool.ROOT=root
                bundle=root/'bundle';source=root/'source';cache=root/'cache'
                for d in (bundle,source,cache,root/'data',root/'configs/stage1',root/'configs/stage2'):d.mkdir(parents=True)
                p,s,e=fixture()
                def csvwrite(path,rows):
                    with path.open('w',newline='') as f:
                        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
                csvwrite(bundle/'patches.csv',p);csvwrite(bundle/'final_outer_inner_split_manifest.csv',s)
                expected={n:tool.sha(bundle/n) for n in ('patches.csv','final_outer_inner_split_manifest.csv')}
                (root/'data/manifest_bundle_identity.json').write_text(json.dumps(dict(files=expected,expected=e)))
                for stage in ('stage1','stage2'):
                    (root/'configs'/stage/'SYNTH.json').write_text(json.dumps(dict(split_sha256=expected['final_outer_inner_split_manifest.csv'])))
                for row in p:
                    (source/row['source_relative_path']).write_bytes(b'NOT_AN_IMAGE_SYNTH')
                    (cache/(row['uid']+'.jpg')).write_bytes(b'NOT_AN_IMAGE_SYNTH')
                weight=root/'SYNTH.weight';weight.write_bytes(b'NOT_MODEL_TENSORS')
                csvwrite(root/'data/pretrained_sources.csv',[dict(model_name='SYNTH',weight_sha256=tool.sha(weight))])
                wm=root/'weights.json';wm.write_text(json.dumps(dict(SYNTH=str(weight))))
                output=root/'local'
                result=tool.prepare(bundle,source,cache,wm,output)
                self.assertEqual(result['patches'],6)
                self.assertEqual((output/'final_outer_inner_split_manifest.csv').read_bytes(),(bundle/'final_outer_inner_split_manifest.csv').read_bytes())
                self.assertFalse(result['model_initialization_verified'])
                with self.assertRaisesRegex(ValueError,'collision'):tool.prepare(bundle,source,cache,wm,output)
                (bundle/'patches.csv').write_text('MODIFIED_SYNTH')
                with self.assertRaisesRegex(ValueError,'SHA mismatch'):tool.prepare(bundle,source,cache,wm,root/'other')
                self.assertFalse((root/'other').exists())
        finally:
            tool.ROOT=original


if __name__=='__main__':unittest.main(verbosity=2)
