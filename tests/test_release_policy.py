"""Artificial disclosure fixtures; no research observations or generated predictions."""
import csv
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import release_data
import extract_reproduction_bundle as extractor
spec=importlib.util.spec_from_file_location('public_release_scan',ROOT/'scripts/check_public_package.py')
scan=importlib.util.module_from_spec(spec);spec.loader.exec_module(scan)


class ReleasePolicyTests(unittest.TestCase):
    def fixture(self,root,text=None,compressed=False):
        (root/'data/reproduction').mkdir(parents=True)
        text=text or 'patient_id,value\n'+'o-'+'0-'+'99-'+'99'+',SYNTHETIC\n'
        raw=text.encode();payload=gzip.compress(raw,mtime=0) if compressed else raw
        name='data/reproduction/SYNTH.csv'+('.gz' if compressed else '')
        (root/name).write_bytes(payload)
        entry=dict(path=name,sha256=hashlib.sha256(payload).hexdigest(),content_sha256=hashlib.sha256(raw).hexdigest(),kind='oof',columns=['patient_id','value'],rows=1)
        policy=dict(status='AUTHOR_AUTHORIZED_FIELD_REVIEWED_DISTRIBUTION_CANDIDATE',files=[entry])
        (root/'data/RELEASE_DATA_POLICY.json').write_text(json.dumps(policy))
        return entry

    def test_exact_reviewed_identifier_scope(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);self.fixture(root)
            self.assertTrue(scan.scan(root)['pass'])

    def test_modified_data_fails(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);entry=self.fixture(root);(root/entry['path']).write_text('MODIFIED')
            self.assertFalse(scan.scan(root)['pass'])

    def test_private_paths_never_waived(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);self.fixture(root,'patient_id,value\nSYNTH,'+'/home/'+'PRIVATE_ACCOUNT/file\n')
            result=scan.scan(root)
            self.assertFalse(result['pass']);self.assertTrue(any(r['rule']=='private_absolute_path' for r in result['findings']))

    def test_unapproved_identifier_still_fails(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);self.fixture(root)
            (root/'UNAPPROVED.csv').write_text('o-'+'0-'+'99-'+'99'+'\n')
            self.assertFalse(scan.scan(root)['pass'])

    def test_gzip_schema_and_content_hash(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);entry=self.fixture(root,compressed=True)
            self.assertTrue(scan.scan(root)['pass'])
            bad=dict(entry,content_sha256='0'*64)
            with self.assertRaises(ValueError):release_data.read_reviewed(root,bad)
            with self.assertRaises(ValueError):release_data.read_reviewed(root,dict(entry,rows=2))

    def test_extraction_preserves_bytes_and_rejects_collision(self):
        old=extractor.ROOT
        try:
            with tempfile.TemporaryDirectory() as temp:
                root=Path(temp)/'repo';root.mkdir();extractor.ROOT=root
                patch=b'uid\nSYNTH\n';split=b'outer_fold\n0\n';entries=[]
                for name,relative,kind,raw in [('patches.csv','data/reproduction/patches.csv.gz','patches',patch),('final_outer_inner_split_manifest.csv','data/reproduction/final_outer_inner_split_manifest.csv','split',split)]:
                    path=root/relative;path.parent.mkdir(parents=True,exist_ok=True)
                    payload=gzip.compress(raw,mtime=0) if relative.endswith('.gz') else raw;path.write_bytes(payload)
                    entries.append(dict(path=relative,sha256=hashlib.sha256(payload).hexdigest(),content_sha256=hashlib.sha256(raw).hexdigest(),kind=kind,columns=raw.decode().splitlines()[0].split(','),rows=1))
                (root/'data/RELEASE_DATA_POLICY.json').write_text(json.dumps(dict(status='AUTHOR_AUTHORIZED_FIELD_REVIEWED_DISTRIBUTION_CANDIDATE',files=entries)))
                (root/'data/manifest_bundle_identity.json').write_text(json.dumps(dict(files={'patches.csv':hashlib.sha256(patch).hexdigest(),'final_outer_inner_split_manifest.csv':hashlib.sha256(split).hexdigest()})))
                output=Path(temp)/'new_bundle';extractor.extract(output)
                self.assertEqual((output/'patches.csv').read_bytes(),patch);self.assertEqual((output/'final_outer_inner_split_manifest.csv').read_bytes(),split)
                with self.assertRaises(ValueError):extractor.extract(output)
        finally:extractor.ROOT=old


if __name__=='__main__':unittest.main()
