"""Artificial package/config checks only; no research data, model or bootstrap."""
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def load(name):
    spec=importlib.util.spec_from_file_location(name,ROOT/'scripts'/f'{name}.py')
    obj=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


check=load('check_public_package')
config=load('configure_local_run')


class PackagingChecks(unittest.TestCase):
    def setup_fixture(self,root):
        file=root/'fixture.txt';file.write_text('SYNTHETIC ONLY\n')
        (root/'SHA256_MANIFEST.txt').write_text(hashlib.sha256(file.read_bytes()).hexdigest()+'  fixture.txt\n')

    def test_valid_manifest(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);self.setup_fixture(root)
            self.assertFalse(check.verify_manifest(root)['findings'])

    def test_unlisted_file(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);self.setup_fixture(root);(root/'extra.txt').write_text('SYNTH')
            self.assertIn('unlisted_file',{x['rule'] for x in check.verify_manifest(root)['findings']})

    def test_duplicate_entry(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);self.setup_fixture(root);file=root/'SHA256_MANIFEST.txt';file.write_text(file.read_text()*2)
            self.assertIn('duplicate_manifest_entry',{x['rule'] for x in check.verify_manifest(root)['findings']})

    def test_traversal_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);self.setup_fixture(root)
            (root/'SHA256_MANIFEST.txt').write_text('0'*64+'  ../outside.txt\n')
            self.assertIn('unsafe_manifest_path',{x['rule'] for x in check.verify_manifest(root)['findings']})

    def test_missing_malformed_and_changed(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);self.setup_fixture(root)
            (root/'fixture.txt').write_text('CHANGED_SYNTH')
            self.assertIn('sha_mismatch',{x['rule'] for x in check.verify_manifest(root)['findings']})
            (root/'SHA256_MANIFEST.txt').write_text('invalid\n'+'0'*64+'  missing.txt\n')
            rules={x['rule'] for x in check.verify_manifest(root)['findings']}
            self.assertTrue({'malformed_entry','manifest_file_missing'}<=rules)

    def test_symlink_rejected_without_reading_target(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);self.setup_fixture(root);(root/'linked').symlink_to(root/'fixture.txt')
            self.assertIn('symlink',{x['rule'] for x in check.scan(root)['findings']})
            with (root/'SHA256_MANIFEST.txt').open('a') as f:f.write('0'*64+'  linked\n')
            self.assertIn('manifest_symlink',{x['rule'] for x in check.verify_manifest(root)['findings']})

    def test_all_frozen_templates_stage_match(self):
        for stage in ('stage1','stage2'):
            for file in (ROOT/'configs'/stage).glob('*.json'):
                c=json.loads(file.read_text());config.validate_stage(stage,c)
                other='stage2' if stage=='stage1' else 'stage1'
                with self.assertRaises(RuntimeError):config.validate_stage(other,c)

    def test_wrong_criterion_rejected(self):
        file=next((ROOT/'configs/stage1').glob('*.json'))
        c=json.loads(file.read_text());c['checkpoint_criterion']='SYNTH_WRONG'
        with self.assertRaises(RuntimeError):config.validate_stage('stage1',c)

    def test_historical_bootstrap_rejects_opposing_label_ties(self):
        spec=importlib.util.spec_from_file_location('guarded_statistics',ROOT/'analysis/stage2_statistics.py')
        obj=importlib.util.module_from_spec(spec);spec.loader.exec_module(obj)
        y=np.array([0,1,2]);p=np.array([[.5,.3,.2],[.5,.2,.3],[.1,.1,.8]])
        with self.assertRaisesRegex(ValueError,'ties unsupported'):
            obj.bootstrap_metrics(y,p,np.array([[0,1,2]]))

    def test_historical_bootstrap_rejects_unstratified_indices(self):
        spec=importlib.util.spec_from_file_location('guarded_statistics',ROOT/'analysis/stage2_statistics.py')
        obj=importlib.util.module_from_spec(spec);spec.loader.exec_module(obj)
        y=np.array([0,1,2]);p=np.array([[.8,.1,.1],[.1,.8,.1],[.1,.1,.8]])
        with self.assertRaisesRegex(ValueError,'class-stratified'):
            obj.bootstrap_metrics(y,p,np.array([[0,0,2]]))


if __name__=='__main__':unittest.main(verbosity=2)
