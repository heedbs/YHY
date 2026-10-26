"""Artificial metadata only; no images, tensors, model execution or patient data."""
import importlib.util
import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
spec = importlib.util.spec_from_file_location('preflight_local', ROOT / 'scripts/preflight_local_run.py')
tool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tool)


class LocalPreflightTests(unittest.TestCase):
    def fixture(self, root):
        source = root / 'SYNTH_SOURCE.png'
        cache = root / 'SYNTH_PATCH.jpg'
        source.write_bytes(b'NOT_AN_IMAGE')
        cache.write_bytes(b'NOT_AN_IMAGE')
        row = dict(uid='SYNTH_PATCH', x='0', y='0', w='512', h='512', cls='normal',
                   patient_key='SYNTH_PATIENT', source_sha256='a' * 64,
                   source_relative_path=source.name)
        local = dict(row, path=str(source), cache_path=str(cache))
        return local, row

    def test_exact_binding_and_no_writes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); local, frozen = self.fixture(root)
            before = {p.name: p.read_bytes() for p in root.iterdir()}
            tool.verify_local_rows([local], [frozen])
            self.assertEqual(before, {p.name: p.read_bytes() for p in root.iterdir()})

    def test_changed_target_or_crop_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            local, frozen = self.fixture(Path(temp))
            for key, value in [('cls', 'oscc'), ('x', '1'), ('source_sha256', 'b' * 64)]:
                with self.assertRaisesRegex(ValueError, 'semantics'):
                    tool.verify_local_rows([dict(local, **{key: value})], [frozen])

    def test_duplicate_missing_and_wrong_source_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); local, frozen = self.fixture(root)
            for rows in ([], [local, local]):
                with self.assertRaises(ValueError): tool.verify_local_rows(rows, [frozen])
            with self.assertRaises(ValueError):
                tool.verify_local_rows([dict(local, path=str(root / 'WRONG.png'))], [frozen])
            Path(local['cache_path']).unlink()
            with self.assertRaises(ValueError): tool.verify_local_rows([local], [frozen])

    def test_existing_output_and_symlink_fail_before_inputs(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for output in (root, root / 'link'):
                if output != root: output.symlink_to(root / 'absent')
                with self.assertRaisesRegex(ValueError, 'collision'):
                    tool.check('stage2', root / 'missing.json', root, root, output)

    def test_code_digest_matches_config_binder(self):
        import hashlib
        for stage in ('stage1', 'stage2'):
            root = ROOT / stage
            files = [root / 'train_one_run.py', *sorted((root / 'orchid').glob('*.py'))]
            if stage == 'stage1': files += [root / 'validate_before_run.py', root / 'evaluate_oof.py']
            digest = hashlib.sha256()
            for path in sorted(files):
                digest.update(str(path.relative_to(root)).encode()); digest.update(tool.sha(path).encode())
            self.assertEqual(tool.code_digest(stage), digest.hexdigest())

    def test_full_synthetic_preflight_and_identity_failures(self):
        from test_manifest_preparation import fixture
        original_root = tool.ROOT
        try:
            with tempfile.TemporaryDirectory() as temp:
                root = Path(temp); tool.ROOT = root
                for name in ('data', 'bundle', 'local', 'configs/stage2', 'stage2/orchid'):
                    (root / name).mkdir(parents=True)
                def write_csv(path, rows):
                    with path.open('w', newline='') as handle:
                        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator='\n')
                        writer.writeheader(); writer.writerows(rows)
                frozen, splits, expected = fixture()
                local = []
                for row in frozen:
                    source = root / row['source_relative_path']; source.write_bytes(b'SYNTH_NOT_IMAGE')
                    cache = root / (row['uid'] + '.jpg'); cache.write_bytes(b'SYNTH_NOT_IMAGE')
                    local.append(dict(row, path=str(source), cache_path=str(cache)))
                write_csv(root / 'bundle/patches.csv', frozen)
                write_csv(root / 'bundle/final_outer_inner_split_manifest.csv', splits)
                write_csv(root / 'local/final_unique_patch_manifest.csv', local)
                write_csv(root / 'local/final_outer_inner_split_manifest.csv', splits)
                hashes = {n: tool.sha(root / 'bundle' / n) for n in ('patches.csv', 'final_outer_inner_split_manifest.csv')}
                (root / 'data/manifest_bundle_identity.json').write_text(json.dumps(dict(files=hashes, expected=expected)))
                weight = root / 'SYNTH_WEIGHT'; weight.write_bytes(b'NOT_TENSORS')
                weights = [dict(model_name='EfficientNet-Lite2+Mamba', weight_sha256=tool.sha(weight), local_cache_path=str(weight))]
                write_csv(root / 'data/pretrained_sources.csv', weights)
                write_csv(root / 'local/selected_model_pretrained_provenance.csv', weights)
                receipt = dict(files={n: tool.sha(root / 'local' / n) for n in tool.LOCAL_FILES})
                (root / 'local/LOCAL_BINDING_RECEIPT.json').write_text(json.dumps(receipt))
                (root / 'stage2/train_one_run.py').write_text('# SYNTHETIC, not executable training\n')
                template = json.loads((ROOT / 'configs/stage2/stage2_correction_composite_mamba_seed42_fold0.json').read_text())
                template.update(split_sha256=hashes['final_outer_inner_split_manifest.csv'], pretrained_sha256=tool.sha(weight))
                (root / 'configs/stage2/SYNTH.json').write_text(json.dumps(template))
                config = dict(template, code_sha256=tool.code_digest('stage2'))
                config_path = root / 'SYNTH_CONFIG.json'; config_path.write_text(json.dumps(config))
                def check(): return tool.check('stage2', config_path, root / 'local', root / 'bundle', root / 'future_output')
                result = check()
                self.assertFalse(result['model_execution']); self.assertFalse((root / 'future_output').exists())
                config_path.write_text(json.dumps(dict(config, lr=9.0)))
                with self.assertRaisesRegex(ValueError, 'scientific template'): check()
                config_path.write_text(json.dumps(dict(config, code_sha256='0' * 64)))
                with self.assertRaisesRegex(ValueError, 'code/config/split'): check()
                config_path.write_text(json.dumps(config))
                weight.write_bytes(b'CORRUPTED_SYNTH')
                with self.assertRaisesRegex(ValueError, 'pretrained content'): check()
                (root / 'bundle/patches.csv').write_text('CORRUPTED_SYNTH')
                with self.assertRaisesRegex(ValueError, 'bundle SHA'): check()
        finally:
            tool.ROOT = original_root


if __name__ == '__main__':
    unittest.main()
