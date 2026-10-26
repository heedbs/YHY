"""Small artificial RGB arrays only, never research images or model inference."""
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
spec = importlib.util.spec_from_file_location('preflight_content', ROOT / 'scripts/preflight_local_run.py')
tool = importlib.util.module_from_spec(spec); spec.loader.exec_module(tool)


class ContentVerificationTests(unittest.TestCase):
    def fixture(self, root):
        source = root / 'SYNTH_SOURCE.png'
        cache = root / 'SYNTH_CACHE.jpg'
        pixels = np.arange(48, dtype=np.uint8).reshape(4, 4, 3)
        source.write_bytes(cv2.imencode('.png', pixels)[1].tobytes())
        cache.write_bytes(cv2.imencode('.jpg', pixels)[1].tobytes())
        return dict(uid='SYNTH_PATCH', path=str(source), source_sha256=tool.sha(source),
                    cache_path=str(cache), w='4', h='4')

    def test_inventory_deterministic_and_inputs_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); row = self.fixture(root)
            before = {p.name: tool.sha(p) for p in root.iterdir()}
            first = tool.verify_content([row]); second = tool.verify_content([row])
            self.assertEqual(first, second)
            self.assertEqual(first['cache_decoded'], 1)
            self.assertFalse(first['historical_first_generation_verified'])
            self.assertEqual(before, {p.name: tool.sha(p) for p in root.iterdir()})

    def test_wrong_source_sha_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            row = self.fixture(Path(tmp)); row['source_sha256'] = '0' * 64
            with self.assertRaisesRegex(ValueError, 'source SHA'): tool.verify_content([row])

    def test_corrupt_cache_and_wrong_dimensions_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            row = self.fixture(Path(tmp))
            with self.assertRaisesRegex(ValueError, 'dimension'):
                tool.verify_content([dict(row, w='8')])
            Path(row['cache_path']).write_bytes(b'SYNTH_CORRUPT')
            with self.assertRaisesRegex(ValueError, 'decode'): tool.verify_content([row])

    def test_duplicate_uid_and_source_contradiction_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            row = self.fixture(Path(tmp))
            with self.assertRaisesRegex(ValueError, 'duplicate UID'):
                tool.verify_content([row, row])
            with self.assertRaisesRegex(ValueError, 'contradictory source'):
                tool.verify_content([row, dict(row, source_sha256='0' * 64)])


if __name__ == '__main__':
    unittest.main()
