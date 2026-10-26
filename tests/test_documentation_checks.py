"""Artificial files exercise the research-data-free documentation checker."""
import csv
import hashlib
import importlib.util
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('evidence_links', ROOT / 'scripts/check_evidence_links.py')
tool = importlib.util.module_from_spec(spec); spec.loader.exec_module(tool)


class DocumentationChecks(unittest.TestCase):
    def fixture(self, root):
        (root / 'docs/provenance').mkdir(parents=True)
        (root / 'SYNTH.txt').write_text('ARTIFICIAL_FIXTURE_ONLY\n')
        (root / 'README.md').write_text('[Fixture](SYNTH.txt)\n')
        digest = hashlib.sha256((root / 'SYNTH.txt').read_bytes()).hexdigest()
        for name, fields in [('CLAIM_SOURCE_INDEX.csv', ['repository_path', 'repository_sha256']),
                             ('SOURCE_FILE_FINGERPRINTS.csv', ['public_path', 'public_sha256'])]:
            with (root / 'docs/provenance' / name).open('w', newline='') as handle:
                writer = csv.writer(handle); writer.writerow(fields); writer.writerow(['SYNTH.txt', digest])

    def test_valid_and_read_only(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); self.fixture(root)
            before = {str(p): p.read_bytes() for p in root.rglob('*') if p.is_file()}
            self.assertTrue(tool.check(root)['passed'])
            self.assertEqual(before, {str(p): p.read_bytes() for p in root.rglob('*') if p.is_file()})

    def test_broken_and_escaping_links(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); self.fixture(root)
            for link in ('MISSING.md', '../outside.md'):
                (root / 'README.md').write_text('[Bad](' + link + ')')
                self.assertFalse(tool.check(root)['passed'])

    def test_changed_and_duplicate_index(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); self.fixture(root)
            (root / 'SYNTH.txt').write_text('CHANGED')
            self.assertFalse(tool.check(root)['passed'])
            self.fixture_rewrite(root)
            index = root / 'docs/provenance/CLAIM_SOURCE_INDEX.csv'
            lines = index.read_text().splitlines(); index.write_text('\n'.join(lines + lines[1:]) + '\n')
            self.assertFalse(tool.check(root)['passed'])

    def fixture_rewrite(self, root):
        (root / 'SYNTH.txt').write_text('ARTIFICIAL_FIXTURE_ONLY\n')

    def test_symlink_to_outside_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / 'root'; root.mkdir(); self.fixture(root)
            outside = Path(temp) / 'OUTSIDE.txt'; outside.write_text('SYNTHETIC')
            (root / 'SYNTH.txt').unlink(); (root / 'SYNTH.txt').symlink_to(outside)
            self.assertFalse(tool.check(root)['passed'])


if __name__ == '__main__': unittest.main()
