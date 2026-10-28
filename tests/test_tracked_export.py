"""Synthetic git-archive regression: no research models or patient data."""
import hashlib
import importlib.util
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('export_package_check', ROOT/'scripts/check_public_package.py')
check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check)


@unittest.skipUnless(shutil.which('git'), 'git required for tracked-export regression')
class TrackedExportTests(unittest.TestCase):
    def test_clone_rejected_archive_passes_other_hidden_files_still_checked(self):
        with tempfile.TemporaryDirectory() as temp:
            checkout = Path(temp)/'clone'; checkout.mkdir()
            subprocess.run(['git', 'init', '-q', str(checkout)], check=True)
            synthetic = checkout/'fixture.txt'; synthetic.write_text('SYNTHETIC_FIXTURE_ONLY\n')
            digest = hashlib.sha256(synthetic.read_bytes()).hexdigest()
            (checkout/'SHA256_MANIFEST.txt').write_text(digest+'  fixture.txt\n')
            subprocess.run(['git', '-C', str(checkout), 'add', '.'], check=True)
            subprocess.run(['git', '-C', str(checkout), '-c', 'user.name=Synthetic Test',
                            '-c', 'user.email=fixture'+'@'+'example.invalid', 'commit', '-qm',
                            'Synthetic fixture, not research data'], check=True)
            self.assertTrue(check.verify_manifest(checkout)['findings'])
            # Untracked and modified contents intentionally are not validated by archive HEAD.
            (checkout/'.untracked-fixture').write_text('NOT_EXPORTED')
            synthetic.write_text('UNCOMMITTED_EDIT')
            before = {p.relative_to(checkout):p.read_bytes() for p in checkout.rglob('*') if p.is_file()}
            archive = Path(temp)/'HEAD.tar'
            subprocess.run(['git', '-C', str(checkout), 'archive', '--format=tar',
                            '--output='+str(archive), 'HEAD'], check=True)
            export = Path(temp)/'fresh-tracked'; export.mkdir()
            with tarfile.open(archive) as handle:
                for member in handle.getmembers():
                    self.assertFalse(Path(member.name).is_absolute())
                    self.assertNotIn('..', Path(member.name).parts)
                    self.assertFalse(member.issym() or member.islnk())
                handle.extractall(export, filter='data')
            self.assertFalse((export/'.git').exists())
            self.assertFalse(check.verify_manifest(export)['findings'])
            self.assertFalse(check.scan(export)['findings'])
            self.assertEqual(before, {p.relative_to(checkout):p.read_bytes() for p in checkout.rglob('*') if p.is_file()})
            self.assertEqual(hashlib.sha256((export/'fixture.txt').read_bytes()).hexdigest(), digest)
            (export/'.unexpected-hidden-file').write_text('SYNTHETIC')
            self.assertIn('unlisted_file', {f['rule'] for f in check.verify_manifest(export)['findings']})
            (export/'fixture.txt').write_text('ALTERED_SYNTHETIC')
            self.assertIn('sha_mismatch', {f['rule'] for f in check.verify_manifest(export)['findings']})


if __name__ == '__main__': unittest.main()
