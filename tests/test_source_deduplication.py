"""Canonical aggregate metadata checks; no inference or statistical recalculation."""
import csv
import hashlib
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]


class SourceDeduplicationTests(unittest.TestCase):
    def test_csvs_have_no_byte_identical_copies(self):
        seen={}
        for path in ROOT.rglob('*.csv'):
            digest=hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertNotIn(digest,seen,'redundant CSV: '+str(path.relative_to(ROOT)))
            seen[digest]=path

    def test_removed_paths_and_canonical_hashes(self):
        with (ROOT/'docs/provenance/FILE_REDIRECTIONS.csv').open(newline='') as handle:rows=list(csv.DictReader(handle))
        self.assertEqual(len(rows),24)
        self.assertEqual(sum(r['kind']=='IDENTICAL_AGGREGATE_ALIAS' for r in rows),15)
        for row in rows:
            self.assertFalse((ROOT/row['removed_path']).exists())
            canonical=ROOT/row['canonical_path']
            self.assertTrue(canonical.is_file())
            self.assertEqual(hashlib.sha256(canonical.read_bytes()).hexdigest(),row['canonical_sha256'])
            if row['kind']=='IDENTICAL_AGGREGATE_ALIAS':self.assertEqual(row['removed_sha256'],row['canonical_sha256'])


if __name__=='__main__':unittest.main()
