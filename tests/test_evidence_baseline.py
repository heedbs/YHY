"""Aggregate evidence only; no patient rows, inference or metric recomputation."""
import csv
import hashlib
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
def rows(name):
    with (ROOT / name).open(newline='') as handle: return list(csv.DictReader(handle))


class EvidenceBaselineTests(unittest.TestCase):
    def test_cohort_units_and_exports(self):
        flow = json.loads((ROOT / 'data/cohort_flow.json').read_text())
        self.assertEqual(flow['source_images_before'] - flow['source_images_after'], 156)
        self.assertEqual(flow['source_images_excluded'], 156)
        self.assertEqual(flow['duplicate_source_sha_groups'], 78)
        self.assertEqual(flow['patch_rows_before'] - flow['patch_rows_after'], 1248)
        self.assertEqual(flow['patch_rows_removed'], 1248)
        self.assertFalse(flow['removed_patch_hash_deduplication_claim'])
        figure = rows('figure_source_data/01_COHORT_AND_STUDY_DESIGN/FIG1_COHORT_COUNTS.csv')[0]
        self.assertEqual(int(figure['patches_before_duplicate_exclusion']), flow['patch_rows_before'])
        self.assertEqual(int(figure['patches_after_duplicate_exclusion']), flow['patch_rows_after'])
        for row in rows('table_source_data/table02_cohort.csv'):
            self.assertIn('156 source-image occurrences', row['duplicate_exclusion_scope'])
            self.assertIn('1248 associated patch rows', row['duplicate_exclusion_scope'])

    def test_source_index_hashes_and_paths(self):
        index = rows('docs/provenance/CLAIM_SOURCE_INDEX.csv')
        self.assertGreaterEqual(len(index), 8)
        self.assertEqual(len({r['topic'] for r in index}), len(index))
        for row in index:
            relative = Path(row['repository_path'])
            self.assertFalse(relative.is_absolute()); self.assertNotIn('..', relative.parts)
            self.assertEqual(hashlib.sha256((ROOT / relative).read_bytes()).hexdigest(), row['repository_sha256'])

    def test_corrections_retain_original_fingerprints(self):
        corrections = rows('docs/provenance/DATA_CORRECTIONS.csv')
        fingerprints = rows('docs/provenance/SOURCE_FILE_FINGERPRINTS.csv')
        self.assertEqual(len(corrections), 2)
        for row in corrections:
            record = next(r for r in fingerprints if r['public_path'] == row['file'])
            self.assertNotEqual(record['original_sha256'], record['public_sha256'])
            self.assertEqual(record['public_sha256'], hashlib.sha256((ROOT / row['file']).read_bytes()).hexdigest())
            self.assertIn('CORRECTION', record['transfer_method'])


if __name__ == '__main__': unittest.main()
