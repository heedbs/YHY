# Distribution Data Candidate

This directory now includes the author-reviewed minimum exact input metadata and saved OOF predictions. The repository is public as verified on 28 October 2026; a fixed software Release and archive remain separate actions.

| Location | Contents |
| --- | --- |
| `reproduction/patches.csv.gz` | Losslessly compressed original minimum patch CSV, 232,689 rows |
| `reproduction/final_outer_inner_split_manifest.csv` | Exact frozen split bytes, 745 rows; 149 patients × 5 folds |
| `oof/stage1/` | Four frozen model OOF files, each with 149 patients |
| `oof/stage2/` | Six corrected model-seed OOF files, each with the same 149 patients |
| `RELEASE_DATA_POLICY.json` | Exact reviewed paths, hashes, fields, study/model/seed and counts |

The patient keys and relative filenames are source-linked pseudonyms, not private real identities and not a claim of irreversible anonymization. Repeated model/seed rows do not represent additional independent patients. Frozen numeric strings and selected checkpoints were not changed. No image, cache or weight is included.

## Source and Attribution

ORCHID training: [Chaudhary et al., Zenodo 12636426](https://doi.org/10.5281/zenodo.12636426). Related original validation/test record: [Chaudhary and Ahmad, Zenodo 12646943](https://doi.org/10.5281/zenodo.12646943). The minimum patch source paths resolve relative to the training archive used by this study; original archive partitions are not the study folds. Official metadata is retained in `dataset_metadata.json`.

Original ORCHID materials are CC BY 4.0. Preserve creator/DOI/license attribution and identify the duplicate-clean selection, crop hierarchy, study partitions and added model outputs as modifications/additions. Source-image bytes are not redistributed. The author's new metadata/prediction contributions in `reproduction/` and `oof/` are offered under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), separately from the code MIT grant. Third-party rights and privacy rights are not overridden. Other provenance files in this directory document their sources; no weight license is inferred.

## Use

Extract exact bundle files **outside the checkout**:

```bash
python -B scripts/extract_reproduction_bundle.py --output ../orchid-exact-bundle
python -B scripts/verify_release_data.py
```

The first command restores the two CSVs with exact historical hashes, rejects collisions and never generates cache. The second checks alignment, not model performance metrics. Follow [REPRODUCIBILITY](../REPRODUCIBILITY.md) for data/cache/weight prerequisites. Current data availability does not establish historical first-generation cache equivalence or CUDA reproduction.
