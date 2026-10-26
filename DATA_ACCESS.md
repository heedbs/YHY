# Data Access

[Execution guide](REPRODUCIBILITY.md) · [Disclosure review](docs/data/DATA_DISCLOSURE_REVIEW.md)

## Official Source Data

| Source archive | DOI |
| --- | --- |
| Training: `train.zip` | https://doi.org/10.5281/zenodo.12636426 |
| Validation/test: `val.zip`, `test.zip` | https://doi.org/10.5281/zenodo.12646943 |

Official API metadata checked on 21 October 2026 records **CC BY 4.0** for both datasets; retained metadata is in [data/dataset_metadata.json](data/dataset_metadata.json). Credit the creators, link the DOI and [license](https://creativecommons.org/licenses/by/4.0/), and disclose crop/resize/overlay changes. Original archive partitions are distinct from this study's patient-wise folds.

## Study-Specific Inputs

| Artifact | In this package? | Route |
| --- | --- | --- |
| Source images and patch cache | No | Official datasets; validated study cache still needed |
| Exact patch/split metadata | Yes, reviewed candidate | `data/reproduction/`; extract exact CSVs outside checkout |
| Pretrained/checkpoint weights | No | Identities documented under `data/`; obtain matching files separately |
| Patient-level frozen predictions | Yes, reviewed candidate | `data/oof/stage1/` (4 sets), `data/oof/stage2/` (6 corrected sets) |
| Aggregate figure/table data | Yes | [Results guide](results/README.md) |

The author approved the minimum distribution candidate on 26 October 2026. See [data/README](data/README.md) for fields, attribution and data licensing. Extract the exact metadata using [AUTHORIZED_MANIFEST_BUNDLE](data/AUTHORIZED_MANIFEST_BUNDLE.md). The repository remains private; no external download endpoint or public Release is claimed.

Approved identifiers, hashes and coordinates remain linkable to public source images, not newly anonymized. The [current policy](data/RELEASE_DATA_POLICY.json) authorizes exact reviewed bytes/fields, not the broader historical [field inventory](docs/data/DATA_FIELD_RELEASE_REVIEW.csv). Private paths, extra clinical attributes, logs and images remain excluded. Dataset licensing does not substitute for an ethics determination.

The single register of unresolved scientific/reproduction issues is [LIMITATIONS.md](LIMITATIONS.md). Synthetic fixtures contain artificial records, not patient observations.
