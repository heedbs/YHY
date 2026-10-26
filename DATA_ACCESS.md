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
| Exact patch/split bundle | No | Server-only review draft; distribution route not finalized |
| Pretrained/checkpoint weights | No | Identities documented under `data/`; obtain matching files separately |
| Patient-level frozen predictions | No | Separate disclosure decision; aggregate outputs are included |
| Aggregate figure/table data | Yes | [Results guide](results/README.md) |

Exact bundle schema and local preparation are specified in [AUTHORIZED_MANIFEST_BUNDLE.md](data/AUTHORIZED_MANIFEST_BUNDLE.md). Whole-file fingerprints do not supply the rows themselves. There is no promised download endpoint or approved request-access process yet.

Public-source-linked manifests are not categorically prohibited. Their identifiers, image hashes and crop coordinates can link back to source images; review the minimal required fields and attribution before release. See [field review](docs/data/DATA_FIELD_RELEASE_REVIEW.csv) and [file decisions](docs/data/RELEASE_FILE_DECISIONS.csv). Dataset licensing does not substitute for an ethics determination.

The single register of unresolved scientific/reproduction issues is [LIMITATIONS.md](LIMITATIONS.md). Synthetic fixtures contain artificial records, not patient observations.
