# ORCHID

**Patient-wise oral histopathology comparison and matched Mamba ablation**

[한국어 안내](readme_kor.md) · [Reproduce](REPRODUCIBILITY.md) · [Results](results/README.md) · [Documentation](docs/README.md) · [Current Status](docs/STATUS.md)

Research software and frozen aggregate source data supporting a manuscript intended for **BioData Mining**. This repository is a private review candidate, not a published article or an independently validated clinical system.

## Study at a Glance

**149 patients · 10,072 source images · 232,689 duplicate-clean patches**

The task is H&E oral histopathology classification: **Normal (0), OSMF (1), OSCC (2)**. Predictions are averaged within each source image, then images receive equal weight within each patient. Evaluation uses frozen patient-wise outer folds and separate inner validation.

| Study | Models and runs | Checkpoint selection | Epoch cap |
| --- | --- | --- | --- |
| Stage 1 | InceptionV3, EfficientNet-Lite2, MobileViT-S, Composite-Mamba; 5 folds each, seed 42; **20 runs** | Inner-validation patient macro OVR AUROC first | 80 |
| Corrected Stage 2 | Composite-NoMamba versus Composite-Mamba; 5 folds × 3 seeds each; **30 runs** | Inner-validation patient macro F1 first | 40 |

Stage 2 uses seeds 42, 20260824 and 314159 on the **same patients and folds**, not 447 independent patients. Its Mamba seed-42 results come from corrected Stage 2 runs, never historical Stage 1 replacements. The Identity control retains the shared composite architecture; Stage 1 EfficientNet-Lite2 is not that matched control.

## Start Here

For manuscript cross-checking, start with the [evidence guide](docs/MANUSCRIPT_EVIDENCE_GUIDE.md) and its hashed source index. Curated export corrections are logged separately from frozen model results.

| Goal | Entry point |
| --- | --- |
| Read frozen results and table-ready data | [Results guide](results/README.md), [table data](table_source_data/) |
| Inspect models and training settings | [Stage 1](stage1/README.md), [Stage 2](stage2/README.md), [config catalog](configs/README.md) |
| Obtain data and understand missing prerequisites | [Data access](DATA_ACCESS.md), [reproduction instructions](REPRODUCIBILITY.md) |
| Check the package without research data | [CPU environment](environment/CPU_TEST_ENVIRONMENT.md), commands below |
| Trace code, results and AI assistance | [Provenance and audit index](docs/README.md) |
| Understand what is not yet reproduced | [Limitations](LIMITATIONS.md), [current status](docs/STATUS.md) |

### CPU Checks

From the repository root, in the documented CPU environment:

```bash
python -B scripts/check_public_package.py . --manifest
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' python -B -m unittest discover -s tests -v
```

Verification results and their scope are maintained in the [test report](docs/audits/TEST_REPORT.md). Synthetic fixtures are not research observations.

## Reproduction Workflow

Follow [REPRODUCIBILITY.md](REPRODUCIBILITY.md): obtain authorized inputs → prepare manifests → bind config → run CPU input preflight → execute the appropriate trainer → analyze saved predictions → generate figures/tables. Exact prerequisites and command verification status are kept there, not duplicated here.

## Scope and Interpretation

Included: model/training/evaluation source, 50 config examples, synthetic tests and frozen aggregate figure/table data. Research images, cache, weights and patient-linked rows are not bundled; see [data access](DATA_ACCESS.md).

Stage 2 is same-cohort matched ablation, not external validation. Read the single [limitations register](LIMITATIONS.md) before interpreting results or claiming complete reproduction.

## License and Citation

[MIT](LICENSE) is selected for authorized original software contributions. Third-party software retains its own terms; see [license scope](docs/licensing/LICENSE_SCOPE.md) and [third-party notices](THIRD_PARTY_NOTICES.md). Dataset CC BY 4.0 is separate from the code license. No new pathology, ethics or clinical approval is asserted.

Citation metadata: [CITATION.cff](CITATION.cff). Cite the exact reviewed commit. No manuscript DOI or archival release DOI is assigned. Private access, release decisions and dated historical records are explained in [Current Status](docs/STATUS.md).
