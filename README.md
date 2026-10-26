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

The isolated CPU environment passed **28 synthetic tests**. These check package integrity and selected scientific logic, not real-data training, CUDA equivalence or retrospective reconstruction of the experiments. Synthetic fixtures are not research observations. See the [test report](docs/audits/TEST_REPORT.md).

## Reproduction Workflow

1. Obtain the original ORCHID datasets from the official DOIs in [DATA_ACCESS.md](DATA_ACCESS.md).
2. Supply authorized exact manifests and a validated patch cache. Patient-linked rows and the cache are **not bundled**; historical cache-generation lineage has unresolved limitations.
3. Bind local paths and hashes using [REPRODUCIBILITY.md](REPRODUCIBILITY.md). Public code copies contain documented portability and verified-weight-binding changes.
4. Use the corresponding Stage 1 or Stage 2 trainer. Config examples preserve their distinct checkpoint rules and epoch caps; InceptionV3 uses batch 8 × accumulation 4, the other Stage 1 models 16 × 2.
5. Evaluate authorized saved predictions with [analysis code](analysis/README.md), then use the [figure](figures/README.md) and table source data.

This is an execution map, **not a claim that the entire workflow has been rerun from this package**. Benchmark and explainability scripts are reference material with additional prerequisites.

## Scope and Interpretation

- Included: model/training/evaluation source, 50 config examples, synthetic tests and frozen aggregate figure/table data.
- Excluded: source images, patch cache, weights, patient-level predictions, exact patient-linked manifests and private server logs.
- Stage 2 is a same-cohort matched ablation/stability analysis, **not external validation**. Its three primary seed-specific confidence intervals include zero; neither superiority nor equivalence is established.
- Historical cache-fit lineage, development-time outer access and final externally edited figure mapping remain explicit [limitations](LIMITATIONS.md).

## License and Citation

[MIT](LICENSE) is selected for authorized original software contributions. Third-party software retains its own terms; see [license scope](docs/licensing/LICENSE_SCOPE.md) and [third-party notices](THIRD_PARTY_NOTICES.md). Dataset CC BY 4.0 is separate from the code license. No new pathology, ethics or clinical approval is asserted.

Citation metadata: [CITATION.cff](CITATION.cff). Cite the exact reviewed commit. No manuscript DOI or archival release DOI is assigned. Private access, release decisions and dated historical records are explained in [Current Status](docs/STATUS.md).
