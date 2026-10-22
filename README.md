# ORCHID comparative study: reproducibility candidate

**Private review candidate for a manuscript intended for submission to BioData Mining. This is not an accepted or published article, and no public software release has been issued. MIT is selected for authorized original software contributions; explicit composite-module exceptions and remaining rights review are documented in LICENSE_SCOPE.md.**

English is the primary documentation for reviewers and readers. [한국어 보조 안내](readme_kor.md).

For the intended journal submission, see [BioData Mining submission preparation](BIODATA_MINING_SUBMISSION_CHECKLIST.md), including outstanding author decisions and a current-state availability-statement draft. This checklist is not a journal compliance certification.

Repository: https://github.com/heedbs/YHY . The review candidate is on `review/orchid-reproducibility-20261022`, not the default `main` branch. A private repository link alone does not grant reviewers access; agree an appropriate access or supplementary-package route before submission. Cite the exact reviewed commit, not an unspecified moving branch. MIT selection does not resolve remaining component-rights, public-release or archival decisions.

This package documents the existing internal, patient-wise ORCHID study. Class order: Normal=0, OSMF=1, OSCC=2. Cohort: 149 patients, 10,072 source images, 232,689 duplicate-clean patches. Stage 1 compares four models with seed42, five outer folds, AUROC-first checkpoint selection and an 80-epoch cap. Corrected Stage 2 compares Composite-Mamba with an exact Identity control using seeds42/20260824/314159, F1-first selection and a 40-epoch cap. Historical Stage 1 Mamba seed42 is not substituted into corrected Stage 2.

## What can be checked now

| Reader goal | Entry point | Current limit |
| --- | --- | --- |
| Inspect study results | `results/`, `table_source_data/`, `figure_source_data/` | Aggregate frozen outputs, not a new analysis |
| Inspect model/training settings | `stage1/`, `stage2/`, `configs/run_catalog.csv` | Exact private manifests and validated cache still required |
| Run research-data-free checks | `tests/` | CPU synthetic checks, not full experiment reproduction |
| Understand data access | [DATA_ACCESS.md](DATA_ACCESS.md) | Patient-level release review incomplete |
| Prepare authorized local inputs | [Minimum manifest bundle](data/AUTHORIZED_MANIFEST_BUNDLE.md) | CPU-only binding; private patient bundle and validated cache required |
| Understand licensing | [LICENSE_SCOPE.md](LICENSE_SCOPE.md) | Composite-module source exceptions remain |
| Reuse benchmark/Grad-CAM scripts | `benchmark/reference/`, `explainability/reference/` | Reference only; not a turnkey public workflow |

Read aggregate frozen outputs under results/, table_source_data/ and figure_source_data/. Compare scientific parameters in configs/ with configs/run_catalog.csv, and original/public code fingerprints in CODE_CHANGES.csv and code_semantic_fingerprints.json. Run the CPU-only synthetic tests below; fixtures are not research observations.

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' python -B -m unittest discover -s tests -v
sha256sum -c SHA256_MANIFEST.txt
```

## Research execution order

1. Obtain ORCHID from the official DOIs in DATA_ACCESS.md, preserving the original repository split separately from study folds.
2. Obtain author-reviewed duplicate-clean patch metadata and exact patient split metadata; these are excluded pending release approval. A historical notebook builder and reference constants have been located (PREPROCESSING_AND_ENVIRONMENT_EVIDENCE.md); first-generation/full-cache and final fold-fit lineage are not fully verified. Supply the already validated cache or resolve this prerequisite with the authors.
3. Follow REPRODUCIBILITY.md to configure local paths and bind relocated code/config hashes. Public copies are path-adapted, not byte-identical historical executable files.
4. Stage 1 training uses stage1/train_one_run.py; Stage 2 uses stage2/train_one_run.py. Their Trainer/model/data functions preserve the verified scientific logic. New research execution examples below were not run during packaging.
5. Frozen prediction evaluation/statistical source code is in analysis/. Individual predictions are excluded. Corrected paired analysis uses 10,000 class-stratified paired-patient bootstrap repetitions and seed20260830; no new bootstrap was performed here.
6. Figure/table scripts and aggregate source data are included. Final externally edited manuscript figures are not identified in the local evidence and are not certified as reproduced.

GPU measurement/explainability code is reference material under benchmark/reference/ and explainability/reference/. Read limitations before execution. No new training, inference, benchmark or GPU action was used to prepare this package. Public launch approval and automatic resume are not provided.

See LICENSE_DECISION_REQUIRED.md and RELEASE_FILE_DECISIONS.csv before deciding on any public release. Dataset CC BY 4.0 does not license this code or resolve ethics/IRB questions.

Empty historical figure-helper efficiency CSVs have been omitted. Use `results/efficiency/EFFICIENCY_MODEL_SUMMARY.csv` and `EFFICIENCY_FIGURE_PLOTTING_DATA.csv` instead. Historical sources are preserved outside the candidate; no result was recalculated. See MAINTENANCE_CHANGES.csv for public-copy-only fixes.

See [REPRODUCTION_READINESS.csv](REPRODUCTION_READINESS.csv) for an explicit distinction between checked local input/config binding and unverified clean-environment model execution. A matching pretrained-file hash does not alone prove that timm initialized from that exact file.

See [license scope](LICENSE_SCOPE.md), [data review](DATA_DISCLOSURE_REVIEW.md) and [recovered preprocessing evidence](PREPROCESSING_AND_ENVIRONMENT_EVIDENCE.md). No new training or patient-data upload was performed.
