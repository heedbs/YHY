# ORCHID comparative study: reproducibility candidate

**Private review candidate for a manuscript intended for submission to BioData Mining. This is not an accepted or published article, and no public software release has been issued. Code license and ownership are undecided.**

English is the primary documentation for reviewers and readers. [한국어 보조 안내](readme_kor.md).

For the intended journal submission, see [BioData Mining submission preparation](BIODATA_MINING_SUBMISSION_CHECKLIST.md), including outstanding author decisions and a current-state availability-statement draft. This checklist is not a journal compliance certification.

Repository: https://github.com/heedbs/YHY . The review candidate is on `review/orchid-reproducibility-20261022`, not the default `main` branch. A private repository link alone does not grant reviewers access; agree an appropriate access or supplementary-package route before submission. Cite the exact reviewed commit, not an unspecified moving branch. Public release, archival identifiers and license selection remain separate author decisions.

This package documents the existing internal, patient-wise ORCHID study. Class order: Normal=0, OSMF=1, OSCC=2. Cohort: 149 patients, 10,072 source images, 232,689 duplicate-clean patches. Stage 1 compares four models with seed42, five outer folds, AUROC-first checkpoint selection and an 80-epoch cap. Corrected Stage 2 compares Composite-Mamba with an exact Identity control using seeds42/20260824/314159, F1-first selection and a 40-epoch cap. Historical Stage 1 Mamba seed42 is not substituted into corrected Stage 2.

## What can be checked now

Read aggregate frozen outputs under results/, table_source_data/ and figure_source_data/. Compare scientific parameters in configs/ with configs/run_catalog.csv, and original/public code fingerprints in CODE_CHANGES.csv and code_semantic_fingerprints.json. Run the CPU-only synthetic tests below; fixtures are not research observations.

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' python tests/test_cpu.py
sha256sum -c SHA256_MANIFEST.txt
```

## Research execution order

1. Obtain ORCHID from the official DOIs in DATA_ACCESS.md, preserving the original repository split separately from study folds.
2. Obtain author-reviewed duplicate-clean patch metadata and exact patient split metadata; these are excluded pending release approval. The historical Reinhard cache builder/reference is not reconstructed here. Supply the already validated cache or resolve this prerequisite with the authors.
3. Follow REPRODUCIBILITY.md to configure local paths and bind relocated code/config hashes. Public copies are path-adapted, not byte-identical historical executable files.
4. Stage 1 training uses stage1/train_one_run.py; Stage 2 uses stage2/train_one_run.py. Their Trainer/model/data functions preserve the verified scientific logic. New research execution examples below were not run during packaging.
5. Frozen prediction evaluation/statistical source code is in analysis/. Individual predictions are excluded. Corrected paired analysis uses 10,000 class-stratified paired-patient bootstrap repetitions and seed20260830; no new bootstrap was performed here.
6. Figure/table scripts and aggregate source data are included. Final externally edited manuscript figures are not identified in the local evidence and are not certified as reproduced.

GPU measurement/explainability code is reference material under benchmark/reference/ and explainability/reference/. Read limitations before execution. No new training, inference, benchmark or GPU action was used to prepare this package. Public launch approval and automatic resume are not provided.

See LICENSE_DECISION_REQUIRED.md and RELEASE_FILE_DECISIONS.csv before deciding on any public release. Dataset CC BY 4.0 does not license this code or resolve ethics/IRB questions.
