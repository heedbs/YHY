# Evidence Guide for Manuscript Comparison

Check statements against evidence, not the reverse. The [claim-source index](provenance/CLAIM_SOURCE_INDEX.csv) identifies the intended repository file and exact current SHA for each topic.

## Evidence Order

1. Frozen original manifests and execution config/code establish what was run. Public run/config/checkpoint identities are in `configs/run_catalog.csv`; private path mappings are not bundled.
2. Frozen final analyses establish stored estimates. Stage 1 and corrected Stage 2 are different source families.
3. Curated figure/table inputs are conveniences, not independent experiments. Record corrections rather than copying export errors.
4. Manuscript text is a comparison target, not an authority for choosing runs. Rounded numerical agreement cannot uniquely identify a checkpoint; provenance and hashes are required.

## Where to Check

| Topic | Evidence |
| --- | --- |
| Cohort flow / exclusion units | [Verified cohort flow](../data/cohort_flow.json) |
| Stage 1 performance | [Frozen results](../results/README.md) |
| Corrected Stage 2 | Seed-specific metrics, paired effects and corrected run catalog in the index |
| Training and selection | [Config guide](../configs/README.md), catalog and code provenance |
| Efficiency | Batch/mode-specific summaries; benchmark checkpoint identity stays unchanged |
| Learning curves | Observed histories only; missing fields and aggregation units remain explicit |
| Rights / image credits / AI use | [Documentation index](README.md); separate from numeric checks |

## Metadata Corrections

An earlier export mislabeled **156 source-image occurrences** as patch rows. Excluding all occurrences in **78 exact source-SHA groups** removes **1,248 associated patch rows**: 233,937 → 232,689. This does not demonstrate independently detected duplicate patch hashes. The earlier Figure 1 helper's pre-exclusion count of 232,845 is superseded.

[DATA_CORRECTIONS.csv](provenance/DATA_CORRECTIONS.csv) records affected fields and replacement evidence. Original files remain unchanged outside this candidate and in prior Git history. Fingerprint records retain the original source SHA alongside corrected public SHA. Model estimates, predictions, splits and checkpoint selection are unchanged.

## Interpretation

Keep pooled OOF, fold summaries, seed-specific paired intervals and descriptive seed averages separate. Use corrected Stage 2 Mamba seed 42, never Stage 1 substitution. Repeated seeds are not independent cohorts; a CI spanning zero does not establish equivalence. Keep benchmark identity, batch, timing mode and units intact.

Current cache integrity, historical cache-generation equivalence and CUDA validation are different claims. Use [LIMITATIONS](../LIMITATIONS.md) and [TEST_REPORT](audits/TEST_REPORT.md) for those boundaries. Hash verification is not ethics, pathology or publication approval. Cite a fixed reviewed Git commit, not an unspecified moving branch.
