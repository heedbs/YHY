# Results provenance

Current claim-to-file mapping is in [CLAIM_SOURCE_INDEX.csv](CLAIM_SOURCE_INDEX.csv). Cohort export unit/count corrections are recorded in [DATA_CORRECTIONS.csv](DATA_CORRECTIONS.csv); corrected public copies retain original source fingerprints. These corrections do not change frozen model estimates.

Byte-identical result aliases and obsolete review snapshots were removed from the current tree; [FILE_REDIRECTIONS.csv](FILE_REDIRECTIONS.csv) preserves their hashes and canonical replacements. Active fingerprints index retained files only. Original server artifacts and prior commits are unchanged.

Source families were selected from frozen final analysis manifests and corrected Stage2 source provenance, not timestamps. results/ CSVs preserve stored numerical strings. SOURCE_FILE_FINGERPRINTS.csv links the indexed source-derived files to original artifact labels and SHA values; it does not cover every newly authored document or helper. SHA256_MANIFEST.txt covers the complete candidate file set except itself. CLAIM_SOURCE_INDEX.csv maps selected evidence topics to current repository files. configs/run_catalog.csv links all 50 runs to original config/code/split/pretrained/best-checkpoint digests. Private absolute-path mapping and exact code diffs remain outside the public ZIP.

Stage1: 20 completed runs, seed42, AUROC-first lexicographic tuple (AUROC, macro F1, negative validation patch loss, earlier epoch), max80; pooled OOF149 patients/model. Fold means are separate descriptive outputs.

Corrected Stage2: 30 matched runs (15 per condition), seeds42/20260824/314159, F1-first tuple (macro F1, AUROC, negative validation patch loss, earlier epoch), max40. All five Mamba seed42 folds come from correction outputs bound by the corrected analysis queue/provenance. No historical Stage1 Mamba seed42 is used here. Primary Mamba-minus-NoMamba macro F1: +0.0396 CI[-0.0270,+0.1187], +0.0135 CI[-0.0123,+0.0406], -0.0087 CI[-0.0468,+0.0215]. These are display roundings of frozen full-precision values. Bootstrap uses 10,000 paired, class-stratified patient samples, seed20260830. Three-seed descriptive SD uses ddof=1; no independent447-patient sample or pooled CI is created.

Efficiency: five models, RTX4500 Ada, FP32, TF32 disabled,256×256,30 warm-up/100 timed iterations×3 blocks. model_only excludes preprocessing, H2D, file I/O, initialization. The distinct preprocessing-plus-forward mode remains separate. Benchmark Composite-Mamba checkpoint precedes the corrected seed42 training; it must not be described as measured from the correction checkpoint. Digests are retained in efficiency summaries; performance and timing are different estimands. Mamba FLOPs/MACs remain NOT_RELIABLY_ESTIMATED.

The author-designated current manuscript was supplied for a separate internal read-only comparison on 26 October 2026: 144 numeric entries in Tables 2/4/5/6/7 matched stored sources at display precision. This was not a check of every narrative claim, embedded figure or supplementary file. Final pathology-panel source identities and FigureLabs edit mappings remain unconfirmed. Repository evidence is not altered merely to fit manuscript text; documented cohort metadata corrections follow frozen manifests instead.

Plotting-source caveat: the original figure manifest excludes `build_figures.py`; the current script SHA is recorded in SOURCE_FILE_FINGERPRINTS.csv, but a historical execution SHA is unavailable for that generator. This is preserved as an unresolved source-provenance limitation.

## Development outer-test access

Clarification added 29 October 2026; historical audit records remain unchanged.

An early v2.3 development smoke evaluated the fold-0 outer partition for the four Stage 1 models at seed 42 and saved predictions and metrics for the same 30 patients and 45,539 patches per model. This access is confirmed. The runs were invalidated because of smoke outer-test access, output-path binding discrepancies and missing process-level resume evidence.

The limited read-only lineage audit reviewed 50 final production runs and ten final OOF sets. No reuse of the v2.3 smoke checkpoints, predictions or metrics was identified in those final production or numerical-analysis records. The ten OOF sets were linked to their production prediction inputs; recorded checkpoint/config identities and analysis input paths were checked. This supports a scoped final-artifact lineage conclusion, not a guarantee that the early values never influenced model or configuration choices, or that every externally edited manuscript panel has been traced.

Historical access to the saved outputs by people or AI, and any influence on subsequent development decisions, could not be fully established. No explicit record attributing a later choice to the smoke performance was located in the reviewed material; absence of such a record does not establish non-use. Later corrective documentation prohibited using the invalid outputs for selection, but that policy is not evidence of no prior viewing or influence. Stage 2 was explicitly designed after Stage 1; this audit does not establish that all study-design decisions were independent of every outer-test result.

In the audited final production runs, checkpoint selection used inner-validation metrics and outer evaluation occurred once after reloading the selected best checkpoint. That production rule is distinct from the confirmed development access and does not negate it.

The internal audit artifacts are identified below by basename and SHA-256; internal absolute paths and patient-level records are not added here. They document the review scope rather than a new experiment or an independently archived public release.

| Internal audit artifact | SHA-256 |
| --- | --- |
| `OUTER_TEST_ACCESS_PROVENANCE_AUDIT.md` | 25dcc205453094ea785a0206936d31f8509d0dfdc0debd1d2a45046ff8ce4011 |
| `FINAL_ARTIFACT_LINEAGE_CHECK.csv` | 979f0278a16c54b951cd4b9098f771379ad6161a5a51da2d39fd9f92810c4c6c |
| `CLAIM_EVIDENCE_MATRIX.csv` | 5974f0d983ff0bdbbfc6a2f62626189c1f67dc1c621a1e88641c6a2c18f6d9e5 |

This development-access issue is separate from the [normalization-reference limitations](../../LIMITATIONS.md): the actual reference patch roster and original full-cache generation lineage remain UNVERIFIED. Legacy-index population overlap is not established overlap of the actual reference sample; neither actual leakage nor absence of exposure or performance impact follows from the missing records. CPU/SHA/OOF checks do not establish complete historical GPU reproduction or preprocessing independence.
