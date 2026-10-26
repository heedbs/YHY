# Frozen Results

[Home](../README.md) · [Results provenance](../docs/provenance/RESULTS_PROVENANCE.md)

Saved aggregate outputs, not recalculated estimates. Fold summaries and pooled patient-level OOF metrics must not be interchanged.

| Question | Source |
| --- | --- |
| Stage 1 pooled performance | [Pooled OOF](stage1/stage1_pooled_oof_metrics.csv) |
| Fold variability | [Fold metrics](stage1/stage1_fold_metrics.csv), [CV summary](stage1/stage1_model_cv_summary.csv) |
| Class-level results | [Classwise metrics](stage1/stage1_classwise_metrics.csv), [confusion matrices](stage1/stage1_pooled_confusion_matrices.csv) |
| Corrected Stage 2 by seed | [Seed metrics](stage2/STAGE2_CORRECTED_SEED_SPECIFIC_METRICS.csv) |
| Corrected matched differences | [Paired effects](stage2/STAGE2_CORRECTED_PAIRED_EFFECTS.csv) |
| Descriptive seed stability | [Three-seed summary](stage2/STAGE2_CORRECTED_THREE_SEED_DESCRIPTIVE.csv) |
| Efficiency and plotting input | [Canonical model summary](efficiency/EFFICIENCY_MODEL_SUMMARY.csv) |
| Manuscript tables | [Table source data](../table_source_data/) |

Stage 2 compares Mamba minus NoMamba on the same patients. Primary CIs include zero; non-significance is not equivalence. Keep efficiency modes/batches separate. Historical benchmark identity is not relabeled as the corrected checkpoint. Aggregates here are unchanged; reviewed frozen probabilities are separately included under [data/oof](../data/oof/).
