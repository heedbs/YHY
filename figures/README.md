# Figure Source

[Figure data](../figure_source_data/) · [Table data](../table_source_data/) · [Results](../results/README.md)

Byte-identical aggregate copies were removed from the figure/table folders. [FILE_REDIRECTIONS.csv](../docs/provenance/FILE_REDIRECTIONS.csv) maps old paths to canonical results. The former `SUPP_FOLD_CONFUSION_MATRICES.csv` was pooled OOF data, not fold-specific data, and was removed rather than relabeled as a fold result. Actual per-fold matrices are in `results/stage1/stage1_fold_metrics.csv`.

Historical reference scripts still target the original external source package, which was not changed. They are not turnkey drivers for the deduplicated public folder; use the mapping when adapting inputs. No figure generation is performed by this cleanup.

Code and aggregate plotting inputs support inspection of saved analyses. They do not certify equivalence to manuscript panels edited externally; final mapping remains incomplete.

Learning curves use observed train/inner-validation history only. Outer performance is not an epoch-level convergence curve. Fold/seed dispersion is not a patient-level CI. Different early-stop epochs do not justify padding or carry-forward.

Pathology panels, images and weights are not included. Reviewed frozen patient predictions are separately available in `data/oof/`. Confirm attribution and pathology-review status before manuscript reuse. See [AI use](../docs/provenance/AI_USE_INVENTORY.md) and [limitations](../LIMITATIONS.md).
