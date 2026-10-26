# Stage 1 Source

[Reproduction](../REPRODUCIBILITY.md) · [Configs](../configs/README.md) · [Results](../results/README.md)

Entry point: `train_one_run.py`; model, data, aggregation and training implementation: `orchid/`.

Four models, seed 42, folds 0–4: 20 runs. Selection is inner-validation patient macro OVR AUROC first; maximum 80 epochs. InceptionV3 uses physical batch 8, accumulation 4; the others use 16 and 2.

These are portable review copies, not byte-identical historical executables. Read [code changes](../docs/provenance/CODE_CHANGES.csv) and [limitations](../LIMITATIONS.md). Research data, cache and weights are not bundled.
