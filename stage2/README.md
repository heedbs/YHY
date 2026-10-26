# Corrected Stage 2 Source

[Reproduction](../REPRODUCIBILITY.md) · [Configs](../configs/README.md) · [Results](../results/README.md)

Entry point: `train_one_run.py`. Mamba and NoMamba share the composite architecture except the Mamba-versus-Identity operation. Plain EfficientNet-Lite2 is not the matched control.

Seeds 42, 20260824 and 314159, folds 0–4: 30 analyzed runs. Mamba seed 42 uses corrected Stage 2, not Stage 1 reuse. Selection is inner-validation patient macro F1 first; maximum 40 epochs.

Repeated seeds reuse the same patients/folds. This is matched ablation/stability analysis, not external confirmation. Read [provenance](../docs/provenance/RESULTS_PROVENANCE.md) and [limitations](../LIMITATIONS.md).
