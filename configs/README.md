# Run Configurations

[Home](../README.md) · [Local binding](../REPRODUCIBILITY.md)

[run_catalog.csv](run_catalog.csv) indexes 50 scientific conditions: 20 Stage 1 and 30 corrected Stage 2 runs. Config copies are examples for local input binding, not execution approvals.

| Condition | Selection priority | Epoch cap | Batch / accumulation |
| --- | --- | --- | --- |
| Stage 1 InceptionV3 | Patient macro OVR AUROC, then macro F1 | 80 | 8 / 4 |
| Other Stage 1 models | Patient macro OVR AUROC, then macro F1 | 80 | 16 / 2 |
| Corrected Stage 2 | Patient macro F1, then macro OVR AUROC | 40 | 16 / 2 |

The complete comparison tuple also uses negative validation patch loss and earlier epoch. Use the actual config and trainer comparator, not loss minimization alone. Do not substitute historical Stage 1 Mamba seed 42 into Stage 2. Path/code hash rebinding is not proof of historical execution or safe historical resume.
