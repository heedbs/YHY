# Minimum exact-input bundle and local preparation

## Current disclosure status

The authors have prepared a server-only review draft; the patient/source rows are NOT included in this repository. The public `manifest_bundle_identity.json` contains whole-file fingerprints and aggregate expectations only. Possession of that identity document is not access to the actual bundle or permission to distribute linked records. Ask the authors about the finalized access route; no automatic request-access promise is made here.

Minimum draft files:

1. `patches.csv`: `uid`, `x`, `y`, `w`, `h`, `cls`, `patient_key`, `source_sha256`, `source_relative_path` (relative to the original training archive root).
2. `final_outer_inner_split_manifest.csv`: exact historical bytes, including original study patient IDs, folds, partitions, classes and existing count metadata. Its SHA must remain identical to the frozen configs.

The draft removes absolute server/cache paths and redundant path-existence checks; it does NOT anonymize the source-linked records. Original IDs, image hashes, crop coordinates and relative filenames remain linkable to downloadable source images. New aliases alone would not break that linkage and would change the split SHA. No split-hash check is bypassed to accommodate aliases.

OOF probabilities are not necessary to initialize a training Dataset, but are needed for independent reanalysis of frozen results without retraining. They remain a separate disclosure decision. Exclusion evidence is scientifically important but not read by the current Dataset; it remains in original provenance, not replaced by a fabricated empty file.

## CPU-only local preparation

After obtaining an authorized exact bundle, original source images, the existing validated JPEG cache, and matching pretrained artifacts, prepare a private JSON mapping from the four Stage 1 model names in `pretrained_sources.csv` to your local weight files. Both Stage 2 variants use the `EfficientNet-Lite2+Mamba` backbone provenance row. Do not publish this JSON if it contains personal paths.

```bash
python scripts/prepare_local_manifests.py --bundle /your/authorized/bundle --source-root /your/ORCHID/train --cache-root /your/validated/cache --weights-map /your/private/weights.json --output local_manifests
export ORCHID_MANIFEST_DIR="$PWD/local_manifests"
python scripts/configure_local_run.py --stage stage1 --template configs/stage1/stage1_inceptionv3_seed42_fold0.json --output local_config_stage1.json
```

This is a documented path example, not a fresh research execution. The preparation tool uses the standard library only: it checks bundle hashes/metadata, source/cache file presence and pretrained-file SHA; relocates paths; and preserves exact split bytes. It never loads tensors, decodes images, generates cache, trains, infers or downloads files. Existing output is rejected. File presence is NOT full source/cache content verification or proof of actual model initialization. Keep generated local manifests private.

`weight_artifact_identity.csv` identifies frozen safetensors artifact revisions from local provenance. Alternative `.pth` URLs must not be assumed to have the same byte SHA. No weight was downloaded in this follow-up. Public-copy factories now explicitly pass the SHA-verified file to timm; an offline CPU audit compared old and new initialization states bitwise under seed 42. See PRETRAINED_INITIALIZATION_AUDIT.csv. This change does not modify original production code, regenerate research checkpoints, or prove historical/CUDA equivalence. The generated local binding receipt still correctly reports that the preparation tool itself does not initialize models.

## What was actually checked

On the server, the local bundle was checked against the exact 149-patient, 10,072-source, 232,689-patch study inputs. Generated local patch semantic fields matched all original rows, and split bytes were identical. Fifty local config examples were bound separately without model execution; see `../docs/audits/REPRODUCTION_READINESS.csv`. This is input preparation evidence, not a clean-install or complete numerical reproduction. Full cache generation, reference fitting scope, full historical environment and composite source clearance remain explicit limitations.
