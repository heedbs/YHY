# Minimum exact-input bundle and local preparation

## Current disclosure status

The author approved this exact metadata candidate on 26 October 2026. `data/reproduction/` contains losslessly compressed patches and the original split CSV. `manifest_bundle_identity.json` retains historical hashes and its earlier draft annotation as provenance; current inclusion is governed by [RELEASE_DATA_POLICY.json](RELEASE_DATA_POLICY.json). The repository is public as verified on 28 October 2026; fixed Release/archive registration is separate.

Extract outside the checkout: `python -B scripts/extract_reproduction_bundle.py --output ../orchid-exact-bundle`. Existing outputs are rejected.

Minimum draft files:

1. `patches.csv`: `uid`, `x`, `y`, `w`, `h`, `cls`, `patient_key`, `source_sha256`, `source_relative_path` (relative to the original training archive root).
2. `final_outer_inner_split_manifest.csv`: exact historical bytes, including original study patient IDs, folds, partitions, classes and existing count metadata. Its SHA must remain identical to the frozen configs.

The draft removes absolute server/cache paths and redundant path-existence checks; it does NOT anonymize the source-linked records. Original IDs, image hashes, crop coordinates and relative filenames remain linkable to downloadable source images. New aliases alone would not break that linkage and would change the split SHA. No split-hash check is bypassed to accommodate aliases.

OOF probabilities are not needed to initialize a Dataset. Ten reviewed frozen sets are now included under `data/oof/` for later reanalysis. Extraction and integrity checks perform no metric/CI recalculation. Exclusion evidence remains in original provenance and verified aggregate counts, not a fabricated empty exclusion file.

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

On the server, the local bundle was checked against the exact 149-patient, 10,072-source, 232,689-patch study inputs. Generated local semantic fields matched the originals and split bytes were identical. See [readiness evidence](../docs/audits/REPRODUCTION_READINESS.csv). After preparation, use the read-only preflight in [REPRODUCIBILITY.md](../REPRODUCIBILITY.md) to recheck the frozen bundle, local rows, config/code/split/weights and prospective output. Content-level cache verification and unresolved historical evidence remain in [LIMITATIONS.md](../LIMITATIONS.md).
