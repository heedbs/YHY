# Reproduce the Study

[Home](README.md) · [Data access](DATA_ACCESS.md) · [Verification status](docs/STATUS.md) · [Scientific limitations](LIMITATIONS.md)

This is the operational guide. Historical evidence is indexed under [docs](docs/README.md); scientific limitations are maintained in one place, [LIMITATIONS.md](LIMITATIONS.md). Commands below distinguish verified CPU tooling from research execution examples.

## 1. Check the Package

Use a clean checkout with no local datasets or environment directory inside it:

```bash
python3 -B scripts/check_public_package.py . --manifest
```

For dependency setup and synthetic tests, follow [the tested CPU environment guide](environment/CPU_TEST_ENVIRONMENT.md). This environment is for validation, not a replacement for CUDA/Mamba training. Historical GPU records are in `environment/`; unresolved build details are listed in [LIMITATIONS.md](LIMITATIONS.md).

## 2. Prepare Authorized Inputs

Obtain the official data and exact input bundle as described in [DATA_ACCESS.md](DATA_ACCESS.md). Use existing validated cache files, not a newly guessed cache implementation. Prepare a private JSON mapping of model names in `data/pretrained_sources.csv` to local matching weight files.

Set paths outside the repository. These shell variables are placeholders to replace with actual authorized locations:

```bash
export ORCHID_BUNDLE=/your/authorized/exact-bundle
export ORCHID_SOURCE_ROOT=/your/ORCHID/train
export ORCHID_CACHE_ROOT=/your/validated/cache
export ORCHID_WEIGHTS_MAP=/your/private/weights.json
export ORCHID_MANIFEST_DIR=/your/new/local-manifests
python -B scripts/prepare_local_manifests.py \
  --bundle "$ORCHID_BUNDLE" --source-root "$ORCHID_SOURCE_ROOT" \
  --cache-root "$ORCHID_CACHE_ROOT" --weights-map "$ORCHID_WEIGHTS_MAP" \
  --output "$ORCHID_MANIFEST_DIR"
```

The standard-library preparation tool checks exact bundle/weight hashes, patient partitions and file presence. It creates local manifests with exclusive output creation; it does not decode images, generate cache or execute models. See [bundle schema](data/AUTHORIZED_MANIFEST_BUNDLE.md) for required fields and private handling.

## 3. Bind and Preflight One Run

Example using **corrected Stage 2 Mamba seed 42 fold 0**:

```bash
export LOCAL_CONFIG=/your/new/local-config.json
export RUN_OUTPUT=/your/new/run-output
python -B scripts/configure_local_run.py --stage stage2 \
  --template configs/stage2/stage2_correction_composite_mamba_seed42_fold0.json \
  --output "$LOCAL_CONFIG"
python -B scripts/preflight_local_run.py --stage stage2 \
  --config "$LOCAL_CONFIG" --manifest-dir "$ORCHID_MANIFEST_DIR" \
  --bundle "$ORCHID_BUNDLE" --output "$RUN_OUTPUT"
```

Preflight is read-only and requires no PyTorch, image decoder or GPU. It compares local patch semantics with the SHA-verified bundle, preserves the exact split, verifies scientific config equality and current code digest, hashes pretrained files, checks source/cache presence and rejects output collisions. A pass is **input binding only**, not cache-content, model-runtime or full reproduction certification. It does not create the future run output.

For Stage 1 use `--stage stage1`, a template under `configs/stage1/` and its trainer. The [config catalog](configs/README.md) defines the correct stage/model family. Never use historical Stage 1 Mamba seed 42 as corrected Stage 2 input.

## 4. Research Execution

The next command is a documented example, **not run during package maintenance**. A separate compatible CUDA/Mamba environment and authorized execution are required:

```bash
python stage2/train_one_run.py --config "$LOCAL_CONFIG" \
  --device cuda:0 --output "$RUN_OUTPUT"
```

Local configs record both historical and relocated code identities. Do not bypass historical checkpoint identity checks or resume a historical run under changed code.

## 5. Analysis and Figures

| Purpose | Entry point | Execution scope |
| --- | --- | --- |
| Inspect frozen tables | [results](results/README.md), `table_source_data/` | Read existing aggregates; no reanalysis needed |
| Patient metrics / paired statistics | [analysis](analysis/README.md) | Authorized saved probabilities required; reference orchestration is not turnkey |
| Figures | [figure guide](figures/README.md) | Aggregate inputs included; final external panel mapping is incomplete |
| Efficiency | `benchmark/reference/` | Historical driver checks old helper/spec hashes; review a new identity spec before new execution |
| Explainability | `explainability/reference/` | Cases excluded; review identifier handling and method limits before execution |

`ORCHID_PRIVATE_ROOT` is used only by historical reference orchestration, not ordinary training input binding. Scripts under `reference/` are preserved source, not launch approvals. See [code-change provenance](docs/provenance/CODE_CHANGES.csv) for portability and guard changes. Current test results are maintained only in the [test report](docs/audits/TEST_REPORT.md).
