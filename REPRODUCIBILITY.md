# Reproducibility and execution status

## Environment

Historical training/benchmark records, not the current pip environment, are in environment/. They report Python3.12.7, PyTorch2.5.1+cu124, CUDA12.4, cuDNN90100, timm1.0.19, mamba_ssm2.2.4, NumPy1.26.4, pandas2.2.2, sklearn1.5.1, OpenCV4.10.0. Some plotting/packaging dependency versions are not fully recorded. The current CPU packaging interpreter is separately identified in TEST_REPORT.md.

Installation order (documentation example, not executed here): create a Python3.12 environment; install the CUDA12.4 build of torch2.5.1 from the official PyTorch index; install compatible torchvision (historical exact version must be verified); install recorded scientific packages; install a CUDA/compiler-compatible mamba-ssm2.2.4 and required causal-conv1d dependency only after torch. The exact historical causal-conv1d/compiler build is unresolved. A recovered legacy notebook records torchvision0.20.1; this is not proof of the final-training wheel/build identity. See PREPROCESSING_AND_ENVIRONMENT_EVIDENCE.md. No CPU/Identity fallback is permitted for Composite-Mamba. Do not install new packages into the historical research environment.

## Local data configuration

`ORCHID_MANIFEST_DIR` points to a private directory containing final_unique_patch_manifest.csv, final_outer_inner_split_manifest.csv, final_duplicate_exclusion_manifest.csv and selected_model_pretrained_provenance.csv. The pretrained table must have correct local_cache_path entries and verified weight_sha256. Set local cache/source paths in privately supplied manifests; keep logical patient/source/UID relations unchanged. `ORCHID_PRIVATE_ROOT` is a relocated research-artifact root used only by historical reference orchestration scripts. No frozen patient split or metadata content is distributed by default.

## Path-adapted training copies

Public configs preserve scientific settings and keep the historical code hash as historical_code_sha256. They are examples, not launch approvals. Create a separate local config with scripts/configure_local_run.py: it calculates the relocated code hash and requires the exact historical split SHA, then writes a new output config without overwriting any existing file. This explicitly distinguishes historical identity from local reproduction identity. The extracted Mamba module has the same class ASTs and parameter paths but a different source hash. Historical checkpoints must not be silently resumed under the changed source identity; original-code runtime provenance and strict checkpoint identity requirements remain documented.

Documentation examples, not research commands executed during this task:

```bash
export ORCHID_MANIFEST_DIR=/your/private/manifests
export ORCHID_PRIVATE_ROOT=/your/private/research-artifacts
python scripts/configure_local_run.py --stage stage1 --template configs/stage1/stage1_inceptionv3_seed42_fold0.json --output local_config.json
python stage1/train_one_run.py --config local_config.json --device cuda:0 --output /your/new/run
```

Use Stage2 corrected config examples and the Stage2 trainer for the matched study. Do not replace corrected Stage2 seed42 with Stage1 checkpoint output. Phase-specific selection tuples, physical batch8×accumulation4 for Inception and batch16×2 for other models, FP32, patient-balanced sampling, max80/max40 and no TTA are retained in the 50 examples.

## Analysis, figures, benchmark, explainability

analysis/stage1_metrics.py and analysis/stage2_statistics.py expose exact extracted metric/statistic functions. analysis/reference/ retains final filesystem orchestration for audit; it requires original private metadata/outputs and has not been re-executed. Reference plotting scripts expect the named figure-source layout and can write files; select a new output root in a reviewed local copy. Bootstrap/reference analyses must not be run as part of a simple import/test.

Public-copy maintenance: `analysis/stage2_statistics.py` now guards the historical bootstrap entry point against invalid/unstratified inputs and opposing-label score ties. The original numerical implementation and `analysis/reference/analyze_stage2_corrected.py` remain preserved; no frozen estimate/CI was regenerated. Unsupported tied-score data must not be silently evaluated by this helper. Config generation also rejects stage/template selection-rule mismatches. SHA validation now checks complete file coverage, duplicates and unsafe paths. See MAINTENANCE_CHANGES.csv.

benchmark/reference/ is the actual measured driver and source helper, with only path portability edits. It still validates historical source-runner/spec hashes. Because a path-adapted helper has a new file hash, the historical driver is intentionally NOT claimed turnkey or revalidated for new measurement. An explicit reviewed local identity spec is needed before any future benchmark. The original telemetry parser can return an error on multiple GPU lines; this limitation is preserved, not silently fixed.

explainability/reference/ is the actual r1 runner. Frozen cases are excluded; output includes IDs unless separately reviewed. It selects median/high evidence patches within each model's own scores; identical-patch matched heatmap comparisons and patient-OOF-to-patch argmax agreement were not established by this runner. Do not claim validated localization, deterministic repeatability, or pathology validation from file presence.

Tested now: syntax, exact selected class/function AST parity, synthetic aggregation/metric tests, config scientific settings, frozen aggregate-source integrity, privacy/path/secret/file-size scans, SHA and ZIP checks. Not tested now: package installation, research training/inference, CUDA equivalence of adapted models, bootstrap execution, benchmark, Grad-CAM, or bitwise reproduction of all paper figures.
