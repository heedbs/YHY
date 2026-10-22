# Technical verification report

Status: **PASS_TECHNICAL_VALIDATION_ONLY — PUBLIC_RELEASE_NOT_APPROVED**.

11 CPU tests passed with CUDA_VISIBLE_DEVICES empty: all Python syntax; scientific class/function AST equality; synthetic image-equal patient aggregation; contradictory target rejection; invalid probability rejection; patient-level metric checks; different Stage1/Stage2 checkpoint priorities and earlier-epoch ties; deterministic synthetic preprocessing; 50-run config catalog checks; local config hash binding/collision/mismatch; extracted final metric functions on artificial data. Fixtures are synthetic and are not research runs.

The AST fingerprint check normalizes only the empty Python3.12 type_params field absent in Python3.9; it does not discard function bodies, constants or scientific expressions. Runtime code digests were independently compared with all 50 historical configs; scientific values in public config examples match their original files exactly. Best-checkpoint/history digests and final aggregate-source numeric strings were also checked. Both absolute and relative SHA manifest entry styles were resolved.

Authoritative Stage1/Stage2/benchmark results were not reselected or recomputed. Bootstrap code was preserved without execution. Existing saved scores have some same-binary-label ties but no opposing-label tie groups; the vectorized historical rank/AP implementation should not be assumed generally tie-correct for a new dataset. No new CI or p-value was produced.

Not run: dependency installation; real-data training/inference; CUDA smoke or architecture execution; bootstrap; efficiency measurement; Grad-CAM; original-cache reconstruction; final manuscript figure equivalence; independent pathology/ethics assessment. Current packaging environment is separately recorded in environment/current_cpu_packaging_record.json.
