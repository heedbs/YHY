# Verification Results

Updated 26 October 2026. Status: **technical checks only**, not complete experiment reproduction or public-release approval.

| Check | Result | What it establishes |
| --- | --- | --- |
| Isolated CPU synthetic suite | 34 tests passed | Syntax, selected scientific AST/metric/aggregation parity, config guards, manifest binding, package checks and local preflight regressions |
| Full synthetic preflight | Passed | Wrong config/code/weight/bundle and output collision fail before model execution; successful preflight creates no run directory |
| Existing authorized local inputs | 50/50 configurations passed | Exact bundle/local patch semantics, split/config/code/pretrained SHA and source/cache presence; no input changes or run-output creation; [aggregate receipt](LOCAL_INPUT_PREFLIGHT_SUMMARY.csv) |
| Historical CPU initialization receipts | Six factory cases, full state equality | Verified-file binding matched baseline initialization; zero forward calls; not trained checkpoint/CUDA identity |
| Frozen aggregate/code/config integrity | Rechecked during packaging | No research source, config or result recalculation/change in this maintenance |

The suite uses artificial fixtures only. Commands and environment setup are maintained in [the CPU guide](../../environment/CPU_TEST_ENVIRONMENT.md), not repeated here. Local-input verification is documented in [REPRODUCIBILITY.md](../../REPRODUCIBILITY.md).

The six new preflight tests cover exact binding/no writes, crop/target/SHA alterations, duplicates/missing files, existing/symlink outputs, trainer code-digest agreement and a full synthetic success/failure sequence.

No real-data training, model forward, CUDA execution, bootstrap, benchmark, Grad-CAM or cache generation was performed. The unresolved scientific limitations are maintained in [LIMITATIONS.md](../../LIMITATIONS.md).

Earlier test counts and follow-up narratives are preserved in [the dated verification snapshot](../history/TECHNICAL_VERIFICATION_PRE_CLEANUP_20261026.md). They are historical evidence, not competing current test totals.
