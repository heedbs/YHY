# Verification Results

Updated 26 October 2026. Status: **technical checks only**, not complete experiment reproduction or public-release approval.

| Check | Result | What it establishes |
| --- | --- | --- |
| Isolated CPU synthetic suite | 38 tests passed | Earlier 34 checks plus four CPU cache-content regressions: determinism/no writes, wrong source SHA, damaged/dimension-mismatched cache and duplicate/conflicting identities |
| Full synthetic preflight | Passed | Wrong config/code/weight/bundle and output collision fail before model execution; successful preflight creates no run directory |
| Existing authorized local inputs | 50/50 configurations passed | Exact bundle/local patch semantics, split/config/code/pretrained SHA and source/cache presence; no input changes or run-output creation; [aggregate receipt](LOCAL_INPUT_PREFLIGHT_SUMMARY.csv) |
| Current source/cache content | 10,072 source SHA matches; 232,689 cache files decoded and hashed | All cache crop dimensions/channels matched the frozen rows; [aggregate content receipt](SOURCE_CACHE_CONTENT_VERIFICATION.json); no image/model output created |
| Historical CPU initialization receipts | Six factory cases, full state equality | Verified-file binding matched baseline initialization; zero forward calls; not trained checkpoint/CUDA identity |
| Frozen aggregate/code/config integrity | Rechecked during packaging | No research source, config or result recalculation/change in this maintenance |

The unit suite uses artificial fixtures only. The separately labeled content census reads real files but does not execute a model. Commands and environment setup are maintained in [the CPU guide](../../environment/CPU_TEST_ENVIRONMENT.md). Local-input verification is documented in [REPRODUCIBILITY.md](../../REPRODUCIBILITY.md).

The six input preflight tests cover binding, identities and collisions. Four content regressions additionally check deterministic inventories without file changes, source corruption, cache corruption/dimensions and duplicate/conflicting identities. Default preflight remains standard-library-only; image decoding is opt-in.

The current complete-cache digest is a new inventory of existing files, not a comparison against an unavailable first-generation inventory. It does not prove historical generation/fitting equivalence. A source-linked exact-input ZIP was prepared for server-side review only; no patient rows, images or per-image hash list were uploaded.

No real-data training, model forward, CUDA execution, bootstrap, benchmark, Grad-CAM or cache generation was performed. The unresolved scientific limitations are maintained in [LIMITATIONS.md](../../LIMITATIONS.md).

Earlier test counts and follow-up narratives are preserved in [the dated verification snapshot](../history/TECHNICAL_VERIFICATION_PRE_CLEANUP_20261026.md). They are historical evidence, not competing current test totals.
