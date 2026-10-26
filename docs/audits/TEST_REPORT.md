# Verification Results

Updated 26 October 2026. Status: **technical checks only**, not complete experiment reproduction or public-release approval.

| Check | Result | What it establishes |
| --- | --- | --- |
| Isolated CPU/metadata suite | 53 tests passed | Earlier 51 checks plus CSV deduplication and removed-file/canonical-source integrity guards |
| Full synthetic preflight | Passed | Wrong config/code/weight/bundle and output collision fail before model execution; successful preflight creates no run directory |
| Existing authorized local inputs | 50/50 configurations passed | Exact bundle/local patch semantics, split/config/code/pretrained SHA and source/cache presence; no input changes or run-output creation; [aggregate receipt](LOCAL_INPUT_PREFLIGHT_SUMMARY.csv) |
| Current source/cache content | 10,072 source SHA matches; 232,689 cache files decoded and hashed | All cache crop dimensions/channels matched the frozen rows; [aggregate content receipt](SOURCE_CACHE_CONTENT_VERIFICATION.json); no image/model output created |
| Historical CPU initialization receipts | Six factory cases, full state equality | Verified-file binding matched baseline initialization; zero forward calls; not trained checkpoint/CUDA identity |
| Frozen aggregate/code/config integrity | Rechecked during packaging | No research source, config or result recalculation/change in this maintenance |

Model/metric regressions use artificial fixtures. The evidence-index tests inspect public aggregate metadata; the separately labeled content census reads real files without executing a model. Commands and environment setup are maintained in [the CPU guide](../../environment/CPU_TEST_ENVIRONMENT.md). Local-input verification is documented in [REPRODUCIBILITY.md](../../REPRODUCIBILITY.md).

The six input preflight tests cover binding, identities and collisions. Four content regressions additionally check deterministic inventories without file changes, source corruption, cache corruption/dimensions and duplicate/conflicting identities. Default preflight remains standard-library-only; image decoding is opt-in.

The current complete-cache digest is a new inventory, not first-generation equivalence. The author subsequently approved a field-reviewed minimum metadata/OOF candidate. Twelve exact files now contain intentionally source-linked records; images, cache and private source maps remain excluded. Release scanning allows only the patient-pattern rule for exact reviewed path/SHA, never secret/private-path rules.

`verify_release_data.py` checks 232,689 patch rows, 745 split rows and ten 149-patient OOF sets against the frozen split and run catalog. It rejects target/fold/argmax/probability and hierarchy mismatches, including historical Stage 1 substitution into corrected Stage 2. This is data-integrity validation, not new inference or performance/CI calculation.

No real-data training, model forward, CUDA execution, bootstrap, benchmark, Grad-CAM or cache generation was performed. The unresolved scientific limitations are maintained in [LIMITATIONS.md](../../LIMITATIONS.md).

A SHA-pinned GitHub-hosted CPU workflow is configured; local equivalents were checked separately. Hosted success must be confirmed on the exact commit's Actions run, not inferred from this local report. See [continuous validation](../CONTINUOUS_VALIDATION.md).

Earlier test narratives remain in Git history rather than duplicate current-tree snapshots. [FILE_REDIRECTIONS.csv](../provenance/FILE_REDIRECTIONS.csv) records removed paths and current replacements. No historical research artifact was deleted.
