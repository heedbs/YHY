# Recovered preprocessing and environment evidence

This read-only review found the pre-existing notebook cache implementation; it did not execute it or create images/cache. Earlier wording that no builder could be located was too broad.

The recovered notebook estimates LAB target mean/std as the component-wise median over up to 200 legacy train patches (seed 42). Each source crop is normalized with OpenCV BGR-to-LAB, per-patch mean/std (epsilon 1e-6), target rescaling, clipping to 0..255, uint8 conversion and LAB-to-BGR. Cache uses UID JPEG names, JPEG quality 95, and skips nonempty existing files. Crop size is 512 and grid stride 256; the final Dataset reads cache, converts RGB, cubic-resizes to 256 and applies model-specific normalization.

Saved notebook output records target mean [159.31796264648438, 156.23876953125, 105.43659973144531] and std [41.599281311035156, 9.994132995605469, 9.240299224853516]. Its saved cache execution reports zero newly written files and all existing files skipped. Consequently the notebook is evidence of an algorithm and a recorded reference, NOT proof of the first complete cache-generation run or of an immutable historical notebook revision.

An independent pre-existing August 2026 Gate A record used those same constants and reports byte-identical regenerated JPEGs for 30 of 30 sampled patches (10 per class). Its script SHA matches the saved environment receipt. Sample results were read, not rerun. That historical sample audit did not regenerate the complete cache. The separate current read-only content census is indexed in [TEST_REPORT.md](TEST_REPORT.md); hashing/decoding existing files is not a full regeneration test or proof of the original fitting scope.

The notebook reference is fitted on a LEGACY train partition, not independently inside each final outer fold. The current legacy index shares patients with current outer-test partitions. This does not establish which exact 200 patches were originally sampled; the final reference sample lineage and full fit-scope isolation remain unresolved. Do not claim fold-specific training-only normalization fitting or conclude absence of preprocessing exposure. Label-free fitting is distinct from supervised test-metric tuning but still requires transparent disclosure.

The notebook stores torchvision 0.20.1, torch 2.5.1+cu124, timm 1.0.19, NumPy 1.26.4, pandas 2.2.2 and scikit-learn 1.5.1. These are notebook-session records, not a complete final-training environment lock. Existing final experiment records remain separately authoritative. Exact compiler, causal-conv1d build, all plotting dependencies and complete cache-generation environment are not established. Current environment is not substituted for historical evidence.

No automatic cache builder or new reproduction claim is added. Public training still requires validated cache and approved exact manifests; adapted code has not been CUDA-equivalence tested. Missing dependency versions are not guessed.

## Submission recheck — 28 October 2026

This recheck reads preserved code, outputs and manifests only; it does not execute fitting or recreate patches. Notebook indices below are zero-based cell indices, not execution order. Source fingerprints are current read-only artifact identities, not proof of immutability since first generation.

| Item | Preserved evidence and finding | Status / boundary |
| --- | --- | --- |
| Crop/grid | dataset.ipynb cell 5: PATCH=512, STRIDE=256; crop precedes LAB matching | SERVER_VERIFIED_CODE; no reconstruction executed |
| LAB formula | Cell 8: BGR→LAB float32, patch mean and population std+1e-6; (LAB−mean)/std × target_std + target_mean; clip [0,255], uint8, LAB→BGR | SERVER_VERIFIED_CODE |
| Target | Component-wise medians of eligible sampled patch statistics; recorded mean [159.31796264648438,156.23876953125,105.43659973144531], std [41.599281311035156,9.994132995605469,9.240299224853516] | SERVER_VERIFIED_SAVED_OUTPUT, not first-generation lineage |
| Sampling | Cell 4 SEED=42; cell 8 pandas sample min(200,len(train_df)), RandomState(seed), unreadable/empty crops skipped; fallback constants if none readable | SERVER_VERIFIED_CODE_AND_SAVED_EXECUTION; exact contributing patches/count NOT_RECORDED |
| Legacy partition | Cell 6 uses patient-stratified splits and class-stratified patch-budget subsampling to construct train_df; then target estimator uses pixels, not diagnosis/performance directly | Labels affect upstream partition composition; no observed test-metric fitting objective in inspected estimator |
| Cache then resize | Cell 8 JPEG quality95, UID filenames, skips nonempty files; final Stage1/2 OrchidPatchDataset.__getitem__: BGR→RGB then INTER_CUBIC 256×256, then training-only augmentation and model normalization | SERVER_VERIFIED_CODE; no new builder supplied |
| Saved execution | Cell 8 execution_count=6; target log and 0 new / 168600 train + 46800 val + 18537 test skipped | Establishes recorded reference + reuse session; not first cache write, sampled UID identity or historical revision |
| JPEG sample | Preserved Gate A: 30/30 exact SHA matches, 10/class, audit seed20260823 | Historical sampled reproduction only, not the normalization fit sample or entire-cache equivalence |
| Content census | Existing receipt: 10072 source hashes checked,232689 cache decodes; inventory4250d64d45bd8109e124fa7141178c75707238532978ee745f1cf54c69e6eedd | Current content only; no proof of original fold-isolated fit |
| Actual reference roster | No saved selected-UID roster or execution-bound ordered input/RNG/decoder receipt located in inspected preserved evidence | UNVERIFIED; current seed+index alone cannot establish historical sample |
| Human selection history | No immutable evidence of whether reference/constants were selected using test outcomes | UNVERIFIED; absence of evidence is not proof of non-use |

The current legacy index overlaps final outer-test patients/sources. This is **index-population overlap**, not established overlap of the unknown actual normalization reference sample:

| Outer fold | Legacy-index patient overlap | Legacy-index source-path overlap | Source-SHA overlap for those matched source rows | Actual fit-sample overlap |
| --- | ---: | ---: | ---: | --- |
| 0 | 17 | 1147 | 1147 | UNVERIFIED |
| 1 | 22 | 1692 | 1692 | UNVERIFIED |
| 2 | 24 | 1584 | 1584 | UNVERIFIED |
| 3 | 19 | 1306 | 1306 | UNVERIFIED |
| 4 | 22 | 1520 | 1520 | UNVERIFIED |

Source-SHA counts above use the already frozen source catalog for exact path matches; images are not rehashed or decoded here. They do not establish additional byte-duplicate paths outside that mapping. Actual fit-sample patient/source overlap cannot be computed without its execution-bound roster. A label-free reference can expose test-distribution information; this differs from supervised label/test-metric leakage. We establish neither actual test-sample fitting nor absence of preprocessing exposure, and do not estimate or correct its performance impact.

### Author follow-up (only evidence unavailable on the server)

| Specific question | Required record | Already established | Wording if unavailable |
| --- | --- | --- | --- |
| Is the original normalization-fit sample roster preserved elsewhere? | Selected patch UID list, ordered input hash, actual read-success list and execution receipt | Saved estimator output exists; roster not found | Exact reference sample and its outer-test overlap remain unverified |
| Is the first complete cache-writing run preserved elsewhere? | Immutable notebook revision plus original write log, input/output manifests and environment | Saved session wrote zero files; 30-sample check is limited | Original full-cache generation lineage is not established |
| Was the reference chosen independently of test outcomes? | Dated decision record or pre-outcome protocol, not memory alone | Inspected estimator consumes pixels; upstream partition uses labels | Historical decision/viewing history is unverified |

No author recollection alone is promoted to immutable execution proof. Algorithm, recorded constants, index-population overlap and existing census/sample scope are already server-verifiable and do not require the author to rediscover them.

### Read-only source fingerprints

Server absolute paths are retained only in the separate internal handoff. Basenames and SHA below locate the preserved evidence without exposing server paths.

| Source | Current SHA-256 |
| --- | --- |
| dataset.ipynb | 1cc552eabe9efa56c3b2a4fe0f521d0524173aab2a771c82843e705987d894bf |
| train_index.csv | 1bf0d190a2187dfb544e3d94f79b69f0831a356c31431be5bb1529a9f125423f |
| final_outer_inner_split_manifest.csv | e3aa21fa00af400d11287251e67027f001d433b7c231bc14526bb85f3fcd66e7 |
| final_unique_source_manifest.csv | 9ce3c5ebbd0d9ce2ec013463f2a13a383797e65c0a9f5eed89b5f6eafd9e2f0a |
| final_unique_patch_manifest.csv | 0a11c1fbf95be513220a09da19fa5bd42eb6749ec384b1c09aab52b3b29a9a99 |
| run_gate_a_preprocessing_equivalence.py | cca467ec7c13591794cc0257eb0ea654e0c885a21bc7d42261f84bf224e223bc |
| gate_a_environment.txt | 255dc9b2ae65433a02a4c01511c7984eb6915136ca5bc8ed59efa9c79e55df73 |
| gate_a_comparison_results.csv | 19af46428d621b2ce388f7c56c244aeda61431710f6893b2b0ca8a3d41254662 |
| gate_a_sha256_manifest.txt | 1241795ac04f7b51340050fe4db80c35e7e2c9d86aae03279b9ee4c15d78557a |
| SOURCE_CACHE_CONTENT_VERIFICATION.json | 1b2f62119239b80583b97a5067db00ba8a0cc6e75d94f4bd2bf827400798de18 |
