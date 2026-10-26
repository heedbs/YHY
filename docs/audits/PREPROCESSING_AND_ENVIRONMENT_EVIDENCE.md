# Recovered preprocessing and environment evidence

This read-only review found the pre-existing notebook cache implementation; it did not execute it or create images/cache. Earlier wording that no builder could be located was too broad.

The recovered notebook estimates LAB target mean/std as the component-wise median over up to 200 legacy train patches (seed 42). Each source crop is normalized with OpenCV BGR-to-LAB, per-patch mean/std (epsilon 1e-6), target rescaling, clipping to 0..255, uint8 conversion and LAB-to-BGR. Cache uses UID JPEG names, JPEG quality 95, and skips nonempty existing files. Crop size is 512 and grid stride 256; the final Dataset reads cache, converts RGB, cubic-resizes to 256 and applies model-specific normalization.

Saved notebook output records target mean [159.31796264648438, 156.23876953125, 105.43659973144531] and std [41.599281311035156, 9.994132995605469, 9.240299224853516]. Its saved cache execution reports zero newly written files and all existing files skipped. Consequently the notebook is evidence of an algorithm and a recorded reference, NOT proof of the first complete cache-generation run or of an immutable historical notebook revision.

An independent pre-existing August 2026 Gate A record used those same constants and reports byte-identical regenerated JPEGs for 30 of 30 sampled patches (10 per class). Its script SHA matches the saved environment receipt. Sample results were read, not rerun. That historical sample audit did not regenerate the complete cache. The separate current read-only content census is indexed in [TEST_REPORT.md](TEST_REPORT.md); hashing/decoding existing files is not a full regeneration test or proof of the original fitting scope.

The notebook reference is fitted on a LEGACY train partition, not independently inside each final outer fold. The current legacy index shares patients with current outer-test partitions. This does not establish which exact 200 patches were originally sampled; the final reference sample lineage and full fit-scope isolation remain unresolved. Do not claim fold-specific training-only normalization fitting or conclude absence of preprocessing exposure. Label-free fitting is distinct from supervised test-metric tuning but still requires transparent disclosure.

The notebook stores torchvision 0.20.1, torch 2.5.1+cu124, timm 1.0.19, NumPy 1.26.4, pandas 2.2.2 and scikit-learn 1.5.1. These are notebook-session records, not a complete final-training environment lock. Existing final experiment records remain separately authoritative. Exact compiler, causal-conv1d build, all plotting dependencies and complete cache-generation environment are not established. Current environment is not substituted for historical evidence.

No automatic cache builder or new reproduction claim is added. Public training still requires validated cache and approved exact manifests; adapted code has not been CUDA-equivalence tested. Missing dependency versions are not guessed.
