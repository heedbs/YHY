> Historical snapshot from 22 October 2026. For current status, see [docs/STATUS.md](../STATUS.md). Statements about branch location or pending choices describe that earlier snapshot.

# Publication candidate: verified scope and remaining author decisions

Status: technical preparation substantially checked; NOT full numerical reproduction or automatic public-release approval.

## Completed technical checks

- 50 scientific config templates retain stage-specific settings and exact split identity.
- The server-only minimal manifest draft preserves all 232,689 original patch semantic rows and the original split bytes.
- An offline CPU initialization audit traced the actual pretrained file and all applied state tensors for the five publication model variants in the current default interpreter environment. No model forward occurred. See data/PRETRAINED_INITIALIZATION_AUDIT.csv.
- Feature-only Composite backbones discard the original classification head's eight keys by design. A first audit incorrectly counted those absent keys as mismatched; the audit was corrected and independently rerun for the two variants. All 378 applied backbone tensors matched. Initial records remain in private provenance; no model code was changed.
- A new isolated CPU-only environment installed successfully; 28 synthetic tests and pip check passed. This does not validate the full historical CUDA environment.
- Source/config/result bytes and public-package SHA/security checks are maintained. No new training, inference, bootstrap, benchmark or cache generation occurred.

## What cannot be certified retrospectively by code changes

1. Exact authorship/derivation of historical MixStyle/GeM/local composite files and institution/contributor permissions. Official method-reference licenses are MIT and now retained in licenses/ for conservative attribution; this is not proof of actual derivation or a substitute for rights confirmation. ../licensing/LICENSE_SCOPE.md exceptions remain.
2. Permission and acceptable linkage scope for the actual patient/source-linked bundle and OOF records. The review draft remains server-only; only schema, aggregate counts and whole-file identity are published in the candidate. Public source images and CC BY do not automatically establish all privacy/ethical conditions.
3. The first complete cache-generation run and exact reference-fit sample lineage. Historical notebook/sample equivalence evidence is available, but reference fitting is legacy-train based, not current fold-specific train fitting. Do not erase this limitation or fabricate a new historical pipeline.

## Remaining release sequence

Resolve the explicit component/data decisions above; reconcile the final manuscript figures and availability statement with actual shared files; then review and merge this branch to main. Make the repository public only after the agreed review and record the exact submission commit/release. No main merge, visibility change, release or DOI creation is performed by this document.

More generic audit documents are not a substitute for authorized exact inputs, a truthful Methods description or a validated historical environment. The original study results are preserved even when a reproducibility limitation remains.

The public-copy factory now explicitly decodes the SHA-verified safetensors file and supplies its state_dict to timm. This avoids both mutable default cache selection and filename-suffix guessing when HF symlinks resolve to extensionless blobs. A file-overlay-only intermediate attempt failed on that suffix issue; its private audit record is retained and no such failing intermediate code was uploaded. Final old/new seeded full-state hashes match for all six audited factory cases. Only public copies change; original source/checkpoints remain untouched.
