# Approved Minimal Data Candidate

On 26 October 2026 the author authorized field review and inclusion of minimal public-ORCHID-linked split/manifest and frozen OOF data in the distribution candidate. The repository was private at that review; it is public as verified on 28 October 2026. See [current status](../STATUS.md). This is not consent to publish all server records.

## Included Fields

- Patch metadata: UID, crop coordinates/dimensions, class, original public-source-linked patient key, image SHA and archive-relative source path. No server/cache absolute path is included.
- Exact split: the seven original columns are retained to preserve historical bytes/SHA. They contain fold/partition, public-source-linked patient/class, constant source token and image/patch counts, not new clinical attributes.
- Frozen OOF: run/model/seed/fold, patient key/target, three probabilities, image/patch counts and argmax class. Four Stage 1 sets and six corrected Stage 2 model-seed sets are included separately. Mamba seed 42 in Stage 2 is corrected, not historical Stage 1 reuse.

These records are **linkable to public source images**, not newly anonymized. No real-name/contact/date-of-birth fields or additional clinical attributes are included. The review is not a guarantee against every possible reidentification route. Data-origin licensing does not substitute for an ethics determination.

## Controls

[RELEASE_DATA_POLICY.json](../../data/RELEASE_DATA_POLICY.json) lists exact candidate paths, byte/content hashes, fields and counts. Only these exact reviewed files can bypass the source-patient-pattern rule. Credentials, private paths, emails and other rules remain active. Modified files or unknown gzip content fail; uncompressed content is scanned within bounded size limits.

The earlier [field inventory](DATA_FIELD_RELEASE_REVIEW.csv) remains a dated inventory of the broader original schemas, not the current permission to disclose every listed field. Current file inclusion/exclusion is in [RELEASE_FILE_DECISIONS.csv](RELEASE_FILE_DECISIONS.csv). No image, checkpoint, cache, raw log, private source map or manuscript is added.
