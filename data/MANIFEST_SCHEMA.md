# Private inputs required for full reproduction

These schemas describe inputs, not permission to disclose their records. No real
patient row, image identifier, UID list or image hash list is shipped.

- `final_unique_patch_manifest.csv`: `uid`, `path`, `x`, `y`, `w`, `h`, `cls`,
  `patient_key`, `source_sha256`, `cache_path`, plus historical audit columns.
  Expected frozen count: 232,689 rows. The Dataset reads the existing cache JPEG,
  converts BGR to RGB, cubic-resizes to 256 and applies model-specific normalization.
- `final_outer_inner_split_manifest.csv`: `outer_fold`, `split`, `patient_id`,
  `class`, `source_id`, `unique_source_images`, `patch_count`. Partition values are
  `train`, `inner_val`, `outer_test`. Every patient's sources and patches move together.
- `final_duplicate_exclusion_manifest.csv`: historical exact-content exclusion
  records; retained for private provenance, not reconstructed by this package.
- `selected_model_pretrained_provenance.csv`: model name, exact timm identifier,
  official mean/std, `local_cache_path`, `weight_sha256`; public source/normalization
  fields are available in `pretrained_sources.csv`. The private local file must bind
  actual downloaded weights to their expected digests.

Image acquisition, cache-generation parameters/reference image, exclusion records,
and approved split access must be resolved before a new full run. An independent
third party cannot yet regenerate all original numbers using only this ZIP.

Follow-up: the historical notebook builder and recorded target constants have
been found. See PREPROCESSING_AND_ENVIRONMENT_EVIDENCE.md; first-generation
cache identity and exact reference fit-sample lineage are still not established.
