# Data disclosure review: no raw patient rows uploaded

The author authorized investigation and expressed willingness to disclose reproducibility data. A technical review cannot certify all legal or reidentification conditions. Original split, patient, source, patch, exclusion and corrected OOF schemas were inspected read-only. Counts and field-level decisions are public; original values/mappings remain server-only.

Observed frozen corpus: 149 patients, 10,072 sources, 232,689 patches; patient outer-test membership exactly once across 5 folds; all per-fold train/inner-validation/outer-test patient sets disjoint. This is a metadata identity check, not new prediction evaluation.

Original patient keys and filenames link directly to downloadable source records; source hashes allow exact linking; patch UIDs/cache names and crop coordinates permit linkage. Paths additionally expose server layout. Class and per-patient image/patch counts can act as quasi-identifiers in combination. Replacing only the patient column does not establish anonymization. No names/contacts are schema fields in these manifests, but that is not proof of patient anonymity.

Recommended release design: disclose only the authorized minimum deterministic source-to-study mapping, fold/partition, target and crop hierarchy needed for exact reruns, after explicitly reviewing source-data linkage. If only opaque study aliases are released, acknowledge that exact source-linked reruns remain impossible without the withheld mapping. Avoid publishing redundant patient rows, server/cache paths and raw logs. OOF probabilities require the same explicit linkage review. Do not claim a need for secrecy as a legal fact if none has been established.

This update adds only aggregate fold counts and schema-level review. A final field/list decision is still required before posting original or derived patient-level split/OOF rows; no claim that those data have been publicly released is made. The internal source inventory retains paths and SHA for author review only.
