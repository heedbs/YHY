# Data access and disclosure decisions

Official ORCHID training source: https://doi.org/10.5281/zenodo.12636426 (train.zip). Validation/test source: https://doi.org/10.5281/zenodo.12646943 (val.zip and test.zip). Official API metadata checked 2026-10-21 reports CC BY 4.0 for both; selected metadata is preserved in data/dataset_metadata.json. Credit the dataset creators, link the DOI and license, and describe crop/resize/overlay modifications. License text: https://creativecommons.org/licenses/by/4.0/ .

The original archive split is distinct from this study's patient-wise folds. Exact duplicate-clean patient/source/patch manifests, frozen splits, image hashes and OOF probabilities are reproducibility-critical, but their public disclosure/linkability and authorized distribution need explicit review; they are excluded pending that decision. See RELEASE_FILE_DECISIONS.csv. Publicly accessible original data does not automatically authorize redistribution of study-linked patient records or resolve IRB requirements.

No raw image, cache, model weight, checkpoint, individual prediction, source-image hash list or identity map is included. Cache-generation equivalence, source DOI mapping for final panels, and unavailable metadata are not reconstructed. Synthetic test records contain artificial identities prefixed SYNTH and no research observations.

Follow-up schema/linkability review: DATA_DISCLOSURE_REVIEW.md and DATA_FIELD_RELEASE_REVIEW.csv. Only aggregate fold composition is newly included; no patient-level disclosure approval is inferred. Cache evidence is updated in PREPROCESSING_AND_ENVIRONMENT_EVIDENCE.md.
