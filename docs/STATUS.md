# Private Pre-release Handoff

Updated 26 October 2026. **Author-approved code and minimum linked-data candidate prepared for final private review.** Repository `heedbs/YHY` remains private. No public Release, visibility change or archive registration has been performed.

## Candidate Scope

| Included | Excluded |
| --- | --- |
| Original project software under MIT; third-party notices retained | Third-party library binaries and weight files |
| Fifty scientific config examples and frozen aggregate outputs | Source images, patch cache, checkpoints and full server logs |
| Exact minimum patch metadata and original split bytes | Additional clinical information, real-name/contact fields and private path maps |
| Four Stage 1 OOF sets and six corrected Stage 2 OOF sets | Manuscript and final Figure editing |
| CPU tests, input/data verification and CI | New training, inference, GPU execution or metric/CI recalculation |

See [license scope](licensing/LICENSE_SCOPE.md), [data candidate](../data/README.md) and [current inclusion decisions](data/RELEASE_FILE_DECISIONS.csv). Source-linked records are not claimed anonymous. The earlier composite exclusion and blanket patient-row withholding are superseded only for the expressly authorized, field-reviewed candidate.

## Technical Gates

Complete package SHA coverage, source/claim indexes, field/hash-restricted release scanning, exact bundle extraction, frozen patient/fold/target/run alignment and CPU regressions are checked. Local evidence is in [TEST_REPORT](audits/TEST_REPORT.md). Check the exact commit's [Actions run](https://github.com/heedbs/YHY/actions) rather than inferring hosted success from local tests.

The earlier reachable-history scan remains a scoped baseline; approved linked data is intentionally added afterward and verified by its release policy, not mislabeled as a secret-free anonymous dataset. History scanning does not cover commit author metadata or unreachable/deleted objects.

## Author's Remaining Actions

Review the fixed candidate commit and its actual Actions result, then separately authorize public visibility when ready. A fixed Release and any Zenodo registration are separate later actions. About description/topics are optional and cannot be edited through the current connector.

No further Figure/Table or manuscript work is required to organize this candidate. It is not certified as complete historical experiment reproduction: public-copy CUDA execution, first-generation cache lineage and unresolved environment details remain honestly documented in [LIMITATIONS](../LIMITATIONS.md). Official images, matching weights and validated cache are still prerequisites for new model runs.

Navigation: [documentation](README.md), [data access](../DATA_ACCESS.md), [execution guide](../REPRODUCIBILITY.md), [evidence guide](MANUSCRIPT_EVIDENCE_GUIDE.md).
