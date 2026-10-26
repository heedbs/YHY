# Current Review Status

Updated: 26 October 2026. Earlier dated records are retained under [history](history/README.md).

| Item | Current state |
| --- | --- |
| Repository | `heedbs/YHY`, private; reviewed files on default branch `main` |
| Public release / archive DOI | Not issued; no visibility change in this maintenance |
| Manuscript | Intended for BioData Mining; acceptance is not asserted |
| Software license | MIT selected for authorized original contributions; see [scope](licensing/LICENSE_SCOPE.md) |
| CPU checks | 28 synthetic tests passed in the isolated CPU environment |
| Initialization audit | Six CPU factory cases; zero forward passes; not retrospective CUDA/checkpoint validation |
| Research inputs | Images, cache, weights, exact patient-linked manifests and predictions not bundled |
| Stage 2 source | Corrected matched comparison, never historical Stage 1 Mamba seed-42 substitution |
| Full reproduction | Not certified; cache lineage and historical-environment gaps remain |
| Final figure mapping | Not fully supplied or verified |

## Before Public Release

1. Confirm declared code provenance and any applicable institutional/third-party obligations. AI assistance is not evidence of copying from another repository.
2. Decide whether to distribute minimal source-linked manifests or predictions after field/linkage review. Their current omission does not mean licensed ORCHID derivatives are categorically prohibited.
3. Keep cache/environment limitations and development-time outer-access history aligned with the manuscript.
4. Provide reviewers an actual access route. A private URL alone does not grant access. Cite a fixed commit; archival release requires separate authorization.

Technical checks do not certify ethics/IRB status, pathology interpretation, clinical safety or publication eligibility. This refresh performs no training, inference or real-data statistical calculation.
