# BioData Mining submission preparation

Checked 22 October 2026 against https://link.springer.com/journal/13040/submission-guidelines . This is an author workflow, not journal approval. Research is the working article-type assumption; confirm the chosen article type before submission. Requirements specific to a software article must not be applied automatically to a research article.

## Repository scope

English README.md is primary; readme_kor.md is optional supplementary guidance. The repository supports a research manuscript, not a clinical product. Maintain stage-specific training settings and corrected Stage 2 provenance. Do not change frozen results to improve reproducibility claims. Review-only documentation and tests do not establish complete end-to-end reproducibility.

## Author workflow

1. Confirm the actual code rightsholders and institutional/coauthor restrictions. Record consent privately. MIT is a proposed option for original code, NOT a selected license. Preserve third-party licenses irrespective of that decision. Do not label the repository open-source before selecting an applicable license.
2. Review minimal exact fold/partition, duplicate-exclusion, patch hierarchy and OOF fields for distribution rights and linkability. User willingness to disclose is not proof that all fields are safe. Keep these files out until a field-level review is complete. Do not invent a legal reason to withhold data or promise access that cannot be provided.
3. Resolve or explicitly document the historical cache-generation/reference gap and incomplete environment pins. Distinguish tested CPU synthetic checks, unexecuted examples, and reference-only benchmark/explainability code. No new model execution is needed merely to edit these documents.
4. Compare tables/figures with their frozen aggregate sources and record the specific final code commit. Retain scientific limitations and AI-assistance disclosures. Add final panel identities only when supplied.
5. After author approval, merge the review branch to main; verify tests, SHA and public-file scans again. Make the approved repository public before submission as planned. Check access without authentication. No merge or visibility change is performed by this checklist.
6. Identify the version supplied with the submission. An archival deposit and persistent identifier are useful for stable preservation; no release, archive or DOI exists merely because this document mentions one. Create/cite these only after a separate authorized action. The journal's general software-availability guidance includes archived-version information.
7. Keep the submitted commit unchanged as an identifiable reference. Make revisions in new commits and describe changes; never replace frozen research observations silently. Git history is not a substitute for separate backups of private research artifacts.

## Submission readiness

| Item | Current status | Action |
| --- | --- | --- |
| Code and aggregate review candidate | Uploaded on review branch | Review before merging |
| Default branch contents | Review candidate not yet merged | Merge only after approval |
| Code license and rightsholder | Unresolved | Author decision; third-party notices retained |
| Exact split/manifest/OOF disclosure | Pending field-level review | Decide minimal safe, authorized distribution |
| Cache reconstruction and historical environment | Partly unresolved | Resolve evidence or disclose limitations |
| Version/archive identifier | No release/DOI issued | Identify final submission version; do not invent DOI |
| Data/code availability paragraph | Draft below, not ready to paste unchanged | Update after actual release decisions |
| Public access | Private at preparation | Verify after separately authorized public transition |

## Availability statement: current review state

The original ORCHID data are available from Zenodo at https://doi.org/10.5281/zenodo.12636426 and https://doi.org/10.5281/zenodo.12646943. A private review candidate containing path-adapted code, configuration examples, aggregate results and figure/table source data has been prepared at https://github.com/heedbs/YHY. Exact study-specific split/patch manifests and individual predictions are not included in the current candidate, pending disclosure review. Complete reconstruction of the historical preprocessing cache and some environment details remains unresolved. A final public code version, license and data-access conditions have not yet been assigned.

This paragraph describes the current state, not the intended submission state. Replace it with an accurate final statement after publication/access decisions; do not simply replace 'private' with 'public' while leaving other claims unchecked. Cite both original datasets formally. A public repository does not itself establish ethics approval or authorization to redistribute linked patient records.

## Software information to finalize

- Project: ORCHID comparative study reproducibility package.
- Home page: https://github.com/heedbs/YHY .
- Archived version: NOT_ASSIGNED; final commit/release and any archive identifier must be recorded after creation.
- OS: historical Linux/CUDA workflow; other operating systems not validated.
- Language: Python; shell components are reference orchestration.
- Requirements: see environment/ and REPRODUCIBILITY.md; missing historical dependencies remain stated.
- License: NOT_SELECTED. Dataset CC BY 4.0 is separate from source-code licensing.
- Non-academic restrictions: cannot be finalized until applicable code and third-party licenses are reviewed.

The general journal guidance requires an Availability of data and materials statement, recommends machine-readable supporting data, and requests dataset identifiers and software access/requirements information. It does not prescribe a special GitHub folder layout or a Korean README. The layout and workflow here are project recommendations, not editorial requirements.
