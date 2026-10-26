# Repository Handoff

Updated 26 October 2026. **Technical organization completed for the code-and-aggregate-data candidate.** This is not full experiment reproduction or authorization to publish every server artifact.

Repository `heedbs/YHY` remains **private**, with `main` as the maintained branch. No Release or archive DOI has been issued. The author will decide when to make it public; maintenance does not change visibility.

## Completed Scope

| Area | Evidence |
| --- | --- |
| Documentation roles | Overview in README; commands in REPRODUCIBILITY; availability in DATA_ACCESS; unresolved scientific limits in LIMITATIONS |
| Final study source families | Stage 1 and corrected Stage 2 separated; 50 config/run identities indexed |
| Aggregate evidence | Topic-to-file SHA index and explicit cohort corrections; no model estimates changed |
| CPU validation | 45 local tests; independent hosted baseline [7cf75a8 succeeded](https://github.com/heedbs/YHY/actions/runs/40291231421) |
| Ongoing validation | Push/PR workflow; check the exact final commit on the [Actions page](https://github.com/heedbs/YHY/actions) |
| File hygiene | Complete package SHA, source-index checks and heuristic secret/path/identifier/large-file scan |
| Historical files | Reachable history reviewed at the baseline; scope and exclusions in [final technical check](audits/REPOSITORY_FINAL_CHECK.json) |
| Current data integrity | Source/cache read-only census recorded; exact input bundle prepared for server-only review |

## Release Boundary

The candidate contains software, config examples, synthetic fixtures and aggregate source data. It excludes source images, patch cache, weights, exact patient-linked manifests, individual OOF predictions, private reports and the manuscript. Exclusion from this package is not a blanket statement that those materials may never be shared. No data access route is promised until actually established.

The following decisions cannot be silently resolved by a technical check:

1. **License scope:** MIT is selected for authorized original contributions; the retained composite/source exception is still recorded in [license scope](licensing/LICENSE_SCOPE.md). Resolve that exact scope before representing the entire repository as MIT-cleared.
2. **Linked-data distribution:** the review-only input ZIP stays outside GitHub. Choose an approved field/distribution policy before adding it or OOF rows. The current code-and-aggregate scope is usable for inspection but not self-contained full reproduction.
3. **Public release:** after these decisions, review the final commit's Actions result and separately authorize public visibility, a fixed release and any archive registration.

## Not Part of This Cleanup

No manuscript revision, final Figure/Table editing, new model training, inference, bootstrap, benchmark or GPU execution is performed. Public-copy CUDA validation remains unrun and must not be claimed; it is separate from repository organization. Historical cache/environment uncertainty remains in [LIMITATIONS](../LIMITATIONS.md), not an endlessly repeated cleanup task.

About description/topics are optional presentation settings, not a numerical-integrity blocker. They are unset and cannot be edited through the current connector. Suggested description: “Patient-wise oral histopathology benchmarking and corrected matched Mamba ablation: code and aggregate evidence.”

Documentation index: [docs/README.md](README.md). For file inclusion policy use [RELEASE_FILE_DECISIONS.csv](data/RELEASE_FILE_DECISIONS.csv); for actual availability use [DATA_ACCESS.md](../DATA_ACCESS.md).
