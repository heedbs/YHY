# Results provenance

Source families were selected from frozen final analysis manifests and corrected Stage2 source provenance, not timestamps. results/ CSVs preserve stored numerical strings. SOURCE_FILE_FINGERPRINTS.csv connects every exported file to an original artifact label and SHA without server paths. configs/run_catalog.csv links all 50 runs to original config/code/split/pretrained/best-checkpoint digests. Private absolute-path mapping and exact code diffs remain outside the public ZIP.

Stage1: 20 completed runs, seed42, AUROC-first lexicographic tuple (AUROC, macro F1, negative validation patch loss, earlier epoch), max80; pooled OOF149 patients/model. Fold means are separate descriptive outputs.

Corrected Stage2: 30 matched runs (15 per condition), seeds42/20260824/314159, F1-first tuple (macro F1, AUROC, negative validation patch loss, earlier epoch), max40. All five Mamba seed42 folds come from correction outputs bound by the corrected analysis queue/provenance. No historical Stage1 Mamba seed42 is used here. Primary Mamba-minus-NoMamba macro F1: +0.0396 CI[-0.0270,+0.1187], +0.0135 CI[-0.0123,+0.0406], -0.0087 CI[-0.0468,+0.0215]. These are display roundings of frozen full-precision values. Bootstrap uses 10,000 paired, class-stratified patient samples, seed20260830. Three-seed descriptive SD uses ddof=1; no independent447-patient sample or pooled CI is created.

Efficiency: five models, RTX4500 Ada, FP32, TF32 disabled,256×256,30 warm-up/100 timed iterations×3 blocks. model_only excludes preprocessing, H2D, file I/O, initialization. The distinct preprocessing-plus-forward mode remains separate. Benchmark Composite-Mamba checkpoint precedes the corrected seed42 training; it must not be described as measured from the correction checkpoint. Digests are retained in efficiency summaries; performance and timing are different estimands. Mamba FLOPs/MACs remain NOT_RELIABLY_ESTIMATED.

Final manuscript files edited outside the server, pathology panel choices and FigureLabs mappings remain unconfirmed. Current aggregate CSVs are not rewritten to match an unseen manuscript.

Plotting-source caveat: the original figure manifest excludes `build_figures.py`; the current script SHA is recorded in SOURCE_FILE_FINGERPRINTS.csv, but a historical execution SHA is unavailable for that generator. This is preserved as an unresolved source-provenance limitation.
