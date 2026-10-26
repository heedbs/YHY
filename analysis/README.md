# Analysis Source

[Frozen results](../results/README.md) · [Provenance](../docs/provenance/RESULTS_PROVENANCE.md)

Saved-prediction evaluation and statistical source. Reading aggregates needs no reanalysis. Reviewed frozen probabilities are now in [data/oof](../data/oof/), separated into Stage 1 and corrected Stage 2. Reference orchestration may still require original private audit metadata and is not certified turnkey.

Corrected Stage 2 contrasts Mamba minus NoMamba using class-stratified paired-patient bootstrap, 10,000 repetitions, seed 20260830. No exclusions, new thresholds or ad-hoc inferential rules are authorized here. Three training repeats are not independent patient groups.

Reference scripts can require private inputs. CPU tests use artificial fixtures and do not reproduce the real bootstrap or manuscript estimates. See [execution prerequisites](../REPRODUCIBILITY.md).
