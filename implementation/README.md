# Implementation Notes

This directory contains the stage scripts and the small derived outputs needed to understand and reproduce the manuscript figures.

## Scripts
- `rank_age_labeled_datasets.py`: score candidate age-labeled datasets.
- `run_stage1_longevity_mechinterp.py`: donor-aware frozen age probes.
- `run_stage2_manifold_robustness.py`: manifold diagnostics and geometry checks.
- `run_stage3_sae_pilot.py`: sparse autoencoder feature discovery.
- `run_stage4_cross_model_convergence.py`: cross-model pathway matching.
- `run_stage5_intervention_validation.py`: representation-space interventions and donor-held-out readouts.
- `run_stage34_multiseed_hardening.py`: seed hardening for the Stage 3/4 pipeline.
- `summarize_stage5_*`, `summarize_stage6_*`, `summarize_stage7_*`, `summarize_stage8_*`, `summarize_stage9_*`: stage-specific aggregation and audit helpers.
- `build_composition_matched_sampled_obs.py`: sampled composition-matching helper used in the stricter control stages.
- `download_age_datasets.py`: manifest-driven downloader for external age-diverse cohorts.

## Included outputs
Only the summary artifacts needed for the paper are included here. The full private workspace contains many additional intermediate files and heavier outputs that are intentionally omitted from this public repository.

See `implementation/outputs/README.md` for the exact contents of the included output subset.
