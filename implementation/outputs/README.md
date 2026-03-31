# Included Output Subset

This repository includes a minimal subset of derived outputs that are sufficient to reproduce the paper figures and tables.

## Included artifacts
- `stage1_full_contextual_20260303/stage1_run_aggregate.csv`
- `stage2_manifold_robustness_20260303/stage2_run_aggregate.csv`
- `stage3_sae_pilot_scgpt_20260303_fast/stage3_run_aggregate.csv`
- `stage3_sae_pilot_geneformer_20260303_fast/stage3_run_aggregate.csv`
- `stage4_cross_model_convergence_20260303/stage4_consensus_pathway_summary.csv`
- `stage6_inflammation_followup_20260303/model_specific_claims_strict_gates/stage6_model_specific_claim_gates.csv`
- `stage8_contrast_ci_gate_audit_20260303/stage8_contrast_gate_summary_by_run.csv`
- `stage8_monocyte_aida_v1_scale_summary_20260303/stage8_scale_focus_contrast_ci.csv`
- `stage8_monocyte_aida_v2_scale_summary_20260303/stage8_scale_focus_contrast_ci.csv`
- `stage9_composition_reweighting_sensitivity_20260304.csv`
- `stage9_aida_v2_global_scale2_seedexpansion_20260304/donor_bootstrap_ci/stage5_donor_bootstrap_consensus.csv`
- `stage10_aida_v2_global_scale2_compmatched_forwardpass_20260304/donor_bootstrap_ci/stage5_donor_bootstrap_consensus.csv`
- `stage11_compmatched_seedpanel_summary_20260304/stage11_compmatched_seedpanel_summary.csv`
- `stage11_compmatched_seedpanel_summary_20260304/stage11_compmatched_seedpanel_metrics.csv`
- donor-threshold sweep summaries under `stage9_contrast_gate_aida_v2_seedexpansion_min*_20260304/`

## Why this subset exists
The full project produced many larger intermediate directories, repeated reruns, and local-only artifacts. Those are not required to rebuild the paper figures, so they are intentionally excluded from the public repository.
