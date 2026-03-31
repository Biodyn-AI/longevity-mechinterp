#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from textwrap import fill

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap


def _save_fig(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


DATASET_LABELS = {
    "aida_phase1_v1": "AIDA phase 1 v1",
    "aida_phase1_v2": "AIDA phase 1 v2",
    "allen_imm_of_aging_b_plasma_cells": "Allen aging plasma cells",
    "allen_immune_health_atlas_full": "Allen immune atlas",
    "yazar_eqtl_981_donors": "Yazar donor cohort",
}

PATHWAY_LABELS = {
    "inflammation_nfkb": "Inflammation / NF-kappaB",
    "interferon_antiviral": "Interferon / antiviral response",
    "proteostasis_upr": "Proteostasis / unfolded protein response",
    "senescence_sasp": "Senescence / SASP",
}

RUN_LABELS = {
    "stage9_seedexpansion": "Baseline\nexpanded splits\nn=424",
    "stage10_compmatched_seed42": "Matched rerun\nseed 42\nn=340",
    "stage11_compmatched_seed101": "Matched rerun\nseed 101\nn=334",
    "stage11_compmatched_seed202": "Matched rerun\nseed 202\nn=333",
    "stage11_compmatched_seed303": "Matched rerun\nseed 303\nn=337",
}

RUN_TABLE_LABELS = {
    "stage9_seedexpansion": "Baseline expanded-split run",
    "stage10_compmatched_seed42": "Composition-matched rerun, seed 42",
    "stage11_compmatched_seed101": "Composition-matched rerun, seed 101",
    "stage11_compmatched_seed202": "Composition-matched rerun, seed 202",
    "stage11_compmatched_seed303": "Composition-matched rerun, seed 303",
}

MODEL_LABELS = {
    "scgpt": "scGPT",
    "geneformer": "Geneformer",
    "scGPT": "scGPT",
    "Geneformer": "Geneformer",
}

INTERVENTION_LABELS = {
    "old_push": "Aging program\npush",
    "random_push": "Random feature\ncontrol",
    "young_push": "Young program\npush",
}

PASS_FAIL_LABELS = {False: "Fail", True: "Pass"}


def _friendly_dataset_label(dataset_id: str, width: int = 16) -> str:
    return fill(DATASET_LABELS.get(dataset_id, dataset_id.replace("_", " ")), width=width)


def _friendly_pathway_label(pathway: str) -> str:
    return PATHWAY_LABELS.get(pathway, pathway.replace("_", " "))


def _friendly_model_label(model: str) -> str:
    return MODEL_LABELS.get(model, model)


def _friendly_run_label(run_label: str) -> str:
    return RUN_LABELS.get(run_label, run_label.replace("_", " "))


def _parse_args() -> argparse.Namespace:
    default_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Build publication-ready figures and tables for the longevity mechinterp paper.")
    parser.add_argument("--project-root", type=Path, default=default_root, help="Repository root containing implementation/ and paper/.")
    parser.add_argument("--paper-dir", type=Path, default=None, help="Output directory for paper assets. Defaults to <project-root>/paper.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    root = args.project_root
    outputs = root / "implementation" / "outputs"
    paper_dir = args.paper_dir or (root / "paper")
    fig_dir = paper_dir / "figures"
    table_dir = paper_dir / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
        }
    )

    # Load core artifacts.
    stage1 = pd.read_csv(outputs / "stage1_full_contextual_20260303" / "stage1_run_aggregate.csv")
    stage2 = pd.read_csv(outputs / "stage2_manifold_robustness_20260303" / "stage2_run_aggregate.csv")
    stage3_sc = pd.read_csv(outputs / "stage3_sae_pilot_scgpt_20260303_fast" / "stage3_run_aggregate.csv")
    stage3_gf = pd.read_csv(outputs / "stage3_sae_pilot_geneformer_20260303_fast" / "stage3_run_aggregate.csv")
    stage4_cons = pd.read_csv(outputs / "stage4_cross_model_convergence_20260303" / "stage4_consensus_pathway_summary.csv")
    stage6_gates = pd.read_csv(outputs / "stage6_inflammation_followup_20260303" / "model_specific_claims_strict_gates" / "stage6_model_specific_claim_gates.csv")
    stage8_gate_all = pd.read_csv(outputs / "stage8_contrast_ci_gate_audit_20260303" / "stage8_contrast_gate_summary_by_run.csv")
    stage9_reweight_sens = pd.read_csv(outputs / "stage9_composition_reweighting_sensitivity_20260304.csv")
    stage11_seedpanel = pd.read_csv(outputs / "stage11_compmatched_seedpanel_summary_20260304" / "stage11_compmatched_seedpanel_summary.csv")
    stage11_metrics = pd.read_csv(outputs / "stage11_compmatched_seedpanel_summary_20260304" / "stage11_compmatched_seedpanel_metrics.csv")

    stage8_v1_contrast = pd.read_csv(outputs / "stage8_monocyte_aida_v1_scale_summary_20260303" / "stage8_scale_focus_contrast_ci.csv")
    stage8_v2_contrast = pd.read_csv(outputs / "stage8_monocyte_aida_v2_scale_summary_20260303" / "stage8_scale_focus_contrast_ci.csv")

    stage9_boot = pd.read_csv(outputs / "stage9_aida_v2_global_scale2_seedexpansion_20260304" / "donor_bootstrap_ci" / "stage5_donor_bootstrap_consensus.csv")
    stage10_boot = pd.read_csv(outputs / "stage10_aida_v2_global_scale2_compmatched_forwardpass_20260304" / "donor_bootstrap_ci" / "stage5_donor_bootstrap_consensus.csv")

    # Stage-9 donor-threshold sweep.
    thresholds = [20, 30, 40, 50, 100, 200, 300, 400, 500]
    rows = []
    for t in thresholds:
        p = outputs / f"stage9_contrast_gate_aida_v2_seedexpansion_min{t}_20260304" / "stage8_contrast_gate_summary_by_run.csv"
        df = pd.read_csv(p)
        row = df.iloc[0].to_dict()
        row["min_donors"] = t
        rows.append(row)
    stage9_sweep = pd.DataFrame(rows)

    # Harmonize Stage-3.
    stage3_sc = stage3_sc.assign(model="scGPT")
    stage3_gf = stage3_gf.assign(model="Geneformer")
    stage3 = pd.concat([stage3_sc, stage3_gf], ignore_index=True)

    # Merge Stage1/Stage2 for combined geometry view.
    s12 = stage1.merge(stage2[["dataset_id", "topk_max_pc1_abs_corr", "topk_max_silhouette", "g2_geometry_pass"]], on="dataset_id", how="left")
    s12 = s12.sort_values("best_balanced_accuracy_mean", ascending=False)
    s12["dataset_label"] = s12["dataset_id"].map(lambda x: _friendly_dataset_label(x, width=14))

    # Figure 1: Frozen probe and geometry summary.
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    x = np.arange(len(s12))
    axes[0].bar(x, s12["best_balanced_accuracy_mean"], color="#4c78a8")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(s12["dataset_label"])
    axes[0].set_ylabel("Balanced accuracy")
    axes[0].set_title("Frozen embeddings carry detectable age information")
    axes[0].set_ylim(0, max(0.45, float(s12["best_balanced_accuracy_mean"].max()) + 0.05))
    for idx, value in enumerate(s12["best_balanced_accuracy_mean"]):
        axes[0].text(idx, float(value) + 0.01, f"{value:.3f}", ha="center", va="bottom", fontsize=8)

    axes[1].bar(x - 0.18, s12["topk_max_pc1_abs_corr"], width=0.36, color="#f58518", label="|corr(age, PC1)|")
    axes[1].bar(x + 0.18, s12["topk_max_silhouette"], width=0.36, color="#54a24b", label="Silhouette(age)")
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(s12["dataset_label"])
    axes[1].set_title("Global age geometry is weak and unstable")
    axes[1].legend(loc="upper right", fontsize=8)
    _save_fig(fig, fig_dir / "fig1_stage1_stage2_overview.png")

    # Figure 2: robust SAE feature counts.
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    stage3_plot = stage3.copy()
    stage3_plot["dataset_model"] = stage3_plot["dataset_id"].map(lambda x: _friendly_dataset_label(x, width=14)) + "\n" + stage3_plot["model"].map(_friendly_model_label)
    order = stage3_plot.sort_values("n_robust_features", ascending=False)
    ax.bar(order["dataset_model"], order["n_robust_features"], color=["#e45756" if m == "Geneformer" else "#72b7b2" for m in order["model"]])
    ax.set_ylabel("Number of robust SAE features")
    ax.set_title("Robust sparse age features are concentrated in the AIDA cohorts")
    ax.tick_params(axis="x", rotation=0)
    for idx, value in enumerate(order["n_robust_features"]):
        ax.text(idx, float(value) + 1.5, str(int(value)), ha="center", va="bottom", fontsize=8)
    _save_fig(fig, fig_dir / "fig2_stage3_robust_features.png")

    # Figure 3: cross-model pathway pair counts.
    fig, ax = plt.subplots(figsize=(9.4, 4.8))
    p4 = stage4_cons.copy()
    p4["dataset_pathway"] = p4.apply(lambda r: f"{DATASET_LABELS.get(r['dataset_id'], r['dataset_id'])}: {_friendly_pathway_label(r['pathway'])}", axis=1)
    p4 = p4.sort_values("n_cross_model_pairs", ascending=True)
    ax.barh(p4["dataset_pathway"], p4["n_cross_model_pairs"], color="#4c78a8")
    ax.set_xlabel("Number of cross-model feature pairs")
    ax.set_title("Inflammation dominates cross-model pathway overlap")
    for idx, value in enumerate(p4["n_cross_model_pairs"]):
        ax.text(float(value) + 1.5, idx, str(int(value)), va="center", fontsize=8)
    _save_fig(fig, fig_dir / "fig3_stage4_pathway_convergence.png")

    # Figure 4: stage8 monocyte scale focus (old-minus-random expected-age contrast).
    focus = pd.concat([stage8_v1_contrast, stage8_v2_contrast], ignore_index=True)
    focus = focus[(focus["metric"] == "delta_expected_age_mean") & (focus["contrast"] == "old_minus_random")].copy()
    focus["dataset_short"] = focus["dataset_id"].map({"aida_phase1_v1": "AIDA v1", "aida_phase1_v2": "AIDA v2"})
    focus["label"] = focus["dataset_short"] + ", " + focus["model"].map(_friendly_model_label)
    focus = focus.sort_values(["dataset_short", "model", "intervention_scale"])

    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    label_order = ["AIDA v1, Geneformer", "AIDA v1, scGPT", "AIDA v2, Geneformer", "AIDA v2, scGPT"]
    y_base = np.arange(len(label_order))
    y_map = {k: v for k, v in zip(label_order, y_base)}
    offsets = {1.0: -0.12, 2.0: 0.12}
    colors = {1.0: "#f58518", 2.0: "#4c78a8"}

    for _, r in focus.iterrows():
        y = y_map[r["label"]] + offsets[float(r["intervention_scale"])]
        mean = float(r["mean_delta"])
        lo = float(r["ci_low"])
        hi = float(r["ci_high"])
        ax.errorbar(mean, y, xerr=[[mean - lo], [hi - mean]], fmt="o", color=colors[float(r["intervention_scale"])] , capsize=3)

    ax.axvline(0.0, color="black", linewidth=0.8)
    ax.set_yticks(y_base)
    ax.set_yticklabels(label_order)
    ax.set_xlabel("Change in expected age after aging-program push vs random control")
    ax.set_title("Monocyte inflammation signal is strongest in AIDA v1")
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", label="Intervention scale 1.0", markerfacecolor=colors[1.0], markersize=8),
        Line2D([0], [0], marker="o", color="w", label="Intervention scale 2.0", markerfacecolor=colors[2.0], markersize=8),
    ]
    ax.legend(handles=legend_elements, loc="lower right")
    _save_fig(fig, fig_dir / "fig4_stage8_monocyte_scale_contrast.png")

    # Figure 5: donor-threshold strict pass sweep.
    fig, ax = plt.subplots(figsize=(7.8, 4.2))
    ax.plot(stage9_sweep["min_donors"], stage9_sweep["n_full_strict"], marker="o", color="#4c78a8", label="Full strict gate")
    ax.plot(stage9_sweep["min_donors"], stage9_sweep["n_expected_age_strict"], marker="s", color="#e45756", label="Expected-age gate only")
    ax.set_xlabel("Minimum donors required per comparison")
    ax.set_ylabel("Number of passing rows")
    ax.set_title("The strongest Geneformer branch survives donor-threshold tightening")
    ax.set_ylim(-0.05, 1.1)
    ax.legend(loc="upper right")
    _save_fig(fig, fig_dir / "fig5_stage9_donor_threshold_sweep.png")

    # Figure 6: reweighting sensitivity + compmatched seed panel outcome.
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    rs = stage9_reweight_sens.sort_values("ridge_alpha")
    axes[0].plot(rs["ridge_alpha"], rs["ea_old_minus_random_shift"], marker="o", label="Aging-program vs random")
    axes[0].plot(rs["ridge_alpha"], rs["ea_young_minus_random_shift"], marker="s", label="Young-program vs random")
    axes[0].set_xscale("log")
    axes[0].axhline(0.0, color="black", linewidth=0.8)
    axes[0].set_xlabel("Ridge penalty used for reweighting")
    axes[0].set_ylabel("Shift after reweighting")
    axes[0].set_title("Composition reweighting barely changes the effect estimates")
    axes[0].legend(fontsize=8)

    panel = stage11_seedpanel.copy()
    run_order = ["stage9_seedexpansion", "stage10_compmatched_seed42", "stage11_compmatched_seed101", "stage11_compmatched_seed202", "stage11_compmatched_seed303"]
    panel = panel[panel["run_label"].isin(run_order)].copy()
    panel["run_label"] = pd.Categorical(panel["run_label"], categories=run_order, ordered=True)
    panel = panel.sort_values("run_label")
    heatmap = np.vstack(
        [
            panel["passes_full_strict"].astype(int).to_numpy(),
            panel["geneformer_directional_pattern_ok"].astype(int).to_numpy(),
        ]
    )
    axes[1].imshow(heatmap, aspect="auto", cmap=ListedColormap(["#e45756", "#54a24b"]), vmin=0, vmax=1)
    axes[1].set_xticks(np.arange(len(panel)))
    axes[1].set_xticklabels([_friendly_run_label(label) for label in panel["run_label"].astype(str)], rotation=0)
    axes[1].set_yticks([0, 1])
    axes[1].set_yticklabels(["Full strict\nclaim", "Geneformer\ndirectional pattern"])
    axes[1].set_title("The baseline signal does not survive composition-matched reruns")
    for row_idx in range(heatmap.shape[0]):
        for col_idx in range(heatmap.shape[1]):
            passed = bool(heatmap[row_idx, col_idx])
            axes[1].text(col_idx, row_idx, PASS_FAIL_LABELS[passed], ha="center", va="center", color="white", fontsize=8, fontweight="bold")
    _save_fig(fig, fig_dir / "fig6_composition_controls_and_seedpanel.png")

    # Figure 7: donor-bootstrap expected-age intervention CIs (geneformer, stage9 vs stage10).
    def prep_boot(df: pd.DataFrame, run_label: str) -> pd.DataFrame:
        d = df[(df["dataset_id"] == "aida_phase1_v2") & (df["model"] == "geneformer") & (df["pathway"] == "inflammation_nfkb")].copy()
        d["run_label"] = run_label
        return d

    b = pd.concat([prep_boot(stage9_boot, "stage9_seedexpansion"), prep_boot(stage10_boot, "stage10_compmatched")], ignore_index=True)
    keep = ["old_push", "random_push", "young_push"]
    b = b[b["intervention"].isin(keep)].copy()
    order = {"old_push": 0, "random_push": 1, "young_push": 2}
    b["x"] = b["intervention"].map(order).astype(float)
    b["x"] += b["run_label"].map({"stage9_seedexpansion": -0.12, "stage10_compmatched": 0.12})

    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = {"stage9_seedexpansion": "#4c78a8", "stage10_compmatched": "#f58518"}
    for _, r in b.iterrows():
        mean = float(r["delta_expected_age_mean_mean"])
        lo = float(r["delta_expected_age_mean_ci_low"])
        hi = float(r["delta_expected_age_mean_ci_high"])
        ax.errorbar(r["x"], mean, yerr=[[mean - lo], [hi - mean]], fmt="o", color=colors[r["run_label"]], capsize=3)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels([INTERVENTION_LABELS["old_push"], INTERVENTION_LABELS["random_push"], INTERVENTION_LABELS["young_push"]])
    ax.set_ylabel("Expected-age change (mean ± 95% CI)")
    ax.set_title("The strongest Geneformer effect weakens after full composition matching")
    legend = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=colors["stage9_seedexpansion"], label="Baseline expanded-split run", markersize=8),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=colors["stage10_compmatched"], label="Composition-matched rerun", markersize=8),
    ]
    ax.legend(handles=legend, loc="upper right", fontsize=8)
    _save_fig(fig, fig_dir / "fig7_geneformer_bootstrap_stage9_vs_stage10.png")

    # Build tables for manuscript.
    table_stage_overview = pd.DataFrame(
        [
            {"stage": "Stage 1", "objective": "Frozen age probe", "key_result": f"Best balanced accuracy range: {stage1['best_balanced_accuracy_mean'].min():.3f}-{stage1['best_balanced_accuracy_mean'].max():.3f}"},
            {"stage": "Stage 2", "objective": "Manifold geometry", "key_result": f"Geometry pass datasets: {int(stage2['g2_geometry_pass'].sum())}/{stage2.shape[0]}"},
            {"stage": "Stage 3", "objective": "SAE feature discovery", "key_result": f"Robust features discovered (scGPT+Geneformer): {int(stage3['n_robust_features'].sum())}"},
            {"stage": "Stage 4", "objective": "Cross-model convergence", "key_result": f"Cross-model pathway-matched pairs: {int(stage4_cons['n_cross_model_pairs'].sum())}"},
            {"stage": "Stage 6 strict", "objective": "Model-specific claim promotion", "key_result": f"Strict promotion passes: {int(stage6_gates['gate_pass_all_required'].sum())}/{stage6_gates.shape[0]}"},
            {"stage": "Stage 8 strict audit", "objective": "Contrast+direction gate", "key_result": f"Runs with >=1 full strict row: {int((stage8_gate_all['n_full_strict']>0).sum())}/{stage8_gate_all.shape[0]}"},
            {"stage": "Stage 9", "objective": "Deep-dive on surviving branch", "key_result": f"Full strict at min_donors=200: {int(stage9_sweep.loc[stage9_sweep['min_donors']==200,'n_full_strict'].iloc[0])}"},
            {"stage": "Stage 10", "objective": "Composition-matched forward pass", "key_result": "Full strict no longer passed (n_full_strict=0)"},
            {"stage": "Stage 11", "objective": "Compmatched seed panel", "key_result": f"Compmatched full strict pass rate: {float(stage11_metrics['share_compmatched_full_strict_passes'].iloc[0]):.2f}"},
        ]
    )
    table_stage_overview.to_csv(table_dir / "table_stage_overview.csv", index=False)

    table_best_positive = stage11_seedpanel[[
        "run_label",
        "n_full_strict",
        "n_expected_age_contrast_strict",
        "n_full_contrast_strict",
        "geneformer_old_push_expected_age_ci_flag",
        "geneformer_random_push_expected_age_ci_flag",
        "geneformer_young_push_expected_age_ci_flag",
    ]].copy()
    table_best_positive["run_label"] = table_best_positive["run_label"].map(RUN_TABLE_LABELS)
    table_best_positive.to_csv(table_dir / "table_compmatched_seedpanel.csv", index=False)

    # Summary metrics for manuscript text insertion.
    metrics = {
        "n_datasets_stage1": int(stage1.shape[0]),
        "stage1_best_bacc_min": float(stage1["best_balanced_accuracy_mean"].min()),
        "stage1_best_bacc_max": float(stage1["best_balanced_accuracy_mean"].max()),
        "stage2_geometry_pass_count": int(stage2["g2_geometry_pass"].sum()),
        "stage2_geometry_total": int(stage2.shape[0]),
        "stage3_total_robust_features": int(stage3["n_robust_features"].sum()),
        "stage3_scgpt_robust_features_total": int(stage3_sc["n_robust_features"].sum()),
        "stage3_geneformer_robust_features_total": int(stage3_gf["n_robust_features"].sum()),
        "stage4_cross_model_pairs_total": int(stage4_cons["n_cross_model_pairs"].sum()),
        "stage6_strict_claim_passes": int(stage6_gates["gate_pass_all_required"].sum()),
        "stage6_strict_claim_total": int(stage6_gates.shape[0]),
        "stage8_runs_with_full_strict": int((stage8_gate_all["n_full_strict"] > 0).sum()),
        "stage8_total_runs": int(stage8_gate_all.shape[0]),
        "stage9_full_strict_min200": int(stage9_sweep.loc[stage9_sweep["min_donors"] == 200, "n_full_strict"].iloc[0]),
        "stage9_full_strict_min400": int(stage9_sweep.loc[stage9_sweep["min_donors"] == 400, "n_full_strict"].iloc[0]),
        "stage9_full_strict_min500": int(stage9_sweep.loc[stage9_sweep["min_donors"] == 500, "n_full_strict"].iloc[0]),
        "stage11_compmatched_runs": int(stage11_metrics["n_compmatched_runs"].iloc[0]),
        "stage11_compmatched_full_strict_passes": int(stage11_metrics["n_compmatched_full_strict_passes"].iloc[0]),
        "stage11_compmatched_full_strict_share": float(stage11_metrics["share_compmatched_full_strict_passes"].iloc[0]),
        "stage11_compmatched_geneformer_directional_ok_share": float(stage11_metrics["share_compmatched_geneformer_directional_pattern_ok"].iloc[0]),
    }
    with open(table_dir / "summary_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # Write a markdown quick-look for manual drafting.
    quick = [
        "# Paper Asset Build Summary",
        "",
        "## Metrics",
        json.dumps(metrics, indent=2),
        "",
        "## Generated Figures",
    ]
    for p in sorted(fig_dir.glob("*.png")):
        quick.append(f"- {p.name}")
    quick += ["", "## Generated Tables"]
    for p in sorted(table_dir.glob("*.csv")):
        quick.append(f"- {p.name}")

    (paper_dir / "asset_build_summary.md").write_text("\n".join(quick), encoding="utf-8")
    print("[done] paper assets generated")
    print(f"[figures] {fig_dir}")
    print(f"[tables] {table_dir}")


if __name__ == "__main__":
    main()
