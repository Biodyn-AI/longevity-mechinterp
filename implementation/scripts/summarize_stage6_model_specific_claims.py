#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Build strict model-specific evidence claims for Stage-6 inflammation follow-up.")
    parser.add_argument(
        "--bootstrap-consensus-csv",
        type=Path,
        default=root / "implementation" / "outputs" / "stage5_donor_bootstrap_ci_20260303" / "stage5_donor_bootstrap_consensus.csv",
    )
    parser.add_argument(
        "--dose-trend-csv",
        type=Path,
        default=root
        / "implementation"
        / "outputs"
        / "stage6_inflammation_followup_20260303"
        / "dose_response_summary"
        / "dose_response_trend_summary.csv",
    )
    parser.add_argument(
        "--specificity-compare-csv",
        type=Path,
        default=root
        / "implementation"
        / "outputs"
        / "stage6_inflammation_followup_20260303"
        / "specificity_summary"
        / "specificity_inflammation_vs_controls.csv",
    )
    parser.add_argument(
        "--celltype-directional-csv",
        type=Path,
        default=root
        / "implementation"
        / "outputs"
        / "stage6_inflammation_followup_20260303"
        / "celltype_anchor_summary"
        / "celltype_anchor_directional.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "implementation" / "outputs" / "stage6_inflammation_followup_20260303" / "model_specific_claims",
    )
    parser.add_argument("--pathway", type=str, default="inflammation_nfkb")
    parser.add_argument("--require-young-ci-negative-for-promotion", action="store_true")
    parser.add_argument("--require-celltype-ci-for-promotion", action="store_true")
    return parser.parse_args()


def _as_bool(x: object) -> bool:
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        return bool(int(x))
    if isinstance(x, str):
        t = x.strip().lower()
        if t in {"true", "1", "yes"}:
            return True
        if t in {"false", "0", "no", ""}:
            return False
    return False


def _claim_tier(row: pd.Series) -> str:
    old_ci_pos = _as_bool(row.get("old_ci_positive", False))
    young_ci_neg = _as_bool(row.get("young_ci_negative", False))
    dose_shape = _as_bool(row.get("dose_shape_expected_age_ok", False))
    spec_all = row.get("specificity_all_expected_positive", np.nan)
    spec_ok = bool(spec_all) if pd.notna(spec_all) else False
    spec_missing = bool(int(row.get("n_specificity_controls", 0)) == 0)
    ct_ci_any = _as_bool(row.get("celltype_any_ci_directional_expected_age", False))

    if old_ci_pos and young_ci_neg and dose_shape and spec_ok and ct_ci_any:
        return "strong_model_specific"
    if old_ci_pos and dose_shape and (spec_ok or spec_missing):
        return "promising_model_specific"
    if dose_shape and spec_ok:
        return "supportive_but_ci_weak"
    return "exploratory"


def _claim_rank(tier: str) -> int:
    order = {
        "strong_model_specific": 0,
        "promising_model_specific": 1,
        "supportive_but_ci_weak": 2,
        "exploratory": 3,
    }
    return int(order.get(str(tier), 99))


def _gate_status(row: pd.Series) -> str:
    required = _as_bool(row.get("gate_required_all", False))
    passed = _as_bool(row.get("gate_pass_all_required", False))
    if required and passed:
        return "promoted"
    if required and not passed:
        return "blocked"
    return "conditional"


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cards_dir = args.output_dir / "claim_cards"
    cards_dir.mkdir(parents=True, exist_ok=True)

    boot = pd.read_csv(args.bootstrap_consensus_csv).copy()
    boot = boot[boot["pathway"].astype(str) == str(args.pathway)].copy()
    boot = boot[boot["intervention"].astype(str).isin({"old_push", "young_push"})].copy()

    boot_wide = (
        boot.pivot_table(
            index=["dataset_id", "model"],
            columns="intervention",
            values=[
                "delta_expected_age_mean_ci_flag",
                "delta_expected_age_mean_mean",
                "delta_expected_age_mean_ci_low",
                "delta_expected_age_mean_ci_high",
            ],
            aggfunc="first",
        )
        .reset_index()
    )
    boot_wide.columns = [f"{a}_{b}" if b else str(a) for a, b in boot_wide.columns]

    dose = pd.read_csv(args.dose_trend_csv).copy()
    dose = dose[dose["intervention"].astype(str).isin({"old_push", "young_push"})].copy()
    dose_wide = (
        dose.pivot_table(
            index=["dataset_id", "model"],
            columns="intervention",
            values=["linear_slope_expected_age_mean", "spearman_scale_vs_expected_age_mean"],
            aggfunc="first",
        )
        .reset_index()
    )
    dose_wide.columns = [f"{a}_{b}" if b else str(a) for a, b in dose_wide.columns]

    spec = pd.read_csv(args.specificity_compare_csv).copy()
    spec_rows: List[Dict[str, object]] = []
    for (dataset_id, model), grp in spec.groupby(["dataset_id", "model"], as_index=False):
        vals = grp["infl_minus_ctrl_expected_age"].to_numpy(dtype=np.float64)
        vals = vals[np.isfinite(vals)]
        spec_rows.append(
            {
                "dataset_id": str(dataset_id),
                "model": str(model),
                "n_specificity_controls": int(vals.size),
                "specificity_all_expected_positive": bool(vals.size > 0 and np.all(vals > 0.0)),
                "specificity_mean_expected_delta": float(np.mean(vals)) if vals.size > 0 else float("nan"),
            }
        )
    spec_df = pd.DataFrame(spec_rows)

    ctd = pd.read_csv(args.celltype_directional_csv).copy()
    ctd = ctd[ctd["pathway"].astype(str) == str(args.pathway)].copy()
    ct_rows: List[Dict[str, object]] = []
    for (dataset_id, model), grp in ctd.groupby(["dataset_id", "model"], as_index=False):
        ct_rows.append(
            {
                "dataset_id": str(dataset_id),
                "model": str(model),
                "n_celltype_strata": int(len(grp)),
                "n_celltype_directional_expected_age": int(grp["directional_pattern_expected_age"].astype(bool).sum()),
                "n_celltype_ci_directional_expected_age": int(grp["ci_directional_pattern_expected_age"].astype(bool).sum()),
                "celltype_mean_old_minus_random_expected_age": float(
                    pd.to_numeric(grp["old_minus_random_expected_age"], errors="coerce").mean()
                ),
                "celltype_any_directional_expected_age": bool(grp["directional_pattern_expected_age"].astype(bool).any()),
                "celltype_any_ci_directional_expected_age": bool(
                    grp["ci_directional_pattern_expected_age"].astype(bool).any()
                ),
            }
        )
    ct_df = pd.DataFrame(ct_rows)

    out = boot_wide.merge(dose_wide, on=["dataset_id", "model"], how="outer")
    out = out.merge(spec_df, on=["dataset_id", "model"], how="left")
    out = out.merge(ct_df, on=["dataset_id", "model"], how="left")

    out["old_ci_positive"] = out["delta_expected_age_mean_ci_flag_old_push"].astype(str).eq("positive")
    out["young_ci_negative"] = out["delta_expected_age_mean_ci_flag_young_push"].astype(str).eq("negative")
    out["dose_old_slope_positive"] = out["linear_slope_expected_age_mean_old_push"] > 0.0
    out["dose_young_slope_negative"] = out["linear_slope_expected_age_mean_young_push"] < 0.0
    out["dose_shape_expected_age_ok"] = out["dose_old_slope_positive"] & out["dose_young_slope_negative"]

    # Missing controls should be explicit so review can separate "pass" vs "not tested".
    out["n_specificity_controls"] = pd.to_numeric(out["n_specificity_controls"], errors="coerce").fillna(0).astype(int)
    out["specificity_all_expected_positive"] = out["specificity_all_expected_positive"].astype("boolean")

    out["claim_tier"] = out.apply(_claim_tier, axis=1)
    out["claim_rank"] = out["claim_tier"].map(_claim_rank).astype(int)

    # Explicit promotion gates:
    # base required gates are old-push CI positivity + dose-shape + specificity pass (or unavailable).
    out["gate_old_ci_positive_required"] = out["old_ci_positive"].astype(bool)
    out["gate_dose_shape_required"] = out["dose_shape_expected_age_ok"].astype(bool)
    out["gate_specificity_required_or_na"] = (
        out["specificity_all_expected_positive"].fillna(True).astype(bool)
    )
    out["gate_young_ci_negative_required"] = (
        out["young_ci_negative"].astype(bool) if args.require_young_ci_negative_for_promotion else True
    )
    out["gate_celltype_ci_required"] = (
        out["celltype_any_ci_directional_expected_age"].fillna(False).astype(bool)
        if args.require_celltype_ci_for_promotion
        else True
    )

    required_gate_cols = [
        "gate_old_ci_positive_required",
        "gate_dose_shape_required",
        "gate_specificity_required_or_na",
        "gate_young_ci_negative_required",
        "gate_celltype_ci_required",
    ]
    out["gate_required_all"] = True
    out["gate_pass_all_required"] = out[required_gate_cols].all(axis=1)
    out["promotion_status"] = out.apply(_gate_status, axis=1)

    out = out.sort_values(["claim_rank", "dataset_id", "model"]).reset_index(drop=True)
    out.to_csv(args.output_dir / "stage6_model_specific_claims.csv", index=False)

    compact_cols = [
        "dataset_id",
        "model",
        "claim_tier",
        "old_ci_positive",
        "young_ci_negative",
        "dose_shape_expected_age_ok",
        "specificity_all_expected_positive",
        "n_specificity_controls",
        "celltype_any_directional_expected_age",
        "celltype_any_ci_directional_expected_age",
        "delta_expected_age_mean_mean_old_push",
        "delta_expected_age_mean_ci_low_old_push",
        "delta_expected_age_mean_ci_high_old_push",
        "linear_slope_expected_age_mean_old_push",
        "linear_slope_expected_age_mean_young_push",
        "specificity_mean_expected_delta",
        "promotion_status",
        "gate_pass_all_required",
    ]
    compact = out[compact_cols].copy()
    compact.to_csv(args.output_dir / "stage6_model_specific_claims_compact.csv", index=False)
    out[
        [
            "dataset_id",
            "model",
            "claim_tier",
            "promotion_status",
            "gate_pass_all_required",
            *required_gate_cols,
        ]
    ].to_csv(args.output_dir / "stage6_model_specific_claim_gates.csv", index=False)

    for row in out.itertuples(index=False):
        dataset_id = str(row.dataset_id)
        model = str(row.model)
        card_name = f"{dataset_id}__{model}_claim_card.md"
        card_path = cards_dir / card_name
        lines = [
            f"# Claim Card: {dataset_id} / {model}",
            "",
            f"- pathway: `{args.pathway}`",
            f"- claim_tier: `{row.claim_tier}`",
            f"- promotion_status: `{row.promotion_status}`",
            "",
            "## Required Gates",
            "",
            f"- old-push expected-age CI positive: `{bool(row.gate_old_ci_positive_required)}`",
            f"- dose-shape expected-age (`old_slope>0`, `young_slope<0`): `{bool(row.gate_dose_shape_required)}`",
            f"- specificity expected-age pass (or NA): `{bool(row.gate_specificity_required_or_na)}`",
            f"- young-push CI negative required: `{bool(args.require_young_ci_negative_for_promotion)}` -> `{bool(row.gate_young_ci_negative_required)}`",
            f"- cell-type CI directional required: `{bool(args.require_celltype_ci_for_promotion)}` -> `{bool(row.gate_celltype_ci_required)}`",
            "",
            "## Evidence Snapshot",
            "",
            f"- old_push expected-age mean: `{float(row.delta_expected_age_mean_mean_old_push):.6f}`",
            f"- old_push expected-age CI: `[{float(row.delta_expected_age_mean_ci_low_old_push):.6f}, {float(row.delta_expected_age_mean_ci_high_old_push):.6f}]`",
            f"- old slope expected-age: `{float(row.linear_slope_expected_age_mean_old_push):.6f}`",
            f"- young slope expected-age: `{float(row.linear_slope_expected_age_mean_young_push):.6f}`",
            f"- specificity controls count: `{int(row.n_specificity_controls)}`",
            f"- specificity mean expected-age delta: `{float(row.specificity_mean_expected_delta) if pd.notna(row.specificity_mean_expected_delta) else float('nan')}`",
            "",
        ]
        card_path.write_text("\n".join(lines), encoding="utf-8")

    lines = [
        "# Stage-6 Model-Specific Claims",
        "",
        "Strict evidence framing per dataset/model for inflammation expected-age signal.",
        "",
        "Claim tiers:",
        "- `strong_model_specific`: old/young CI directional support + dose shape + specificity + CI-supported cell-type directional support",
        "- `promising_model_specific`: old CI positive + dose shape + specificity pass (or controls unavailable)",
        "- `supportive_but_ci_weak`: dose shape + specificity pass, but CI support is weak",
        "- `exploratory`: criteria above not met",
        "",
        "Promotion gates are explicit and written to `stage6_model_specific_claim_gates.csv`.",
        "",
        "## Compact Evidence Table",
        "",
        "```text",
        compact.to_string(index=False),
        "```",
        "",
    ]
    (args.output_dir / "stage6_model_specific_claims_report.md").write_text("\n".join(lines), encoding="utf-8")

    run_cfg = {
        "bootstrap_consensus_csv": str(args.bootstrap_consensus_csv.resolve()),
        "dose_trend_csv": str(args.dose_trend_csv.resolve()),
        "specificity_compare_csv": str(args.specificity_compare_csv.resolve()),
        "celltype_directional_csv": str(args.celltype_directional_csv.resolve()),
        "pathway": str(args.pathway),
        "require_young_ci_negative_for_promotion": bool(args.require_young_ci_negative_for_promotion),
        "require_celltype_ci_for_promotion": bool(args.require_celltype_ci_for_promotion),
    }
    (args.output_dir / "run_config.json").write_text(json.dumps(run_cfg, indent=2), encoding="utf-8")
    print(f"[done] model-specific claims written to: {args.output_dir}")


if __name__ == "__main__":
    main()
