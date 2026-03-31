#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def _bootstrap_ci(values: np.ndarray, n_boot: int, rng: np.random.Generator) -> Tuple[float, float, float]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan")
    if arr.size == 1:
        x = float(arr[0])
        return x, x, x
    idx = rng.integers(0, arr.size, size=(int(n_boot), arr.size))
    boot = arr[idx].mean(axis=1)
    return float(np.mean(arr)), float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


def _ci_flag(low: float, high: float) -> str:
    if np.isfinite(low) and np.isfinite(high):
        if low > 0:
            return "positive"
        if high < 0:
            return "negative"
    return "uncertain"


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Summarize Stage-6 cell-type anchored intervention run.")
    parser.add_argument(
        "--stage5-dir",
        type=Path,
        default=root / "implementation" / "outputs" / "stage6_inflammation_followup_20260303" / "celltype_anchor_seed42",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "implementation" / "outputs" / "stage6_inflammation_followup_20260303" / "celltype_anchor_summary",
    )
    parser.add_argument("--analysis-scope", type=str, default="within_cell_type")
    parser.add_argument("--cell-type", type=str, default="__all__")
    parser.add_argument("--bootstrap-iters", type=int, default=4000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260303)
    return parser.parse_args()


def _read_donor_rows(stage5_dir: Path) -> pd.DataFrame:
    top_csv = stage5_dir / "stage5_intervention_donor_results.csv"
    if top_csv.exists():
        return pd.read_csv(top_csv)

    nested = sorted(stage5_dir.rglob("donor_intervention_results.csv"))
    if not nested:
        raise FileNotFoundError(
            f"No donor results found. Missing both {top_csv} and nested donor_intervention_results.csv files."
        )
    return pd.concat((pd.read_csv(p) for p in nested), ignore_index=True)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(int(args.bootstrap_seed))

    raw = _read_donor_rows(args.stage5_dir).copy()
    raw = raw[raw["analysis_scope"].astype(str) == str(args.analysis_scope)].copy()
    if str(args.cell_type) != "__all__":
        raw = raw[raw["cell_type"].astype(str) == str(args.cell_type)].copy()
    raw.to_csv(args.output_dir / "celltype_anchor_donor_rows.csv", index=False)

    donor_means = (
        raw.groupby(
            [
                "dataset_id",
                "model",
                "pathway",
                "analysis_scope",
                "cell_type",
                "intervention",
                "donor_id",
            ],
            as_index=False,
        )
        .agg(
            n_split_rows=("split_idx", "count"),
            delta_expected_age_mean=("delta_expected_age_mean", "mean"),
            delta_old_prob_mean=("delta_old_prob_mean", "mean"),
            delta_pathway_activation_mean=("delta_pathway_activation_mean", "mean"),
        )
        .sort_values(["dataset_id", "model", "pathway", "cell_type", "intervention", "donor_id"])
        .reset_index(drop=True)
    )
    donor_means.to_csv(args.output_dir / "celltype_anchor_donor_means.csv", index=False)

    boot_rows: List[Dict[str, object]] = []
    for (
        dataset_id,
        model,
        pathway,
        analysis_scope,
        cell_type,
        intervention,
    ), grp in donor_means.groupby(
        ["dataset_id", "model", "pathway", "analysis_scope", "cell_type", "intervention"],
        as_index=False,
    ):
        row: Dict[str, object] = {
            "dataset_id": str(dataset_id),
            "model": str(model),
            "pathway": str(pathway),
            "analysis_scope": str(analysis_scope),
            "cell_type": str(cell_type),
            "intervention": str(intervention),
            "n_donors": int(grp["donor_id"].nunique()),
        }
        for metric in ["delta_expected_age_mean", "delta_old_prob_mean", "delta_pathway_activation_mean"]:
            mean, low, high = _bootstrap_ci(grp[metric].to_numpy(dtype=np.float64), args.bootstrap_iters, rng)
            row[f"{metric}_mean"] = mean
            row[f"{metric}_ci_low"] = low
            row[f"{metric}_ci_high"] = high
            row[f"{metric}_ci_flag"] = _ci_flag(low, high)
        boot_rows.append(row)

    boot_df = pd.DataFrame(boot_rows).sort_values(
        ["dataset_id", "model", "pathway", "cell_type", "intervention"]
    ).reset_index(drop=True)
    boot_df.to_csv(args.output_dir / "celltype_anchor_bootstrap.csv", index=False)

    value_pivot = boot_df.pivot_table(
        index=["dataset_id", "model", "pathway", "analysis_scope", "cell_type"],
        columns="intervention",
        values=[
            "delta_expected_age_mean_mean",
            "delta_old_prob_mean_mean",
            "delta_pathway_activation_mean_mean",
        ],
        aggfunc="first",
    )
    value_pivot.columns = [f"{a}_{b}" for a, b in value_pivot.columns]
    value_pivot = value_pivot.reset_index()

    flag_pivot = boot_df.pivot_table(
        index=["dataset_id", "model", "pathway", "analysis_scope", "cell_type"],
        columns="intervention",
        values=[
            "delta_expected_age_mean_ci_flag",
            "delta_old_prob_mean_ci_flag",
            "delta_pathway_activation_mean_ci_flag",
        ],
        aggfunc="first",
    )
    flag_pivot.columns = [f"{a}_{b}" for a, b in flag_pivot.columns]
    flag_pivot = flag_pivot.reset_index()

    directional = value_pivot.merge(
        flag_pivot,
        on=["dataset_id", "model", "pathway", "analysis_scope", "cell_type"],
        how="left",
    )

    directional["old_minus_random_expected_age"] = (
        directional.get("delta_expected_age_mean_mean_old_push", np.nan)
        - directional.get("delta_expected_age_mean_mean_random_push", np.nan)
    )
    directional["young_minus_random_expected_age"] = (
        directional.get("delta_expected_age_mean_mean_young_push", np.nan)
        - directional.get("delta_expected_age_mean_mean_random_push", np.nan)
    )
    directional["old_minus_random_old_prob"] = (
        directional.get("delta_old_prob_mean_mean_old_push", np.nan)
        - directional.get("delta_old_prob_mean_mean_random_push", np.nan)
    )
    directional["young_minus_random_old_prob"] = (
        directional.get("delta_old_prob_mean_mean_young_push", np.nan)
        - directional.get("delta_old_prob_mean_mean_random_push", np.nan)
    )

    directional["directional_pattern_expected_age"] = (
        (directional.get("delta_expected_age_mean_mean_old_push", np.nan) > 0.0)
        & (directional.get("delta_expected_age_mean_mean_young_push", np.nan) < 0.0)
    )
    directional["directional_pattern_old_prob"] = (
        (directional.get("delta_old_prob_mean_mean_old_push", np.nan) > 0.0)
        & (directional.get("delta_old_prob_mean_mean_young_push", np.nan) < 0.0)
    )
    directional["ci_directional_pattern_expected_age"] = (
        (directional.get("delta_expected_age_mean_ci_flag_old_push", "") == "positive")
        & (directional.get("delta_expected_age_mean_ci_flag_young_push", "") == "negative")
    )
    directional["ci_directional_pattern_old_prob"] = (
        (directional.get("delta_old_prob_mean_ci_flag_old_push", "") == "positive")
        & (directional.get("delta_old_prob_mean_ci_flag_young_push", "") == "negative")
    )
    directional = directional.sort_values(
        ["dataset_id", "model", "pathway", "cell_type"]
    ).reset_index(drop=True)
    directional.to_csv(args.output_dir / "celltype_anchor_directional.csv", index=False)

    shared = directional.pivot_table(
        index=["dataset_id", "pathway", "analysis_scope", "cell_type"],
        columns="model",
        values=[
            "old_minus_random_expected_age",
            "old_minus_random_old_prob",
            "directional_pattern_expected_age",
            "directional_pattern_old_prob",
            "ci_directional_pattern_expected_age",
            "ci_directional_pattern_old_prob",
        ],
        aggfunc="first",
    )
    shared.columns = [f"{a}_{b}" for a, b in shared.columns]
    shared = shared.reset_index()

    has_scgpt = shared.filter(regex=r"_scgpt$").notna().any(axis=1)
    has_geneformer = shared.filter(regex=r"_geneformer$").notna().any(axis=1)
    shared = shared[has_scgpt & has_geneformer].copy()
    if not shared.empty:
        def _to_bool(s: pd.Series) -> pd.Series:
            return s.astype("boolean").fillna(False).astype(bool)

        shared["sign_agree_old_minus_random_expected_age"] = (
            np.sign(shared["old_minus_random_expected_age_scgpt"])
            == np.sign(shared["old_minus_random_expected_age_geneformer"])
        )
        shared["sign_agree_old_minus_random_old_prob"] = (
            np.sign(shared["old_minus_random_old_prob_scgpt"])
            == np.sign(shared["old_minus_random_old_prob_geneformer"])
        )
        shared["both_directional_expected_age"] = (
            _to_bool(shared["directional_pattern_expected_age_scgpt"])
            & _to_bool(shared["directional_pattern_expected_age_geneformer"])
        )
        shared["both_ci_directional_expected_age"] = (
            _to_bool(shared["ci_directional_pattern_expected_age_scgpt"])
            & _to_bool(shared["ci_directional_pattern_expected_age_geneformer"])
        )
        shared["both_directional_old_prob"] = (
            _to_bool(shared["directional_pattern_old_prob_scgpt"])
            & _to_bool(shared["directional_pattern_old_prob_geneformer"])
        )
        shared["both_ci_directional_old_prob"] = (
            _to_bool(shared["ci_directional_pattern_old_prob_scgpt"])
            & _to_bool(shared["ci_directional_pattern_old_prob_geneformer"])
        )
    shared = shared.sort_values(["dataset_id", "pathway", "cell_type"]).reset_index(drop=True)
    shared.to_csv(args.output_dir / "celltype_anchor_cross_model.csv", index=False)

    n_unique_strata = directional[["dataset_id", "model", "pathway", "cell_type"]].drop_duplicates().shape[0]
    n_shared = len(shared)
    n_both_dir = int(shared["both_directional_expected_age"].sum()) if not shared.empty else 0
    n_both_ci_dir = int(shared["both_ci_directional_expected_age"].sum()) if not shared.empty else 0
    best_shared = (
        shared.sort_values(
            ["both_ci_directional_expected_age", "both_directional_expected_age", "dataset_id", "cell_type"],
            ascending=[False, False, True, True],
        )
        .head(12)
        .copy()
    )

    lines = [
        "# Cell-Type Anchor Summary",
        "",
        "Donor-bootstrap summary for within-cell-type inflammation intervention effects.",
        "",
        f"- Unique (dataset, model, pathway, cell_type) strata: `{n_unique_strata}`",
        f"- Shared scGPT+Geneformer strata: `{n_shared}`",
        f"- Shared strata with both-model directional pattern (expected age): `{n_both_dir}`",
        f"- Shared strata with both-model CI directional support (expected age): `{n_both_ci_dir}`",
        "",
        "## Top Shared Strata",
        "",
        "```text",
        (best_shared.to_string(index=False) if not best_shared.empty else "No shared strata found."),
        "```",
        "",
    ]
    (args.output_dir / "celltype_anchor_report.md").write_text("\n".join(lines), encoding="utf-8")

    run_cfg = {
        "stage5_dir": str(args.stage5_dir.resolve()),
        "analysis_scope": str(args.analysis_scope),
        "cell_type": str(args.cell_type),
        "bootstrap_iters": int(args.bootstrap_iters),
        "bootstrap_seed": int(args.bootstrap_seed),
    }
    (args.output_dir / "run_config.json").write_text(json.dumps(run_cfg, indent=2), encoding="utf-8")
    print(f"[done] cell-type anchor summary written to: {args.output_dir}")


if __name__ == "__main__":
    main()
