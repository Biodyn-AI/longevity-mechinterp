#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

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
    parser = argparse.ArgumentParser(description="Summarize inflammation specificity vs control pathways.")
    parser.add_argument(
        "--inflammation-dir",
        type=Path,
        default=root / "implementation" / "outputs" / "stage5_donor_bootstrap_runs_20260303" / "stage5_seed42_n1x15_donor",
    )
    parser.add_argument(
        "--control-dir",
        type=Path,
        default=root
        / "implementation"
        / "outputs"
        / "stage6_inflammation_followup_20260303"
        / "specificity"
        / "neighbor_controls_seed42",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root
        / "implementation"
        / "outputs"
        / "stage6_inflammation_followup_20260303"
        / "specificity_summary",
    )
    parser.add_argument("--analysis-scope", type=str, default="global")
    parser.add_argument("--cell-type", type=str, default="__all__")
    parser.add_argument("--bootstrap-iters", type=int, default=4000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260303)
    return parser.parse_args()


def _read_donor_csv(run_dir: Path, label: str) -> pd.DataFrame:
    donor_csv = run_dir / "stage5_intervention_donor_results.csv"
    if not donor_csv.exists():
        raise FileNotFoundError(f"Missing donor csv for {label}: {donor_csv}")
    df = pd.read_csv(donor_csv).copy()
    df["source_run"] = str(label)
    return df


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(int(args.bootstrap_seed))

    infl = _read_donor_csv(args.inflammation_dir, "inflammation")
    ctrl = _read_donor_csv(args.control_dir, "controls")
    all_df = pd.concat([infl, ctrl], ignore_index=True)
    all_df = all_df[
        (all_df["analysis_scope"].astype(str) == str(args.analysis_scope))
        & (all_df["cell_type"].astype(str) == str(args.cell_type))
    ].copy()
    all_df.to_csv(args.output_dir / "specificity_donor_rows.csv", index=False)

    # Collapse split-level rows into donor means.
    donor_means = (
        all_df.groupby(
            ["source_run", "dataset_id", "model", "pathway", "intervention", "donor_id"],
            as_index=False,
        )
        .agg(
            n_split_rows=("split_idx", "count"),
            delta_expected_age_mean=("delta_expected_age_mean", "mean"),
            delta_old_prob_mean=("delta_old_prob_mean", "mean"),
            delta_pathway_activation_mean=("delta_pathway_activation_mean", "mean"),
        )
        .sort_values(["dataset_id", "model", "pathway", "intervention", "donor_id"])
        .reset_index(drop=True)
    )
    donor_means.to_csv(args.output_dir / "specificity_donor_means.csv", index=False)

    # Bootstrap by pathway/model/intervention.
    boot_rows: List[Dict[str, object]] = []
    for (dataset_id, model, pathway, intervention), grp in donor_means.groupby(
        ["dataset_id", "model", "pathway", "intervention"], as_index=False
    ):
        row: Dict[str, object] = {
            "dataset_id": str(dataset_id),
            "model": str(model),
            "pathway": str(pathway),
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
        ["dataset_id", "model", "pathway", "intervention"]
    ).reset_index(drop=True)
    boot_df.to_csv(args.output_dir / "specificity_bootstrap.csv", index=False)

    # Specificity score: old_push minus random_push per pathway.
    pivot = boot_df.pivot_table(
        index=["dataset_id", "model", "pathway"],
        columns="intervention",
        values=["delta_expected_age_mean_mean", "delta_old_prob_mean_mean"],
        aggfunc="first",
    )
    pivot.columns = [f"{a}_{b}" for a, b in pivot.columns]
    pivot = pivot.reset_index()
    for metric in ["delta_expected_age_mean_mean", "delta_old_prob_mean_mean"]:
        old_col = f"{metric}_old_push"
        rnd_col = f"{metric}_random_push"
        if old_col in pivot.columns and rnd_col in pivot.columns:
            pivot[f"{metric}_old_minus_random"] = pivot[old_col] - pivot[rnd_col]
    pivot.to_csv(args.output_dir / "specificity_old_minus_random.csv", index=False)

    # Compare inflammation against controls within same dataset/model.
    compare_rows: List[Dict[str, object]] = []
    for (dataset_id, model), grp in pivot.groupby(["dataset_id", "model"], as_index=False):
        sub = grp.copy()
        infl_sub = sub[sub["pathway"].astype(str) == "inflammation_nfkb"]
        if infl_sub.empty:
            continue
        infl_row = infl_sub.iloc[0]
        for _, ctrl_row in sub.iterrows():
            if str(ctrl_row["pathway"]) == "inflammation_nfkb":
                continue
            compare_rows.append(
                {
                    "dataset_id": str(dataset_id),
                    "model": str(model),
                    "control_pathway": str(ctrl_row["pathway"]),
                    "infl_old_minus_random_expected_age": float(
                        infl_row.get("delta_expected_age_mean_mean_old_minus_random", np.nan)
                    ),
                    "ctrl_old_minus_random_expected_age": float(
                        ctrl_row.get("delta_expected_age_mean_mean_old_minus_random", np.nan)
                    ),
                    "infl_minus_ctrl_expected_age": float(
                        infl_row.get("delta_expected_age_mean_mean_old_minus_random", np.nan)
                        - ctrl_row.get("delta_expected_age_mean_mean_old_minus_random", np.nan)
                    ),
                    "infl_old_minus_random_old_prob": float(
                        infl_row.get("delta_old_prob_mean_mean_old_minus_random", np.nan)
                    ),
                    "ctrl_old_minus_random_old_prob": float(
                        ctrl_row.get("delta_old_prob_mean_mean_old_minus_random", np.nan)
                    ),
                    "infl_minus_ctrl_old_prob": float(
                        infl_row.get("delta_old_prob_mean_mean_old_minus_random", np.nan)
                        - ctrl_row.get("delta_old_prob_mean_mean_old_minus_random", np.nan)
                    ),
                }
            )
    compare_df = pd.DataFrame(compare_rows)
    if not compare_df.empty:
        compare_df = compare_df.sort_values(["dataset_id", "model", "control_pathway"]).reset_index(drop=True)
    compare_df.to_csv(args.output_dir / "specificity_inflammation_vs_controls.csv", index=False)

    report_lines = [
        "# Specificity Report",
        "",
        "Compares inflammation pathway intervention effects against neighboring control pathways.",
        "",
        "## Bootstrap Summary",
        "",
        "```text",
        boot_df.to_string(index=False),
        "```",
        "",
        "## Inflammation vs Controls",
        "",
        "```text",
        (compare_df.to_string(index=False) if not compare_df.empty else "No comparable control rows."),
        "```",
        "",
    ]
    (args.output_dir / "specificity_report.md").write_text("\n".join(report_lines), encoding="utf-8")

    run_cfg = {
        "inflammation_dir": str(args.inflammation_dir.resolve()),
        "control_dir": str(args.control_dir.resolve()),
        "analysis_scope": str(args.analysis_scope),
        "cell_type": str(args.cell_type),
        "bootstrap_iters": int(args.bootstrap_iters),
        "bootstrap_seed": int(args.bootstrap_seed),
    }
    (args.output_dir / "run_config.json").write_text(json.dumps(run_cfg, indent=2), encoding="utf-8")
    print(f"[done] specificity summary written to: {args.output_dir}")


if __name__ == "__main__":
    main()
