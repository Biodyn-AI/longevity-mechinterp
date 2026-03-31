#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd


def _parse_scale_dirs(text: str) -> List[Tuple[float, Path]]:
    out: List[Tuple[float, Path]] = []
    for item in [x.strip() for x in str(text).split(",") if x.strip()]:
        if "=" not in item:
            raise ValueError(f"Invalid scale-dir mapping '{item}'. Use scale=/abs/path.")
        scale_text, path_text = item.split("=", 1)
        out.append((float(scale_text.strip()), Path(path_text.strip()).expanduser().resolve()))
    if not out:
        raise ValueError("No scale directories provided.")
    return sorted(out, key=lambda x: x[0])


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


def _default_scale_dirs(root: Path) -> str:
    base = root / "implementation" / "outputs"
    mapping = [
        (
            0.0,
            base / "stage6_inflammation_followup_20260303" / "dose_response" / "scale_0_seed42",
        ),
        (
            0.5,
            base / "stage6_inflammation_followup_20260303" / "dose_response" / "scale_0p5_seed42",
        ),
        (
            1.0,
            base / "stage5_donor_bootstrap_runs_20260303" / "stage5_seed42_n1x15_donor",
        ),
        (
            2.0,
            base / "stage6_inflammation_followup_20260303" / "dose_response" / "scale_2_seed42",
        ),
    ]
    return ",".join([f"{s}={p}" for s, p in mapping])


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Summarize inflammation dose-response from donor-level Stage-5 outputs.")
    parser.add_argument("--scale-dirs", type=str, default=_default_scale_dirs(root))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "implementation" / "outputs" / "stage6_inflammation_followup_20260303" / "dose_response_summary",
    )
    parser.add_argument("--pathway", type=str, default="inflammation_nfkb")
    parser.add_argument("--analysis-scope", type=str, default="global")
    parser.add_argument("--cell-type", type=str, default="__all__")
    parser.add_argument("--bootstrap-iters", type=int, default=4000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260303)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    scale_dirs = _parse_scale_dirs(args.scale_dirs)

    status_rows: List[Dict[str, object]] = []
    donor_rows: List[pd.DataFrame] = []
    for scale, scale_dir in scale_dirs:
        donor_csv = scale_dir / "stage5_intervention_donor_results.csv"
        has = donor_csv.exists()
        status_rows.append(
            {
                "scale": float(scale),
                "scale_dir": str(scale_dir),
                "donor_csv": str(donor_csv),
                "has_donor_csv": bool(has),
                "status": "ok" if has else "missing",
            }
        )
        if not has:
            continue
        df = pd.read_csv(donor_csv)
        df.insert(0, "intervention_scale", float(scale))
        df = df[
            (df["pathway"].astype(str) == str(args.pathway))
            & (df["analysis_scope"].astype(str) == str(args.analysis_scope))
            & (df["cell_type"].astype(str) == str(args.cell_type))
        ].copy()
        if not df.empty:
            donor_rows.append(df)

    status_df = pd.DataFrame(status_rows).sort_values("scale").reset_index(drop=True)
    status_df.to_csv(args.output_dir / "dose_response_run_status.csv", index=False)
    if not donor_rows:
        (args.output_dir / "dose_response_report.md").write_text(
            "# Dose-Response Report\n\nNo donor rows available.\n", encoding="utf-8"
        )
        return

    donor_all = pd.concat(donor_rows, ignore_index=True)
    donor_all.to_csv(args.output_dir / "dose_response_donor_rows.csv", index=False)

    # Collapse split-level rows into donor means per scale.
    donor_scale_means = (
        donor_all.groupby(
            ["intervention_scale", "dataset_id", "model", "intervention", "donor_id"],
            as_index=False,
        )
        .agg(
            n_split_rows=("split_idx", "count"),
            delta_expected_age_mean=("delta_expected_age_mean", "mean"),
            delta_old_prob_mean=("delta_old_prob_mean", "mean"),
            delta_pathway_activation_mean=("delta_pathway_activation_mean", "mean"),
        )
        .sort_values(["dataset_id", "model", "intervention", "intervention_scale", "donor_id"])
        .reset_index(drop=True)
    )
    donor_scale_means.to_csv(args.output_dir / "dose_response_donor_scale_means.csv", index=False)

    boot_rows: List[Dict[str, object]] = []
    rng = np.random.default_rng(int(args.bootstrap_seed))
    for (dataset_id, model, intervention, scale), grp in donor_scale_means.groupby(
        ["dataset_id", "model", "intervention", "intervention_scale"],
        as_index=False,
    ):
        row: Dict[str, object] = {
            "dataset_id": str(dataset_id),
            "model": str(model),
            "intervention": str(intervention),
            "intervention_scale": float(scale),
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
        ["dataset_id", "model", "intervention", "intervention_scale"]
    ).reset_index(drop=True)
    boot_df.to_csv(args.output_dir / "dose_response_bootstrap_by_scale.csv", index=False)

    # Monotonic trend summary.
    trend_rows: List[Dict[str, object]] = []
    for (dataset_id, model, intervention), grp in boot_df.groupby(
        ["dataset_id", "model", "intervention"], as_index=False
    ):
        g = grp.sort_values("intervention_scale")
        x = g["intervention_scale"].to_numpy(dtype=np.float64)
        y_age = g["delta_expected_age_mean_mean"].to_numpy(dtype=np.float64)
        y_old = g["delta_old_prob_mean_mean"].to_numpy(dtype=np.float64)
        # Spearman using rank correlation through pandas.
        rho_age = float(pd.Series(x).corr(pd.Series(y_age), method="spearman"))
        rho_old = float(pd.Series(x).corr(pd.Series(y_old), method="spearman"))
        slope_age = float(np.polyfit(x, y_age, 1)[0]) if np.unique(x).size >= 2 else float("nan")
        slope_old = float(np.polyfit(x, y_old, 1)[0]) if np.unique(x).size >= 2 else float("nan")
        trend_rows.append(
            {
                "dataset_id": str(dataset_id),
                "model": str(model),
                "intervention": str(intervention),
                "n_scales": int(g.shape[0]),
                "scale_min": float(np.min(x)),
                "scale_max": float(np.max(x)),
                "spearman_scale_vs_expected_age_mean": rho_age,
                "spearman_scale_vs_old_prob_mean": rho_old,
                "linear_slope_expected_age_mean": slope_age,
                "linear_slope_old_prob_mean": slope_old,
            }
        )
    trend_df = pd.DataFrame(trend_rows).sort_values(
        ["dataset_id", "model", "intervention"]
    ).reset_index(drop=True)
    trend_df.to_csv(args.output_dir / "dose_response_trend_summary.csv", index=False)

    # Specificity proxy inside inflammation runs: old_push - random_push at each scale.
    pivot = boot_df.pivot_table(
        index=["dataset_id", "model", "intervention_scale"],
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
    pivot.to_csv(args.output_dir / "dose_response_specificity_old_minus_random.csv", index=False)

    report_lines = [
        "# Inflammation Dose-Response Report",
        "",
        "Donor-level dose-response summary for Stage-5 inflammation interventions.",
        "",
        "## Run Status",
        "",
        "```text",
        status_df.to_string(index=False),
        "```",
        "",
        "## Bootstrap By Scale",
        "",
        "```text",
        boot_df.to_string(index=False),
        "```",
        "",
        "## Trend Summary",
        "",
        "```text",
        trend_df.to_string(index=False),
        "```",
        "",
    ]
    (args.output_dir / "dose_response_report.md").write_text("\n".join(report_lines), encoding="utf-8")

    run_cfg = {
        "scale_dirs": [{"scale": s, "path": str(p)} for s, p in scale_dirs],
        "pathway": str(args.pathway),
        "analysis_scope": str(args.analysis_scope),
        "cell_type": str(args.cell_type),
        "bootstrap_iters": int(args.bootstrap_iters),
        "bootstrap_seed": int(args.bootstrap_seed),
    }
    (args.output_dir / "run_config.json").write_text(json.dumps(run_cfg, indent=2), encoding="utf-8")
    print(f"[done] dose-response summary written to: {args.output_dir}")


if __name__ == "__main__":
    main()
