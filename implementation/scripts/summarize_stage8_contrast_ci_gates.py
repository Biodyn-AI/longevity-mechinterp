#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd


INTERVENTION_OLD = "old_push"
INTERVENTION_RANDOM = "random_push"
INTERVENTION_YOUNG = "young_push"

METRICS = ["delta_expected_age_mean", "delta_old_prob_mean"]
CONTRASTS = [
    ("old_minus_random", INTERVENTION_OLD, INTERVENTION_RANDOM),
    ("young_minus_random", INTERVENTION_YOUNG, INTERVENTION_RANDOM),
    ("old_minus_young", INTERVENTION_OLD, INTERVENTION_YOUNG),
]


def _parse_run_dirs(text: str) -> List[Tuple[str, Path]]:
    out: List[Tuple[str, Path]] = []
    for item in [x.strip() for x in str(text).split(",") if x.strip()]:
        if "=" not in item:
            raise ValueError(f"Invalid run mapping '{item}'. Use label=/abs/path format.")
        label, path_text = item.split("=", 1)
        out.append((str(label.strip()), Path(path_text.strip()).expanduser().resolve()))
    return out


def _discover_run_dirs(root: Path) -> List[Tuple[str, Path]]:
    rows: List[Tuple[str, Path]] = []
    for donor_csv in sorted(root.rglob("stage5_intervention_donor_results.csv")):
        run_dir = donor_csv.parent
        try:
            label = str(run_dir.relative_to(root))
        except ValueError:
            label = run_dir.name
        rows.append((label, run_dir))
    return rows


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


def _ci_flag(ci_low: float, ci_high: float) -> str:
    if np.isfinite(ci_low) and np.isfinite(ci_high):
        if ci_low > 0:
            return "positive"
        if ci_high < 0:
            return "negative"
    return "uncertain"


def _bool_flag(series: pd.Series, value: str) -> pd.Series:
    return series.astype(str) == str(value)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Compute strict contrast-CI gates from donor-level Stage-5 intervention runs."
    )
    parser.add_argument(
        "--discover-root",
        type=Path,
        default=root / "implementation" / "outputs",
        help="Root to scan for stage5_intervention_donor_results.csv when --run-dirs is not provided.",
    )
    parser.add_argument(
        "--run-dirs",
        type=str,
        default="",
        help="Optional explicit mappings: label=/abs/path,label2=/abs/path2",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "implementation" / "outputs" / "stage8_contrast_ci_gate_audit_20260303",
    )
    parser.add_argument("--bootstrap-iters", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260303)
    parser.add_argument("--min-donors", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if str(args.run_dirs).strip():
        run_dirs = _parse_run_dirs(args.run_dirs)
    else:
        run_dirs = _discover_run_dirs(args.discover_root.expanduser().resolve())

    status_rows: List[Dict[str, object]] = []
    all_rows: List[pd.DataFrame] = []
    for run_label, run_dir in run_dirs:
        donor_csv = run_dir / "stage5_intervention_donor_results.csv"
        has_csv = donor_csv.exists()
        status_rows.append(
            {
                "run_label": str(run_label),
                "run_dir": str(run_dir),
                "donor_csv": str(donor_csv),
                "has_donor_csv": bool(has_csv),
                "status": "ok" if has_csv else "missing_donor_csv",
            }
        )
        if not has_csv:
            continue

        df = pd.read_csv(donor_csv)
        keep = df["intervention"].astype(str).isin({INTERVENTION_OLD, INTERVENTION_RANDOM, INTERVENTION_YOUNG})
        df = df.loc[keep].copy()
        if df.empty:
            continue
        df.insert(0, "run_label", str(run_label))
        df.insert(1, "run_dir", str(run_dir))
        all_rows.append(df)

    run_status = pd.DataFrame(status_rows).sort_values("run_label").reset_index(drop=True)
    run_status.to_csv(args.output_dir / "stage8_contrast_gate_run_status.csv", index=False)

    if not all_rows:
        (args.output_dir / "stage8_contrast_gate_report.md").write_text(
            "# Stage-8 Contrast Gate Report\n\nNo donor-level rows were found.\n",
            encoding="utf-8",
        )
        return

    donor_all = pd.concat(all_rows, ignore_index=True)
    donor_all.to_csv(args.output_dir / "stage8_contrast_gate_donor_rows.csv", index=False)

    group_cols = ["run_label", "run_dir", "dataset_id", "model", "pathway", "analysis_scope", "cell_type"]
    donor_means = (
        donor_all.groupby(group_cols + ["intervention", "donor_id"], as_index=False)
        .agg(
            n_rows=("split_idx", "count"),
            delta_expected_age_mean=("delta_expected_age_mean", "mean"),
            delta_old_prob_mean=("delta_old_prob_mean", "mean"),
        )
        .sort_values(group_cols + ["intervention", "donor_id"])
        .reset_index(drop=True)
    )
    donor_means.to_csv(args.output_dir / "stage8_contrast_gate_donor_means.csv", index=False)

    long_rows: List[Dict[str, object]] = []
    marginal_rows: List[Dict[str, object]] = []
    for keys, grp in donor_means.groupby(group_cols, as_index=False):
        key_map = {col: keys[idx] for idx, col in enumerate(group_cols)}
        pivot = grp.pivot_table(
            index="donor_id",
            columns="intervention",
            values=METRICS,
            aggfunc="mean",
        )
        for metric in METRICS:
            for intervention in [INTERVENTION_OLD, INTERVENTION_RANDOM, INTERVENTION_YOUNG]:
                key_int = (metric, intervention)
                if key_int not in pivot.columns:
                    continue
                marginal_values = pivot[key_int].dropna().to_numpy(dtype=np.float64)
                marginal_rng = np.random.default_rng(
                    int(args.bootstrap_seed)
                    + int(
                        abs(
                            hash(
                                (
                                    key_map["run_label"],
                                    key_map["dataset_id"],
                                    key_map["model"],
                                    metric,
                                    intervention,
                                )
                            )
                        )
                        % 10_000_000
                    )
                )
                m_mean, m_low, m_high = _bootstrap_ci(marginal_values, int(args.bootstrap_iters), marginal_rng)
                mrow: Dict[str, object] = dict(key_map)
                mrow.update(
                    {
                        "metric": str(metric),
                        "intervention": str(intervention),
                        "n_donors": int(marginal_values.size),
                        "mean_delta": float(m_mean),
                        "ci_low": float(m_low),
                        "ci_high": float(m_high),
                        "ci_flag": _ci_flag(m_low, m_high),
                    }
                )
                marginal_rows.append(mrow)

            for contrast_name, lhs, rhs in CONTRASTS:
                key_lhs = (metric, lhs)
                key_rhs = (metric, rhs)
                if key_lhs not in pivot.columns or key_rhs not in pivot.columns:
                    continue
                values = (pivot[key_lhs] - pivot[key_rhs]).dropna().to_numpy(dtype=np.float64)
                rng = np.random.default_rng(
                    int(args.bootstrap_seed)
                    + int(abs(hash((key_map["run_label"], key_map["dataset_id"], key_map["model"], metric, contrast_name))) % 10_000_000)
                )
                mean, ci_low, ci_high = _bootstrap_ci(values, int(args.bootstrap_iters), rng)
                row: Dict[str, object] = dict(key_map)
                row.update(
                    {
                        "metric": str(metric),
                        "contrast": str(contrast_name),
                        "n_donors": int(values.size),
                        "mean_delta": float(mean),
                        "ci_low": float(ci_low),
                        "ci_high": float(ci_high),
                        "ci_flag": _ci_flag(ci_low, ci_high),
                    }
                )
                long_rows.append(row)

    contrast_long = (
        pd.DataFrame(long_rows)
        .sort_values(group_cols + ["metric", "contrast"])
        .reset_index(drop=True)
        if long_rows
        else pd.DataFrame()
    )
    contrast_long.to_csv(args.output_dir / "stage8_contrast_ci_long.csv", index=False)

    marginal_long = (
        pd.DataFrame(marginal_rows)
        .sort_values(group_cols + ["metric", "intervention"])
        .reset_index(drop=True)
        if marginal_rows
        else pd.DataFrame()
    )
    marginal_long.to_csv(args.output_dir / "stage8_marginal_ci_long.csv", index=False)

    if contrast_long.empty or marginal_long.empty:
        (args.output_dir / "stage8_contrast_gate_report.md").write_text(
            "# Stage-8 Contrast Gate Report\n\nNo contrast or marginal rows could be computed.\n",
            encoding="utf-8",
        )
        return

    wide_base = contrast_long[group_cols].drop_duplicates().reset_index(drop=True)
    for metric in METRICS:
        for contrast_name, _, _ in CONTRASTS:
            sub = contrast_long[
                (contrast_long["metric"].astype(str) == metric) & (contrast_long["contrast"].astype(str) == contrast_name)
            ][group_cols + ["n_donors", "mean_delta", "ci_low", "ci_high", "ci_flag"]].copy()
            suffix = f"{metric}__{contrast_name}"
            sub = sub.rename(
                columns={
                    "n_donors": f"n_donors__{suffix}",
                    "mean_delta": f"mean__{suffix}",
                    "ci_low": f"ci_low__{suffix}",
                    "ci_high": f"ci_high__{suffix}",
                    "ci_flag": f"ci_flag__{suffix}",
                }
            )
            wide_base = wide_base.merge(sub, on=group_cols, how="left")
        for intervention in [INTERVENTION_OLD, INTERVENTION_RANDOM, INTERVENTION_YOUNG]:
            subm = marginal_long[
                (marginal_long["metric"].astype(str) == metric)
                & (marginal_long["intervention"].astype(str) == intervention)
            ][group_cols + ["n_donors", "mean_delta", "ci_low", "ci_high", "ci_flag"]].copy()
            suffix_m = f"{metric}__{intervention}"
            subm = subm.rename(
                columns={
                    "n_donors": f"n_donors_marginal__{suffix_m}",
                    "mean_delta": f"mean_marginal__{suffix_m}",
                    "ci_low": f"ci_low_marginal__{suffix_m}",
                    "ci_high": f"ci_high_marginal__{suffix_m}",
                    "ci_flag": f"ci_flag_marginal__{suffix_m}",
                }
            )
            wide_base = wide_base.merge(subm, on=group_cols, how="left")

    # Strict gates:
    # 1) expected-age contrasts pass directionally with CI support
    # 2) old-prob contrasts pass directionally with CI support
    # 3) donor count gate for expected-age contrasts
    ea_or = "delta_expected_age_mean__old_minus_random"
    ea_yr = "delta_expected_age_mean__young_minus_random"
    ea_oy = "delta_expected_age_mean__old_minus_young"
    op_or = "delta_old_prob_mean__old_minus_random"
    op_yr = "delta_old_prob_mean__young_minus_random"

    wide_base["gate_min_donors_expected_age"] = (
        wide_base[f"n_donors__{ea_or}"].fillna(0) >= int(args.min_donors)
    ) & (
        wide_base[f"n_donors__{ea_yr}"].fillna(0) >= int(args.min_donors)
    ) & (
        wide_base[f"n_donors__{ea_oy}"].fillna(0) >= int(args.min_donors)
    )
    wide_base["gate_ea_old_minus_random_positive"] = _bool_flag(wide_base[f"ci_flag__{ea_or}"], "positive")
    wide_base["gate_ea_young_minus_random_negative"] = _bool_flag(wide_base[f"ci_flag__{ea_yr}"], "negative")
    wide_base["gate_ea_old_minus_young_positive"] = _bool_flag(wide_base[f"ci_flag__{ea_oy}"], "positive")
    wide_base["gate_oldprob_old_minus_random_positive"] = _bool_flag(wide_base[f"ci_flag__{op_or}"], "positive")
    wide_base["gate_oldprob_young_minus_random_negative"] = _bool_flag(wide_base[f"ci_flag__{op_yr}"], "negative")

    wide_base["pass_expected_age_contrast_strict"] = (
        wide_base["gate_min_donors_expected_age"]
        & wide_base["gate_ea_old_minus_random_positive"]
        & wide_base["gate_ea_young_minus_random_negative"]
        & wide_base["gate_ea_old_minus_young_positive"]
    )
    wide_base["pass_full_contrast_strict"] = (
        wide_base["pass_expected_age_contrast_strict"]
        & wide_base["gate_oldprob_old_minus_random_positive"]
        & wide_base["gate_oldprob_young_minus_random_negative"]
    )

    ea_old_m = "delta_expected_age_mean__old_push"
    ea_yng_m = "delta_expected_age_mean__young_push"
    wide_base["gate_ea_old_push_marginal_positive"] = _bool_flag(
        wide_base[f"ci_flag_marginal__{ea_old_m}"], "positive"
    )
    wide_base["gate_ea_young_push_marginal_negative"] = _bool_flag(
        wide_base[f"ci_flag_marginal__{ea_yng_m}"], "negative"
    )

    wide_base["pass_expected_age_strict"] = (
        wide_base["pass_expected_age_contrast_strict"]
        & wide_base["gate_ea_old_push_marginal_positive"]
        & wide_base["gate_ea_young_push_marginal_negative"]
    )
    wide_base["pass_full_strict"] = (
        wide_base["pass_full_contrast_strict"]
        & wide_base["gate_ea_old_push_marginal_positive"]
        & wide_base["gate_ea_young_push_marginal_negative"]
    )

    wide_base.to_csv(args.output_dir / "stage8_contrast_ci_wide.csv", index=False)

    gate_summary = (
        wide_base.groupby("run_label", as_index=False)
        .agg(
            n_rows=("run_label", "count"),
            n_expected_age_contrast_strict=("pass_expected_age_contrast_strict", "sum"),
            n_full_contrast_strict=("pass_full_contrast_strict", "sum"),
            n_expected_age_strict=("pass_expected_age_strict", "sum"),
            n_full_strict=("pass_full_strict", "sum"),
        )
        .sort_values(
            ["n_full_strict", "n_expected_age_strict", "n_full_contrast_strict"],
            ascending=[False, False, False],
        )
        .reset_index(drop=True)
    )
    gate_summary.to_csv(args.output_dir / "stage8_contrast_gate_summary_by_run.csv", index=False)

    top_pass = wide_base[wide_base["pass_full_strict"]].copy()
    top_pass = top_pass.sort_values(group_cols).reset_index(drop=True)
    top_pass.to_csv(args.output_dir / "stage8_contrast_gate_pass_full_strict.csv", index=False)

    expected_pass = wide_base[wide_base["pass_expected_age_strict"]].copy()
    expected_pass = expected_pass.sort_values(group_cols).reset_index(drop=True)
    expected_pass.to_csv(args.output_dir / "stage8_contrast_gate_pass_expected_age_strict.csv", index=False)

    report_lines = [
        "# Stage-8 Contrast CI Gate Report",
        "",
        "Strict gate definitions:",
        "- `pass_expected_age_contrast_strict`: min donor threshold + CI-supported expected-age contrast directions",
        "  - `old_minus_random > 0`",
        "  - `young_minus_random < 0`",
        "  - `old_minus_young > 0`",
        "- `pass_full_contrast_strict`: `pass_expected_age_contrast_strict` plus CI-supported old-probability contrasts",
        "  - `old_minus_random > 0`",
        "  - `young_minus_random < 0`",
        "- `pass_expected_age_strict`: `pass_expected_age_contrast_strict` plus CI-supported marginal directions",
        "  - `old_push > 0`",
        "  - `young_push < 0`",
        "- `pass_full_strict`: `pass_full_contrast_strict` plus the same marginal direction gates",
        "",
        "## Run Status",
        "",
        "```text",
        run_status.to_string(index=False),
        "```",
        "",
        "## Gate Summary By Run",
        "",
        "```text",
        gate_summary.to_string(index=False),
        "```",
        "",
        "## Full Strict Pass Rows",
        "",
        "```text",
        top_pass[
            group_cols
            + [
                "pass_expected_age_contrast_strict",
                "pass_full_contrast_strict",
                "pass_expected_age_strict",
                "pass_full_strict",
            ]
        ].to_string(index=False)
        if not top_pass.empty
        else "None",
        "```",
        "",
    ]
    (args.output_dir / "stage8_contrast_gate_report.md").write_text("\n".join(report_lines), encoding="utf-8")

    run_cfg = {
        "discover_root": str(args.discover_root),
        "run_dirs_count": int(len(run_dirs)),
        "bootstrap_iters": int(args.bootstrap_iters),
        "bootstrap_seed": int(args.bootstrap_seed),
        "min_donors": int(args.min_donors),
    }
    if str(args.run_dirs).strip():
        run_cfg["run_dirs"] = str(args.run_dirs)
    (args.output_dir / "run_config.json").write_text(json.dumps(run_cfg, indent=2), encoding="utf-8")
    print(f"[done] stage-8 contrast CI gate audit written to: {args.output_dir}")


if __name__ == "__main__":
    main()
