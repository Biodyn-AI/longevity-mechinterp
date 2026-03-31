#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd


def _parse_seed_dirs(text: str) -> Dict[int, Path]:
    out: Dict[int, Path] = {}
    for item in [x.strip() for x in str(text).split(",") if x.strip()]:
        if "=" not in item:
            raise ValueError(f"Invalid seed mapping '{item}'. Use seed=/abs/path format.")
        seed_text, path_text = item.split("=", 1)
        out[int(seed_text.strip())] = Path(path_text.strip()).expanduser().resolve()
    if not out:
        raise ValueError("No seed directories provided.")
    return out


def _resolve_default_seed_dirs(root: Path) -> Dict[int, Path]:
    base = root / "implementation" / "outputs" / "stage5_donor_bootstrap_runs_20260303"
    return {
        42: base / "stage5_seed42_n1x15_donor",
        123: base / "stage5_seed123_n1x15_donor",
        314: base / "stage5_seed314_n1x15_donor",
    }


def _bootstrap_ci(
    values: np.ndarray,
    n_boot: int,
    rng: np.random.Generator,
) -> Dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    if arr.size == 1:
        x = float(arr[0])
        return {"mean": x, "ci_low": x, "ci_high": x}

    idx = rng.integers(0, arr.size, size=(int(n_boot), arr.size))
    boot_means = arr[idx].mean(axis=1)
    return {
        "mean": float(np.mean(arr)),
        "ci_low": float(np.quantile(boot_means, 0.025)),
        "ci_high": float(np.quantile(boot_means, 0.975)),
    }


def _ci_flag(ci_low: float, ci_high: float) -> str:
    if np.isfinite(ci_low) and np.isfinite(ci_high):
        if ci_low > 0:
            return "positive"
        if ci_high < 0:
            return "negative"
    return "uncertain"


def _bootstrap_table(
    df: pd.DataFrame,
    group_cols: Sequence[str],
    metric_cols: Sequence[str],
    n_boot: int,
    seed: int,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    grouped = df.groupby(list(group_cols), as_index=False)
    for _, grp in grouped:
        base: Dict[str, object] = {col: grp.iloc[0][col] for col in group_cols}
        base["n_donors"] = int(grp["donor_id"].nunique())
        local_rng = np.random.default_rng(int(seed) + int(base["n_donors"]) * 97)
        for metric in metric_cols:
            stats = _bootstrap_ci(grp[metric].to_numpy(dtype=np.float64), n_boot=n_boot, rng=local_rng)
            base[f"{metric}_mean"] = stats["mean"]
            base[f"{metric}_ci_low"] = stats["ci_low"]
            base[f"{metric}_ci_high"] = stats["ci_high"]
            base[f"{metric}_ci_flag"] = _ci_flag(stats["ci_low"], stats["ci_high"])
        rows.append(base)

    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    return out.sort_values(list(group_cols)).reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    default_seed_dirs = _resolve_default_seed_dirs(root)
    default_seed_text = ",".join([f"{k}={v}" for k, v in sorted(default_seed_dirs.items())])

    parser = argparse.ArgumentParser(description="Compute donor-bootstrap CIs from Stage-5 donor-level intervention outputs.")
    parser.add_argument("--seed-dirs", type=str, default=default_seed_text)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "implementation" / "outputs" / "stage5_donor_bootstrap_ci_20260303",
    )
    parser.add_argument("--pathway", type=str, default="inflammation_nfkb")
    parser.add_argument("--analysis-scope", type=str, default="global")
    parser.add_argument("--cell-type", type=str, default="__all__")
    parser.add_argument("--interventions", type=str, default="old_push,young_push,random_push,ablate")
    parser.add_argument("--bootstrap-iters", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260303)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    seed_dirs = _parse_seed_dirs(args.seed_dirs)
    interventions = [x.strip() for x in str(args.interventions).split(",") if x.strip()]
    metric_cols = [
        "delta_expected_age_mean",
        "delta_old_prob_mean",
        "delta_balanced_accuracy",
        "delta_pathway_activation_mean",
    ]

    run_status_rows: List[Dict[str, object]] = []
    donor_tables: List[pd.DataFrame] = []
    for seed, seed_dir in sorted(seed_dirs.items()):
        donor_csv = seed_dir / "stage5_intervention_donor_results.csv"
        status = donor_csv.exists()
        run_status_rows.append(
            {
                "seed": int(seed),
                "seed_dir": str(seed_dir),
                "donor_csv": str(donor_csv),
                "has_donor_csv": bool(status),
                "status": "ok" if status else "missing_donor_csv",
            }
        )
        if not status:
            continue

        df = pd.read_csv(donor_csv).copy()
        df.insert(0, "seed_run", int(seed))
        df = df[
            (df["analysis_scope"].astype(str) == str(args.analysis_scope))
            & (df["cell_type"].astype(str) == str(args.cell_type))
            & (df["pathway"].astype(str) == str(args.pathway))
            & (df["intervention"].astype(str).isin(set(interventions)))
        ].copy()
        if not df.empty:
            donor_tables.append(df)

    run_status_df = pd.DataFrame(run_status_rows).sort_values("seed").reset_index(drop=True)
    run_status_df.to_csv(args.output_dir / "stage5_donor_bootstrap_run_status.csv", index=False)

    if not donor_tables:
        report = [
            "# Stage-5 Donor Bootstrap Report",
            "",
            "No donor-level rows were available after filtering.",
            "",
            "## Run Status",
            "",
            "```text",
            run_status_df.to_string(index=False),
            "```",
            "",
        ]
        (args.output_dir / "stage5_donor_bootstrap_report.md").write_text("\n".join(report), encoding="utf-8")
        return

    donor_all = pd.concat(donor_tables, ignore_index=True)
    donor_all.to_csv(args.output_dir / "stage5_donor_rows_filtered.csv", index=False)

    # First collapse split-level rows into donor means within each seed run.
    donor_seed_means = (
        donor_all.groupby(
            ["seed_run", "dataset_id", "model", "pathway", "intervention", "donor_id"],
            as_index=False,
        )
        .agg(
            n_split_rows=("split_idx", "count"),
            delta_expected_age_mean=("delta_expected_age_mean", "mean"),
            delta_old_prob_mean=("delta_old_prob_mean", "mean"),
            delta_balanced_accuracy=("delta_balanced_accuracy", "mean"),
            delta_pathway_activation_mean=("delta_pathway_activation_mean", "mean"),
        )
        .sort_values(["dataset_id", "model", "intervention", "seed_run", "donor_id"])
        .reset_index(drop=True)
    )
    donor_seed_means.to_csv(args.output_dir / "stage5_donor_seed_means.csv", index=False)

    # Then collapse across seeds to avoid pseudo-replicating the same donor IDs across runs.
    donor_consensus_means = (
        donor_seed_means.groupby(
            ["dataset_id", "model", "pathway", "intervention", "donor_id"],
            as_index=False,
        )
        .agg(
            n_seed_runs=("seed_run", "nunique"),
            n_split_rows=("n_split_rows", "sum"),
            delta_expected_age_mean=("delta_expected_age_mean", "mean"),
            delta_old_prob_mean=("delta_old_prob_mean", "mean"),
            delta_balanced_accuracy=("delta_balanced_accuracy", "mean"),
            delta_pathway_activation_mean=("delta_pathway_activation_mean", "mean"),
        )
        .sort_values(["dataset_id", "model", "intervention", "donor_id"])
        .reset_index(drop=True)
    )
    donor_consensus_means.to_csv(args.output_dir / "stage5_donor_consensus_means.csv", index=False)

    by_seed_boot = _bootstrap_table(
        donor_seed_means.rename(columns={"seed_run": "seed"}),
        group_cols=["seed", "dataset_id", "model", "pathway", "intervention"],
        metric_cols=metric_cols,
        n_boot=args.bootstrap_iters,
        seed=args.bootstrap_seed,
    )
    by_seed_boot.to_csv(args.output_dir / "stage5_donor_bootstrap_by_seed.csv", index=False)

    consensus_boot = _bootstrap_table(
        donor_consensus_means,
        group_cols=["dataset_id", "model", "pathway", "intervention"],
        metric_cols=metric_cols,
        n_boot=args.bootstrap_iters,
        seed=args.bootstrap_seed,
    )
    consensus_boot.to_csv(args.output_dir / "stage5_donor_bootstrap_consensus.csv", index=False)

    report_lines = [
        "# Stage-5 Donor Bootstrap Report",
        "",
        "Donor-level bootstrap confidence intervals for Stage-5 intervention deltas.",
        "",
        "## Run Status",
        "",
        "```text",
        run_status_df.to_string(index=False),
        "```",
        "",
        "## Consensus Donor Bootstrap",
        "",
        "```text",
        consensus_boot.to_string(index=False),
        "```",
        "",
    ]
    (args.output_dir / "stage5_donor_bootstrap_report.md").write_text("\n".join(report_lines), encoding="utf-8")

    run_cfg = {
        "seed_dirs": {str(k): str(v) for k, v in sorted(seed_dirs.items())},
        "pathway": str(args.pathway),
        "analysis_scope": str(args.analysis_scope),
        "cell_type": str(args.cell_type),
        "interventions": interventions,
        "bootstrap_iters": int(args.bootstrap_iters),
        "bootstrap_seed": int(args.bootstrap_seed),
    }
    (args.output_dir / "run_config.json").write_text(json.dumps(run_cfg, indent=2), encoding="utf-8")

    print(f"[done] donor bootstrap outputs written to: {args.output_dir}")


if __name__ == "__main__":
    main()
