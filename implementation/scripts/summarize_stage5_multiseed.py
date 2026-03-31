#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def _parse_seed_dirs(text: str) -> Dict[int, Path]:
    pairs = [x.strip() for x in str(text).split(",") if x.strip()]
    out: Dict[int, Path] = {}
    for item in pairs:
        if "=" not in item:
            raise ValueError(f"Invalid seed mapping '{item}'. Expected format seed=/path/to/dir")
        seed_text, path_text = item.split("=", 1)
        out[int(seed_text.strip())] = Path(path_text.strip()).expanduser().resolve()
    if not out:
        raise ValueError("No seed directories provided")
    return out


def _resolve_default_seed_dirs(root: Path) -> Dict[int, Path]:
    return {
        42: (root / "implementation" / "outputs" / "stage5_intervention_validation_20260303_v2").resolve(),
        123: (
            root
            / "implementation"
            / "outputs"
            / "stage34_multiseed_hardening_20260303_v2"
            / "stage5_seed123_n1x15"
        ).resolve(),
        314: (
            root
            / "implementation"
            / "outputs"
            / "stage34_multiseed_hardening_20260303_v2"
            / "stage5_seed314_n1x15"
        ).resolve(),
    }


def _directionality_tier(row: pd.Series) -> str:
    if (
        row["n_seeds_available"] >= 3
        and row["frac_scgpt_pattern_ok"] >= 2.0 / 3.0
        and row["frac_geneformer_pattern_ok"] >= 2.0 / 3.0
        and row["frac_sign_agree_expected_age"] >= 2.0 / 3.0
        and row["frac_sign_agree_old_prob"] >= 2.0 / 3.0
    ):
        return "high"
    if (
        row["n_seeds_available"] >= 2
        and row["frac_sign_agree_expected_age"] >= 2.0 / 3.0
        and row["frac_both_models_stronger_than_random"] >= 2.0 / 3.0
    ):
        return "medium"
    return "low"


def _build_directionality_consensus(seed_level: pd.DataFrame, n_seeds_total: int) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for (dataset_id, pathway), grp in seed_level.groupby(["dataset_id", "pathway"], as_index=False):
        grp = grp.copy()
        row = {
            "dataset_id": str(dataset_id),
            "pathway": str(pathway),
            "n_seeds_total": int(n_seeds_total),
            "n_seeds_available": int(grp["seed"].nunique()),
            "n_cross_model_pairs_median": float(grp["n_cross_model_pairs"].median()),
            "n_cross_model_pairs_max": int(grp["n_cross_model_pairs"].max()),
            "mean_scgpt_old_push_delta_expected_age": float(grp["scgpt_old_push_delta_expected_age"].mean()),
            "mean_geneformer_old_push_delta_expected_age": float(
                grp["geneformer_old_push_delta_expected_age"].mean()
            ),
            "mean_scgpt_old_push_delta_old_prob": float(grp["scgpt_old_push_delta_old_prob"].mean()),
            "mean_geneformer_old_push_delta_old_prob": float(grp["geneformer_old_push_delta_old_prob"].mean()),
            "frac_sign_agree_expected_age": float(grp["sign_agree_expected_age"].astype(bool).mean()),
            "frac_sign_agree_old_prob": float(grp["sign_agree_old_prob"].astype(bool).mean()),
            "frac_scgpt_pattern_ok": float(grp["scgpt_directional_pattern_ok"].astype(bool).mean()),
            "frac_geneformer_pattern_ok": float(grp["geneformer_directional_pattern_ok"].astype(bool).mean()),
            "frac_both_models_stronger_than_random": float(grp["both_models_stronger_than_random"].astype(bool).mean()),
        }
        rows.append(row)

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows)
    out["replication_tier"] = out.apply(_directionality_tier, axis=1)
    tier_rank = {"high": 0, "medium": 1, "low": 2}
    out["tier_rank"] = out["replication_tier"].map(tier_rank).fillna(3).astype(int)
    out = out.sort_values(
        ["tier_rank", "frac_sign_agree_expected_age", "frac_both_models_stronger_than_random"],
        ascending=[True, False, False],
    ).drop(columns=["tier_rank"])
    return out.reset_index(drop=True)


def _build_effect_consensus(summary_seed_level: pd.DataFrame) -> pd.DataFrame:
    keys = ["dataset_id", "pathway", "model", "intervention"]
    metrics = [
        "mean_delta_balanced_accuracy",
        "mean_delta_expected_age",
        "mean_delta_old_prob",
        "mean_delta_pathway_activation",
    ]
    agg = (
        summary_seed_level.groupby(keys, as_index=False)
        .agg(
            n_seeds_available=("seed", "nunique"),
            mean_delta_balanced_accuracy_avg=("mean_delta_balanced_accuracy", "mean"),
            mean_delta_balanced_accuracy_std_across_seeds=("mean_delta_balanced_accuracy", "std"),
            mean_delta_expected_age_avg=("mean_delta_expected_age", "mean"),
            mean_delta_expected_age_std_across_seeds=("mean_delta_expected_age", "std"),
            mean_delta_old_prob_avg=("mean_delta_old_prob", "mean"),
            mean_delta_old_prob_std_across_seeds=("mean_delta_old_prob", "std"),
            mean_delta_pathway_activation_avg=("mean_delta_pathway_activation", "mean"),
            mean_delta_pathway_activation_std_across_seeds=("mean_delta_pathway_activation", "std"),
        )
        .sort_values(keys)
        .reset_index(drop=True)
    )
    # Explicitly keep these metrics present for readability and downstream checks.
    _ = metrics
    return agg


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    defaults = _resolve_default_seed_dirs(root)
    default_seed_dirs = ",".join([f"{seed}={path}" for seed, path in sorted(defaults.items())])

    parser = argparse.ArgumentParser(description="Aggregate Stage-5 intervention outputs across seeds.")
    parser.add_argument("--seed-dirs", type=str, default=default_seed_dirs)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "implementation" / "outputs" / "stage5_multiseed_consensus_20260303",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    seed_dirs = _parse_seed_dirs(args.seed_dirs)
    status_rows: List[Dict[str, object]] = []
    direction_rows: List[pd.DataFrame] = []
    summary_rows: List[pd.DataFrame] = []

    for seed, seed_dir in sorted(seed_dirs.items()):
        direction_csv = seed_dir / "stage5_cross_model_directionality.csv"
        summary_csv = seed_dir / "stage5_intervention_summary.csv"
        ok = direction_csv.exists() and summary_csv.exists()
        status_rows.append(
            {
                "seed": int(seed),
                "seed_dir": str(seed_dir),
                "has_directionality_csv": bool(direction_csv.exists()),
                "has_summary_csv": bool(summary_csv.exists()),
                "status": "ok" if ok else "missing_files",
            }
        )
        if not ok:
            continue

        direction_df = pd.read_csv(direction_csv).copy()
        direction_df.insert(0, "seed", int(seed))
        direction_rows.append(direction_df)

        summary_df = pd.read_csv(summary_csv).copy()
        summary_df.insert(0, "seed", int(seed))
        summary_rows.append(summary_df)

    status_df = pd.DataFrame(status_rows).sort_values("seed").reset_index(drop=True)
    status_df.to_csv(args.output_dir / "stage5_multiseed_run_status.csv", index=False)

    if not direction_rows or not summary_rows:
        report = [
            "# Stage-5 Multi-Seed Consensus Report",
            "",
            "No complete seed outputs found for aggregation.",
            "",
            "## Run Status",
            "",
            "```text",
            status_df.to_string(index=False),
            "```",
            "",
        ]
        (args.output_dir / "stage5_multiseed_consensus_report.md").write_text("\n".join(report), encoding="utf-8")
        return

    direction_seed = pd.concat(direction_rows, ignore_index=True)
    direction_seed.to_csv(args.output_dir / "stage5_multiseed_directionality_seed_level.csv", index=False)

    summary_seed = pd.concat(summary_rows, ignore_index=True)
    summary_seed.to_csv(args.output_dir / "stage5_multiseed_effects_seed_level.csv", index=False)

    direction_consensus = _build_directionality_consensus(direction_seed, n_seeds_total=len(seed_dirs))
    direction_consensus.to_csv(args.output_dir / "stage5_multiseed_directionality_consensus.csv", index=False)

    effects_consensus = _build_effect_consensus(summary_seed)
    effects_consensus.to_csv(args.output_dir / "stage5_multiseed_effects_consensus.csv", index=False)

    report: List[str] = [
        "# Stage-5 Multi-Seed Consensus Report",
        "",
        "Aggregates intervention directionality/effect metrics across available Stage-5 seed runs.",
        "",
        "## Run Status",
        "",
        "```text",
        status_df.to_string(index=False),
        "```",
        "",
        "## Directionality Consensus",
        "",
        "```text",
        direction_consensus.to_string(index=False),
        "```",
        "",
        "## Effect Consensus",
        "",
        "```text",
        effects_consensus.to_string(index=False),
        "```",
        "",
    ]
    (args.output_dir / "stage5_multiseed_consensus_report.md").write_text("\n".join(report), encoding="utf-8")

    run_config = {
        "seed_dirs": {str(k): str(v) for k, v in sorted(seed_dirs.items())},
        "output_dir": str(args.output_dir.resolve()),
        "n_seeds_total": int(len(seed_dirs)),
    }
    (args.output_dir / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")

    print(f"[done] Stage-5 multiseed consensus written to: {args.output_dir}")


if __name__ == "__main__":
    main()
