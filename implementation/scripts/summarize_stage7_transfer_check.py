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
    parser = argparse.ArgumentParser(description="Summarize AIDA -> external transfer consistency for intervention directions.")
    parser.add_argument(
        "--source-bootstrap-csv",
        type=Path,
        default=root / "implementation" / "outputs" / "stage5_donor_bootstrap_ci_20260303" / "stage5_donor_bootstrap_consensus.csv",
    )
    parser.add_argument(
        "--target-comparison-csv",
        type=Path,
        default=root / "implementation" / "outputs" / "stage6_inflammation_followup_20260303" / "external_relaxed_comparison_20260303.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "implementation" / "outputs" / "stage7_transfer_check_20260303",
    )
    parser.add_argument("--source-pathway", type=str, default="inflammation_nfkb")
    parser.add_argument("--source-datasets", type=str, default="aida_phase1_v1,aida_phase1_v2")
    return parser.parse_args()


def _sign_from_ci_flag(flag: str) -> int:
    t = str(flag).strip().lower()
    if t == "positive":
        return 1
    if t == "negative":
        return -1
    return 0


def _transfer_tier(row: pd.Series) -> str:
    src_sign = int(row.get("source_consensus_sign", 0))
    tgt_sign = int(row.get("target_ci_sign", 0))
    tgt_uncertain = bool(int(row.get("target_is_uncertain", 0)) == 1)
    n_pairs = int(row.get("target_n_cross_model_pairs", 0))
    if n_pairs <= 0:
        return "no_target_pairing"
    if src_sign != 0 and tgt_sign != 0 and src_sign == tgt_sign:
        return "directional_transfer_supported"
    if tgt_uncertain:
        return "target_uncertain"
    if src_sign != 0 and tgt_sign != 0 and src_sign != tgt_sign:
        return "directional_transfer_conflict"
    return "insufficient"


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    source_datasets = [x.strip() for x in str(args.source_datasets).split(",") if x.strip()]
    src = pd.read_csv(args.source_bootstrap_csv).copy()
    src = src[
        (src["pathway"].astype(str) == str(args.source_pathway))
        & (src["intervention"].astype(str) == "old_push")
        & (src["dataset_id"].astype(str).isin(set(source_datasets)))
    ].copy()

    if src.empty:
        raise RuntimeError("No source rows after filtering. Check source datasets/pathway/intervention.")

    src_model = (
        src.groupby("model", as_index=False)
        .agg(
            n_source_rows=("dataset_id", "count"),
            n_source_positive=("delta_expected_age_mean_ci_flag", lambda s: int((s.astype(str) == "positive").sum())),
            n_source_negative=("delta_expected_age_mean_ci_flag", lambda s: int((s.astype(str) == "negative").sum())),
            mean_source_expected_age=("delta_expected_age_mean_mean", "mean"),
        )
        .sort_values("model")
        .reset_index(drop=True)
    )
    src_model["source_consensus_sign"] = np.select(
        [
            src_model["n_source_positive"] > src_model["n_source_negative"],
            src_model["n_source_negative"] > src_model["n_source_positive"],
        ],
        [1, -1],
        default=0,
    ).astype(int)
    src_model["source_consensus_label"] = src_model["source_consensus_sign"].map({1: "positive", -1: "negative", 0: "uncertain"})
    src_model.to_csv(args.output_dir / "transfer_source_consensus.csv", index=False)

    tgt = pd.read_csv(args.target_comparison_csv).copy()
    needed = {"cohort", "pathway", "n_cross_model_pairs", "model", "old_push_expected_ci_flag"}
    missing = needed - set(tgt.columns)
    if missing:
        raise KeyError(f"Target comparison file missing required columns: {sorted(missing)}")

    rows: List[Dict[str, object]] = []
    for r in tgt.itertuples(index=False):
        model = str(r.model)
        src_sub = src_model[src_model["model"].astype(str) == model]
        if src_sub.empty:
            continue
        source_sign = int(src_sub.iloc[0]["source_consensus_sign"])
        target_flag = str(r.old_push_expected_ci_flag)
        target_sign = _sign_from_ci_flag(target_flag)
        out: Dict[str, object] = {
            "cohort": str(r.cohort),
            "pathway": str(r.pathway),
            "model": model,
            "target_n_cross_model_pairs": int(r.n_cross_model_pairs),
            "source_consensus_sign": int(source_sign),
            "source_consensus_label": str(src_sub.iloc[0]["source_consensus_label"]),
            "target_ci_flag": target_flag,
            "target_ci_sign": int(target_sign),
            "target_is_uncertain": int(target_sign == 0),
            "transfer_same_sign": bool(source_sign != 0 and target_sign != 0 and source_sign == target_sign),
        }
        rows.append(out)

    out_df = pd.DataFrame(rows)
    if out_df.empty:
        raise RuntimeError("No transfer rows were produced; verify model overlap between source and target files.")
    out_df["transfer_tier"] = out_df.apply(_transfer_tier, axis=1)
    tier_rank = {
        "directional_transfer_supported": 0,
        "target_uncertain": 1,
        "directional_transfer_conflict": 2,
        "no_target_pairing": 3,
        "insufficient": 4,
    }
    out_df["tier_rank"] = out_df["transfer_tier"].map(tier_rank).fillna(99).astype(int)
    out_df = out_df.sort_values(["tier_rank", "cohort", "model"]).reset_index(drop=True)
    out_df.to_csv(args.output_dir / "stage7_transfer_table.csv", index=False)

    summary = (
        out_df.groupby(["cohort", "transfer_tier"], as_index=False)
        .agg(n_rows=("model", "count"))
        .sort_values(["cohort", "transfer_tier"])
        .reset_index(drop=True)
    )
    summary.to_csv(args.output_dir / "stage7_transfer_summary.csv", index=False)

    lines = [
        "# Stage-7 Transfer Check",
        "",
        f"Source consensus pathway: `{args.source_pathway}` on datasets `{','.join(source_datasets)}`.",
        "",
        "## Source Consensus by Model",
        "",
        "```text",
        src_model.to_string(index=False),
        "```",
        "",
        "## Transfer Table",
        "",
        "```text",
        out_df.to_string(index=False),
        "```",
        "",
        "## Transfer Summary",
        "",
        "```text",
        summary.to_string(index=False),
        "```",
        "",
    ]
    (args.output_dir / "stage7_transfer_report.md").write_text("\n".join(lines), encoding="utf-8")

    run_cfg = {
        "source_bootstrap_csv": str(args.source_bootstrap_csv.resolve()),
        "target_comparison_csv": str(args.target_comparison_csv.resolve()),
        "source_pathway": str(args.source_pathway),
        "source_datasets": source_datasets,
    }
    (args.output_dir / "run_config.json").write_text(json.dumps(run_cfg, indent=2), encoding="utf-8")
    print(f"[done] transfer check written to: {args.output_dir}")


if __name__ == "__main__":
    main()
