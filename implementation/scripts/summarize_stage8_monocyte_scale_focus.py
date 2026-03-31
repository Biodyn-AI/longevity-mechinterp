#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd


def _parse_scale_runs(text: str) -> List[Tuple[float, Path]]:
    out: List[Tuple[float, Path]] = []
    for item in [x.strip() for x in str(text).split(",") if x.strip()]:
        if "=" not in item:
            raise ValueError(f"Invalid mapping '{item}'. Use scale=/abs/path format.")
        scale_text, path_text = item.split("=", 1)
        out.append((float(scale_text.strip()), Path(path_text.strip()).expanduser().resolve()))
    if not out:
        raise ValueError("No scale runs provided.")
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


def _default_scale_runs(root: Path) -> str:
    base = root / "implementation" / "outputs"
    mapping = [
        (1.0, base / "stage8_monocyte_aida_v1_highpower_scale1_20260303"),
        (2.0, base / "stage8_monocyte_aida_v1_highpower_scale2_20260303"),
    ]
    return ",".join([f"{scale}={path}" for scale, path in mapping])


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Summarize focused Stage-8 monocyte intervention runs across intervention scales."
    )
    parser.add_argument("--scale-runs", type=str, default=_default_scale_runs(root))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "implementation" / "outputs" / "stage8_monocyte_aida_v1_scale_summary_20260303",
    )
    parser.add_argument("--dataset-id", type=str, default="aida_phase1_v1")
    parser.add_argument("--pathway", type=str, default="inflammation_nfkb")
    parser.add_argument("--cell-type", type=str, default="CD14-positive monocyte")
    parser.add_argument("--analysis-scope", type=str, default="within_cell_type")
    parser.add_argument("--bootstrap-iters", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260303)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    scale_runs = _parse_scale_runs(args.scale_runs)

    status_rows: List[Dict[str, object]] = []
    direction_rows: List[pd.DataFrame] = []
    bootstrap_rows: List[pd.DataFrame] = []
    contrast_rows: List[Dict[str, object]] = []
    rng = np.random.default_rng(int(args.bootstrap_seed))

    for scale, run_dir in scale_runs:
        direction_csv = run_dir / "stage5_cross_model_directionality.csv"
        donor_bootstrap_csv = run_dir / "donor_bootstrap_ci" / "stage5_donor_bootstrap_consensus.csv"
        donor_means_csv = run_dir / "donor_bootstrap_ci" / "stage5_donor_consensus_means.csv"

        has_direction = direction_csv.exists()
        has_bootstrap = donor_bootstrap_csv.exists()
        has_donor_means = donor_means_csv.exists()
        status_rows.append(
            {
                "scale": float(scale),
                "run_dir": str(run_dir),
                "has_directionality_csv": bool(has_direction),
                "has_bootstrap_consensus_csv": bool(has_bootstrap),
                "has_donor_consensus_means_csv": bool(has_donor_means),
                "status": "ok" if (has_direction and has_bootstrap and has_donor_means) else "missing_inputs",
            }
        )

        if has_direction:
            ddir = pd.read_csv(direction_csv)
            ddir = ddir[
                (ddir["dataset_id"].astype(str) == str(args.dataset_id))
                & (ddir["pathway"].astype(str) == str(args.pathway))
                & (ddir["cell_type"].astype(str) == str(args.cell_type))
            ].copy()
            if not ddir.empty:
                ddir.insert(0, "intervention_scale", float(scale))
                direction_rows.append(ddir)

        if has_bootstrap:
            db = pd.read_csv(donor_bootstrap_csv)
            db = db[
                (db["dataset_id"].astype(str) == str(args.dataset_id))
                & (db["pathway"].astype(str) == str(args.pathway))
                & (db["intervention"].astype(str).isin(["old_push", "random_push", "young_push"]))
            ].copy()
            if not db.empty:
                db.insert(0, "intervention_scale", float(scale))
                bootstrap_rows.append(db)

        if has_donor_means:
            dm = pd.read_csv(donor_means_csv)
            dm = dm[
                (dm["dataset_id"].astype(str) == str(args.dataset_id))
                & (dm["pathway"].astype(str) == str(args.pathway))
                & (dm["intervention"].astype(str).isin(["old_push", "random_push", "young_push"]))
            ].copy()
            if dm.empty:
                continue

            for model, dmm in dm.groupby("model", as_index=False):
                pivot = dmm.pivot_table(
                    index="donor_id",
                    columns="intervention",
                    values=["delta_expected_age_mean", "delta_old_prob_mean"],
                    aggfunc="mean",
                )
                for metric in ["delta_expected_age_mean", "delta_old_prob_mean"]:
                    for contrast, lhs, rhs in [
                        ("old_minus_random", "old_push", "random_push"),
                        ("young_minus_random", "young_push", "random_push"),
                        ("old_minus_young", "old_push", "young_push"),
                    ]:
                        key_lhs = (metric, lhs)
                        key_rhs = (metric, rhs)
                        if key_lhs not in pivot.columns or key_rhs not in pivot.columns:
                            continue
                        values = (pivot[key_lhs] - pivot[key_rhs]).dropna().to_numpy(dtype=np.float64)
                        mean, low, high = _bootstrap_ci(values, int(args.bootstrap_iters), rng)
                        contrast_rows.append(
                            {
                                "intervention_scale": float(scale),
                                "dataset_id": str(args.dataset_id),
                                "model": str(model),
                                "pathway": str(args.pathway),
                                "analysis_scope": str(args.analysis_scope),
                                "cell_type": str(args.cell_type),
                                "metric": str(metric),
                                "contrast": str(contrast),
                                "n_donors": int(values.size),
                                "mean_delta": mean,
                                "ci_low": low,
                                "ci_high": high,
                                "ci_flag": _ci_flag(low, high),
                            }
                        )

    status_df = pd.DataFrame(status_rows).sort_values("scale").reset_index(drop=True)
    status_df.to_csv(args.output_dir / "stage8_scale_focus_run_status.csv", index=False)

    direction_df = pd.concat(direction_rows, ignore_index=True) if direction_rows else pd.DataFrame()
    direction_df.to_csv(args.output_dir / "stage8_scale_focus_directionality.csv", index=False)

    bootstrap_df = pd.concat(bootstrap_rows, ignore_index=True) if bootstrap_rows else pd.DataFrame()
    bootstrap_df.to_csv(args.output_dir / "stage8_scale_focus_bootstrap_by_scale.csv", index=False)

    contrast_df = (
        pd.DataFrame(contrast_rows)
        .sort_values(["metric", "contrast", "intervention_scale", "model"])
        .reset_index(drop=True)
        if contrast_rows
        else pd.DataFrame()
    )
    contrast_df.to_csv(args.output_dir / "stage8_scale_focus_contrast_ci.csv", index=False)

    focus_view = contrast_df[
        (contrast_df["metric"].astype(str) == "delta_expected_age_mean")
        & (contrast_df["contrast"].astype(str).isin(["old_minus_random", "young_minus_random"]))
    ].copy()
    focus_view.to_csv(args.output_dir / "stage8_scale_focus_expected_age_key_contrasts.csv", index=False)

    lines = [
        "# Stage-8 Monocyte Scale-Focus Report",
        "",
        f"Dataset: `{args.dataset_id}`",
        f"Pathway: `{args.pathway}`",
        f"Analysis scope: `{args.analysis_scope}`",
        f"Cell type: `{args.cell_type}`",
        "",
        "## Run Status",
        "",
        "```text",
        status_df.to_string(index=False),
        "```",
        "",
    ]
    if not direction_df.empty:
        lines.extend(
            [
                "## Cross-Model Directionality",
                "",
                "```text",
                direction_df.to_string(index=False),
                "```",
                "",
            ]
        )
    if not focus_view.empty:
        lines.extend(
            [
                "## Key Expected-Age Contrasts (donor bootstrap)",
                "",
                "`old_minus_random > 0` and `young_minus_random < 0` indicate directional separation from random control.",
                "",
                "```text",
                focus_view.to_string(index=False),
                "```",
                "",
            ]
        )
    if not bootstrap_df.empty:
        lines.extend(
            [
                "## Raw Bootstrap Rows By Scale (old/random/young)",
                "",
                "```text",
                bootstrap_df.to_string(index=False),
                "```",
                "",
            ]
        )

    lines.extend(
        [
            "## Interpretation Notes",
            "",
            "- Contrast CIs are computed at donor level from donor-consensus means (not from split-level rows directly).",
            "- These focused outputs should be interpreted as branch-specific evidence, not broad cross-cohort replication.",
            "",
        ]
    )
    (args.output_dir / "stage8_scale_focus_report.md").write_text("\n".join(lines), encoding="utf-8")

    run_cfg = {
        "scale_runs": [{"scale": scale, "path": str(path)} for scale, path in scale_runs],
        "dataset_id": str(args.dataset_id),
        "pathway": str(args.pathway),
        "analysis_scope": str(args.analysis_scope),
        "cell_type": str(args.cell_type),
        "bootstrap_iters": int(args.bootstrap_iters),
        "bootstrap_seed": int(args.bootstrap_seed),
    }
    (args.output_dir / "run_config.json").write_text(json.dumps(run_cfg, indent=2), encoding="utf-8")
    print(f"[done] stage-8 focused scale summary written to: {args.output_dir}")


if __name__ == "__main__":
    main()
