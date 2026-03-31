#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from run_stage1_longevity_mechinterp import (
    _baseline_expression_representation,
    _group_split_scores,
    _load_subset_anndata,
    _manifold_metrics,
)


def _resolve_default_stage1_output(root: Path) -> Path:
    """
    Pick the most recent stage-1 output directory.

    We prefer contextual runs first because they contain the latest Geneformer
    extraction mode used in this project.
    """
    outputs_dir = root / "implementation" / "outputs"
    contextual = sorted(outputs_dir.glob("stage1_full_contextual_*"))
    if contextual:
        return contextual[-1]
    legacy = sorted(outputs_dir.glob("stage1_full_*"))
    if legacy:
        return legacy[-1]
    return outputs_dir / "stage1_full_contextual_20260303"


def _topk_representation_geometry(
    probe_df: pd.DataFrame,
    manifold_df: pd.DataFrame,
    top_k: int,
) -> pd.DataFrame:
    """Join top-k probe rows with manifold metrics for compact stage-2 summaries."""
    if probe_df.empty or manifold_df.empty:
        return pd.DataFrame()

    top = probe_df.sort_values("balanced_accuracy_mean", ascending=False).head(top_k).copy()
    merged = top.merge(
        manifold_df[
            [
                "representation",
                "participation_ratio",
                "pc1_explained_ratio",
                "pc5_cumulative_explained_ratio",
                "pc1_age_corr",
                "age_label_silhouette",
            ]
        ],
        on="representation",
        how="left",
        validate="one_to_one",
    )
    merged["pc1_age_corr_abs"] = merged["pc1_age_corr"].abs()
    return merged


def _within_celltype_expr_controls(
    dataset_path: Path,
    sampled_meta: pd.DataFrame,
    expr_components: int,
    group_splits: int,
    min_cells_per_celltype: int,
    min_donors_per_celltype: int,
    min_age_classes_per_celltype: int,
    max_celltypes_per_dataset: int,
    seed: int,
) -> pd.DataFrame:
    """
    Run donor-held-out age probes and manifold checks inside each major cell type.

    This is a practical confound stress test: if age signal disappears completely
    once we condition on cell type, we treat stage-1 geometry as composition-driven.
    """
    if sampled_meta.empty:
        return pd.DataFrame()

    adata = _load_subset_anndata(dataset_path, sampled_meta["obs_row"].to_numpy(dtype=np.int64))
    if adata.n_obs != sampled_meta.shape[0]:
        raise RuntimeError("Stage-2 subset mismatch between AnnData and sampled metadata")

    rows: List[Dict[str, Any]] = []
    counts = sampled_meta["cell_type"].astype(str).value_counts()
    selected_celltypes = counts.index[:max_celltypes_per_dataset].tolist()

    for cell_type in selected_celltypes:
        mask = sampled_meta["cell_type"].astype(str).to_numpy() == cell_type
        n_cells = int(mask.sum())
        meta_ct = sampled_meta.loc[mask].reset_index(drop=True)
        n_donors = int(meta_ct["donor_id"].nunique())
        n_age_classes = int(meta_ct["age_label"].nunique())

        row: Dict[str, Any] = {
            "cell_type": str(cell_type),
            "n_cells": n_cells,
            "n_donors": n_donors,
            "n_age_classes": n_age_classes,
            "status": "skipped",
            "skip_reason": "",
        }
        if n_cells < min_cells_per_celltype:
            row["skip_reason"] = "too_few_cells"
            rows.append(row)
            continue
        if n_donors < min_donors_per_celltype:
            row["skip_reason"] = "too_few_donors"
            rows.append(row)
            continue
        if n_age_classes < min_age_classes_per_celltype:
            row["skip_reason"] = "too_few_age_classes"
            rows.append(row)
            continue

        idx = np.where(mask)[0]
        adata_ct = adata[idx].copy()
        X = _baseline_expression_representation(adata_ct, n_components=expr_components, seed=seed)
        y_codes, _ = pd.factorize(meta_ct["age_label"])
        groups = meta_ct["donor_id"].to_numpy(dtype=object)

        probe = _group_split_scores(X=X, y=y_codes, groups=groups, n_splits=group_splits, seed=seed)
        manifold = _manifold_metrics(
            X=X,
            y_label=meta_ct["age_label"].to_numpy(dtype=object),
            age_numeric=meta_ct["age_numeric"].to_numpy(dtype=np.float64),
            seed=seed,
        )

        row.update(probe)
        row.update(manifold)
        row["status"] = "ok"
        rows.append(row)

    df = pd.DataFrame(rows)
    if "balanced_accuracy_mean" in df.columns:
        return df.sort_values("balanced_accuracy_mean", ascending=False, na_position="last")
    return df


def _write_stage2_report(
    aggregate_df: pd.DataFrame,
    topk_geometry_df: pd.DataFrame,
    within_celltype_df: pd.DataFrame,
    output_md: Path,
    g2_pc1_abs_threshold: float,
    g2_silhouette_threshold: float,
) -> None:
    lines: List[str] = [
        "# Stage-2 Manifold Robustness Report",
        "",
        "This stage summarizes top representation geometry and runs within-cell-type confound controls.",
        "",
    ]

    if aggregate_df.empty:
        lines.append("No completed datasets were available from stage-1 outputs.")
        output_md.write_text("\n".join(lines), encoding="utf-8")
        return

    lines.extend(
        [
            "## Dataset Summary",
            "",
            "```text",
            aggregate_df[
                [
                    "dataset_id",
                    "best_representation",
                    "best_balanced_accuracy_mean",
                    "topk_max_pc1_abs_corr",
                    "topk_max_silhouette",
                    "g2_geometry_pass",
                    "n_within_celltype_tests_ok",
                    "within_celltype_best_bacc",
                ]
            ].to_string(index=False),
            "```",
            "",
        ]
    )

    g2_hits = int(aggregate_df["g2_geometry_pass"].sum())
    total = int(aggregate_df.shape[0])
    lines.extend(
        [
            "## G2 Heuristic",
            "",
            f"- Thresholds: `|pc1_age_corr| >= {g2_pc1_abs_threshold:.2f}` and `silhouette >= {g2_silhouette_threshold:.2f}` on top-k representations",
            f"- Datasets passing geometry gate: `{g2_hits} / {total}`",
            "",
        ]
    )

    if not topk_geometry_df.empty:
        lines.extend(
            [
                "## Top-K Representation Geometry",
                "",
                "```text",
                topk_geometry_df[
                    [
                        "dataset_id",
                        "representation",
                        "balanced_accuracy_mean",
                        "pc1_age_corr",
                        "pc1_age_corr_abs",
                        "age_label_silhouette",
                        "participation_ratio",
                    ]
                ]
                .sort_values(["dataset_id", "balanced_accuracy_mean"], ascending=[True, False])
                .to_string(index=False),
                "```",
                "",
            ]
        )

    if not within_celltype_df.empty:
        ok = within_celltype_df[within_celltype_df["status"] == "ok"].copy()
        if not ok.empty:
            lines.extend(
                [
                    "## Within-Cell-Type Controls (expr_svd)",
                    "",
                    "```text",
                    ok[
                        [
                            "dataset_id",
                            "cell_type",
                            "n_cells",
                            "n_donors",
                            "n_age_classes",
                            "balanced_accuracy_mean",
                            "pc1_age_corr",
                            "age_label_silhouette",
                        ]
                    ]
                    .sort_values(["dataset_id", "balanced_accuracy_mean"], ascending=[True, False])
                    .groupby("dataset_id")
                    .head(5)
                    .to_string(index=False),
                    "```",
                    "",
                ]
            )

    lines.extend(
        [
            "## Interpretation",
            "",
            "- Stage-2 output is exploratory and intended to test geometry robustness before SAE/intervention work.",
            "- If within-cell-type signal collapses, prioritize stronger composition controls before mechanistic claims.",
            "",
        ]
    )

    output_md.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Stage-2 manifold robustness analysis based on completed stage-1 outputs."
    )
    parser.add_argument(
        "--stage1-output-dir",
        type=Path,
        default=_resolve_default_stage1_output(root),
        help="Path to a completed stage-1 output directory (contains stage1_run_aggregate.csv).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "implementation" / "outputs" / "stage2_manifold_robustness_20260303",
    )
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--expr-components", type=int, default=64)
    parser.add_argument("--group-splits", type=int, default=5)
    parser.add_argument("--min-cells-per-celltype", type=int, default=120)
    parser.add_argument("--min-donors-per-celltype", type=int, default=20)
    parser.add_argument("--min-age-classes-per-celltype", type=int, default=3)
    parser.add_argument("--max-celltypes-per-dataset", type=int, default=20)
    parser.add_argument("--g2-pc1-abs-threshold", type=float, default=0.10)
    parser.add_argument("--g2-silhouette-threshold", type=float, default=0.00)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    stage1_agg_path = args.stage1_output_dir / "stage1_run_aggregate.csv"
    if not stage1_agg_path.exists():
        raise FileNotFoundError(f"Missing stage-1 aggregate file: {stage1_agg_path}")

    stage1_agg = pd.read_csv(stage1_agg_path)
    ok = stage1_agg[stage1_agg["status"] == "ok"].copy()
    if ok.empty:
        raise RuntimeError("No successful datasets in stage-1 aggregate")

    aggregate_rows: List[Dict[str, Any]] = []
    topk_rows: List[pd.DataFrame] = []
    celltype_rows: List[pd.DataFrame] = []

    for ds in ok.itertuples(index=False):
        dataset_id = str(ds.dataset_id)
        dataset_path = Path(str(ds.dataset_path))
        ds_in = args.stage1_output_dir / dataset_id
        ds_out = args.output_dir / dataset_id
        ds_out.mkdir(parents=True, exist_ok=True)

        probe_path = ds_in / "probe_metrics.csv"
        manifold_path = ds_in / "manifold_metrics.csv"
        sampled_meta_path = ds_in / "sampled_obs.csv"
        if not probe_path.exists() or not manifold_path.exists() or not sampled_meta_path.exists():
            raise FileNotFoundError(
                f"Missing required stage-1 files for {dataset_id}: {probe_path}, {manifold_path}, {sampled_meta_path}"
            )

        probe_df = pd.read_csv(probe_path)
        manifold_df = pd.read_csv(manifold_path)
        sampled_meta = pd.read_csv(sampled_meta_path)

        topk_df = _topk_representation_geometry(probe_df=probe_df, manifold_df=manifold_df, top_k=args.top_k)
        if not topk_df.empty:
            topk_df.insert(0, "dataset_id", dataset_id)
            topk_df.to_csv(ds_out / "topk_representation_geometry.csv", index=False)
            topk_rows.append(topk_df)

        within_df = _within_celltype_expr_controls(
            dataset_path=dataset_path,
            sampled_meta=sampled_meta,
            expr_components=args.expr_components,
            group_splits=args.group_splits,
            min_cells_per_celltype=args.min_cells_per_celltype,
            min_donors_per_celltype=args.min_donors_per_celltype,
            min_age_classes_per_celltype=args.min_age_classes_per_celltype,
            max_celltypes_per_dataset=args.max_celltypes_per_dataset,
            seed=args.seed,
        )
        if not within_df.empty:
            within_df.insert(0, "dataset_id", dataset_id)
            within_df.to_csv(ds_out / "within_celltype_expr_controls.csv", index=False)
            celltype_rows.append(within_df)

        topk_max_pc1 = float(topk_df["pc1_age_corr_abs"].max()) if not topk_df.empty else float("nan")
        topk_max_sil = float(topk_df["age_label_silhouette"].max()) if not topk_df.empty else float("nan")
        g2_pass = bool(
            np.isfinite(topk_max_pc1)
            and np.isfinite(topk_max_sil)
            and topk_max_pc1 >= args.g2_pc1_abs_threshold
            and topk_max_sil >= args.g2_silhouette_threshold
        )

        if within_df.empty:
            n_within_ok = 0
            within_best_bacc = float("nan")
        else:
            within_ok = within_df[within_df["status"] == "ok"]
            n_within_ok = int(within_ok.shape[0])
            within_best_bacc = (
                float(within_ok["balanced_accuracy_mean"].max()) if not within_ok.empty else float("nan")
            )

        aggregate_rows.append(
            {
                "dataset_id": dataset_id,
                "dataset_path": str(dataset_path),
                "best_representation": str(probe_df.iloc[0]["representation"]) if not probe_df.empty else "",
                "best_balanced_accuracy_mean": float(probe_df.iloc[0]["balanced_accuracy_mean"])
                if not probe_df.empty
                else float("nan"),
                "topk_max_pc1_abs_corr": topk_max_pc1,
                "topk_max_silhouette": topk_max_sil,
                "g2_geometry_pass": g2_pass,
                "n_within_celltype_tests_ok": n_within_ok,
                "within_celltype_best_bacc": within_best_bacc,
            }
        )

        ds_summary = {
            "dataset_id": dataset_id,
            "dataset_path": str(dataset_path),
            "n_sampled_cells": int(sampled_meta.shape[0]),
            "n_sampled_donors": int(sampled_meta["donor_id"].nunique()),
            "n_sampled_cell_types": int(sampled_meta["cell_type"].nunique()),
            "n_within_celltype_tests_ok": n_within_ok,
            "g2_geometry_pass": g2_pass,
        }
        (ds_out / "dataset_stage2_summary.json").write_text(json.dumps(ds_summary, indent=2), encoding="utf-8")

        print(
            f"[ok] {dataset_id} | g2_pass={g2_pass} "
            f"| topk_max_pc1_abs={topk_max_pc1:.4f} | within_ok={n_within_ok}"
        )

    aggregate_df = pd.DataFrame(aggregate_rows).sort_values("dataset_id")
    topk_geometry_df = pd.concat(topk_rows, ignore_index=True) if topk_rows else pd.DataFrame()
    within_celltype_df = pd.concat(celltype_rows, ignore_index=True) if celltype_rows else pd.DataFrame()

    aggregate_path = args.output_dir / "stage2_run_aggregate.csv"
    topk_path = args.output_dir / "stage2_topk_geometry.csv"
    within_path = args.output_dir / "stage2_within_celltype_expr_controls.csv"
    report_path = args.output_dir / "stage2_manifold_robustness_report.md"

    aggregate_df.to_csv(aggregate_path, index=False)
    topk_geometry_df.to_csv(topk_path, index=False)
    within_celltype_df.to_csv(within_path, index=False)
    _write_stage2_report(
        aggregate_df=aggregate_df,
        topk_geometry_df=topk_geometry_df,
        within_celltype_df=within_celltype_df,
        output_md=report_path,
        g2_pc1_abs_threshold=args.g2_pc1_abs_threshold,
        g2_silhouette_threshold=args.g2_silhouette_threshold,
    )

    print("[done] stage2 outputs")
    print(f"  - {aggregate_path}")
    print(f"  - {topk_path}")
    print(f"  - {within_path}")
    print(f"  - {report_path}")


if __name__ == "__main__":
    main()
