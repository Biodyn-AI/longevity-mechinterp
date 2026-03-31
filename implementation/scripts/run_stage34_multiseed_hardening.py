#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


def _parse_int_csv(text: str) -> List[int]:
    return [int(x.strip()) for x in str(text).split(",") if x.strip()]


def _parse_csv_list(text: str) -> List[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


def _resolve_default_output_dir(root: Path, prefix: str, fallback_suffix: str) -> Path:
    outputs = root / "implementation" / "outputs"
    candidates = sorted(outputs.glob(f"{prefix}_*"))
    if candidates:
        return candidates[-1]
    return outputs / f"{prefix}_{fallback_suffix}"


def _run_cmd(cmd: Sequence[str]) -> None:
    print("[cmd]", " ".join(cmd))
    subprocess.run(list(cmd), check=True)


def _ensure_reused_seed42(
    seed: int,
    output_root: Path,
    reuse_seed42: bool,
    baseline_scgpt: Path,
    baseline_geneformer: Path,
    baseline_stage4: Path,
) -> None:
    if not reuse_seed42 or int(seed) != 42:
        return

    targets = [
        (output_root / f"stage3_scgpt_seed{seed}", baseline_scgpt),
        (output_root / f"stage3_geneformer_seed{seed}", baseline_geneformer),
        (output_root / f"stage4_seed{seed}", baseline_stage4),
    ]
    for target, source in targets:
        if target.exists():
            continue
        if not source.exists():
            raise FileNotFoundError(f"Baseline source missing for seed-42 reuse: {source}")
        target.symlink_to(source.resolve())
        print(f"[reuse] seed=42 linked {target} -> {source}")


def _build_stage3_scgpt_cmd(
    python_exec: str,
    stage3_script: Path,
    stage1_output_dir: Path,
    output_dir: Path,
    seed: int,
    dataset_ids: Optional[Sequence[str]],
    max_datasets: int,
    max_cells_per_dataset: int,
    scgpt_layers: str,
    scgpt_sae_epochs: int,
    scgpt_perm_iters: int,
    device: str,
) -> List[str]:
    cmd = [
        python_exec,
        str(stage3_script),
        "--stage1-output-dir",
        str(stage1_output_dir),
        "--output-dir",
        str(output_dir),
        "--max-datasets",
        str(max_datasets),
        "--max-cells-per-dataset",
        str(max_cells_per_dataset),
        "--scgpt-layers",
        str(scgpt_layers),
        "--representations",
        "scgpt_layer_09",
        "--sae-epochs",
        str(scgpt_sae_epochs),
        "--perm-iters",
        str(scgpt_perm_iters),
        "--device",
        str(device),
        "--seed",
        str(seed),
    ]
    if dataset_ids:
        cmd.extend(["--dataset-ids", ",".join(dataset_ids)])
    return cmd


def _build_stage3_geneformer_cmd(
    python_exec: str,
    stage3_script: Path,
    stage1_output_dir: Path,
    output_dir: Path,
    seed: int,
    dataset_ids: Optional[Sequence[str]],
    max_datasets: int,
    max_cells_per_dataset: int,
    geneformer_max_genes: int,
    geneformer_batch_size: int,
    geneformer_sae_latent_dim: int,
    geneformer_sae_epochs: int,
    geneformer_perm_iters: int,
    device: str,
) -> List[str]:
    cmd = [
        python_exec,
        str(stage3_script),
        "--stage1-output-dir",
        str(stage1_output_dir),
        "--output-dir",
        str(output_dir),
        "--max-datasets",
        str(max_datasets),
        "--max-cells-per-dataset",
        str(max_cells_per_dataset),
        "--representations",
        "geneformer_contextual",
        "--geneformer-mode",
        "contextual",
        "--geneformer-max-genes",
        str(geneformer_max_genes),
        "--geneformer-batch-size",
        str(geneformer_batch_size),
        "--sae-latent-dim",
        str(geneformer_sae_latent_dim),
        "--sae-epochs",
        str(geneformer_sae_epochs),
        "--perm-iters",
        str(geneformer_perm_iters),
        "--device",
        str(device),
        "--seed",
        str(seed),
    ]
    if dataset_ids:
        cmd.extend(["--dataset-ids", ",".join(dataset_ids)])
    return cmd


def _build_stage4_cmd(
    python_exec: str,
    stage4_script: Path,
    stage3_scgpt_dir: Path,
    stage3_geneformer_dir: Path,
    output_dir: Path,
    seed: int,
    dataset_ids: Optional[Sequence[str]],
    device: str,
) -> List[str]:
    cmd = [
        python_exec,
        str(stage4_script),
        "--stage3-scgpt-dir",
        str(stage3_scgpt_dir),
        "--stage3-geneformer-dir",
        str(stage3_geneformer_dir),
        "--output-dir",
        str(output_dir),
        "--device",
        str(device),
        "--seed",
        str(seed),
    ]
    if dataset_ids:
        cmd.extend(["--dataset-ids", ",".join(dataset_ids)])
    return cmd


def _read_stage3_aggregate(path: Path, seed: int, model: str) -> pd.DataFrame:
    df = pd.read_csv(path).copy()
    df.insert(0, "seed", int(seed))
    df.insert(1, "model", str(model))
    return df


def _read_stage4_consensus(path: Path, seed: int) -> pd.DataFrame:
    df = pd.read_csv(path).copy()
    for col in ["n_scgpt_features", "n_geneformer_features", "n_cross_model_pairs"]:
        if col not in df.columns:
            df[col] = 0
    df["n_scgpt_features"] = pd.to_numeric(df["n_scgpt_features"], errors="coerce").fillna(0).astype(int)
    df["n_geneformer_features"] = pd.to_numeric(df["n_geneformer_features"], errors="coerce").fillna(0).astype(int)
    df["n_cross_model_pairs"] = pd.to_numeric(df["n_cross_model_pairs"], errors="coerce").fillna(0).astype(int)
    if "best_convergence_score" not in df.columns:
        df["best_convergence_score"] = np.nan
    df.insert(0, "seed", int(seed))
    return df


def _confidence_tier(
    seed_replication_rate: float,
    both_model_rate: float,
    median_cross_pairs: float,
) -> str:
    if seed_replication_rate >= 0.67 and both_model_rate >= 0.67 and median_cross_pairs >= 25:
        return "high"
    if seed_replication_rate >= 0.34 and both_model_rate >= 0.34 and median_cross_pairs >= 5:
        return "medium"
    return "low"


def _build_seed_level_table(
    consensus_all: pd.DataFrame,
    seeds: Sequence[int],
) -> pd.DataFrame:
    if consensus_all.empty:
        return pd.DataFrame()

    all_keys = sorted(set(zip(consensus_all["dataset_id"].astype(str), consensus_all["pathway"].astype(str))))
    lookup: Dict[Tuple[int, str, str], Dict[str, Any]] = {}
    for row in consensus_all.itertuples(index=False):
        key = (int(row.seed), str(row.dataset_id), str(row.pathway))
        lookup[key] = {
            "n_scgpt_features": int(row.n_scgpt_features),
            "n_geneformer_features": int(row.n_geneformer_features),
            "n_cross_model_pairs": int(row.n_cross_model_pairs),
            "best_convergence_score": float(row.best_convergence_score)
            if pd.notna(row.best_convergence_score)
            else float("nan"),
        }

    rows: List[Dict[str, Any]] = []
    for dataset_id, pathway in all_keys:
        for seed in seeds:
            item = lookup.get((int(seed), dataset_id, pathway))
            if item is None:
                rows.append(
                    {
                        "seed": int(seed),
                        "dataset_id": dataset_id,
                        "pathway": pathway,
                        "n_scgpt_features": 0,
                        "n_geneformer_features": 0,
                        "n_cross_model_pairs": 0,
                        "best_convergence_score": float("nan"),
                    }
                )
            else:
                rows.append(
                    {
                        "seed": int(seed),
                        "dataset_id": dataset_id,
                        "pathway": pathway,
                        **item,
                    }
                )
    return pd.DataFrame(rows).sort_values(["dataset_id", "pathway", "seed"]).reset_index(drop=True)


def _build_multiseed_summary(seed_level: pd.DataFrame, n_seeds: int) -> pd.DataFrame:
    if seed_level.empty:
        return pd.DataFrame()

    rows: List[Dict[str, Any]] = []
    for (dataset_id, pathway), grp in seed_level.groupby(["dataset_id", "pathway"], as_index=False):
        grp = grp.copy()
        seeds_with_pairs = int((grp["n_cross_model_pairs"] > 0).sum())
        seed_replication_rate = float(seeds_with_pairs / max(n_seeds, 1))

        both_models = (grp["n_scgpt_features"] > 0) & (grp["n_geneformer_features"] > 0)
        seeds_with_both_models = int(both_models.sum())
        both_model_rate = float(seeds_with_both_models / max(n_seeds, 1))

        median_cross_pairs = float(grp["n_cross_model_pairs"].median())
        max_cross_pairs = int(grp["n_cross_model_pairs"].max())
        median_best_score = float(grp["best_convergence_score"].median(skipna=True))
        max_best_score = float(grp["best_convergence_score"].max(skipna=True))
        mean_scgpt_features = float(grp["n_scgpt_features"].mean())
        mean_geneformer_features = float(grp["n_geneformer_features"].mean())

        tier = _confidence_tier(
            seed_replication_rate=seed_replication_rate,
            both_model_rate=both_model_rate,
            median_cross_pairs=median_cross_pairs,
        )
        rows.append(
            {
                "dataset_id": str(dataset_id),
                "pathway": str(pathway),
                "n_seeds": int(n_seeds),
                "seeds_with_pairs": int(seeds_with_pairs),
                "seed_replication_rate": seed_replication_rate,
                "seeds_with_both_models": int(seeds_with_both_models),
                "both_model_rate": both_model_rate,
                "median_cross_model_pairs": median_cross_pairs,
                "max_cross_model_pairs": max_cross_pairs,
                "median_best_convergence_score": median_best_score,
                "max_best_convergence_score": max_best_score,
                "mean_scgpt_features": mean_scgpt_features,
                "mean_geneformer_features": mean_geneformer_features,
                "confidence_tier": tier,
            }
        )

    summary = pd.DataFrame(rows)
    tier_rank = {"high": 0, "medium": 1, "low": 2}
    summary["tier_rank"] = summary["confidence_tier"].map(tier_rank).fillna(3).astype(int)
    summary = summary.sort_values(
        ["tier_rank", "seed_replication_rate", "median_cross_model_pairs", "max_best_convergence_score"],
        ascending=[True, False, False, False],
    ).drop(columns=["tier_rank"])
    return summary.reset_index(drop=True)


def _write_report(
    seed_level: pd.DataFrame,
    summary: pd.DataFrame,
    stage3_agg: pd.DataFrame,
    out_md: Path,
) -> None:
    lines: List[str] = [
        "# Stage-3/4 Multi-Seed Hardening Report",
        "",
        "This run repeats Stage-3 SAE discovery and Stage-4 convergence across multiple seeds and summarizes replication stability.",
        "",
    ]

    if summary.empty:
        lines.extend(["No multiseed summary rows were produced.", ""])
        out_md.write_text("\n".join(lines), encoding="utf-8")
        return

    if not stage3_agg.empty:
        stage3_overview = (
            stage3_agg.groupby(["seed", "model"], as_index=False)
            .agg(
                n_dataset_runs=("dataset_id", "count"),
                total_robust_features=("n_robust_features", "sum"),
                max_top_abs_donor_age_spearman=("top_abs_donor_age_spearman", "max"),
            )
            .sort_values(["seed", "model"])
        )
        lines.extend(
            [
                "## Stage-3 Seed Overview",
                "",
                "```text",
                stage3_overview.to_string(index=False),
                "```",
                "",
            ]
        )

    lines.extend(
        [
            "## Stage-4 Multi-Seed Confidence",
            "",
            "```text",
            summary.to_string(index=False),
            "```",
            "",
        ]
    )

    high = summary[summary["confidence_tier"] == "high"].copy()
    med = summary[summary["confidence_tier"] == "medium"].copy()
    lines.append("## Interpretation")
    lines.append("")
    if not high.empty:
        lines.append(f"- High-confidence pathway rows: `{high.shape[0]}`")
    else:
        lines.append("- High-confidence pathway rows: `0`")
    lines.append(f"- Medium-confidence pathway rows: `{med.shape[0]}`")
    lines.append("- Confidence tiers use seed replication + both-model support + cross-model pair strength.")
    lines.append("")

    top_seed_level = seed_level.sort_values(
        ["n_cross_model_pairs", "best_convergence_score"], ascending=[False, False]
    ).head(40)
    lines.extend(
        [
            "## Top Seed-Level Rows",
            "",
            "```text",
            top_seed_level.to_string(index=False),
            "```",
            "",
        ]
    )

    out_md.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Run Stage-3 + Stage-4 across multiple seeds and build a replication confidence summary."
    )
    parser.add_argument(
        "--stage1-output-dir",
        type=Path,
        default=_resolve_default_output_dir(root, "stage1_full_contextual", "20260303"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "implementation" / "outputs" / "stage34_multiseed_hardening_20260303",
    )
    parser.add_argument("--seeds", type=str, default="42,123,314")
    parser.add_argument("--dataset-ids", type=str, default="")
    parser.add_argument("--max-datasets", type=int, default=2)
    parser.add_argument("--device", type=str, default="cpu", choices=["auto", "cpu", "cuda", "mps"])

    parser.add_argument("--scgpt-max-cells-per-dataset", type=int, default=1000)
    parser.add_argument("--scgpt-layers", type=str, default="9")
    parser.add_argument("--scgpt-sae-epochs", type=int, default=10)
    parser.add_argument("--scgpt-perm-iters", type=int, default=30)

    parser.add_argument("--geneformer-max-cells-per-dataset", type=int, default=700)
    parser.add_argument("--geneformer-max-genes", type=int, default=256)
    parser.add_argument("--geneformer-batch-size", type=int, default=8)
    parser.add_argument("--geneformer-sae-latent-dim", type=int, default=1024)
    parser.add_argument("--geneformer-sae-epochs", type=int, default=8)
    parser.add_argument("--geneformer-perm-iters", type=int, default=20)

    parser.add_argument("--reuse-seed42-baseline", action="store_true")
    parser.add_argument(
        "--baseline-stage3-scgpt-dir",
        type=Path,
        default=root / "implementation" / "outputs" / "stage3_sae_pilot_scgpt_20260303_fast",
    )
    parser.add_argument(
        "--baseline-stage3-geneformer-dir",
        type=Path,
        default=root / "implementation" / "outputs" / "stage3_sae_pilot_geneformer_20260303_fast",
    )
    parser.add_argument(
        "--baseline-stage4-dir",
        type=Path,
        default=root / "implementation" / "outputs" / "stage4_cross_model_convergence_20260303",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip stage runs when output aggregate files already exist.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    seeds = _parse_int_csv(args.seeds)
    if not seeds:
        raise ValueError("No seeds provided")
    dataset_ids = _parse_csv_list(args.dataset_ids) if args.dataset_ids else []

    python_exec = sys.executable
    script_dir = Path(__file__).resolve().parent
    stage3_script = script_dir / "run_stage3_sae_pilot.py"
    stage4_script = script_dir / "run_stage4_cross_model_convergence.py"

    _ensure_reused_seed42(
        seed=42,
        output_root=args.output_dir,
        reuse_seed42=bool(args.reuse_seed42_baseline),
        baseline_scgpt=args.baseline_stage3_scgpt_dir,
        baseline_geneformer=args.baseline_stage3_geneformer_dir,
        baseline_stage4=args.baseline_stage4_dir,
    )

    stage3_rows: List[pd.DataFrame] = []
    stage4_rows: List[pd.DataFrame] = []
    run_status: List[Dict[str, Any]] = []

    for seed in seeds:
        scgpt_out = args.output_dir / f"stage3_scgpt_seed{seed}"
        geneformer_out = args.output_dir / f"stage3_geneformer_seed{seed}"
        stage4_out = args.output_dir / f"stage4_seed{seed}"

        sc_agg = scgpt_out / "stage3_run_aggregate.csv"
        gf_agg = geneformer_out / "stage3_run_aggregate.csv"
        s4_consensus = stage4_out / "stage4_consensus_pathway_summary.csv"

        status_row: Dict[str, Any] = {
            "seed": int(seed),
            "stage3_scgpt": "pending",
            "stage3_geneformer": "pending",
            "stage4": "pending",
        }

        if args.skip_existing and sc_agg.exists():
            print(f"[skip] seed={seed} stage3 scgpt (existing aggregate)")
            status_row["stage3_scgpt"] = "skipped_existing"
        else:
            scgpt_out.mkdir(parents=True, exist_ok=True)
            cmd = _build_stage3_scgpt_cmd(
                python_exec=python_exec,
                stage3_script=stage3_script,
                stage1_output_dir=args.stage1_output_dir,
                output_dir=scgpt_out,
                seed=seed,
                dataset_ids=dataset_ids if dataset_ids else None,
                max_datasets=args.max_datasets,
                max_cells_per_dataset=args.scgpt_max_cells_per_dataset,
                scgpt_layers=args.scgpt_layers,
                scgpt_sae_epochs=args.scgpt_sae_epochs,
                scgpt_perm_iters=args.scgpt_perm_iters,
                device=args.device,
            )
            _run_cmd(cmd)
            status_row["stage3_scgpt"] = "ok"

        if args.skip_existing and gf_agg.exists():
            print(f"[skip] seed={seed} stage3 geneformer (existing aggregate)")
            status_row["stage3_geneformer"] = "skipped_existing"
        else:
            geneformer_out.mkdir(parents=True, exist_ok=True)
            cmd = _build_stage3_geneformer_cmd(
                python_exec=python_exec,
                stage3_script=stage3_script,
                stage1_output_dir=args.stage1_output_dir,
                output_dir=geneformer_out,
                seed=seed,
                dataset_ids=dataset_ids if dataset_ids else None,
                max_datasets=args.max_datasets,
                max_cells_per_dataset=args.geneformer_max_cells_per_dataset,
                geneformer_max_genes=args.geneformer_max_genes,
                geneformer_batch_size=args.geneformer_batch_size,
                geneformer_sae_latent_dim=args.geneformer_sae_latent_dim,
                geneformer_sae_epochs=args.geneformer_sae_epochs,
                geneformer_perm_iters=args.geneformer_perm_iters,
                device=args.device,
            )
            _run_cmd(cmd)
            status_row["stage3_geneformer"] = "ok"

        if args.skip_existing and s4_consensus.exists():
            print(f"[skip] seed={seed} stage4 (existing aggregate)")
            status_row["stage4"] = "skipped_existing"
        else:
            stage4_out.mkdir(parents=True, exist_ok=True)
            cmd = _build_stage4_cmd(
                python_exec=python_exec,
                stage4_script=stage4_script,
                stage3_scgpt_dir=scgpt_out,
                stage3_geneformer_dir=geneformer_out,
                output_dir=stage4_out,
                seed=seed,
                dataset_ids=dataset_ids if dataset_ids else None,
                device=args.device,
            )
            _run_cmd(cmd)
            status_row["stage4"] = "ok"

        run_status.append(status_row)

        if not sc_agg.exists():
            raise FileNotFoundError(f"Missing stage3 scgpt aggregate for seed={seed}: {sc_agg}")
        if not gf_agg.exists():
            raise FileNotFoundError(f"Missing stage3 geneformer aggregate for seed={seed}: {gf_agg}")
        if not s4_consensus.exists():
            raise FileNotFoundError(f"Missing stage4 consensus for seed={seed}: {s4_consensus}")

        stage3_rows.append(_read_stage3_aggregate(sc_agg, seed=seed, model="scgpt"))
        stage3_rows.append(_read_stage3_aggregate(gf_agg, seed=seed, model="geneformer"))
        stage4_rows.append(_read_stage4_consensus(s4_consensus, seed=seed))

    stage3_agg = pd.concat(stage3_rows, ignore_index=True) if stage3_rows else pd.DataFrame()
    stage4_consensus_all = pd.concat(stage4_rows, ignore_index=True) if stage4_rows else pd.DataFrame()
    stage4_seed_level = _build_seed_level_table(stage4_consensus_all, seeds=seeds)
    stage4_summary = _build_multiseed_summary(stage4_seed_level, n_seeds=len(seeds))
    status_df = pd.DataFrame(run_status)

    stage3_csv = args.output_dir / "stage3_multiseed_aggregate.csv"
    stage4_seed_csv = args.output_dir / "stage4_multiseed_seed_level.csv"
    stage4_summary_csv = args.output_dir / "stage4_multiseed_consensus_summary.csv"
    status_csv = args.output_dir / "stage34_multiseed_run_status.csv"
    report_md = args.output_dir / "stage34_multiseed_hardening_report.md"
    run_cfg = args.output_dir / "run_config.json"

    stage3_agg.to_csv(stage3_csv, index=False)
    stage4_seed_level.to_csv(stage4_seed_csv, index=False)
    stage4_summary.to_csv(stage4_summary_csv, index=False)
    status_df.to_csv(status_csv, index=False)
    _write_report(
        seed_level=stage4_seed_level,
        summary=stage4_summary,
        stage3_agg=stage3_agg,
        out_md=report_md,
    )

    run_meta = {
        "stage1_output_dir": str(args.stage1_output_dir),
        "seeds": seeds,
        "dataset_ids": dataset_ids,
        "max_datasets": int(args.max_datasets),
        "device": str(args.device),
        "reuse_seed42_baseline": bool(args.reuse_seed42_baseline),
        "baseline_stage3_scgpt_dir": str(args.baseline_stage3_scgpt_dir),
        "baseline_stage3_geneformer_dir": str(args.baseline_stage3_geneformer_dir),
        "baseline_stage4_dir": str(args.baseline_stage4_dir),
    }
    run_cfg.write_text(json.dumps(run_meta, indent=2), encoding="utf-8")

    print("[done] stage34 multiseed hardening outputs")
    print(f"  - {stage3_csv}")
    print(f"  - {stage4_seed_csv}")
    print(f"  - {stage4_summary_csv}")
    print(f"  - {status_csv}")
    print(f"  - {report_md}")


if __name__ == "__main__":
    main()
