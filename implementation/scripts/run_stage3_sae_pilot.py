#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import scipy.stats as stats
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from run_stage1_longevity_mechinterp import (
    GeneformerRuntime,
    ScGPTRuntime,
    _build_sample_selection,
    _dataset_id_from_path,
    _device_auto,
    _ensure_single_cell_src_on_path,
    _load_subset_anndata,
    _select_dataset_columns,
)


@dataclass
class SAETrainResult:
    latent: np.ndarray
    recon_mse: float
    train_history: List[Dict[str, float]]
    input_mean: np.ndarray
    input_std: np.ndarray
    encoder_weight: np.ndarray
    encoder_bias: np.ndarray
    decoder_weight: np.ndarray
    decoder_bias: np.ndarray


class SparseAutoencoder(nn.Module):
    """Minimal ReLU SAE for representation feature discovery."""

    def __init__(self, input_dim: int, latent_dim: int):
        super().__init__()
        self.encoder = nn.Linear(input_dim, latent_dim)
        self.decoder = nn.Linear(latent_dim, input_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = torch.relu(self.encoder(x))
        x_hat = self.decoder(z)
        return z, x_hat


def _resolve_default_stage1_output(root: Path) -> Path:
    outputs_dir = root / "implementation" / "outputs"
    contextual = sorted(outputs_dir.glob("stage1_full_contextual_*"))
    if contextual:
        return contextual[-1]
    legacy = sorted(outputs_dir.glob("stage1_full_*"))
    if legacy:
        return legacy[-1]
    return outputs_dir / "stage1_full_contextual_20260303"


def _parse_csv_list(text: str) -> List[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def _select_stage3_datasets(
    stage1_aggregate_path: Path,
    user_dataset_ids: Optional[Sequence[str]],
    max_datasets: int,
) -> pd.DataFrame:
    agg = pd.read_csv(stage1_aggregate_path)
    ok = agg[agg["status"] == "ok"].copy()
    if ok.empty:
        raise RuntimeError("No successful stage-1 datasets available for stage-3")

    if user_dataset_ids:
        keep = set(user_dataset_ids)
        ok = ok[ok["dataset_id"].isin(keep)].copy()
        if ok.empty:
            raise ValueError(f"None of requested dataset IDs found in stage-1 aggregate: {sorted(keep)}")

    ok = ok.sort_values("best_balanced_accuracy_mean", ascending=False).head(max_datasets).reset_index(drop=True)
    return ok


def _build_representations(
    adata,
    scgpt_runtime: Optional[ScGPTRuntime],
    geneformer_runtime: Optional[GeneformerRuntime],
    scgpt_layers: Sequence[int],
    rep_keep: Sequence[str],
    scgpt_batch_size: int,
    scgpt_max_genes: int,
    geneformer_batch_size: int,
    geneformer_max_genes: int,
) -> Dict[str, np.ndarray]:
    reps: Dict[str, np.ndarray] = {}
    need_scgpt = any(name.startswith("scgpt_") for name in rep_keep)
    need_geneformer = any(name.startswith("geneformer_") for name in rep_keep)

    if need_scgpt:
        if scgpt_runtime is None:
            raise ValueError("scGPT runtime is required for requested scGPT representations")
        sc_reps = scgpt_runtime.extract_representations(
            adata=adata,
            batch_size=scgpt_batch_size,
            max_genes=scgpt_max_genes,
            layer_indices=list(scgpt_layers),
        )
        reps.update(sc_reps)

    if need_geneformer:
        if geneformer_runtime is None:
            raise ValueError("Geneformer runtime is required for requested Geneformer representations")
        gf_name, gf_rep = geneformer_runtime.extract_representation(
            adata=adata,
            max_genes_per_cell=geneformer_max_genes,
            batch_size=geneformer_batch_size,
        )
        reps[gf_name] = gf_rep
    return reps


def _train_sae(
    X: np.ndarray,
    latent_dim: int,
    epochs: int,
    batch_size: int,
    lr: float,
    l1_coef: float,
    device: str,
    seed: int,
) -> SAETrainResult:
    torch.manual_seed(seed)
    np.random.seed(seed)

    Xf = np.asarray(X, dtype=np.float32)
    mean = Xf.mean(axis=0, keepdims=True)
    std = Xf.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    Xn = (Xf - mean) / std

    ds = TensorDataset(torch.from_numpy(Xn))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)

    model = SparseAutoencoder(input_dim=Xn.shape[1], latent_dim=latent_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    history: List[Dict[str, float]] = []
    model.train()
    for epoch in range(1, epochs + 1):
        n_batches = 0
        total_loss = 0.0
        total_mse = 0.0
        total_l1 = 0.0

        for (xb,) in loader:
            xb = xb.to(device)
            z, xh = model(xb)
            mse = torch.mean((xh - xb) ** 2)
            l1 = torch.mean(torch.abs(z))
            loss = mse + l1_coef * l1

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            total_loss += float(loss.item())
            total_mse += float(mse.item())
            total_l1 += float(l1.item())
            n_batches += 1

        history.append(
            {
                "epoch": float(epoch),
                "loss_mean": total_loss / max(n_batches, 1),
                "mse_mean": total_mse / max(n_batches, 1),
                "l1_mean": total_l1 / max(n_batches, 1),
            }
        )

    model.eval()
    with torch.no_grad():
        xt = torch.from_numpy(Xn).to(device)
        z_all, xh_all = model(xt)
        z_np = z_all.cpu().numpy().astype(np.float32)
        recon_mse = float(torch.mean((xh_all - xt) ** 2).item())

    return SAETrainResult(
        latent=z_np,
        recon_mse=recon_mse,
        train_history=history,
        input_mean=mean.astype(np.float32).ravel(),
        input_std=std.astype(np.float32).ravel(),
        encoder_weight=model.encoder.weight.detach().cpu().numpy().astype(np.float32),
        encoder_bias=model.encoder.bias.detach().cpu().numpy().astype(np.float32),
        decoder_weight=model.decoder.weight.detach().cpu().numpy().astype(np.float32),
        decoder_bias=model.decoder.bias.detach().cpu().numpy().astype(np.float32),
    )


def _eta_squared(values: np.ndarray, labels: np.ndarray) -> float:
    if values.size == 0:
        return float("nan")
    vals = np.asarray(values, dtype=np.float64)
    labs = np.asarray(labels, dtype=object)
    if np.unique(labs).size < 2:
        return float("nan")
    total_ss = float(np.sum((vals - vals.mean()) ** 2))
    if total_ss <= 1e-12:
        return 0.0
    between_ss = 0.0
    for label in pd.unique(labs):
        idx = labs == label
        if idx.sum() == 0:
            continue
        mean_g = float(vals[idx].mean())
        between_ss += float(idx.sum()) * (mean_g - float(vals.mean())) ** 2
    return float(between_ss / total_ss)


def _safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 10:
        return float("nan")
    x_m = x[mask]
    y_m = y[mask]
    if np.std(x_m) <= 1e-12 or np.std(y_m) <= 1e-12:
        return float("nan")
    corr, _ = stats.spearmanr(x_m, y_m)
    return float(corr)


def _donor_aware_feature_scores(
    latent: np.ndarray,
    meta: pd.DataFrame,
    perm_iters: int,
    seed: int,
) -> pd.DataFrame:
    n_cells, n_features = latent.shape
    age_numeric = meta["age_numeric"].to_numpy(dtype=np.float64)
    age_label = meta["age_label"].astype(str).to_numpy(dtype=object)
    donor_id = meta["donor_id"].astype(str).to_numpy(dtype=object)
    cell_type = meta["cell_type"].astype(str).to_numpy(dtype=object)

    base = pd.DataFrame({"donor_id": donor_id, "age_numeric": age_numeric})
    donor_age = base.groupby("donor_id", as_index=True)["age_numeric"].mean()
    donor_ids = donor_age.index.to_numpy(dtype=object)
    donor_age_arr = donor_age.to_numpy(dtype=np.float64)
    donor_lookup = {d: i for i, d in enumerate(donor_ids)}
    donor_index = np.array([donor_lookup[d] for d in donor_id], dtype=np.int64)

    rng = np.random.default_rng(seed + 2718)
    rows: List[Dict[str, Any]] = []

    for feat in range(n_features):
        z = latent[:, feat].astype(np.float64, copy=False)
        frac_active = float(np.mean(z > 1e-6))
        mean_act = float(np.mean(z))
        std_act = float(np.std(z))

        cell_age_spear = _safe_spearman(z, age_numeric)

        donor_means = np.zeros(donor_ids.shape[0], dtype=np.float64)
        counts = np.zeros(donor_ids.shape[0], dtype=np.int64)
        for i, d_idx in enumerate(donor_index):
            donor_means[d_idx] += z[i]
            counts[d_idx] += 1
        nonzero = counts > 0
        donor_means[nonzero] = donor_means[nonzero] / counts[nonzero]

        donor_age_spear = _safe_spearman(donor_means, donor_age_arr)
        donor_perm_p = float("nan")
        if math.isfinite(donor_age_spear) and donor_ids.size >= 20 and perm_iters > 0:
            null_scores = np.zeros(perm_iters, dtype=np.float64)
            for i in range(perm_iters):
                perm_age = rng.permutation(donor_age_arr)
                null_scores[i] = _safe_spearman(donor_means, perm_age)
            null_scores = np.nan_to_num(null_scores, nan=0.0)
            donor_perm_p = float((1 + np.sum(np.abs(null_scores) >= abs(donor_age_spear))) / (perm_iters + 1))

        row = {
            "feature_id": int(feat),
            "n_cells": int(n_cells),
            "n_donors": int(donor_ids.size),
            "mean_activation": mean_act,
            "std_activation": std_act,
            "frac_active": frac_active,
            "cell_age_spearman": cell_age_spear,
            "donor_age_spearman": donor_age_spear,
            "donor_age_perm_p": donor_perm_p,
            "age_eta2": _eta_squared(z, age_label),
            "celltype_eta2": _eta_squared(z, cell_type),
            "donor_eta2": _eta_squared(z, donor_id),
        }
        rows.append(row)

    out = pd.DataFrame(rows)
    out["abs_donor_age_spearman"] = out["donor_age_spearman"].abs()
    out["abs_cell_age_spearman"] = out["cell_age_spearman"].abs()
    return out.sort_values("abs_donor_age_spearman", ascending=False).reset_index(drop=True)


def _annotate_decoder_dimensions(
    feature_scores: pd.DataFrame,
    decoder_weight: np.ndarray,
    top_n: int,
) -> pd.DataFrame:
    """Attach top decoder dimensions per SAE feature for quick inspection."""
    rows: List[Dict[str, Any]] = []
    for feat in feature_scores["feature_id"].astype(int).tolist():
        if feat < 0 or feat >= decoder_weight.shape[1]:
            continue
        vec = decoder_weight[:, feat]
        order = np.argsort(-np.abs(vec))[:top_n]
        dims = [f"{int(i)}:{float(vec[i]):.6f}" for i in order]
        rows.append({"feature_id": int(feat), "top_decoder_dims": ";".join(dims)})
    return pd.DataFrame(rows)


def _write_stage3_report(
    aggregate_df: pd.DataFrame,
    output_md: Path,
    rep_keep: Sequence[str],
    donor_corr_min: float,
    donor_p_max: float,
    max_celltype_eta2: float,
) -> None:
    lines: List[str] = [
        "# Stage-3 SAE Pilot Report",
        "",
        "This stage trains sparse autoencoders on shortlisted frozen representations and scores donor-aware age associations.",
        "",
        f"Representations evaluated: `{', '.join(rep_keep)}`",
        "",
    ]

    if aggregate_df.empty:
        lines.append("No successful stage-3 runs.")
        output_md.write_text("\n".join(lines), encoding="utf-8")
        return

    lines.extend(
        [
            "## Aggregate Summary",
            "",
            "```text",
            aggregate_df[
                [
                    "dataset_id",
                    "representation",
                    "latent_dim",
                    "recon_mse",
                    "n_robust_features",
                    "top_abs_donor_age_spearman",
                    "top_feature_donor_perm_p",
                    "top_feature_celltype_eta2",
                ]
            ]
            .sort_values(["dataset_id", "top_abs_donor_age_spearman"], ascending=[True, False])
            .to_string(index=False),
            "```",
            "",
            "## Robust Feature Gate (Pilot)",
            "",
            f"- `|donor_age_spearman| >= {donor_corr_min:.2f}`",
            f"- `donor_age_perm_p <= {donor_p_max:.3f}`",
            f"- `celltype_eta2 <= {max_celltype_eta2:.2f}`",
            "",
            f"- Total robust SAE features across all runs: `{int(aggregate_df['n_robust_features'].sum())}`",
            "",
            "Interpretation: these are exploratory candidates for follow-up, not final mechanistic claims.",
            "",
        ]
    )
    output_md.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Stage-3 SAE pilot for longevity mechanistic interpretability.")
    parser.add_argument(
        "--stage1-output-dir",
        type=Path,
        default=_resolve_default_stage1_output(root),
        help="Completed stage-1 output directory (for dataset shortlist).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "implementation" / "outputs" / "stage3_sae_pilot_20260303",
    )
    parser.add_argument(
        "--dataset-ids",
        type=str,
        default="",
        help="Optional comma-separated dataset IDs. Empty means auto-shortlist from stage-1.",
    )
    parser.add_argument("--max-datasets", type=int, default=2)
    parser.add_argument("--max-cells-per-dataset", type=int, default=2500)
    parser.add_argument("--max-cells-per-donor", type=int, default=250)
    parser.add_argument("--min-cells-per-donor", type=int, default=20)
    parser.add_argument("--min-cells-per-age-label", type=int, default=80)
    parser.add_argument("--age-bins", type=int, default=4)
    parser.add_argument("--scgpt-layers", type=str, default="9")
    parser.add_argument(
        "--representations",
        type=str,
        default="scgpt_layer_09,geneformer_contextual",
        help="Comma-separated representation names to evaluate in SAE stage.",
    )
    parser.add_argument("--scgpt-max-genes", type=int, default=600)
    parser.add_argument("--scgpt-batch-size", type=int, default=8)
    parser.add_argument("--geneformer-max-genes", type=int, default=800)
    parser.add_argument("--geneformer-batch-size", type=int, default=8)
    parser.add_argument("--geneformer-mode", type=str, default="contextual", choices=["contextual", "static", "auto"])

    parser.add_argument("--sae-latent-dim", type=int, default=0, help="0 means auto from multiplier.")
    parser.add_argument("--sae-latent-multiplier", type=float, default=4.0)
    parser.add_argument("--sae-epochs", type=int, default=30)
    parser.add_argument("--sae-batch-size", type=int, default=256)
    parser.add_argument("--sae-lr", type=float, default=1e-3)
    parser.add_argument("--sae-l1", type=float, default=1e-3)
    parser.add_argument("--perm-iters", type=int, default=100)
    parser.add_argument("--top-decoder-dims", type=int, default=10)
    parser.add_argument("--donor-corr-min", type=float, default=0.15)
    parser.add_argument("--donor-p-max", type=float, default=0.05)
    parser.add_argument("--max-celltype-eta2", type=float, default=0.35)
    parser.add_argument("--device", type=str, default="cpu", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    stage1_aggregate_path = args.stage1_output_dir / "stage1_run_aggregate.csv"
    if not stage1_aggregate_path.exists():
        raise FileNotFoundError(f"Missing stage-1 aggregate file: {stage1_aggregate_path}")

    requested_datasets = _parse_csv_list(args.dataset_ids) if args.dataset_ids else []
    shortlist = _select_stage3_datasets(
        stage1_aggregate_path=stage1_aggregate_path,
        user_dataset_ids=requested_datasets if requested_datasets else None,
        max_datasets=args.max_datasets,
    )

    device = _device_auto(args.device)
    single_cell_root = _ensure_single_cell_src_on_path(Path(__file__))
    sc_layers = [int(x) for x in _parse_csv_list(args.scgpt_layers)]
    rep_keep = _parse_csv_list(args.representations)
    need_scgpt = any(name.startswith("scgpt_") for name in rep_keep)
    need_geneformer = any(name.startswith("geneformer_") for name in rep_keep)

    scgpt_runtime: Optional[ScGPTRuntime] = None
    geneformer_runtime: Optional[GeneformerRuntime] = None
    if need_scgpt:
        scgpt_runtime = ScGPTRuntime(single_cell_root=single_cell_root, device=device)
    if need_geneformer:
        geneformer_runtime = GeneformerRuntime(mode=args.geneformer_mode, device=device)

    aggregate_rows: List[Dict[str, Any]] = []
    for ds in shortlist.itertuples(index=False):
        dataset_id = str(ds.dataset_id)
        dataset_path = Path(str(ds.dataset_path))
        ds_out = args.output_dir / dataset_id
        ds_out.mkdir(parents=True, exist_ok=True)

        cols = _select_dataset_columns(dataset_path)
        selection = _build_sample_selection(
            path=dataset_path,
            cols=cols,
            max_cells=args.max_cells_per_dataset,
            max_cells_per_donor=args.max_cells_per_donor,
            min_cells_per_donor=args.min_cells_per_donor,
            min_cells_per_age_label=args.min_cells_per_age_label,
            age_bins=args.age_bins,
            seed=args.seed,
        )
        meta = selection.metadata.copy()
        meta.to_csv(ds_out / "sampled_obs.csv", index=False)

        adata = _load_subset_anndata(dataset_path, selection.obs_index)
        if adata.n_obs != meta.shape[0]:
            raise RuntimeError(f"Subset size mismatch for {dataset_id}")

        reps = _build_representations(
            adata=adata,
            scgpt_runtime=scgpt_runtime,
            geneformer_runtime=geneformer_runtime,
            scgpt_layers=sc_layers,
            rep_keep=rep_keep,
            scgpt_batch_size=args.scgpt_batch_size,
            scgpt_max_genes=args.scgpt_max_genes,
            geneformer_batch_size=args.geneformer_batch_size,
            geneformer_max_genes=args.geneformer_max_genes,
        )

        available_rep = set(reps.keys())
        target_reps = [r for r in rep_keep if r in available_rep]
        if not target_reps:
            raise ValueError(f"None of requested representations available for {dataset_id}. Available: {sorted(available_rep)}")

        for rep_name in target_reps:
            rep_out = ds_out / rep_name
            rep_out.mkdir(parents=True, exist_ok=True)
            X = np.asarray(reps[rep_name], dtype=np.float32)

            latent_dim = int(args.sae_latent_dim) if args.sae_latent_dim > 0 else int(
                max(32, min(2048, round(X.shape[1] * args.sae_latent_multiplier)))
            )
            sae = _train_sae(
                X=X,
                latent_dim=latent_dim,
                epochs=args.sae_epochs,
                batch_size=args.sae_batch_size,
                lr=args.sae_lr,
                l1_coef=args.sae_l1,
                device=device,
                seed=args.seed,
            )

            history_df = pd.DataFrame(sae.train_history)
            history_df.to_csv(rep_out / "sae_train_history.csv", index=False)

            scores = _donor_aware_feature_scores(
                latent=sae.latent,
                meta=meta,
                perm_iters=args.perm_iters,
                seed=args.seed,
            )
            scores["is_robust_feature"] = (
                (scores["abs_donor_age_spearman"] >= args.donor_corr_min)
                & (scores["donor_age_perm_p"] <= args.donor_p_max)
                & (scores["celltype_eta2"] <= args.max_celltype_eta2)
            )

            decoder_anno = _annotate_decoder_dimensions(
                feature_scores=scores,
                decoder_weight=sae.decoder_weight,
                top_n=args.top_decoder_dims,
            )
            scores = scores.merge(decoder_anno, on="feature_id", how="left")
            scores.to_csv(rep_out / "sae_feature_scores.csv", index=False)
            scores.head(100).to_csv(rep_out / "sae_top_features.csv", index=False)

            model_artifacts = {
                "input_mean": sae.input_mean,
                "input_std": sae.input_std,
                "encoder_weight": sae.encoder_weight,
                "encoder_bias": sae.encoder_bias,
                "decoder_weight": sae.decoder_weight,
                "decoder_bias": sae.decoder_bias,
            }
            np.savez_compressed(rep_out / "sae_model_artifacts.npz", **model_artifacts)

            top_row = scores.iloc[0].to_dict() if not scores.empty else {}
            robust_count = int(scores["is_robust_feature"].sum()) if "is_robust_feature" in scores.columns else 0
            summary = {
                "dataset_id": dataset_id,
                "dataset_path": str(dataset_path),
                "representation": rep_name,
                "n_cells": int(X.shape[0]),
                "input_dim": int(X.shape[1]),
                "latent_dim": int(latent_dim),
                "recon_mse": float(sae.recon_mse),
                "n_robust_features": robust_count,
                "top_feature_id": int(top_row.get("feature_id", -1)),
                "top_abs_donor_age_spearman": float(top_row.get("abs_donor_age_spearman", float("nan"))),
                "top_feature_donor_perm_p": float(top_row.get("donor_age_perm_p", float("nan"))),
                "top_feature_celltype_eta2": float(top_row.get("celltype_eta2", float("nan"))),
            }
            (rep_out / "sae_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
            aggregate_rows.append(summary)
            print(
                f"[ok] {dataset_id}::{rep_name} "
                f"| recon_mse={summary['recon_mse']:.4f} "
                f"| robust_features={robust_count}"
            )

    aggregate_df = pd.DataFrame(aggregate_rows)
    aggregate_path = args.output_dir / "stage3_run_aggregate.csv"
    aggregate_df.to_csv(aggregate_path, index=False)

    report_path = args.output_dir / "stage3_sae_pilot_report.md"
    _write_stage3_report(
        aggregate_df=aggregate_df,
        output_md=report_path,
        rep_keep=rep_keep,
        donor_corr_min=args.donor_corr_min,
        donor_p_max=args.donor_p_max,
        max_celltype_eta2=args.max_celltype_eta2,
    )

    run_meta = {
        "stage1_output_dir": str(args.stage1_output_dir),
        "selected_dataset_ids": shortlist["dataset_id"].astype(str).tolist(),
        "representations": rep_keep,
        "device": device,
        "seed": int(args.seed),
        "sae_epochs": int(args.sae_epochs),
        "sae_batch_size": int(args.sae_batch_size),
        "sae_l1": float(args.sae_l1),
        "perm_iters": int(args.perm_iters),
    }
    (args.output_dir / "run_config.json").write_text(json.dumps(run_meta, indent=2), encoding="utf-8")

    print("[done] stage3 outputs")
    print(f"  - {aggregate_path}")
    print(f"  - {report_path}")


if __name__ == "__main__":
    main()
