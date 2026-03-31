#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import hypergeom

from run_stage1_longevity_mechinterp import (
    GeneformerRuntime,
    ScGPTRuntime,
    _device_auto,
    _ensure_single_cell_src_on_path,
    _load_subset_anndata,
)


PATHWAY_GENESETS: Dict[str, List[str]] = {
    "inflammation_nfkb": [
        "IL1B",
        "IL6",
        "TNF",
        "NFKB1",
        "NFKBIA",
        "RELA",
        "STAT1",
        "STAT3",
        "CXCL8",
        "CCL2",
        "LST1",
        "S100A8",
        "S100A9",
        "PTGS2",
    ],
    "senescence_sasp": [
        "CDKN1A",
        "CDKN2A",
        "TP53",
        "GDF15",
        "SERPINE1",
        "MMP1",
        "MMP3",
        "MMP9",
        "IL1A",
        "IL1B",
        "IL6",
        "CXCL8",
        "GLB1",
    ],
    "mtor_igf_akt": [
        "MTOR",
        "RPTOR",
        "RICTOR",
        "AKT1",
        "AKT2",
        "PIK3CA",
        "PIK3CD",
        "TSC1",
        "TSC2",
        "RHEB",
        "RPS6KB1",
        "EIF4EBP1",
        "IGF1R",
    ],
    "autophagy_lysosome": [
        "BECN1",
        "ATG5",
        "ATG7",
        "MAP1LC3B",
        "SQSTM1",
        "LAMP1",
        "LAMP2",
        "CTSB",
        "CTSD",
        "TFEB",
        "ULK1",
    ],
    "proteostasis_upr": [
        "HSPA1A",
        "HSPA1B",
        "HSP90AA1",
        "HSP90AB1",
        "HSPH1",
        "DNAJB1",
        "HSPD1",
        "HSPE1",
        "ATF4",
        "DDIT3",
        "XBP1",
        "UBB",
        "UBC",
    ],
    "mitochondria_oxphos": [
        "NDUFA1",
        "NDUFS1",
        "UQCRC1",
        "UQCRC2",
        "COX4I1",
        "COX5A",
        "ATP5F1A",
        "ATP5F1B",
        "TFAM",
        "SOD2",
        "PPARGC1A",
    ],
    "dna_damage_repair": [
        "BRCA1",
        "BRCA2",
        "RAD51",
        "ATM",
        "ATR",
        "CHEK1",
        "CHEK2",
        "TP53BP1",
        "XRCC5",
        "XRCC6",
        "PARP1",
        "MRE11",
    ],
    "interferon_antiviral": [
        "IFIT1",
        "IFIT2",
        "IFIT3",
        "IFI6",
        "IFI27",
        "ISG15",
        "MX1",
        "OAS1",
        "OAS2",
        "STAT1",
        "IRF7",
    ],
}

ANNOTATION_COLUMNS: List[str] = [
    "feature_id",
    "n_cells",
    "n_donors",
    "mean_activation",
    "std_activation",
    "frac_active",
    "cell_age_spearman",
    "donor_age_spearman",
    "donor_age_perm_p",
    "age_eta2",
    "celltype_eta2",
    "donor_eta2",
    "abs_donor_age_spearman",
    "abs_cell_age_spearman",
    "is_robust_feature",
    "top_abs_genes",
    "top_abs_corr_values",
    "primary_pathway",
    "pathway_hypergeom_p",
    "pathway_overlap_count",
    "pathway_overlap_genes",
    "pathway_overlap_mean_abs_corr",
]

PAIR_COLUMNS: List[str] = [
    "dataset_id",
    "pathway",
    "scgpt_feature_id",
    "geneformer_feature_id",
    "scgpt_donor_age_spearman",
    "geneformer_donor_age_spearman",
    "same_direction",
    "gene_jaccard_top50",
    "max_celltype_eta2",
    "convergence_score",
]

SUMMARY_COLUMNS: List[str] = [
    "dataset_id",
    "pathway",
    "n_scgpt_features",
    "n_geneformer_features",
    "n_cross_model_pairs",
    "best_convergence_score",
]


def _resolve_default_stage3_dir(root: Path, prefix: str) -> Path:
    outputs = root / "implementation" / "outputs"
    candidates = sorted(outputs.glob(f"{prefix}_*"))
    if candidates:
        return candidates[-1]
    return outputs / f"{prefix}_20260303"


def _clean_gene_symbol(g: str) -> str:
    text = str(g).strip()
    if not text:
        return ""
    return text.upper()


def _parse_top_gene_list(text: str, max_genes: int = 50) -> List[str]:
    if not isinstance(text, str) or not text:
        return []
    out = [x.strip() for x in text.split(";") if x.strip()]
    dedup: List[str] = []
    seen = set()
    for g in out:
        if g in seen:
            continue
        seen.add(g)
        dedup.append(g)
        if len(dedup) >= max_genes:
            break
    return dedup


def _jaccard(a: Sequence[str], b: Sequence[str]) -> float:
    sa = set(a)
    sb = set(b)
    if not sa and not sb:
        return 0.0
    return float(len(sa & sb) / max(len(sa | sb), 1))


def _pick_gene_labels(adata: ad.AnnData) -> np.ndarray:
    candidates = [
        "feature_name",
        "gene_name",
        "gene_symbol",
        "gene_symbols",
        "symbol",
        "hgnc_symbol",
    ]
    for col in candidates:
        if col in adata.var.columns:
            vals = adata.var[col].astype(str).to_numpy(dtype=object)
            non_empty = np.array([bool(str(v).strip()) for v in vals], dtype=bool)
            if non_empty.mean() >= 0.8:
                return np.array([_clean_gene_symbol(v) for v in vals], dtype=object)
    return np.array([_clean_gene_symbol(v) for v in adata.var_names], dtype=object)


def _layer_index_from_representation(name: str) -> int:
    m = re.search(r"scgpt_layer_(\d+)", name)
    if m is None:
        raise ValueError(f"Could not parse scGPT layer from representation '{name}'")
    return int(m.group(1))


def _extract_representation(
    model: str,
    rep_name: str,
    adata: ad.AnnData,
    scgpt_runtime: Optional[ScGPTRuntime],
    geneformer_runtime: Optional[GeneformerRuntime],
    scgpt_batch_size: int,
    scgpt_max_genes: int,
    geneformer_batch_size: int,
    geneformer_max_genes: int,
) -> np.ndarray:
    if model == "scgpt":
        if scgpt_runtime is None:
            raise ValueError("scGPT runtime not initialized")
        layer_idx = _layer_index_from_representation(rep_name)
        reps = scgpt_runtime.extract_representations(
            adata=adata,
            batch_size=scgpt_batch_size,
            max_genes=scgpt_max_genes,
            layer_indices=[layer_idx],
        )
        if rep_name not in reps:
            raise KeyError(f"scGPT representation '{rep_name}' not found in extraction output")
        return np.asarray(reps[rep_name], dtype=np.float32)

    if model == "geneformer":
        if geneformer_runtime is None:
            raise ValueError("Geneformer runtime not initialized")
        name, rep = geneformer_runtime.extract_representation(
            adata=adata,
            max_genes_per_cell=geneformer_max_genes,
            batch_size=geneformer_batch_size,
        )
        if name != rep_name:
            raise KeyError(f"Expected Geneformer representation '{rep_name}' but got '{name}'")
        return np.asarray(rep, dtype=np.float32)

    raise ValueError(f"Unsupported model: {model}")


def _load_sae_artifacts(npz_path: Path) -> Dict[str, np.ndarray]:
    arr = np.load(npz_path)
    required = [
        "input_mean",
        "input_std",
        "encoder_weight",
        "encoder_bias",
    ]
    for k in required:
        if k not in arr:
            raise KeyError(f"Missing '{k}' in SAE artifacts: {npz_path}")
    return {k: np.asarray(arr[k]) for k in arr.files}


def _compute_latent(X: np.ndarray, artifacts: Dict[str, np.ndarray]) -> np.ndarray:
    mean = artifacts["input_mean"].reshape(1, -1)
    std = artifacts["input_std"].reshape(1, -1)
    std = np.where(std < 1e-6, 1.0, std)

    w = artifacts["encoder_weight"]  # (latent, input)
    b = artifacts["encoder_bias"]  # (latent,)
    Xn = (X - mean) / std
    z = Xn @ w.T + b.reshape(1, -1)
    return np.maximum(z, 0.0).astype(np.float32, copy=False)


def _dense_log1p_expression(adata: ad.AnnData) -> np.ndarray:
    X = adata.X
    if sp.issparse(X):
        dense = X.toarray()
    else:
        dense = np.asarray(X)
    dense = np.asarray(dense, dtype=np.float32)
    dense = np.maximum(dense, 0.0)
    return np.log1p(dense, dtype=np.float32)


def _standardize_columns(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mean = X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, keepdims=True)
    keep = (std.ravel() > 1e-6)
    Xk = X[:, keep]
    mean_k = mean[:, keep]
    std_k = std[:, keep]
    Xz = (Xk - mean_k) / std_k
    return np.asarray(Xz, dtype=np.float32), keep


def _pathway_annotation_for_feature(
    top_genes: Sequence[str],
    gene_abs_corr: Dict[str, float],
    universe: set[str],
    pathway_p_threshold: float,
    min_pathway_overlap: int,
) -> Dict[str, Any]:
    top_unique = []
    seen = set()
    for g in top_genes:
        if g and g not in seen:
            seen.add(g)
            top_unique.append(g)

    M = max(len(universe), 1)
    n = max(len(top_unique), 1)

    best_name = "unassigned"
    best_p = 1.0
    best_overlap = 0
    best_overlap_genes: List[str] = []
    best_mean_abs_corr = float("nan")

    for name, genes in PATHWAY_GENESETS.items():
        path_set = {g for g in genes if g in universe}
        K = len(path_set)
        if K == 0:
            continue
        overlap = sorted(set(top_unique) & path_set)
        x = len(overlap)
        if x == 0:
            p = 1.0
            mean_abs = 0.0
        else:
            p = float(hypergeom.sf(x - 1, M, K, n))
            mean_abs = float(np.mean([gene_abs_corr.get(g, 0.0) for g in overlap]))

        if (p < best_p) or (math.isclose(p, best_p) and x > best_overlap):
            best_name = name
            best_p = p
            best_overlap = x
            best_overlap_genes = overlap
            best_mean_abs_corr = mean_abs

    if best_p > pathway_p_threshold or best_overlap < min_pathway_overlap:
        best_name = "unassigned"

    return {
        "primary_pathway": best_name,
        "pathway_hypergeom_p": float(best_p),
        "pathway_overlap_count": int(best_overlap),
        "pathway_overlap_genes": ";".join(best_overlap_genes),
        "pathway_overlap_mean_abs_corr": best_mean_abs_corr,
    }


def _annotate_robust_features(
    feature_scores: pd.DataFrame,
    latent: np.ndarray,
    expr_z: np.ndarray,
    gene_labels: np.ndarray,
    top_genes_per_feature: int,
    pathway_p_threshold: float,
    min_pathway_overlap: int,
) -> pd.DataFrame:
    if "is_robust_feature" not in feature_scores.columns:
        raise KeyError("Feature score table missing 'is_robust_feature'")

    robust = feature_scores[feature_scores["is_robust_feature"] == True].copy()  # noqa: E712
    if robust.empty:
        return pd.DataFrame(columns=ANNOTATION_COLUMNS)

    n_cells = latent.shape[0]
    if expr_z.shape[0] != n_cells:
        raise ValueError("Latent and expression row counts do not match")

    universe = set([g for g in gene_labels.tolist() if g])
    out_rows: List[Dict[str, Any]] = []

    for row in robust.itertuples(index=False):
        fid = int(row.feature_id)
        if fid < 0 or fid >= latent.shape[1]:
            continue

        z = latent[:, fid].astype(np.float32, copy=False)
        z_std = float(np.std(z))
        if z_std <= 1e-6:
            continue
        z_norm = (z - float(np.mean(z))) / z_std

        # Fast vectorized Pearson correlation of one feature against all genes.
        corr = (expr_z.T @ z_norm.astype(np.float32)) / max(n_cells - 1, 1)
        abs_corr = np.abs(corr)
        k = int(min(top_genes_per_feature, abs_corr.shape[0]))
        if k <= 0:
            continue

        idx = np.argpartition(-abs_corr, k - 1)[:k]
        idx = idx[np.argsort(-abs_corr[idx])]

        top_genes = [str(gene_labels[i]) for i in idx if str(gene_labels[i])]
        top_genes = [g for g in top_genes if g]
        top_abs = [float(abs_corr[i]) for i in idx if str(gene_labels[i])]
        gene_abs_corr: Dict[str, float] = {}
        for g, c in zip(top_genes, top_abs):
            prev = gene_abs_corr.get(g)
            if prev is None or c > prev:
                gene_abs_corr[g] = c

        pathway = _pathway_annotation_for_feature(
            top_genes=top_genes,
            gene_abs_corr=gene_abs_corr,
            universe=universe,
            pathway_p_threshold=pathway_p_threshold,
            min_pathway_overlap=min_pathway_overlap,
        )

        out = {
            "feature_id": fid,
            "n_cells": int(getattr(row, "n_cells")),
            "n_donors": int(getattr(row, "n_donors")),
            "mean_activation": float(getattr(row, "mean_activation")),
            "std_activation": float(getattr(row, "std_activation")),
            "frac_active": float(getattr(row, "frac_active")),
            "cell_age_spearman": float(getattr(row, "cell_age_spearman")),
            "donor_age_spearman": float(getattr(row, "donor_age_spearman")),
            "donor_age_perm_p": float(getattr(row, "donor_age_perm_p")),
            "age_eta2": float(getattr(row, "age_eta2")),
            "celltype_eta2": float(getattr(row, "celltype_eta2")),
            "donor_eta2": float(getattr(row, "donor_eta2")),
            "abs_donor_age_spearman": float(getattr(row, "abs_donor_age_spearman")),
            "abs_cell_age_spearman": float(getattr(row, "abs_cell_age_spearman")),
            "is_robust_feature": bool(getattr(row, "is_robust_feature")),
            "top_abs_genes": ";".join(top_genes[:50]),
            "top_abs_corr_values": ";".join([f"{x:.6f}" for x in top_abs[:50]]),
        }
        out.update(pathway)
        out_rows.append(out)

    if not out_rows:
        return pd.DataFrame(columns=ANNOTATION_COLUMNS)
    return pd.DataFrame(out_rows).sort_values("abs_donor_age_spearman", ascending=False).reset_index(drop=True)


def _build_convergence_pairs(annot_df: pd.DataFrame) -> pd.DataFrame:
    if annot_df.empty:
        return pd.DataFrame(columns=PAIR_COLUMNS)

    rows: List[Dict[str, Any]] = []
    for dataset_id, ds_df in annot_df.groupby("dataset_id"):
        sc = ds_df[ds_df["model"] == "scgpt"].copy()
        gf = ds_df[ds_df["model"] == "geneformer"].copy()
        if sc.empty or gf.empty:
            continue
        sc = sc[sc["primary_pathway"] != "unassigned"].copy()
        gf = gf[gf["primary_pathway"] != "unassigned"].copy()
        if sc.empty or gf.empty:
            continue

        for s in sc.itertuples(index=False):
            s_genes = _parse_top_gene_list(str(s.top_abs_genes), max_genes=50)
            s_sign = np.sign(float(s.donor_age_spearman))
            for g in gf.itertuples(index=False):
                if str(s.primary_pathway) != str(g.primary_pathway):
                    continue
                g_genes = _parse_top_gene_list(str(g.top_abs_genes), max_genes=50)
                g_sign = np.sign(float(g.donor_age_spearman))
                same_direction = bool(s_sign != 0 and g_sign != 0 and s_sign == g_sign)
                gene_j = _jaccard(s_genes, g_genes)
                min_corr = float(min(abs(float(s.donor_age_spearman)), abs(float(g.donor_age_spearman))))
                max_celltype_eta2 = float(max(float(s.celltype_eta2), float(g.celltype_eta2)))
                confound_factor = max(0.0, 1.0 - min(max_celltype_eta2, 1.0))
                direction_factor = 1.0 if same_direction else 0.5
                score = float(min_corr * confound_factor * (0.5 + 0.5 * gene_j) * direction_factor)

                rows.append(
                    {
                        "dataset_id": str(dataset_id),
                        "pathway": str(s.primary_pathway),
                        "scgpt_feature_id": int(s.feature_id),
                        "geneformer_feature_id": int(g.feature_id),
                        "scgpt_donor_age_spearman": float(s.donor_age_spearman),
                        "geneformer_donor_age_spearman": float(g.donor_age_spearman),
                        "same_direction": same_direction,
                        "gene_jaccard_top50": gene_j,
                        "max_celltype_eta2": max_celltype_eta2,
                        "convergence_score": score,
                    }
                )

    if not rows:
        return pd.DataFrame(columns=PAIR_COLUMNS)
    return pd.DataFrame(rows).sort_values("convergence_score", ascending=False).reset_index(drop=True)


def _build_consensus_summary(annot_df: pd.DataFrame, pair_df: pd.DataFrame) -> pd.DataFrame:
    if annot_df.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)

    rows: List[Dict[str, Any]] = []
    grouped = annot_df[annot_df["primary_pathway"] != "unassigned"].groupby(["dataset_id", "primary_pathway"])
    for (dataset_id, pathway), grp in grouped:
        sc_count = int((grp["model"] == "scgpt").sum())
        gf_count = int((grp["model"] == "geneformer").sum())
        pair_sub = pair_df[(pair_df["dataset_id"] == dataset_id) & (pair_df["pathway"] == pathway)] if not pair_df.empty else pd.DataFrame()
        pair_count = int(pair_sub.shape[0]) if not pair_sub.empty else 0
        best_score = float(pair_sub["convergence_score"].max()) if pair_count > 0 else float("nan")
        rows.append(
            {
                "dataset_id": str(dataset_id),
                "pathway": str(pathway),
                "n_scgpt_features": sc_count,
                "n_geneformer_features": gf_count,
                "n_cross_model_pairs": pair_count,
                "best_convergence_score": best_score,
            }
        )

    if not rows:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)
    return pd.DataFrame(rows).sort_values(["dataset_id", "best_convergence_score"], ascending=[True, False])


def _write_report(
    annotation_df: pd.DataFrame,
    pair_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    out_md: Path,
) -> None:
    lines: List[str] = [
        "# Stage-4 Cross-Model Convergence Report",
        "",
        "This report links donor-aware robust SAE features across scGPT and Geneformer via pathway annotations and feature overlap.",
        "",
    ]

    if annotation_df.empty:
        lines.append("No robust feature annotations were produced.")
        out_md.write_text("\n".join(lines), encoding="utf-8")
        return

    overview = (
        annotation_df.groupby(["dataset_id", "model"])
        .agg(
            n_annotated_features=("feature_id", "count"),
            max_abs_donor_age=("abs_donor_age_spearman", "max"),
            min_perm_p=("donor_age_perm_p", "min"),
        )
        .reset_index()
    )
    lines.extend(
        [
            "## Annotated Robust Features",
            "",
            "```text",
            overview.to_string(index=False),
            "```",
            "",
        ]
    )

    if not summary_df.empty:
        lines.extend(
            [
                "## Pathway Consensus Summary",
                "",
                "```text",
                summary_df.to_string(index=False),
                "```",
                "",
            ]
        )

    if not pair_df.empty:
        top = pair_df.head(25).copy()
        lines.extend(
            [
                "## Top Cross-Model Pairs",
                "",
                "```text",
                top.to_string(index=False),
                "```",
                "",
            ]
        )
    else:
        lines.extend(["No cross-model pathway-matched pairs were found.", ""])

    out_md.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Stage-4 cross-model convergence and pathway annotation for Stage-3 SAE outputs."
    )
    parser.add_argument(
        "--stage3-scgpt-dir",
        type=Path,
        default=_resolve_default_stage3_dir(root, "stage3_sae_pilot_scgpt"),
    )
    parser.add_argument(
        "--stage3-geneformer-dir",
        type=Path,
        default=_resolve_default_stage3_dir(root, "stage3_sae_pilot_geneformer"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "implementation" / "outputs" / "stage4_cross_model_convergence_20260303",
    )
    parser.add_argument(
        "--dataset-ids",
        type=str,
        default="",
        help="Optional comma-separated dataset IDs to process (default: all common datasets).",
    )
    parser.add_argument("--top-genes-per-feature", type=int, default=120)
    parser.add_argument("--pathway-p-threshold", type=float, default=0.05)
    parser.add_argument("--min-pathway-overlap", type=int, default=3)
    parser.add_argument("--scgpt-max-genes", type=int, default=600)
    parser.add_argument("--scgpt-batch-size", type=int, default=8)
    parser.add_argument("--geneformer-max-genes", type=int, default=256)
    parser.add_argument("--geneformer-batch-size", type=int, default=8)
    parser.add_argument("--geneformer-mode", type=str, default="contextual", choices=["contextual", "static", "auto"])
    parser.add_argument("--device", type=str, default="cpu", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    sc_agg_path = args.stage3_scgpt_dir / "stage3_run_aggregate.csv"
    gf_agg_path = args.stage3_geneformer_dir / "stage3_run_aggregate.csv"
    if not sc_agg_path.exists() or not gf_agg_path.exists():
        raise FileNotFoundError(f"Missing stage3 aggregate files: {sc_agg_path}, {gf_agg_path}")

    sc_agg = pd.read_csv(sc_agg_path).copy()
    gf_agg = pd.read_csv(gf_agg_path).copy()
    common = sorted(set(sc_agg["dataset_id"].astype(str)) & set(gf_agg["dataset_id"].astype(str)))
    if args.dataset_ids:
        requested = {x.strip() for x in str(args.dataset_ids).split(",") if x.strip()}
        common = [d for d in common if d in requested]
    if not common:
        raise RuntimeError("No common datasets between scGPT and Geneformer stage-3 outputs")

    device = _device_auto(args.device)
    single_cell_root = _ensure_single_cell_src_on_path(Path(__file__))
    scgpt_runtime = ScGPTRuntime(single_cell_root=single_cell_root, device=device)
    geneformer_runtime = GeneformerRuntime(mode=args.geneformer_mode, device=device)

    annot_rows: List[pd.DataFrame] = []

    for dataset_id in common:
        # Process each model independently using its own sampled cells and SAE artifacts.
        for model, run_dir, agg_df in [
            ("scgpt", args.stage3_scgpt_dir, sc_agg),
            ("geneformer", args.stage3_geneformer_dir, gf_agg),
        ]:
            row = agg_df[agg_df["dataset_id"].astype(str) == dataset_id].iloc[0]
            dataset_path = Path(str(row["dataset_path"]))
            rep_name = str(row["representation"])
            ds_dir = run_dir / dataset_id
            rep_dir = ds_dir / rep_name
            sampled_path = ds_dir / "sampled_obs.csv"
            score_path = rep_dir / "sae_feature_scores.csv"
            sae_npz = rep_dir / "sae_model_artifacts.npz"

            for p in [sampled_path, score_path, sae_npz]:
                if not p.exists():
                    raise FileNotFoundError(f"Missing required file for {dataset_id}/{model}: {p}")

            sampled = pd.read_csv(sampled_path)
            obs_idx = sampled["obs_row"].to_numpy(dtype=np.int64)
            adata = _load_subset_anndata(dataset_path, obs_idx)

            rep = _extract_representation(
                model=model,
                rep_name=rep_name,
                adata=adata,
                scgpt_runtime=scgpt_runtime,
                geneformer_runtime=geneformer_runtime,
                scgpt_batch_size=args.scgpt_batch_size,
                scgpt_max_genes=args.scgpt_max_genes,
                geneformer_batch_size=args.geneformer_batch_size,
                geneformer_max_genes=args.geneformer_max_genes,
            )

            artifacts = _load_sae_artifacts(sae_npz)
            latent = _compute_latent(rep, artifacts)

            expr = _dense_log1p_expression(adata)
            expr_z, keep_mask = _standardize_columns(expr)
            gene_labels = _pick_gene_labels(adata)
            gene_labels = gene_labels[keep_mask]

            feature_scores = pd.read_csv(score_path)
            ann = _annotate_robust_features(
                feature_scores=feature_scores,
                latent=latent,
                expr_z=expr_z,
                gene_labels=gene_labels,
                top_genes_per_feature=args.top_genes_per_feature,
                pathway_p_threshold=args.pathway_p_threshold,
                min_pathway_overlap=args.min_pathway_overlap,
            )
            if ann.empty:
                print(f"[warn] {dataset_id}::{model} produced no robust feature annotations")
                continue

            ann.insert(0, "dataset_id", dataset_id)
            ann.insert(1, "model", model)
            ann.insert(2, "representation", rep_name)
            ann.insert(3, "dataset_path", str(dataset_path))
            annot_rows.append(ann)

            ds_out = args.output_dir / dataset_id / model
            ds_out.mkdir(parents=True, exist_ok=True)
            ann.to_csv(ds_out / "robust_feature_pathway_annotations.csv", index=False)
            print(f"[ok] {dataset_id}::{model} | robust_annotated={ann.shape[0]}")

    if not annot_rows:
        raise RuntimeError("No robust feature annotations generated in Stage-4")

    annot_df = pd.concat(annot_rows, ignore_index=True)
    pair_df = _build_convergence_pairs(annot_df)
    summary_df = _build_consensus_summary(annot_df, pair_df)

    annot_csv = args.output_dir / "stage4_feature_annotations.csv"
    pair_csv = args.output_dir / "stage4_cross_model_pairs.csv"
    summary_csv = args.output_dir / "stage4_consensus_pathway_summary.csv"
    report_md = args.output_dir / "stage4_cross_model_convergence_report.md"
    run_cfg = args.output_dir / "run_config.json"

    annot_df.to_csv(annot_csv, index=False)
    pair_df.to_csv(pair_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)
    _write_report(annotation_df=annot_df, pair_df=pair_df, summary_df=summary_df, out_md=report_md)

    run_meta = {
        "stage3_scgpt_dir": str(args.stage3_scgpt_dir),
        "stage3_geneformer_dir": str(args.stage3_geneformer_dir),
        "common_datasets": common,
        "device": device,
        "top_genes_per_feature": int(args.top_genes_per_feature),
        "pathway_p_threshold": float(args.pathway_p_threshold),
        "min_pathway_overlap": int(args.min_pathway_overlap),
    }
    run_cfg.write_text(json.dumps(run_meta, indent=2), encoding="utf-8")

    print("[done] stage4 outputs")
    print(f"  - {annot_csv}")
    print(f"  - {pair_csv}")
    print(f"  - {summary_csv}")
    print(f"  - {report_md}")


if __name__ == "__main__":
    main()
