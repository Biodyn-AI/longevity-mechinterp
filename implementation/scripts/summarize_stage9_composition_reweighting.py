#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import anndata as ad
import numpy as np
import pandas as pd


INTERVENTIONS = ["old_push", "random_push", "young_push"]
METRICS = ["delta_expected_age_mean", "delta_old_prob_mean"]


def _ci_flag(low: float, high: float) -> str:
    if np.isfinite(low) and np.isfinite(high):
        if low > 0:
            return "positive"
        if high < 0:
            return "negative"
    return "uncertain"


def _bootstrap_mean(values: np.ndarray, n_boot: int, seed: int) -> Tuple[float, float, float]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan")
    if arr.size == 1:
        x = float(arr[0])
        return x, x, x
    rng = np.random.default_rng(int(seed))
    idx = rng.integers(0, arr.size, size=(int(n_boot), arr.size))
    boot = arr[idx].mean(axis=1)
    return float(np.mean(arr)), float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


def _bootstrap_weighted_mean(values: np.ndarray, weights: np.ndarray, n_boot: int, seed: int) -> Tuple[float, float, float]:
    arr = np.asarray(values, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    mask = np.isfinite(arr) & np.isfinite(w)
    arr = arr[mask]
    w = w[mask]
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan")
    if arr.size == 1:
        x = float(arr[0])
        return x, x, x
    w = np.maximum(w, 0.0)
    if float(np.sum(w)) <= 0.0:
        w = np.ones_like(w) / float(arr.size)
    else:
        w = w / float(np.sum(w))
    mean = float(np.sum(w * arr))
    rng = np.random.default_rng(int(seed))
    boot = np.zeros(int(n_boot), dtype=np.float64)
    n = arr.size
    for i in range(int(n_boot)):
        idx = rng.choice(n, size=n, replace=True, p=w)
        boot[i] = float(np.mean(arr[idx]))
    return mean, float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


def _default_paths(root: Path) -> Dict[str, Path]:
    return {
        "donor_consensus_csv": root
        / "implementation"
        / "outputs"
        / "stage9_aida_v2_global_scale2_seedexpansion_20260304"
        / "donor_bootstrap_ci"
        / "stage5_donor_consensus_means.csv",
        "sampled_obs_csv": root
        / "implementation"
        / "outputs"
        / "stage3_sae_pilot_geneformer_20260303_fast"
        / "aida_phase1_v2"
        / "sampled_obs.csv",
        "h5ad_path": root / "implementation" / "data_downloads" / "raw" / "aida_phase1_v2.h5ad",
        "output_dir": root / "implementation" / "outputs" / "stage9_composition_reweighting_20260304",
    }


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    defaults = _default_paths(root)
    parser = argparse.ArgumentParser(
        description="Composition-matched donor reweighting for Stage-9 strict-pass branch."
    )
    parser.add_argument("--donor-consensus-csv", type=Path, default=defaults["donor_consensus_csv"])
    parser.add_argument("--sampled-obs-csv", type=Path, default=defaults["sampled_obs_csv"])
    parser.add_argument("--h5ad-path", type=Path, default=defaults["h5ad_path"])
    parser.add_argument("--output-dir", type=Path, default=defaults["output_dir"])
    parser.add_argument("--dataset-id", type=str, default="aida_phase1_v2")
    parser.add_argument("--model", type=str, default="geneformer")
    parser.add_argument("--pathway", type=str, default="inflammation_nfkb")
    parser.add_argument("--top-celltypes", type=int, default=12)
    parser.add_argument("--ridge-alpha", type=float, default=10.0)
    parser.add_argument("--bootstrap-iters", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    return parser.parse_args()


def _build_donor_effect_table(df: pd.DataFrame) -> pd.DataFrame:
    sub = df.copy()
    pivot = sub.pivot_table(
        index="donor_id",
        columns="intervention",
        values=METRICS,
        aggfunc="first",
    )
    out = pd.DataFrame(index=pivot.index)
    for metric in METRICS:
        for intervention in INTERVENTIONS:
            key = (metric, intervention)
            if key in pivot.columns:
                out[f"{metric}__{intervention}"] = pivot[key]
        old_col = f"{metric}__old_push"
        rnd_col = f"{metric}__random_push"
        yng_col = f"{metric}__young_push"
        if old_col in out.columns and rnd_col in out.columns:
            out[f"{metric}__old_minus_random"] = out[old_col] - out[rnd_col]
        if yng_col in out.columns and rnd_col in out.columns:
            out[f"{metric}__young_minus_random"] = out[yng_col] - out[rnd_col]
        if old_col in out.columns and yng_col in out.columns:
            out[f"{metric}__old_minus_young"] = out[old_col] - out[yng_col]
    return out.reset_index().rename(columns={"index": "donor_id"})


def _resolve_donor_age(sampled_obs: pd.DataFrame) -> pd.DataFrame:
    work = sampled_obs.copy()
    work["age_numeric"] = pd.to_numeric(work["age_numeric"], errors="coerce")
    age = (
        work.groupby("donor_id", as_index=False)
        .agg(
            age_numeric_mean=("age_numeric", "mean"),
            age_label_mode=("age_label", lambda s: s.astype(str).mode().iloc[0] if len(s) else ""),
            n_cells_sampled=("donor_id", "count"),
        )
        .sort_values("donor_id")
        .reset_index(drop=True)
    )
    # Stable quantile bins from donor-level mean age.
    valid = age["age_numeric_mean"].dropna().to_numpy(dtype=np.float64)
    if valid.size >= 4:
        q = np.quantile(valid, [0.0, 0.25, 0.5, 0.75, 1.0])
        # ensure strictly increasing edges
        q = np.unique(q)
        if q.size >= 3:
            age["age_bin"] = pd.cut(age["age_numeric_mean"], bins=q, include_lowest=True, duplicates="drop").astype(str)
        else:
            age["age_bin"] = age["age_label_mode"].astype(str)
    else:
        age["age_bin"] = age["age_label_mode"].astype(str)
    return age


def _load_donor_composition_from_h5ad(h5ad_path: Path, donor_ids: List[str]) -> pd.DataFrame:
    adata = ad.read_h5ad(str(h5ad_path), backed="r")
    obs = adata.obs[["donor_id", "cell_type"]].copy()
    obs["donor_id"] = obs["donor_id"].astype(str)
    obs["cell_type"] = obs["cell_type"].astype(str)
    keep = set([str(x) for x in donor_ids])
    obs = obs[obs["donor_id"].isin(keep)].copy()
    counts = pd.crosstab(obs["donor_id"], obs["cell_type"])
    frac = counts.div(counts.sum(axis=1), axis=0).fillna(0.0)
    frac.index.name = "donor_id"
    return frac.reset_index()


def _select_celltype_features(comp: pd.DataFrame, top_k: int) -> Tuple[pd.DataFrame, List[str]]:
    work = comp.copy()
    celltype_cols = [c for c in work.columns if c != "donor_id"]
    totals = work[celltype_cols].sum(axis=0).sort_values(ascending=False)
    top_cols = totals.head(int(top_k)).index.astype(str).tolist()
    if len(top_cols) == 0:
        raise ValueError("No cell-type columns available for composition modeling.")
    other_cols = [c for c in celltype_cols if c not in set(top_cols)]
    work["__other__"] = work[other_cols].sum(axis=1) if len(other_cols) > 0 else 0.0
    model_cols = top_cols + ["__other__"]
    model = work[["donor_id"] + model_cols].copy()
    # row renormalize within selected+other set
    row_sum = model[model_cols].sum(axis=1).to_numpy(dtype=np.float64)
    row_sum = np.where(row_sum <= 0.0, 1.0, row_sum)
    model[model_cols] = model[model_cols].to_numpy(dtype=np.float64) / row_sum[:, None]
    return model, model_cols


def _solve_age_balancing_weights(comp: pd.DataFrame, age: pd.DataFrame, feat_cols: List[str], ridge_alpha: float) -> pd.DataFrame:
    merged = comp.merge(age[["donor_id", "age_bin"]], on="donor_id", how="inner").copy()
    if merged["age_bin"].astype(str).nunique() < 2:
        merged["weight_comp_balanced"] = 1.0 / float(len(merged))
        merged["target_match_l2"] = float("nan")
        return merged[["donor_id", "age_bin", "weight_comp_balanced", "target_match_l2"]]

    bin_means = merged.groupby("age_bin", as_index=True)[feat_cols].mean()
    target = bin_means.mean(axis=0).to_numpy(dtype=np.float64)

    P = merged[feat_cols].to_numpy(dtype=np.float64)  # n x k
    n = P.shape[0]
    u = np.ones(n, dtype=np.float64) / float(n)
    alpha = float(max(ridge_alpha, 1e-8))
    A = P.T  # k x n
    Q = A.T @ A + alpha * np.eye(n, dtype=np.float64)
    c = A.T @ target + alpha * u

    # Equality-constrained solve:
    #   min ||A w - target||^2 + alpha ||w-u||^2, s.t. 1^T w = 1
    K = np.zeros((n + 1, n + 1), dtype=np.float64)
    K[:n, :n] = Q
    K[:n, n] = 1.0
    K[n, :n] = 1.0
    rhs = np.zeros(n + 1, dtype=np.float64)
    rhs[:n] = c
    rhs[n] = 1.0
    sol = np.linalg.lstsq(K, rhs, rcond=None)[0]
    w = sol[:n]
    # enforce non-negativity softly and renormalize
    w = np.clip(w, 0.0, None)
    if float(np.sum(w)) <= 0.0:
        w = u.copy()
    else:
        w = w / float(np.sum(w))

    target_match_l2 = float(np.linalg.norm(P.T @ w - target))
    out = merged[["donor_id", "age_bin"]].copy()
    out["weight_comp_balanced"] = w
    out["target_match_l2"] = target_match_l2
    return out


def _balance_diagnostics(comp: pd.DataFrame, age: pd.DataFrame, weights: pd.DataFrame, feat_cols: List[str]) -> pd.DataFrame:
    merged = comp.merge(age[["donor_id", "age_bin"]], on="donor_id", how="inner")
    merged = merged.merge(weights[["donor_id", "weight_comp_balanced"]], on="donor_id", how="inner")
    bin_means = merged.groupby("age_bin", as_index=True)[feat_cols].mean()
    target = bin_means.mean(axis=0).to_numpy(dtype=np.float64)

    rows: List[Dict[str, object]] = []
    for bin_name, grp in merged.groupby("age_bin", as_index=False):
        x = grp[feat_cols].to_numpy(dtype=np.float64)
        unweighted = np.mean(x, axis=0)
        w = grp["weight_comp_balanced"].to_numpy(dtype=np.float64)
        if float(np.sum(w)) <= 0.0:
            w = np.ones_like(w) / float(len(w))
        else:
            w = w / float(np.sum(w))
        weighted = np.sum(x * w[:, None], axis=0)
        rows.append(
            {
                "age_bin": str(bin_name),
                "n_donors": int(grp.shape[0]),
                "l1_unweighted_to_target": float(np.sum(np.abs(unweighted - target))),
                "l1_weighted_to_target": float(np.sum(np.abs(weighted - target))),
                "l2_unweighted_to_target": float(np.linalg.norm(unweighted - target)),
                "l2_weighted_to_target": float(np.linalg.norm(weighted - target)),
            }
        )
    return pd.DataFrame(rows).sort_values("age_bin").reset_index(drop=True)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    donor_consensus = pd.read_csv(args.donor_consensus_csv)
    donor_sub = donor_consensus[
        (donor_consensus["dataset_id"].astype(str) == str(args.dataset_id))
        & (donor_consensus["model"].astype(str) == str(args.model))
        & (donor_consensus["pathway"].astype(str) == str(args.pathway))
    ].copy()
    donor_sub = donor_sub[donor_sub["intervention"].astype(str).isin(set(INTERVENTIONS))].copy()
    if donor_sub.empty:
        raise ValueError("No donor rows after filtering donor-consensus table.")

    donor_effect = _build_donor_effect_table(donor_sub)
    sampled_obs = pd.read_csv(args.sampled_obs_csv)
    sampled_obs["donor_id"] = sampled_obs["donor_id"].astype(str)
    donor_age = _resolve_donor_age(sampled_obs)

    donor_ids = donor_effect["donor_id"].astype(str).tolist()
    donor_comp_full = _load_donor_composition_from_h5ad(args.h5ad_path, donor_ids=donor_ids)
    donor_comp, feat_cols = _select_celltype_features(donor_comp_full, top_k=int(args.top_celltypes))

    merged = donor_effect.merge(donor_age[["donor_id", "age_numeric_mean", "age_bin"]], on="donor_id", how="inner")
    merged = merged.merge(donor_comp, on="donor_id", how="inner")
    merged = merged.sort_values("donor_id").reset_index(drop=True)
    if merged.empty:
        raise ValueError("No donors remain after joining effects, age, and composition tables.")

    weights = _solve_age_balancing_weights(
        comp=merged[["donor_id"] + feat_cols],
        age=merged[["donor_id", "age_bin"]],
        feat_cols=feat_cols,
        ridge_alpha=float(args.ridge_alpha),
    )
    merged = merged.merge(weights[["donor_id", "weight_comp_balanced"]], on="donor_id", how="left")
    merged["weight_comp_balanced"] = merged["weight_comp_balanced"].fillna(0.0)
    sum_w = float(merged["weight_comp_balanced"].sum())
    if sum_w <= 0.0:
        merged["weight_comp_balanced"] = 1.0 / float(merged.shape[0])
    else:
        merged["weight_comp_balanced"] = merged["weight_comp_balanced"] / sum_w

    balance_diag = _balance_diagnostics(
        comp=merged[["donor_id"] + feat_cols],
        age=merged[["donor_id", "age_bin"]],
        weights=merged[["donor_id", "weight_comp_balanced"]],
        feat_cols=feat_cols,
    )
    balance_diag.to_csv(args.output_dir / "composition_balance_diagnostics.csv", index=False)

    effect_rows: List[Dict[str, object]] = []
    effect_cols = [c for c in merged.columns if "__" in c and c.startswith("delta_")]
    for col in effect_cols:
        arr = merged[col].to_numpy(dtype=np.float64)
        w = merged["weight_comp_balanced"].to_numpy(dtype=np.float64)
        naive_mean, naive_low, naive_high = _bootstrap_mean(
            values=arr,
            n_boot=int(args.bootstrap_iters),
            seed=int(args.bootstrap_seed) + int(abs(hash(col)) % 10_000),
        )
        w_mean, w_low, w_high = _bootstrap_weighted_mean(
            values=arr,
            weights=w,
            n_boot=int(args.bootstrap_iters),
            seed=int(args.bootstrap_seed) + int(abs(hash(("w", col))) % 10_000),
        )
        metric, effect = col.split("__", 1)
        row = {
            "dataset_id": str(args.dataset_id),
            "model": str(args.model),
            "pathway": str(args.pathway),
            "metric": str(metric),
            "effect": str(effect),
            "n_donors": int(np.sum(np.isfinite(arr))),
            "naive_mean": float(naive_mean),
            "naive_ci_low": float(naive_low),
            "naive_ci_high": float(naive_high),
            "naive_ci_flag": _ci_flag(naive_low, naive_high),
            "reweighted_mean": float(w_mean),
            "reweighted_ci_low": float(w_low),
            "reweighted_ci_high": float(w_high),
            "reweighted_ci_flag": _ci_flag(w_low, w_high),
            "reweighted_minus_naive": float(w_mean - naive_mean),
        }
        effect_rows.append(row)

    effect_df = pd.DataFrame(effect_rows).sort_values(["metric", "effect"]).reset_index(drop=True)

    # Expected-age strict gate under reweighted effects.
    gate = {"pass_expected_age_reweighted_strict": False, "pass_full_reweighted_strict": False}
    try:
        e = effect_df
        def _flag(metric: str, effect: str, colname: str) -> str:
            sub = e[(e["metric"] == metric) & (e["effect"] == effect)]
            if sub.empty:
                return "uncertain"
            return str(sub.iloc[0][colname])

        ea_or = _flag("delta_expected_age_mean", "old_minus_random", "reweighted_ci_flag")
        ea_yr = _flag("delta_expected_age_mean", "young_minus_random", "reweighted_ci_flag")
        ea_oy = _flag("delta_expected_age_mean", "old_minus_young", "reweighted_ci_flag")
        ea_old = _flag("delta_expected_age_mean", "old_push", "reweighted_ci_flag")
        ea_yng = _flag("delta_expected_age_mean", "young_push", "reweighted_ci_flag")
        op_or = _flag("delta_old_prob_mean", "old_minus_random", "reweighted_ci_flag")
        op_yr = _flag("delta_old_prob_mean", "young_minus_random", "reweighted_ci_flag")

        pass_ea = ea_or == "positive" and ea_yr == "negative" and ea_oy == "positive" and ea_old == "positive" and ea_yng == "negative"
        pass_full = pass_ea and op_or == "positive" and op_yr == "negative"
        gate = {
            "pass_expected_age_reweighted_strict": bool(pass_ea),
            "pass_full_reweighted_strict": bool(pass_full),
            "ea_old_minus_random_flag": ea_or,
            "ea_young_minus_random_flag": ea_yr,
            "ea_old_minus_young_flag": ea_oy,
            "ea_old_push_flag": ea_old,
            "ea_young_push_flag": ea_yng,
            "oldprob_old_minus_random_flag": op_or,
            "oldprob_young_minus_random_flag": op_yr,
        }
    except Exception:
        pass

    donor_weights = merged[
        ["donor_id", "age_numeric_mean", "age_bin", "weight_comp_balanced"] + feat_cols
    ].copy()
    donor_weights.to_csv(args.output_dir / "donor_composition_weights.csv", index=False)
    effect_df.to_csv(args.output_dir / "composition_reweighting_effect_summary.csv", index=False)

    # compact view for key effects
    key_effects = effect_df[
        effect_df["effect"].isin(
            ["old_push", "young_push", "random_push", "old_minus_random", "young_minus_random", "old_minus_young"]
        )
    ].copy()
    key_effects.to_csv(args.output_dir / "composition_reweighting_key_effects.csv", index=False)

    ess = 1.0 / float(np.sum(np.square(merged["weight_comp_balanced"].to_numpy(dtype=np.float64))))
    report_lines = [
        "# Stage-9 Composition Reweighting Report",
        "",
        f"Dataset: `{args.dataset_id}`",
        f"Model: `{args.model}`",
        f"Pathway: `{args.pathway}`",
        "",
        "## Setup",
        "",
        f"- donors after joins: `{merged.shape[0]}`",
        f"- top cell-type features (+other): `{len(feat_cols)}`",
        f"- effective sample size of reweighted donors: `{ess:.2f}`",
        "",
        "## Balance Diagnostics",
        "",
        "```text",
        balance_diag.to_string(index=False),
        "```",
        "",
        "## Key Effect Comparison (Naive vs Composition-Reweighted)",
        "",
        "```text",
        key_effects.to_string(index=False),
        "```",
        "",
        "## Reweighted Strict Gate",
        "",
        "```json",
        json.dumps(gate, indent=2),
        "```",
        "",
        "Interpretation note: if key expected-age directions and strict gates remain after reweighting,",
        "the surviving branch is less likely to be explained purely by donor cell-type composition.",
        "",
    ]
    (args.output_dir / "composition_reweighting_report.md").write_text("\n".join(report_lines), encoding="utf-8")

    run_cfg = {
        "donor_consensus_csv": str(args.donor_consensus_csv),
        "sampled_obs_csv": str(args.sampled_obs_csv),
        "h5ad_path": str(args.h5ad_path),
        "dataset_id": str(args.dataset_id),
        "model": str(args.model),
        "pathway": str(args.pathway),
        "top_celltypes": int(args.top_celltypes),
        "ridge_alpha": float(args.ridge_alpha),
        "bootstrap_iters": int(args.bootstrap_iters),
        "bootstrap_seed": int(args.bootstrap_seed),
    }
    (args.output_dir / "run_config.json").write_text(json.dumps(run_cfg, indent=2), encoding="utf-8")
    print(f"[done] composition reweighting outputs written to: {args.output_dir}")


if __name__ == "__main__":
    main()
