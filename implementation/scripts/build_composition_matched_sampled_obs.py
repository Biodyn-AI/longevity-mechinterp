#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import anndata as ad
import numpy as np
import pandas as pd


def _allocate_with_caps(
    target_n: int,
    proportions: np.ndarray,
    availability: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    p = np.asarray(proportions, dtype=np.float64)
    p = np.where(np.isfinite(p), p, 0.0)
    if float(np.sum(p)) <= 0:
        p = np.ones_like(p) / float(len(p))
    else:
        p = p / float(np.sum(p))

    avail = np.asarray(availability, dtype=np.int64)
    expected = p * float(target_n)
    out = np.floor(expected).astype(np.int64)
    out = np.minimum(out, avail)
    remaining = int(target_n - int(out.sum()))
    if remaining <= 0:
        return out

    # First distribute by largest fractional residual where capacity remains.
    residual = expected - out.astype(np.float64)
    while remaining > 0:
        mask = avail > out
        if not np.any(mask):
            break
        score = np.where(mask, residual, -np.inf)
        best = int(np.argmax(score))
        if not np.isfinite(score[best]):
            break
        out[best] += 1
        remaining -= 1
        residual[best] = expected[best] - float(out[best])

    # If still remaining (rare), distribute among any with spare capacity randomly.
    while remaining > 0:
        spare_idx = np.where(avail > out)[0]
        if spare_idx.size == 0:
            break
        pick = int(rng.choice(spare_idx, size=1)[0])
        out[pick] += 1
        remaining -= 1

    return out


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Build a composition-matched sampled_obs.csv for Stage-5 forward-pass reruns."
    )
    parser.add_argument(
        "--h5ad-path",
        type=Path,
        default=root / "implementation" / "data_downloads" / "raw" / "aida_phase1_v2.h5ad",
    )
    parser.add_argument(
        "--age-source-sampled-obs",
        type=Path,
        default=root
        / "implementation"
        / "outputs"
        / "stage3_sae_pilot_geneformer_20260303_fast"
        / "aida_phase1_v2"
        / "sampled_obs.csv",
    )
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--target-n-cells", type=int, default=700)
    parser.add_argument("--target-age-bins", type=str, default="")
    parser.add_argument("--top-celltypes", type=int, default=16)
    parser.add_argument("--min-count-per-bin", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(int(args.seed))

    age_source = pd.read_csv(args.age_source_sampled_obs)
    age_source["donor_id"] = age_source["donor_id"].astype(str)
    age_source["age_numeric"] = pd.to_numeric(age_source["age_numeric"], errors="coerce")
    donor_age = (
        age_source.groupby("donor_id", as_index=False)
        .agg(
            age_label=("age_label", lambda s: s.astype(str).mode().iloc[0] if len(s) else ""),
            age_numeric=("age_numeric", "mean"),
        )
        .dropna(subset=["age_numeric"])
        .reset_index(drop=True)
    )
    age_order = donor_age.groupby("age_label", as_index=True)["age_numeric"].mean().sort_values()
    age_bins = age_order.index.astype(str).tolist()
    if args.target_age_bins.strip():
        keep_bins = [x.strip() for x in str(args.target_age_bins).split(",") if x.strip()]
        age_bins = [x for x in age_bins if x in set(keep_bins)]
    if len(age_bins) < 2:
        raise ValueError("Need at least 2 age bins for composition matching.")

    donor_age = donor_age[donor_age["age_label"].astype(str).isin(set(age_bins))].copy()
    donor_keep = set(donor_age["donor_id"].astype(str))

    adata = ad.read_h5ad(str(args.h5ad_path), backed="r")
    obs = adata.obs[["donor_id", "cell_type"]].copy()
    obs = obs.reset_index().rename(columns={"index": "obs_name"})
    obs.insert(0, "obs_row", np.arange(obs.shape[0], dtype=np.int64))
    obs["donor_id"] = obs["donor_id"].astype(str)
    obs["cell_type"] = obs["cell_type"].astype(str)
    obs = obs[obs["donor_id"].isin(donor_keep)].copy()
    obs = obs.merge(donor_age, on="donor_id", how="left")
    obs = obs[obs["age_label"].astype(str).isin(set(age_bins))].copy()
    if obs.empty:
        raise RuntimeError("No cells available after donor/age filtering.")

    counts = pd.crosstab(obs["age_label"], obs["cell_type"]).reindex(index=age_bins, fill_value=0)
    min_per_type = counts.min(axis=0)
    eligible_types = min_per_type[min_per_type >= int(args.min_count_per_bin)].sort_values(ascending=False).index.astype(str).tolist()
    if len(eligible_types) == 0:
        raise RuntimeError("No eligible cell types after min-count-per-bin filter.")
    if int(args.top_celltypes) > 0:
        eligible_types = eligible_types[: int(args.top_celltypes)]

    counts_sub = counts[eligible_types].copy()
    # Target composition as average bin-normalized proportions.
    bin_props = counts_sub.div(counts_sub.sum(axis=1), axis=0).fillna(0.0)
    p_target = bin_props.mean(axis=0).to_numpy(dtype=np.float64)
    p_target = p_target / float(np.sum(p_target))

    n_bins = len(age_bins)
    n_total = int(args.target_n_cells)
    n_per_bin = np.full(n_bins, n_total // n_bins, dtype=np.int64)
    n_per_bin[: (n_total % n_bins)] += 1

    target_matrix = pd.DataFrame(0, index=age_bins, columns=eligible_types, dtype=np.int64)
    sampled_idx: List[int] = []

    for i, age_bin in enumerate(age_bins):
        target_bin = int(n_per_bin[i])
        avail = counts_sub.loc[age_bin, eligible_types].to_numpy(dtype=np.int64)
        alloc = _allocate_with_caps(target_n=target_bin, proportions=p_target, availability=avail, rng=rng)
        target_matrix.loc[age_bin, eligible_types] = alloc

        for j, cell_type in enumerate(eligible_types):
            n_take = int(alloc[j])
            if n_take <= 0:
                continue
            cand = obs[(obs["age_label"].astype(str) == str(age_bin)) & (obs["cell_type"].astype(str) == str(cell_type))].copy()
            if cand.shape[0] < n_take:
                n_take = int(cand.shape[0])
            # Donor-debias sampling inside stratum.
            donor_counts = cand["donor_id"].value_counts()
            w = cand["donor_id"].map(lambda d: 1.0 / float(donor_counts.get(d, 1))).to_numpy(dtype=np.float64)
            w = w / float(np.sum(w))
            pick = rng.choice(cand.index.to_numpy(dtype=np.int64), size=n_take, replace=False, p=w)
            sampled_idx.extend(pick.tolist())

    sampled = obs.loc[sampled_idx].copy()
    sampled = sampled.sort_values("obs_row").reset_index(drop=True)
    sampled = sampled[["obs_row", "obs_name", "age_label", "age_numeric", "donor_id", "cell_type"]]

    # Build diagnostics.
    achieved_counts = (
        pd.crosstab(sampled["age_label"], sampled["cell_type"])
        .reindex(index=age_bins, columns=eligible_types, fill_value=0)
        .astype(int)
    )
    achieved_props = achieved_counts.div(achieved_counts.sum(axis=1), axis=0).fillna(0.0)
    l1_by_bin = np.sum(np.abs(achieved_props.to_numpy(dtype=np.float64) - p_target.reshape(1, -1)), axis=1)

    sampled.to_csv(args.output_csv, index=False)
    target_matrix.to_csv(args.output_csv.with_name(args.output_csv.stem + "_target_counts.csv"))
    achieved_counts.to_csv(args.output_csv.with_name(args.output_csv.stem + "_achieved_counts.csv"))
    diag = pd.DataFrame(
        {
            "age_bin": age_bins,
            "target_cells_in_bin": n_per_bin,
            "achieved_cells_in_bin": achieved_counts.sum(axis=1).to_numpy(dtype=np.int64),
            "l1_prop_distance_to_target": l1_by_bin,
        }
    )
    diag.to_csv(args.output_csv.with_name(args.output_csv.stem + "_diagnostics.csv"), index=False)

    run_cfg = {
        "h5ad_path": str(args.h5ad_path),
        "age_source_sampled_obs": str(args.age_source_sampled_obs),
        "output_csv": str(args.output_csv),
        "target_n_cells": int(args.target_n_cells),
        "age_bins": age_bins,
        "top_celltypes": int(args.top_celltypes),
        "min_count_per_bin": int(args.min_count_per_bin),
        "seed": int(args.seed),
        "n_output_cells": int(sampled.shape[0]),
        "n_output_donors": int(sampled["donor_id"].astype(str).nunique()),
        "n_output_celltypes": int(sampled["cell_type"].astype(str).nunique()),
    }
    args.output_csv.with_name(args.output_csv.stem + "_run_config.json").write_text(
        json.dumps(run_cfg, indent=2), encoding="utf-8"
    )
    print(f"[done] wrote composition-matched sampled_obs: {args.output_csv}")
    print(
        f"[stats] cells={sampled.shape[0]} donors={sampled['donor_id'].nunique()} "
        f"celltypes={sampled['cell_type'].nunique()}"
    )


if __name__ == "__main__":
    main()
