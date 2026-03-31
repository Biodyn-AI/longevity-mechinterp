#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from run_stage1_longevity_mechinterp import (
    GeneformerRuntime,
    ScGPTRuntime,
    _device_auto,
    _ensure_single_cell_src_on_path,
    _load_subset_anndata,
)

SPLIT_COLUMNS: List[str] = [
    "dataset_id",
    "model",
    "pathway",
    "representation",
    "analysis_scope",
    "cell_type",
    "n_cells",
    "n_donors",
    "n_age_classes",
    "n_features_in_pathway_test",
    "seed",
    "split_idx",
    "intervention",
    "baseline_balanced_accuracy",
    "intervened_balanced_accuracy",
    "delta_balanced_accuracy",
    "baseline_expected_age_mean",
    "intervened_expected_age_mean",
    "delta_expected_age_mean",
    "baseline_old_prob_mean",
    "intervened_old_prob_mean",
    "delta_old_prob_mean",
    "baseline_pathway_activation_mean",
    "intervened_pathway_activation_mean",
    "delta_pathway_activation_mean",
]

DONOR_COLUMNS: List[str] = [
    "dataset_id",
    "model",
    "pathway",
    "representation",
    "analysis_scope",
    "cell_type",
    "n_cells",
    "n_donors",
    "n_age_classes",
    "n_features_in_pathway_test",
    "seed",
    "split_idx",
    "intervention",
    "donor_id",
    "n_cells_donor_split",
    "baseline_balanced_accuracy",
    "intervened_balanced_accuracy",
    "delta_balanced_accuracy",
    "baseline_expected_age_mean",
    "intervened_expected_age_mean",
    "delta_expected_age_mean",
    "baseline_old_prob_mean",
    "intervened_old_prob_mean",
    "delta_old_prob_mean",
    "baseline_pathway_activation_mean",
    "intervened_pathway_activation_mean",
    "delta_pathway_activation_mean",
]

SUMMARY_COLUMNS: List[str] = [
    "dataset_id",
    "model",
    "pathway",
    "representation",
    "analysis_scope",
    "cell_type",
    "intervention",
    "n_splits",
    "mean_delta_balanced_accuracy",
    "std_delta_balanced_accuracy",
    "mean_delta_expected_age",
    "std_delta_expected_age",
    "mean_delta_old_prob",
    "std_delta_old_prob",
    "mean_delta_pathway_activation",
    "std_delta_pathway_activation",
    "baseline_balanced_accuracy_mean",
    "baseline_old_prob_mean",
    "n_features",
    "n_cells",
    "n_donors",
    "n_age_classes",
]

DIRECTION_COLUMNS: List[str] = [
    "dataset_id",
    "analysis_scope",
    "cell_type",
    "pathway",
    "n_cross_model_pairs",
    "scgpt_old_push_delta_expected_age",
    "geneformer_old_push_delta_expected_age",
    "scgpt_old_push_delta_old_prob",
    "geneformer_old_push_delta_old_prob",
    "sign_agree_expected_age",
    "sign_agree_old_prob",
    "scgpt_directional_pattern_ok",
    "geneformer_directional_pattern_ok",
    "both_models_stronger_than_random",
]


@dataclass
class PreparedModelData:
    dataset_id: str
    model: str
    pathway: str
    representation: str
    dataset_path: Path
    metadata: pd.DataFrame
    representation_matrix: np.ndarray
    latent_matrix: np.ndarray
    sae_artifacts: Dict[str, np.ndarray]
    feature_ids: np.ndarray
    feature_signs: np.ndarray


def _resolve_default_output_dir(root: Path, prefix: str, fallback_suffix: str) -> Path:
    outputs = root / "implementation" / "outputs"
    candidates = sorted(outputs.glob(f"{prefix}_*"))
    if candidates:
        return candidates[-1]
    return outputs / f"{prefix}_{fallback_suffix}"


def _parse_csv_list(text: str) -> List[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


def _parse_int_csv(text: str) -> List[int]:
    return [int(x.strip()) for x in str(text).split(",") if x.strip()]


def _slugify_for_path(text: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", str(text).strip())
    slug = slug.strip("_")
    return slug or "unknown"


def _layer_index_from_representation(name: str) -> int:
    marker = "scgpt_layer_"
    if marker not in name:
        raise ValueError(f"Cannot parse scGPT layer index from representation '{name}'")
    return int(name.split(marker, 1)[1])


def _extract_representation(
    model_name: str,
    representation_name: str,
    adata,
    scgpt_runtime: Optional[ScGPTRuntime],
    geneformer_runtime: Optional[GeneformerRuntime],
    scgpt_batch_size: int,
    scgpt_max_genes: int,
    geneformer_batch_size: int,
    geneformer_max_genes: int,
) -> np.ndarray:
    if model_name == "scgpt":
        if scgpt_runtime is None:
            raise RuntimeError("scGPT runtime is not initialized")
        layer_idx = _layer_index_from_representation(representation_name)
        reps = scgpt_runtime.extract_representations(
            adata=adata,
            batch_size=scgpt_batch_size,
            max_genes=scgpt_max_genes,
            layer_indices=[layer_idx],
        )
        if representation_name not in reps:
            raise KeyError(f"Missing extracted representation: {representation_name}")
        return np.asarray(reps[representation_name], dtype=np.float32)

    if model_name == "geneformer":
        if geneformer_runtime is None:
            raise RuntimeError("Geneformer runtime is not initialized")
        extracted_name, rep = geneformer_runtime.extract_representation(
            adata=adata,
            max_genes_per_cell=geneformer_max_genes,
            batch_size=geneformer_batch_size,
        )
        if extracted_name != representation_name:
            raise KeyError(
                f"Geneformer representation mismatch: expected '{representation_name}', got '{extracted_name}'"
            )
        return np.asarray(rep, dtype=np.float32)

    raise ValueError(f"Unsupported model: {model_name}")


def _load_sae_artifacts(npz_path: Path) -> Dict[str, np.ndarray]:
    arr = np.load(npz_path)
    required = [
        "input_mean",
        "input_std",
        "encoder_weight",
        "encoder_bias",
        "decoder_weight",
        "decoder_bias",
    ]
    missing = [k for k in required if k not in arr]
    if missing:
        raise KeyError(f"Missing keys in SAE artifacts {npz_path}: {missing}")
    return {k: np.asarray(arr[k]) for k in arr.files}


def _compute_latent(X: np.ndarray, artifacts: Dict[str, np.ndarray]) -> np.ndarray:
    input_mean = artifacts["input_mean"].reshape(1, -1)
    input_std = artifacts["input_std"].reshape(1, -1)
    input_std = np.where(input_std < 1e-6, 1.0, input_std)

    encoder_weight = artifacts["encoder_weight"]
    encoder_bias = artifacts["encoder_bias"].reshape(1, -1)

    X_norm = (X - input_mean) / input_std
    latent_pre = X_norm @ encoder_weight.T + encoder_bias
    return np.maximum(latent_pre, 0.0).astype(np.float32, copy=False)


def _decode_from_latent(latent: np.ndarray, artifacts: Dict[str, np.ndarray]) -> np.ndarray:
    decoder_weight = artifacts["decoder_weight"]
    decoder_bias = artifacts["decoder_bias"].reshape(1, -1)
    input_mean = artifacts["input_mean"].reshape(1, -1)
    input_std = artifacts["input_std"].reshape(1, -1)
    input_std = np.where(input_std < 1e-6, 1.0, input_std)

    X_norm_hat = latent @ decoder_weight.T + decoder_bias
    X_hat = X_norm_hat * input_std + input_mean
    return np.asarray(X_hat, dtype=np.float32)


def _expected_age_and_old_prob(model: Pipeline, X: np.ndarray, old_class_idx: int) -> Tuple[np.ndarray, np.ndarray]:
    clf = model.named_steps["clf"]
    classes = clf.classes_.astype(np.int64, copy=False)
    proba = model.predict_proba(X)

    expected_age = proba @ classes.astype(np.float64)
    old_prob = np.zeros(proba.shape[0], dtype=np.float64)
    old_pos = np.where(classes == int(old_class_idx))[0]
    if old_pos.size > 0:
        old_prob = proba[:, int(old_pos[0])]
    return expected_age, old_prob


def _sample_random_feature_ids(
    latent_dim: int,
    feature_ids: np.ndarray,
    n_pick: int,
    rng: np.random.Generator,
) -> np.ndarray:
    pool = np.setdiff1d(np.arange(latent_dim, dtype=np.int64), feature_ids, assume_unique=False)
    if pool.size == 0:
        return np.array([], dtype=np.int64)
    replace = pool.size < n_pick
    sampled = rng.choice(pool, size=n_pick, replace=replace)
    return np.asarray(sampled, dtype=np.int64)


def _run_probe_interventions(
    data: PreparedModelData,
    n_splits: int,
    split_seeds: Sequence[int],
    intervention_scale: float,
    analysis_scope: str,
    cell_type: str,
    collect_donor_level: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    meta = data.metadata
    X = data.representation_matrix
    latent = data.latent_matrix
    artifacts = data.sae_artifacts
    feature_ids = data.feature_ids
    feature_signs = data.feature_signs

    # Build age-class codes in chronological order using age_numeric means.
    # This avoids arbitrary factorization order and makes "older-direction"
    # readouts interpretable across splits.
    age_label_series = meta["age_label"].astype(str)
    age_numeric_series = pd.to_numeric(meta["age_numeric"], errors="coerce")
    age_order = (
        pd.DataFrame({"age_label": age_label_series, "age_numeric": age_numeric_series})
        .groupby("age_label", as_index=True)["age_numeric"]
        .mean()
        .sort_values()
    )
    ordered_labels = age_order.index.astype(str).tolist()
    label_to_code = {lbl: i for i, lbl in enumerate(ordered_labels)}
    y_codes = age_label_series.map(label_to_code).to_numpy(dtype=np.int64)
    age_classes = np.array(ordered_labels, dtype=object)
    groups = meta["donor_id"].astype(str).to_numpy(dtype=object)
    old_class_idx = int(len(age_classes) - 1)

    if feature_ids.size == 0:
        return pd.DataFrame()

    latent_dim = int(latent.shape[1])
    latent_std = np.std(latent, axis=0).astype(np.float32, copy=False)
    feature_steps = intervention_scale * np.maximum(latent_std[feature_ids], 1e-4)

    rows: List[Dict[str, Any]] = []
    donor_rows: List[Dict[str, Any]] = []
    split_counter = 0
    for seed in split_seeds:
        splitter = GroupShuffleSplit(n_splits=n_splits, test_size=0.25, random_state=int(seed))
        for local_split, (train_idx, test_idx) in enumerate(splitter.split(X, y_codes, groups=groups)):
            y_train = y_codes[train_idx]
            y_test = y_codes[test_idx]
            if np.unique(y_train).size < 2 or np.unique(y_test).size < 2:
                continue

            probe = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "clf",
                        LogisticRegression(
                            max_iter=2000,
                            class_weight="balanced",
                            solver="lbfgs",
                        ),
                    ),
                ]
            )
            probe.fit(X[train_idx], y_train)

            X_test = X[test_idx]
            latent_test = latent[test_idx].copy()
            base_pred = probe.predict(X_test)
            base_bacc = float(balanced_accuracy_score(y_test, base_pred))
            base_expected_age, base_old_prob = _expected_age_and_old_prob(
                probe,
                X_test,
                old_class_idx=old_class_idx,
            )
            base_pathway_act = float(np.mean(latent_test[:, feature_ids]))

            donor_index_map: Dict[str, np.ndarray] = {}
            donor_base: Dict[str, Dict[str, float]] = {}
            if collect_donor_level:
                donor_ids_test = groups[test_idx].astype(str)
                donor_unique = np.unique(donor_ids_test)
                for donor_id in donor_unique:
                    donor_mask = donor_ids_test == donor_id
                    donor_index_map[str(donor_id)] = donor_mask
                    y_d = y_test[donor_mask]
                    base_bacc_d = float("nan")
                    if np.unique(y_d).size >= 2:
                        base_bacc_d = float(balanced_accuracy_score(y_d, base_pred[donor_mask]))
                    donor_base[str(donor_id)] = {
                        "n_cells_donor_split": float(np.sum(donor_mask)),
                        "baseline_balanced_accuracy": base_bacc_d,
                        "baseline_expected_age_mean": float(np.mean(base_expected_age[donor_mask])),
                        "baseline_old_prob_mean": float(np.mean(base_old_prob[donor_mask])),
                        "baseline_pathway_activation_mean": float(np.mean(latent_test[donor_mask][:, feature_ids])),
                    }

            rng = np.random.default_rng(int(seed) * 10_000 + int(local_split) + 17)
            random_ids = _sample_random_feature_ids(
                latent_dim=latent_dim,
                feature_ids=feature_ids,
                n_pick=int(feature_ids.size),
                rng=rng,
            )

            interventions: List[Tuple[str, np.ndarray, np.ndarray, bool]] = [
                ("old_push", feature_ids, feature_signs, False),
                ("young_push", feature_ids, -feature_signs, False),
                ("ablate", feature_ids, feature_signs, True),
            ]
            if random_ids.size > 0:
                random_signs = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=random_ids.size)
                interventions.append(("random_push", random_ids, random_signs, False))

            for intervention_name, target_ids, target_signs, do_ablate in interventions:
                latent_mod = latent_test.copy()
                if do_ablate:
                    latent_mod[:, target_ids] = 0.0
                else:
                    if intervention_name == "random_push":
                        step = float(np.mean(feature_steps)) if feature_steps.size > 0 else 0.0
                        latent_mod[:, target_ids] += step * target_signs.reshape(1, -1)
                    else:
                        latent_mod[:, target_ids] += feature_steps.reshape(1, -1) * target_signs.reshape(1, -1)

                X_mod = _decode_from_latent(latent_mod, artifacts)
                mod_pred = probe.predict(X_mod)
                mod_bacc = float(balanced_accuracy_score(y_test, mod_pred))
                mod_expected_age, mod_old_prob = _expected_age_and_old_prob(
                    probe,
                    X_mod,
                    old_class_idx=old_class_idx,
                )
                mod_pathway_act = float(np.mean(latent_mod[:, feature_ids]))

                rows.append(
                    {
                        "dataset_id": data.dataset_id,
                        "model": data.model,
                        "pathway": data.pathway,
                        "representation": data.representation,
                        "analysis_scope": str(analysis_scope),
                        "cell_type": str(cell_type),
                        "n_cells": int(X.shape[0]),
                        "n_donors": int(meta["donor_id"].nunique()),
                        "n_age_classes": int(len(age_classes)),
                        "n_features_in_pathway_test": int(feature_ids.size),
                        "seed": int(seed),
                        "split_idx": int(split_counter),
                        "intervention": intervention_name,
                        "baseline_balanced_accuracy": base_bacc,
                        "intervened_balanced_accuracy": mod_bacc,
                        "delta_balanced_accuracy": float(mod_bacc - base_bacc),
                        "baseline_expected_age_mean": float(np.mean(base_expected_age)),
                        "intervened_expected_age_mean": float(np.mean(mod_expected_age)),
                        "delta_expected_age_mean": float(np.mean(mod_expected_age - base_expected_age)),
                        "baseline_old_prob_mean": float(np.mean(base_old_prob)),
                        "intervened_old_prob_mean": float(np.mean(mod_old_prob)),
                        "delta_old_prob_mean": float(np.mean(mod_old_prob - base_old_prob)),
                        "baseline_pathway_activation_mean": base_pathway_act,
                        "intervened_pathway_activation_mean": mod_pathway_act,
                        "delta_pathway_activation_mean": float(mod_pathway_act - base_pathway_act),
                    }
                )

                if collect_donor_level and donor_base:
                    for donor_id, base_stats in donor_base.items():
                        donor_mask = donor_index_map[donor_id]
                        y_d = y_test[donor_mask]
                        mod_pred_d = mod_pred[donor_mask]
                        mod_bacc_d = float("nan")
                        if np.unique(y_d).size >= 2:
                            mod_bacc_d = float(balanced_accuracy_score(y_d, mod_pred_d))

                        donor_rows.append(
                            {
                                "dataset_id": data.dataset_id,
                                "model": data.model,
                                "pathway": data.pathway,
                                "representation": data.representation,
                                "analysis_scope": str(analysis_scope),
                                "cell_type": str(cell_type),
                                "n_cells": int(X.shape[0]),
                                "n_donors": int(meta["donor_id"].nunique()),
                                "n_age_classes": int(len(age_classes)),
                                "n_features_in_pathway_test": int(feature_ids.size),
                                "seed": int(seed),
                                "split_idx": int(split_counter),
                                "intervention": intervention_name,
                                "donor_id": str(donor_id),
                                "n_cells_donor_split": int(base_stats["n_cells_donor_split"]),
                                "baseline_balanced_accuracy": float(base_stats["baseline_balanced_accuracy"]),
                                "intervened_balanced_accuracy": mod_bacc_d,
                                "delta_balanced_accuracy": float(mod_bacc_d - base_stats["baseline_balanced_accuracy"])
                                if np.isfinite(mod_bacc_d) and np.isfinite(base_stats["baseline_balanced_accuracy"])
                                else float("nan"),
                                "baseline_expected_age_mean": float(base_stats["baseline_expected_age_mean"]),
                                "intervened_expected_age_mean": float(np.mean(mod_expected_age[donor_mask])),
                                "delta_expected_age_mean": float(
                                    np.mean(mod_expected_age[donor_mask] - base_expected_age[donor_mask])
                                ),
                                "baseline_old_prob_mean": float(base_stats["baseline_old_prob_mean"]),
                                "intervened_old_prob_mean": float(np.mean(mod_old_prob[donor_mask])),
                                "delta_old_prob_mean": float(
                                    np.mean(mod_old_prob[donor_mask] - base_old_prob[donor_mask])
                                ),
                                "baseline_pathway_activation_mean": float(base_stats["baseline_pathway_activation_mean"]),
                                "intervened_pathway_activation_mean": float(np.mean(latent_mod[donor_mask][:, feature_ids])),
                                "delta_pathway_activation_mean": float(
                                    np.mean(latent_mod[donor_mask][:, feature_ids]) - base_stats["baseline_pathway_activation_mean"]
                                ),
                            }
                        )
            split_counter += 1

    if not rows:
        return pd.DataFrame(columns=SPLIT_COLUMNS), pd.DataFrame(columns=DONOR_COLUMNS)
    split_df = pd.DataFrame(rows)
    donor_df = pd.DataFrame(donor_rows) if donor_rows else pd.DataFrame(columns=DONOR_COLUMNS)
    return split_df, donor_df


def _summarize_intervention_results(split_df: pd.DataFrame) -> pd.DataFrame:
    if split_df.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)

    summary = (
        split_df.groupby(
            ["dataset_id", "model", "pathway", "representation", "analysis_scope", "cell_type", "intervention"],
            as_index=False,
        )
        .agg(
            n_splits=("split_idx", "count"),
            mean_delta_balanced_accuracy=("delta_balanced_accuracy", "mean"),
            std_delta_balanced_accuracy=("delta_balanced_accuracy", "std"),
            mean_delta_expected_age=("delta_expected_age_mean", "mean"),
            std_delta_expected_age=("delta_expected_age_mean", "std"),
            mean_delta_old_prob=("delta_old_prob_mean", "mean"),
            std_delta_old_prob=("delta_old_prob_mean", "std"),
            mean_delta_pathway_activation=("delta_pathway_activation_mean", "mean"),
            std_delta_pathway_activation=("delta_pathway_activation_mean", "std"),
            baseline_balanced_accuracy_mean=("baseline_balanced_accuracy", "mean"),
            baseline_old_prob_mean=("baseline_old_prob_mean", "mean"),
            n_features=("n_features_in_pathway_test", "max"),
            n_cells=("n_cells", "max"),
            n_donors=("n_donors", "max"),
            n_age_classes=("n_age_classes", "max"),
        )
        .sort_values(["dataset_id", "analysis_scope", "cell_type", "pathway", "model", "intervention"])
        .reset_index(drop=True)
    )
    return summary


def _cross_model_directionality(
    summary_df: pd.DataFrame,
    consensus_df: pd.DataFrame,
) -> pd.DataFrame:
    if summary_df.empty:
        return pd.DataFrame(columns=DIRECTION_COLUMNS)

    old_push = summary_df[summary_df["intervention"] == "old_push"].copy()
    young_push = summary_df[summary_df["intervention"] == "young_push"].copy()
    random_push = summary_df[summary_df["intervention"] == "random_push"].copy()

    rows: List[Dict[str, Any]] = []
    keys = sorted(set(zip(old_push["dataset_id"], old_push["analysis_scope"], old_push["cell_type"], old_push["pathway"])))
    for dataset_id, analysis_scope, cell_type, pathway in keys:
        old_sub = old_push[
            (old_push["dataset_id"] == dataset_id)
            & (old_push["analysis_scope"] == analysis_scope)
            & (old_push["cell_type"] == cell_type)
            & (old_push["pathway"] == pathway)
        ]
        if old_sub["model"].nunique() < 2:
            continue

        def _metric(sub: pd.DataFrame, model: str, column: str) -> float:
            tmp = sub[sub["model"] == model]
            if tmp.empty:
                return float("nan")
            return float(tmp.iloc[0][column])

        sc_old_exp = _metric(old_sub, "scgpt", "mean_delta_expected_age")
        gf_old_exp = _metric(old_sub, "geneformer", "mean_delta_expected_age")
        sc_old_prob = _metric(old_sub, "scgpt", "mean_delta_old_prob")
        gf_old_prob = _metric(old_sub, "geneformer", "mean_delta_old_prob")

        young_sub = young_push[
            (young_push["dataset_id"] == dataset_id)
            & (young_push["analysis_scope"] == analysis_scope)
            & (young_push["cell_type"] == cell_type)
            & (young_push["pathway"] == pathway)
        ]
        sc_young_exp = _metric(young_sub, "scgpt", "mean_delta_expected_age")
        gf_young_exp = _metric(young_sub, "geneformer", "mean_delta_expected_age")

        random_sub = random_push[
            (random_push["dataset_id"] == dataset_id)
            & (random_push["analysis_scope"] == analysis_scope)
            & (random_push["cell_type"] == cell_type)
            & (random_push["pathway"] == pathway)
        ]
        sc_rand_prob = _metric(random_sub, "scgpt", "mean_delta_old_prob")
        gf_rand_prob = _metric(random_sub, "geneformer", "mean_delta_old_prob")

        sc_direction_ok = bool(np.isfinite(sc_old_exp) and np.isfinite(sc_young_exp) and sc_old_exp > 0 and sc_young_exp < 0)
        gf_direction_ok = bool(np.isfinite(gf_old_exp) and np.isfinite(gf_young_exp) and gf_old_exp > 0 and gf_young_exp < 0)

        sign_agree_expected = bool(np.sign(sc_old_exp) == np.sign(gf_old_exp)) if np.isfinite(sc_old_exp) and np.isfinite(gf_old_exp) else False
        sign_agree_old_prob = bool(np.sign(sc_old_prob) == np.sign(gf_old_prob)) if np.isfinite(sc_old_prob) and np.isfinite(gf_old_prob) else False
        stronger_than_random = bool(
            np.isfinite(sc_old_prob)
            and np.isfinite(gf_old_prob)
            and np.isfinite(sc_rand_prob)
            and np.isfinite(gf_rand_prob)
            and abs(sc_old_prob) > abs(sc_rand_prob)
            and abs(gf_old_prob) > abs(gf_rand_prob)
        )

        consensus_sub = consensus_df[
            (consensus_df["dataset_id"].astype(str) == str(dataset_id))
            & (consensus_df["pathway"].astype(str) == str(pathway))
        ]
        n_pairs = int(consensus_sub.iloc[0]["n_cross_model_pairs"]) if not consensus_sub.empty else 0

        rows.append(
            {
                "dataset_id": str(dataset_id),
                "analysis_scope": str(analysis_scope),
                "cell_type": str(cell_type),
                "pathway": str(pathway),
                "n_cross_model_pairs": int(n_pairs),
                "scgpt_old_push_delta_expected_age": sc_old_exp,
                "geneformer_old_push_delta_expected_age": gf_old_exp,
                "scgpt_old_push_delta_old_prob": sc_old_prob,
                "geneformer_old_push_delta_old_prob": gf_old_prob,
                "sign_agree_expected_age": sign_agree_expected,
                "sign_agree_old_prob": sign_agree_old_prob,
                "scgpt_directional_pattern_ok": sc_direction_ok,
                "geneformer_directional_pattern_ok": gf_direction_ok,
                "both_models_stronger_than_random": stronger_than_random,
            }
        )

    if not rows:
        return pd.DataFrame(columns=DIRECTION_COLUMNS)
    return (
        pd.DataFrame(rows)
        .sort_values(["dataset_id", "analysis_scope", "cell_type", "n_cross_model_pairs"], ascending=[True, True, True, False])
        .reset_index(drop=True)
    )


def _select_targets(
    consensus_df: pd.DataFrame,
    dataset_ids: Optional[Sequence[str]],
    pathways: Optional[Sequence[str]],
    min_cross_model_pairs: int,
    max_pathways_per_dataset: int,
) -> pd.DataFrame:
    subset = consensus_df.copy()
    subset = subset[subset["n_cross_model_pairs"].fillna(0).astype(int) >= int(min_cross_model_pairs)].copy()
    if dataset_ids:
        subset = subset[subset["dataset_id"].astype(str).isin(set(dataset_ids))].copy()
    if pathways:
        subset = subset[subset["pathway"].astype(str).isin(set(pathways))].copy()

    if subset.empty:
        return subset

    selected_rows: List[pd.DataFrame] = []
    for dataset_id, ds in subset.groupby("dataset_id"):
        ds_sorted = ds.sort_values(["n_cross_model_pairs", "best_convergence_score"], ascending=[False, False])
        selected_rows.append(ds_sorted.head(int(max_pathways_per_dataset)))
    return pd.concat(selected_rows, ignore_index=True)


def _iter_celltype_strata(
    metadata: pd.DataFrame,
    selected_celltypes: Optional[Sequence[str]],
    max_celltypes: int,
    min_cells: int,
    min_donors: int,
    min_age_classes: int,
) -> Tuple[List[Tuple[str, np.ndarray]], List[Dict[str, Any]]]:
    if metadata.empty:
        return [], []

    cell_types = metadata["cell_type"].astype(str)
    counts = cell_types.value_counts()
    ordered = counts.index.tolist()
    if selected_celltypes:
        keep = set([str(x) for x in selected_celltypes])
        ordered = [ct for ct in ordered if ct in keep]
    ordered = ordered[: int(max_celltypes)]

    strata: List[Tuple[str, np.ndarray]] = []
    skip_rows: List[Dict[str, Any]] = []
    ct_array = cell_types.to_numpy(dtype=object)
    for ct in ordered:
        mask = ct_array == ct
        meta_ct = metadata.loc[mask]
        n_cells = int(meta_ct.shape[0])
        n_donors = int(meta_ct["donor_id"].astype(str).nunique())
        n_age_classes = int(meta_ct["age_label"].astype(str).nunique())

        reason = ""
        if n_cells < int(min_cells):
            reason = "too_few_cells"
        elif n_donors < int(min_donors):
            reason = "too_few_donors"
        elif n_age_classes < int(min_age_classes):
            reason = "too_few_age_classes"

        if reason:
            skip_rows.append(
                {
                    "cell_type": str(ct),
                    "n_cells": n_cells,
                    "n_donors": n_donors,
                    "n_age_classes": n_age_classes,
                    "status": "skipped",
                    "skip_reason": reason,
                }
            )
            continue

        strata.append((str(ct), mask.astype(bool)))
    return strata, skip_rows


def _write_report(
    target_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    direction_df: pd.DataFrame,
    strat_skip_df: pd.DataFrame,
    out_md: Path,
) -> None:
    lines: List[str] = [
        "# Stage-5 Intervention Validation Report",
        "",
        "This stage evaluates representation-space interventions on pathway-linked SAE features and measures directional effects on donor-held-out age probes.",
        "",
    ]

    if target_df.empty:
        lines.extend(["No target pathways were selected for Stage-5.", ""])
        out_md.write_text("\n".join(lines), encoding="utf-8")
        return

    lines.extend(
        [
            "## Selected Targets",
            "",
            "```text",
            target_df[
                ["dataset_id", "pathway", "n_scgpt_features", "n_geneformer_features", "n_cross_model_pairs", "best_convergence_score"]
            ].to_string(index=False),
            "```",
            "",
        ]
    )

    if not summary_df.empty:
        key_summary = summary_df[
            [
                "dataset_id",
                "analysis_scope",
                "cell_type",
                "model",
                "pathway",
                "intervention",
                "n_splits",
                "mean_delta_expected_age",
                "mean_delta_old_prob",
                "mean_delta_balanced_accuracy",
                "mean_delta_pathway_activation",
            ]
        ].copy()
        lines.extend(
            [
                "## Intervention Summary",
                "",
                "```text",
                key_summary.to_string(index=False),
                "```",
                "",
            ]
        )

    if not strat_skip_df.empty:
        lines.extend(
            [
                "## Stratification Skips",
                "",
                "```text",
                strat_skip_df.to_string(index=False),
                "```",
                "",
            ]
        )

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
    else:
        lines.extend(["No cross-model directional comparisons were available.", ""])

    lines.extend(
        [
            "## Interpretation Guardrails",
            "",
            "- Directionality evidence is stronger when `old_push > 0`, `young_push < 0`, and random-feature effects are smaller.",
            "- This is still representation-level evidence, not causal biology proof.",
            "",
        ]
    )
    out_md.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Stage-5 intervention validation for longevity mechanistic interpretability.")
    parser.add_argument(
        "--stage4-dir",
        type=Path,
        default=_resolve_default_output_dir(root, "stage4_cross_model_convergence", "20260303"),
    )
    parser.add_argument(
        "--stage3-scgpt-dir",
        type=Path,
        default=_resolve_default_output_dir(root, "stage3_sae_pilot_scgpt", "20260303_fast"),
    )
    parser.add_argument(
        "--stage3-geneformer-dir",
        type=Path,
        default=_resolve_default_output_dir(root, "stage3_sae_pilot_geneformer", "20260303_fast"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "implementation" / "outputs" / "stage5_intervention_validation_20260303",
    )
    parser.add_argument("--dataset-ids", type=str, default="")
    parser.add_argument(
        "--pathways",
        type=str,
        default="inflammation_nfkb,senescence_sasp",
        help="Comma-separated pathways to test. Empty means auto-select by convergence ranking.",
    )
    parser.add_argument("--min-cross-model-pairs", type=int, default=2)
    parser.add_argument("--max-pathways-per-dataset", type=int, default=2)
    parser.add_argument("--top-features-per-model", type=int, default=5)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--split-seeds", type=str, default="42,123,314")
    parser.add_argument("--intervention-scale", type=float, default=1.0)
    parser.add_argument(
        "--write-donor-level-results",
        action="store_true",
        help="If set, also write donor-level intervention deltas for downstream donor-bootstrap analysis.",
    )
    parser.add_argument(
        "--stratify-by-cell-type",
        action="store_true",
        help="If set, run interventions within major cell-type strata instead of the full sampled cohort.",
    )
    parser.add_argument(
        "--celltypes",
        type=str,
        default="",
        help="Optional comma-separated cell types to include when --stratify-by-cell-type is set.",
    )
    parser.add_argument("--max-celltypes-per-dataset", type=int, default=8)
    parser.add_argument("--min-cells-per-celltype", type=int, default=120)
    parser.add_argument("--min-donors-per-celltype", type=int, default=20)
    parser.add_argument("--min-age-classes-per-celltype", type=int, default=3)
    parser.add_argument("--scgpt-max-genes", type=int, default=600)
    parser.add_argument("--scgpt-batch-size", type=int, default=8)
    parser.add_argument("--geneformer-max-genes", type=int, default=256)
    parser.add_argument("--geneformer-batch-size", type=int, default=8)
    parser.add_argument("--geneformer-mode", type=str, default="contextual", choices=["contextual", "static", "auto"])
    parser.add_argument("--device", type=str, default="cpu", choices=["auto", "cpu", "cuda", "mps"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    annotations_csv = args.stage4_dir / "stage4_feature_annotations.csv"
    pairs_csv = args.stage4_dir / "stage4_cross_model_pairs.csv"
    consensus_csv = args.stage4_dir / "stage4_consensus_pathway_summary.csv"
    for path in [annotations_csv, pairs_csv, consensus_csv]:
        if not path.exists():
            raise FileNotFoundError(f"Missing required stage-4 artifact: {path}")

    annotations = pd.read_csv(annotations_csv)
    consensus = pd.read_csv(consensus_csv)

    dataset_ids = _parse_csv_list(args.dataset_ids) if args.dataset_ids else []
    pathways = _parse_csv_list(args.pathways) if args.pathways else []
    target_df = _select_targets(
        consensus_df=consensus,
        dataset_ids=dataset_ids if dataset_ids else None,
        pathways=pathways if pathways else None,
        min_cross_model_pairs=args.min_cross_model_pairs,
        max_pathways_per_dataset=args.max_pathways_per_dataset,
    )
    if target_df.empty:
        raise RuntimeError("No Stage-5 targets selected; relax filters or verify Stage-4 outputs.")

    device = _device_auto(args.device)
    single_cell_root = _ensure_single_cell_src_on_path(Path(__file__))
    scgpt_runtime = ScGPTRuntime(single_cell_root=single_cell_root, device=device)
    geneformer_runtime = GeneformerRuntime(mode=args.geneformer_mode, device=device)
    split_seeds = _parse_int_csv(args.split_seeds)
    selected_celltypes = _parse_csv_list(args.celltypes) if args.celltypes else []

    cache: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    split_tables: List[pd.DataFrame] = []
    donor_tables: List[pd.DataFrame] = []
    skip_rows: List[Dict[str, Any]] = []

    for target in target_df.itertuples(index=False):
        dataset_id = str(target.dataset_id)
        pathway = str(target.pathway)
        for model_name, stage3_dir in [("scgpt", args.stage3_scgpt_dir), ("geneformer", args.stage3_geneformer_dir)]:
            sub = annotations[
                (annotations["dataset_id"].astype(str) == dataset_id)
                & (annotations["model"].astype(str) == model_name)
                & (annotations["primary_pathway"].astype(str) == pathway)
            ].copy()
            if sub.empty:
                continue

            sub = sub.sort_values("abs_donor_age_spearman", ascending=False).head(int(args.top_features_per_model))
            representation = str(sub.iloc[0]["representation"])
            dataset_path = Path(str(sub.iloc[0]["dataset_path"]))
            feature_ids = sub["feature_id"].astype(int).to_numpy(dtype=np.int64)
            feature_signs = np.sign(sub["donor_age_spearman"].to_numpy(dtype=np.float32))
            feature_signs = np.where(feature_signs == 0, 1.0, feature_signs).astype(np.float32)

            cache_key = (dataset_id, model_name, representation)
            if cache_key not in cache:
                ds_dir = stage3_dir / dataset_id
                sampled_csv = ds_dir / "sampled_obs.csv"
                sae_npz = ds_dir / representation / "sae_model_artifacts.npz"
                for req in [sampled_csv, sae_npz]:
                    if not req.exists():
                        raise FileNotFoundError(f"Missing stage-3 file: {req}")

                sampled_meta = pd.read_csv(sampled_csv)
                obs_idx = sampled_meta["obs_row"].to_numpy(dtype=np.int64)
                adata = _load_subset_anndata(dataset_path, obs_idx)
                rep = _extract_representation(
                    model_name=model_name,
                    representation_name=representation,
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
                cache[cache_key] = {
                    "metadata": sampled_meta,
                    "representation_matrix": rep,
                    "latent_matrix": latent,
                    "sae_artifacts": artifacts,
                    "dataset_path": dataset_path,
                }
                print(
                    f"[prep] {dataset_id}::{model_name}::{representation} "
                    f"| cells={rep.shape[0]} dim={rep.shape[1]} latent={latent.shape[1]}"
                )

            prepared = PreparedModelData(
                dataset_id=dataset_id,
                model=model_name,
                pathway=pathway,
                representation=representation,
                dataset_path=cache[cache_key]["dataset_path"],
                metadata=cache[cache_key]["metadata"],
                representation_matrix=cache[cache_key]["representation_matrix"],
                latent_matrix=cache[cache_key]["latent_matrix"],
                sae_artifacts=cache[cache_key]["sae_artifacts"],
                feature_ids=feature_ids,
                feature_signs=feature_signs,
            )

            if args.stratify_by_cell_type:
                strata, local_skips = _iter_celltype_strata(
                    metadata=prepared.metadata,
                    selected_celltypes=selected_celltypes if selected_celltypes else None,
                    max_celltypes=args.max_celltypes_per_dataset,
                    min_cells=args.min_cells_per_celltype,
                    min_donors=args.min_donors_per_celltype,
                    min_age_classes=args.min_age_classes_per_celltype,
                )
                for row in local_skips:
                    row.update(
                        {
                            "dataset_id": dataset_id,
                            "model": model_name,
                            "pathway": pathway,
                            "analysis_scope": "within_cell_type",
                        }
                    )
                    skip_rows.append(row)

                if not strata:
                    print(f"[warn] no eligible cell-type strata for {dataset_id}::{model_name}::{pathway}")
                    continue

                for cell_type, mask in strata:
                    meta_ct = prepared.metadata.loc[mask].reset_index(drop=True)
                    rep_ct = prepared.representation_matrix[mask]
                    latent_ct = prepared.latent_matrix[mask]

                    prepared_ct = PreparedModelData(
                        dataset_id=prepared.dataset_id,
                        model=prepared.model,
                        pathway=prepared.pathway,
                        representation=prepared.representation,
                        dataset_path=prepared.dataset_path,
                        metadata=meta_ct,
                        representation_matrix=rep_ct,
                        latent_matrix=latent_ct,
                        sae_artifacts=prepared.sae_artifacts,
                        feature_ids=prepared.feature_ids,
                        feature_signs=prepared.feature_signs,
                    )
                    split_df, donor_df = _run_probe_interventions(
                        data=prepared_ct,
                        n_splits=args.n_splits,
                        split_seeds=split_seeds,
                        intervention_scale=args.intervention_scale,
                        analysis_scope="within_cell_type",
                        cell_type=cell_type,
                        collect_donor_level=bool(args.write_donor_level_results),
                    )
                    if split_df.empty:
                        print(
                            f"[warn] no valid splits for {dataset_id}::{model_name}::{pathway}::cell_type={cell_type}"
                        )
                        continue

                    model_out = args.output_dir / dataset_id / model_name / pathway / _slugify_for_path(cell_type)
                    model_out.mkdir(parents=True, exist_ok=True)
                    split_df.to_csv(model_out / "intervention_split_results.csv", index=False)
                    if args.write_donor_level_results and not donor_df.empty:
                        donor_df.to_csv(model_out / "donor_intervention_results.csv", index=False)
                        donor_tables.append(donor_df)
                    split_tables.append(split_df)
                    print(
                        f"[ok] {dataset_id}::{model_name}::{pathway}::cell_type={cell_type} "
                        f"| splits={split_df['split_idx'].nunique()} rows={split_df.shape[0]}"
                    )
            else:
                split_df, donor_df = _run_probe_interventions(
                    data=prepared,
                    n_splits=args.n_splits,
                    split_seeds=split_seeds,
                    intervention_scale=args.intervention_scale,
                    analysis_scope="global",
                    cell_type="__all__",
                    collect_donor_level=bool(args.write_donor_level_results),
                )
                if split_df.empty:
                    print(f"[warn] no valid intervention splits for {dataset_id}::{model_name}::{pathway}")
                    continue

                model_out = args.output_dir / dataset_id / model_name / pathway
                model_out.mkdir(parents=True, exist_ok=True)
                split_df.to_csv(model_out / "intervention_split_results.csv", index=False)
                if args.write_donor_level_results and not donor_df.empty:
                    donor_df.to_csv(model_out / "donor_intervention_results.csv", index=False)
                    donor_tables.append(donor_df)
                split_tables.append(split_df)
                print(
                    f"[ok] {dataset_id}::{model_name}::{pathway} "
                    f"| splits={split_df['split_idx'].nunique()} rows={split_df.shape[0]}"
                )

    if not split_tables:
        raise RuntimeError("Stage-5 produced no intervention split results.")

    split_results = pd.concat(split_tables, ignore_index=True)
    summary = _summarize_intervention_results(split_results)
    direction = _cross_model_directionality(summary, consensus)
    skip_df = pd.DataFrame(skip_rows)

    split_csv = args.output_dir / "stage5_intervention_split_results.csv"
    summary_csv = args.output_dir / "stage5_intervention_summary.csv"
    direction_csv = args.output_dir / "stage5_cross_model_directionality.csv"
    donor_csv = args.output_dir / "stage5_intervention_donor_results.csv"
    skip_csv = args.output_dir / "stage5_celltype_stratification_skips.csv"
    report_md = args.output_dir / "stage5_intervention_validation_report.md"
    run_cfg = args.output_dir / "run_config.json"

    split_results.to_csv(split_csv, index=False)
    summary.to_csv(summary_csv, index=False)
    direction.to_csv(direction_csv, index=False)
    if args.write_donor_level_results and donor_tables:
        donor_results = pd.concat(donor_tables, ignore_index=True)
        donor_results.to_csv(donor_csv, index=False)
    if not skip_df.empty:
        skip_df.to_csv(skip_csv, index=False)
    _write_report(
        target_df=target_df,
        summary_df=summary,
        direction_df=direction,
        strat_skip_df=skip_df,
        out_md=report_md,
    )

    run_meta = {
        "stage4_dir": str(args.stage4_dir),
        "stage3_scgpt_dir": str(args.stage3_scgpt_dir),
        "stage3_geneformer_dir": str(args.stage3_geneformer_dir),
        "selected_targets": target_df[["dataset_id", "pathway"]].to_dict(orient="records"),
        "stratify_by_cell_type": bool(args.stratify_by_cell_type),
        "selected_celltypes": selected_celltypes,
        "max_celltypes_per_dataset": int(args.max_celltypes_per_dataset),
        "min_cells_per_celltype": int(args.min_cells_per_celltype),
        "min_donors_per_celltype": int(args.min_donors_per_celltype),
        "min_age_classes_per_celltype": int(args.min_age_classes_per_celltype),
        "top_features_per_model": int(args.top_features_per_model),
        "n_splits": int(args.n_splits),
        "split_seeds": split_seeds,
        "intervention_scale": float(args.intervention_scale),
        "write_donor_level_results": bool(args.write_donor_level_results),
        "device": device,
    }
    run_cfg.write_text(json.dumps(run_meta, indent=2), encoding="utf-8")

    print("[done] stage5 outputs")
    print(f"  - {split_csv}")
    print(f"  - {summary_csv}")
    print(f"  - {direction_csv}")
    if args.write_donor_level_results and donor_tables:
        print(f"  - {donor_csv}")
    if not skip_df.empty:
        print(f"  - {skip_csv}")
    print(f"  - {report_md}")


if __name__ == "__main__":
    main()
