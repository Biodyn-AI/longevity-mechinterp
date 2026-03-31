#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import pickle
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from safetensors.torch import load_file as load_safetensors_file
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, silhouette_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader


# We keep explicit candidate lists because metadata schemas differ by cohort/provider.
AGE_COLUMN_CANDIDATES = [
    "sample.subjectAgeAtDraw",
    "subject.ageAtFirstDraw",
    "subject.ageGroup",
    "development_stage",
    "age",
    "donor_age",
    "age_years",
    "age_at_collection",
    "ageAtEnrollment",
]

DONOR_COLUMN_CANDIDATES = [
    "subject.subjectGuid",
    "donor_id",
    "participant_id",
    "subject",
    "individual",
    "sample_id",
    "specimen.specimenGuid",
]

CELLTYPE_COLUMN_CANDIDATES = [
    "cell_type",
    "predicted_AIFI_L2",
    "AIFI_L2",
    "predicted_AIFI_L1",
    "AIFI_L1",
    "predicted_AIFI_L3",
    "AIFI_L3",
    "cell_type_ontology_term_id",
]

AGE_REGEX_PATTERNS = [
    re.compile(r"(\d{1,3})\s*[- ]?year[- ]old", re.IGNORECASE),
    re.compile(r"^(\d{1,3})(?:\.\d+)?$"),
    re.compile(r"age\s*[:=_-]?\s*(\d{1,3})", re.IGNORECASE),
]

AGE_GROUP_FALLBACK = {
    "children": 12.0,
    "young adult": 30.0,
    "older adult": 68.0,
}


@dataclass
class DatasetColumnSelection:
    age_col: str
    donor_col: str
    celltype_col: Optional[str]


@dataclass
class SampleSelection:
    obs_index: np.ndarray
    metadata: pd.DataFrame


def _decode_scalar(value: object) -> str:
    """Convert HDF5 scalar values into stable UTF-8 text."""
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.generic):
        return str(value.item())
    return str(value)


def _obs_column_names(obs_group: h5py.Group) -> List[str]:
    """Return AnnData obs columns while skipping internal index keys."""
    return [
        key
        for key in obs_group.keys()
        if key not in {"_index", "__categories", "index", "_index_names"}
    ]


def _find_first_existing(candidates: Sequence[str], columns: Iterable[str]) -> Optional[str]:
    """Pick the first candidate column that exists (case-insensitive)."""
    lowered = {c.lower(): c for c in columns}
    for cand in candidates:
        match = lowered.get(cand.lower())
        if match is not None:
            return match
    return None


def _read_obs_column(path: Path, column: str) -> np.ndarray:
    """
    Read one obs column as an object array of text values.

    Using h5py here keeps I/O robust across very large files and lets us avoid
    importing heavy high-level tooling just to inspect metadata.
    """
    with h5py.File(path, "r") as handle:
        obs = handle["obs"]
        obj = obs[column]
        if isinstance(obj, h5py.Group) and {"categories", "codes"}.issubset(obj.keys()):
            categories = np.array([_decode_scalar(v) for v in obj["categories"][:]], dtype=object)
            codes = np.asarray(obj["codes"][:], dtype=np.int64)
            out = np.empty(codes.shape[0], dtype=object)
            out[:] = ""
            valid = codes >= 0
            if np.any(valid):
                out[valid] = categories[codes[valid]]
            return out
        if isinstance(obj, h5py.Dataset):
            values = np.asarray(obj[()]).reshape(-1)
            return np.array([_decode_scalar(v) for v in values], dtype=object)
    raise ValueError(f"Unsupported obs column format: {column}")


def _read_obs_index(path: Path) -> np.ndarray:
    """Read obs index labels from H5AD."""
    with h5py.File(path, "r") as handle:
        obs = handle["obs"]
        if "_index" not in obs:
            n_rows = None
            for key in obs.keys():
                obj = obs[key]
                if isinstance(obj, h5py.Group) and "codes" in obj:
                    n_rows = int(obj["codes"].shape[0])
                    break
                if isinstance(obj, h5py.Dataset):
                    n_rows = int(obj.shape[0])
                    break
            if n_rows is None:
                raise ValueError("Unable to infer number of obs rows from H5AD")
            return np.arange(n_rows, dtype=np.int64)
        index_values = np.asarray(obs["_index"][:]).reshape(-1)
        return np.array([_decode_scalar(v) for v in index_values], dtype=object)


def _parse_numeric_age(value: str) -> Optional[float]:
    """Extract numeric age where possible; returns None if value is non-numeric."""
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "na", "null"}:
        return None
    text_l = text.lower()
    if text_l in AGE_GROUP_FALLBACK:
        return AGE_GROUP_FALLBACK[text_l]
    for pattern in AGE_REGEX_PATTERNS:
        match = pattern.search(text)
        if match is None:
            continue
        age = float(match.group(1))
        if 0.0 <= age <= 120.0:
            return age
    return None


def _derive_age_labels(age_raw: np.ndarray, n_bins: int, min_per_label: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build age labels for probing.

    We prefer quantile bins over raw-age regression because it is more robust to
    dataset-specific age scales and sparse donor coverage.
    """
    age_numeric = np.array([_parse_numeric_age(v) for v in age_raw], dtype=np.float64)
    valid_numeric = np.isfinite(age_numeric)

    labels = np.empty(age_raw.shape[0], dtype=object)
    labels[:] = ""

    if valid_numeric.sum() >= max(50, n_bins * 20):
        quantiles = np.linspace(0.0, 1.0, n_bins + 1)
        edges = np.quantile(age_numeric[valid_numeric], quantiles)
        edges = np.unique(edges)
        if edges.size >= 3:
            # pandas cut gives robust bins with consistent labels.
            binned = pd.cut(
                age_numeric[valid_numeric],
                bins=edges,
                include_lowest=True,
                duplicates="drop",
            )
            labels_numeric = np.asarray(binned.astype(str), dtype=object)
            labels[valid_numeric] = labels_numeric
        else:
            labels[valid_numeric] = np.array([f"age_{int(round(v))}" for v in age_numeric[valid_numeric]], dtype=object)
    else:
        labels = np.array([str(v).strip() for v in age_raw], dtype=object)

    # Drop infrequent labels because they destabilize donor-held-out classification.
    vc = pd.Series(labels).value_counts()
    keep = set(vc[vc >= min_per_label].index.astype(str))
    cleaned = np.array([lbl if str(lbl) in keep else "" for lbl in labels], dtype=object)
    return cleaned, age_numeric


def _select_dataset_columns(path: Path) -> DatasetColumnSelection:
    """Infer age/donor/celltype column names with strict validation."""
    with h5py.File(path, "r") as handle:
        obs = handle["obs"]
        columns = _obs_column_names(obs)

    age_col = _find_first_existing(AGE_COLUMN_CANDIDATES, columns)
    donor_col = _find_first_existing(DONOR_COLUMN_CANDIDATES, columns)
    celltype_col = _find_first_existing(CELLTYPE_COLUMN_CANDIDATES, columns)

    if age_col is None:
        raise ValueError("No age column found using configured candidates")
    if donor_col is None:
        raise ValueError("No donor column found using configured candidates")
    return DatasetColumnSelection(age_col=age_col, donor_col=donor_col, celltype_col=celltype_col)


def _balanced_subsample_by_donor(
    donor: np.ndarray,
    candidate_idx: np.ndarray,
    max_cells: int,
    max_cells_per_donor: int,
    seed: int,
) -> np.ndarray:
    """
    Subsample cells with an upper bound per donor to reduce donor-size dominance.
    """
    rng = np.random.default_rng(seed)
    selected: List[int] = []
    donor_series = pd.Series(donor[candidate_idx], index=candidate_idx)
    for _, donor_idx in donor_series.groupby(donor_series).groups.items():
        donor_idx_arr = np.array(list(donor_idx), dtype=np.int64)
        if donor_idx_arr.size > max_cells_per_donor:
            donor_idx_arr = rng.choice(donor_idx_arr, size=max_cells_per_donor, replace=False)
        selected.extend(donor_idx_arr.tolist())

    selected_arr = np.array(selected, dtype=np.int64)
    if selected_arr.size > max_cells:
        selected_arr = rng.choice(selected_arr, size=max_cells, replace=False)
    return np.sort(selected_arr)


def _build_sample_selection(
    path: Path,
    cols: DatasetColumnSelection,
    max_cells: int,
    max_cells_per_donor: int,
    min_cells_per_donor: int,
    min_cells_per_age_label: int,
    age_bins: int,
    seed: int,
) -> SampleSelection:
    """Create a clean donor/age/celltype table and deterministic subsample."""
    age_raw = _read_obs_column(path, cols.age_col)
    donor_raw = _read_obs_column(path, cols.donor_col)
    cell_raw = (
        _read_obs_column(path, cols.celltype_col)
        if cols.celltype_col is not None
        else np.array(["unknown"] * age_raw.shape[0], dtype=object)
    )
    obs_names = _read_obs_index(path)

    donor = np.array([str(v).strip() for v in donor_raw], dtype=object)
    cell = np.array([str(v).strip() if str(v).strip() else "unknown" for v in cell_raw], dtype=object)

    age_label, age_numeric = _derive_age_labels(age_raw, n_bins=age_bins, min_per_label=min_cells_per_age_label)

    # Keep only rows with valid donor and age labels.
    valid = (donor != "") & (age_label != "")
    if valid.sum() == 0:
        raise ValueError("No cells with valid donor and age labels after filtering")

    valid_idx = np.where(valid)[0]
    donor_valid = donor[valid_idx]

    # Enforce a minimum number of cells per donor.
    donor_counts = pd.Series(donor_valid).value_counts()
    keep_donors = set(donor_counts[donor_counts >= min_cells_per_donor].index.astype(str))
    donor_keep_mask = np.array([d in keep_donors for d in donor_valid], dtype=bool)
    kept_idx = valid_idx[donor_keep_mask]

    if kept_idx.size == 0:
        raise ValueError("No cells remain after donor minimum-cell filtering")

    sampled_idx = _balanced_subsample_by_donor(
        donor=donor,
        candidate_idx=kept_idx,
        max_cells=max_cells,
        max_cells_per_donor=max_cells_per_donor,
        seed=seed,
    )

    meta = pd.DataFrame(
        {
            "obs_row": sampled_idx,
            "obs_name": obs_names[sampled_idx],
            "age_label": age_label[sampled_idx],
            "age_numeric": age_numeric[sampled_idx],
            "donor_id": donor[sampled_idx],
            "cell_type": cell[sampled_idx],
        }
    ).reset_index(drop=True)

    # Final sanity to avoid tiny classes.
    label_counts = meta["age_label"].value_counts()
    keep_labels = set(label_counts[label_counts >= min_cells_per_age_label].index.astype(str))
    meta = meta[meta["age_label"].isin(keep_labels)].reset_index(drop=True)
    if meta.empty:
        raise ValueError("No cells remain after final age-label frequency filtering")

    return SampleSelection(obs_index=meta["obs_row"].to_numpy(dtype=np.int64), metadata=meta)


def _load_subset_anndata(path: Path, obs_index: np.ndarray) -> ad.AnnData:
    """
    Load only selected cells into memory.

    We open in backed mode first to avoid reading full matrices for large cohorts.
    """
    backed = ad.read_h5ad(path, backed="r")
    try:
        subset = backed[obs_index].to_memory()
    finally:
        backed.file.close()
    return subset


def _baseline_expression_representation(adata: ad.AnnData, n_components: int, seed: int) -> np.ndarray:
    """Compute a compact expression baseline from raw matrix values."""
    X = adata.X
    n = adata.n_obs
    d = adata.n_vars
    target_dim = int(max(2, min(n_components, n - 1, d - 1)))

    if sp.issparse(X):
        svd = TruncatedSVD(n_components=target_dim, random_state=seed)
        Z = svd.fit_transform(X)
    else:
        dense = np.asarray(X, dtype=np.float32)
        pca = PCA(n_components=target_dim, svd_solver="randomized", random_state=seed)
        Z = pca.fit_transform(dense)
    return np.asarray(Z, dtype=np.float32)


def _best_var_labels_for_lookup(
    adata: ad.AnnData,
    lookup_keys: set[str],
) -> Tuple[np.ndarray, str, int]:
    """
    Choose the gene-label column that maximizes overlap with a target lookup table.

    Many large cohorts keep Ensembl IDs in `var_names` but provide symbols in
    `feature_name` or related columns; we evaluate those alternatives explicitly.
    """
    candidates: List[Tuple[str, np.ndarray]] = [("var_names", np.array(adata.var_names, dtype=object))]
    for col in ("feature_name", "gene_name", "gene_symbol", "gene_symbols", "symbol", "hgnc_symbol"):
        if col in adata.var.columns:
            vals = adata.var[col].astype(str).to_numpy(dtype=object)
            candidates.append((col, vals))

    best_name = "var_names"
    best_vals = candidates[0][1]
    best_overlap = int(np.sum(np.isin(best_vals, list(lookup_keys))))
    for name, vals in candidates[1:]:
        overlap = int(np.sum(np.isin(vals, list(lookup_keys))))
        if overlap > best_overlap:
            best_overlap = overlap
            best_name = name
            best_vals = vals
    return best_vals, best_name, best_overlap


def _find_repo_root_from_script(script_path: Path) -> Path:
    # .../biodyn-work/subproject_51_longevity_age_mechinterp/implementation/scripts/...
    return script_path.resolve().parents[4]


def _ensure_single_cell_src_on_path(script_path: Path) -> Path:
    repo_root = _find_repo_root_from_script(script_path)
    single_cell_root = repo_root / "biodyn-work" / "single_cell_mechinterp"
    if str(single_cell_root) not in sys.path:
        sys.path.insert(0, str(single_cell_root))
    return single_cell_root


def _build_scgpt_model_args(scgpt_args: dict, vocab_map: Dict[str, int], pad_token: str) -> dict:
    # This mirrors the model-arg construction used in the existing scGPT pipeline.
    return {
        "ntoken": len(vocab_map),
        "d_model": scgpt_args["embsize"],
        "nhead": scgpt_args["nheads"],
        "d_hid": scgpt_args["d_hid"],
        "nlayers": scgpt_args["nlayers"],
        "nlayers_cls": scgpt_args.get("n_layers_cls", 3),
        "n_cls": 1,
        "vocab": vocab_map,
        "dropout": scgpt_args.get("dropout", 0.5),
        "pad_token": pad_token,
        "pad_value": scgpt_args.get("pad_value", 0),
        "do_mvc": bool(scgpt_args.get("MVC", False)),
        "do_dab": False,
        "use_batch_labels": False,
        "domain_spec_batchnorm": False,
        "input_emb_style": scgpt_args.get("input_emb_style", "continuous"),
        "n_input_bins": scgpt_args.get("n_bins"),
        "cell_emb_style": "avg-pool" if scgpt_args.get("no_cls") else "cls",
        "explicit_zero_prob": False,
        "use_fast_transformer": False,
        "fast_transformer_backend": "flash",
        "pre_norm": False,
    }


def _align_layer_output(layer_tensor: torch.Tensor, batch_size: int, seq_len: int) -> torch.Tensor:
    """Align layer tensor to shape (batch, seq, dim) for pooling."""
    if layer_tensor.dim() != 3:
        raise ValueError(f"Expected 3D layer tensor, got {tuple(layer_tensor.shape)}")
    if layer_tensor.shape[0] == batch_size and layer_tensor.shape[1] == seq_len:
        return layer_tensor
    if layer_tensor.shape[0] == seq_len and layer_tensor.shape[1] == batch_size:
        return layer_tensor.permute(1, 0, 2)
    raise ValueError(
        f"Layer tensor shape mismatch: {tuple(layer_tensor.shape)} vs batch={batch_size}, seq={seq_len}"
    )


class ScGPTRuntime:
    """
    Thin runtime wrapper that loads scGPT once and extracts multiple representations.

    We intentionally keep this simple and explicit so runs are easy to debug.
    """

    def __init__(
        self,
        single_cell_root: Path,
        device: str,
    ):
        self.single_cell_root = single_cell_root
        self.device = device

        from src.model.scgpt_loader import load_scgpt_model
        from src.model.vocab import load_vocab

        self._load_scgpt_model = load_scgpt_model
        self._load_vocab = load_vocab

        self.repo_path = single_cell_root / "external" / "scGPT"
        self.ckpt_path = single_cell_root / "external" / "scGPT_checkpoints" / "whole-human" / "best_model.pt"
        self.vocab_path = single_cell_root / "external" / "scGPT_checkpoints" / "whole-human" / "vocab.json"
        self.args_path = single_cell_root / "external" / "scGPT_checkpoints" / "whole-human" / "args.json"

        self.vocab = self._load_vocab(self.vocab_path)
        self.vocab_map = self.vocab.gene_to_id

        scgpt_args = json.loads(self.args_path.read_text(encoding="utf-8"))
        pad_token = scgpt_args.get("pad_token") or self.vocab.pad_token
        if not pad_token or pad_token not in self.vocab_map:
            for token in ("<pad>", "[PAD]", "<PAD>", "PAD"):
                if token in self.vocab_map:
                    pad_token = token
                    break
        if not pad_token or pad_token not in self.vocab_map:
            raise ValueError("Could not resolve scGPT pad token")

        model_args = _build_scgpt_model_args(scgpt_args, self.vocab_map, pad_token)

        self.model, _, _ = self._load_scgpt_model(
            entrypoint="scgpt.model.TransformerModel",
            repo_path=self.repo_path,
            checkpoint_path=self.ckpt_path,
            device=self.device,
            model_args=model_args,
            prefix_to_strip=None,
        )
        self.model.eval()
        self.model.to(self.device)
        if hasattr(self.model, "transformer_encoder"):
            if hasattr(self.model.transformer_encoder, "enable_nested_tensor"):
                self.model.transformer_encoder.enable_nested_tensor = False
            if hasattr(self.model.transformer_encoder, "use_nested_tensor"):
                self.model.transformer_encoder.use_nested_tensor = False

    def extract_representations(
        self,
        adata: ad.AnnData,
        batch_size: int,
        max_genes: int,
        layer_indices: Optional[Sequence[int]] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Extract final cell embedding plus pooled layerwise residual embeddings.
        """
        from src.data.scgpt_dataset import ScGPTDataset, ScGPTDatasetConfig, collate_scgpt
        from src.interpret.causal_intervention import capture_layer_outputs, find_transformer_layers
        from src.model.wrapper import ScGPTWrapper

        model_gene_labels, gene_label_source, overlap_count = _best_var_labels_for_lookup(
            adata, set(self.vocab_map.keys())
        )
        if overlap_count == 0:
            raise ValueError("No genes overlap with scGPT vocab in selected subset")

        adata_model = adata.copy()
        adata_model.var_names = pd.Index(model_gene_labels.astype(str), dtype=object)

        keep_mask = np.array([gene in self.vocab_map for gene in adata_model.var_names], dtype=bool)
        if keep_mask.sum() == 0:
            raise ValueError("No genes overlap with scGPT vocab in selected subset")
        adata_sc = adata_model[:, keep_mask].copy()

        dataset_cfg = ScGPTDatasetConfig(
            max_genes=max_genes,
            include_zero=False,
            sort_by_expression=True,
            pad_token_id=self.vocab.pad_id,
            cls_token_id=None,
        )
        dataset = ScGPTDataset(adata_sc, self.vocab_map, dataset_cfg)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_scgpt)

        wrapper = ScGPTWrapper(
            self.model,
            {"gene_ids": "src", "gene_values": "values", "src_key_padding_mask": "src_key_padding_mask"},
        )
        layers = find_transformer_layers(self.model)
        n_layers = len(layers)
        if layer_indices is None:
            layer_indices = list(range(n_layers))
        layer_indices = [li for li in layer_indices if 0 <= li < n_layers]
        if not layer_indices:
            raise ValueError("No valid scGPT layer indices selected")

        final_chunks: List[np.ndarray] = []
        layer_chunks: Dict[int, List[np.ndarray]] = {li: [] for li in layer_indices}

        with torch.no_grad():
            for batch in loader:
                batch_cpu = {k: v for k, v in batch.items()}
                batch_dev = {k: v.to(self.device) for k, v in batch.items()}
                bs, seq = batch_dev["gene_ids"].shape

                layer_outputs: List[Optional[torch.Tensor]] = [None for _ in layers]
                hooks = capture_layer_outputs(layers, layer_outputs)
                try:
                    out = wrapper.forward(batch_dev)
                finally:
                    for h in hooks:
                        h.remove()

                cell_emb = out.get("cell_emb")
                if cell_emb is None:
                    raise ValueError("scGPT forward output missing 'cell_emb'")
                final_chunks.append(cell_emb.detach().cpu().numpy().astype(np.float32))

                valid_mask = (batch_cpu["gene_indices"] >= 0).to(torch.float32)  # (B, S)
                denom = torch.clamp(valid_mask.sum(dim=1, keepdim=True), min=1.0)  # (B, 1)

                for li in layer_indices:
                    layer_tensor = layer_outputs[li]
                    if layer_tensor is None:
                        raise ValueError(f"Missing hooked output for layer {li}")
                    aligned = _align_layer_output(layer_tensor, batch_size=bs, seq_len=seq)
                    pooled = (aligned * valid_mask.to(aligned.device).unsqueeze(-1)).sum(dim=1) / denom.to(aligned.device)
                    layer_chunks[li].append(pooled.detach().cpu().numpy().astype(np.float32))

        reps: Dict[str, np.ndarray] = {
            "scgpt_final": np.concatenate(final_chunks, axis=0),
        }
        for li in layer_indices:
            reps[f"scgpt_layer_{li:02d}"] = np.concatenate(layer_chunks[li], axis=0)
        return reps


class GeneformerRuntime:
    """
    Geneformer runtime with contextual and static modes.

    - `contextual`: forward pass through frozen Geneformer encoder, pooled per cell.
    - `static`: expression-weighted average of input token embeddings.
    - `auto`: try contextual first, then fall back to static.
    """

    def __init__(self, mode: str, device: str):
        if mode not in {"contextual", "static", "auto"}:
            raise ValueError(f"Unsupported geneformer mode: {mode}")
        self.mode = mode
        self.device = device

        snapshot = Path(
            "/Users/ihorkendiukhov/.cache/huggingface/hub/models--ctheodoris--Geneformer/snapshots/"
            "05fcbeb8a27d49e0a7a4349152202ee2c1cbfd28"
        )
        self.snapshot = snapshot
        model_path = snapshot / "model.safetensors"
        gene_name_id_pkl = snapshot / "geneformer" / "gene_name_id_dict_gc104M.pkl"
        token_dict_pkl = snapshot / "geneformer" / "token_dictionary_gc104M.pkl"
        gene_median_dict_pkl = snapshot / "geneformer" / "gene_median_dictionary_gc104M.pkl"

        if not model_path.exists():
            raise FileNotFoundError(f"Missing Geneformer safetensors: {model_path}")
        if not gene_name_id_pkl.exists() or not token_dict_pkl.exists() or not gene_median_dict_pkl.exists():
            raise FileNotFoundError("Missing Geneformer mapping pickles in snapshot")

        state = load_safetensors_file(str(model_path), device="cpu")
        emb = state.get("bert.embeddings.word_embeddings.weight")
        if emb is None:
            raise ValueError("Geneformer word embedding matrix not found in safetensors")
        self.embedding = emb.detach().cpu().numpy().astype(np.float32)

        with gene_name_id_pkl.open("rb") as handle:
            self.gene_name_id: Dict[str, str] = pickle.load(handle)
        with token_dict_pkl.open("rb") as handle:
            self.token_dict: Dict[str, int] = pickle.load(handle)
        with gene_median_dict_pkl.open("rb") as handle:
            self.gene_median_dict: Dict[str, float] = pickle.load(handle)

        self.contextual_model = None
        self.contextual_hidden_dim: Optional[int] = None
        self.active_mode = "static"
        if self.mode in {"contextual", "auto"}:
            try:
                from transformers import AutoModel  # type: ignore

                model_dir = snapshot / "Geneformer-V2-316M"
                self.contextual_model = AutoModel.from_pretrained(
                    str(model_dir),
                    local_files_only=True,
                )
                self.contextual_model.eval()
                self.contextual_model.to(self.device)
                self.contextual_hidden_dim = int(self.contextual_model.config.hidden_size)
                self.active_mode = "contextual"
            except Exception as exc:
                if self.mode == "contextual":
                    raise RuntimeError(f"Failed to initialize contextual Geneformer: {exc}") from exc
                print(f"[warn] contextual Geneformer unavailable, falling back to static: {exc}")
                self.active_mode = "static"

    def _build_var_mapping(
        self,
        adata: ad.AnnData,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        model_gene_labels, _, _ = _best_var_labels_for_lookup(adata, set(self.gene_name_id.keys()))
        var_labels = model_gene_labels.astype(str)

        mapped_var_indices: List[int] = []
        mapped_token_ids: List[int] = []
        mapped_medians: List[float] = []
        for i, gene in enumerate(var_labels):
            ens = self.gene_name_id.get(gene)
            if ens is None:
                continue
            tok = self.token_dict.get(ens)
            if tok is None:
                continue
            tok_int = int(tok)
            if 0 <= tok_int < self.embedding.shape[0]:
                mapped_var_indices.append(i)
                mapped_token_ids.append(tok_int)
                mapped_medians.append(float(self.gene_median_dict.get(ens, 1.0)))

        return (
            np.array(mapped_var_indices, dtype=np.int64),
            np.array(mapped_token_ids, dtype=np.int64),
            np.array(mapped_medians, dtype=np.float32),
        )

    @staticmethod
    def _tokenize_cell(
        expression_vector: np.ndarray,
        var_indices: np.ndarray,
        token_ids: np.ndarray,
        medians: np.ndarray,
        max_len: int,
    ) -> Optional[np.ndarray]:
        expr = expression_vector[var_indices]
        nonzero = expr > 0
        if int(nonzero.sum()) == 0:
            return None

        expr_nz = expr[nonzero]
        tokens_nz = token_ids[nonzero]
        medians_nz = medians[nonzero]
        with np.errstate(divide="ignore", invalid="ignore"):
            normalized = expr_nz / medians_nz
        normalized = np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)
        order = np.argsort(-normalized)
        ranked_tokens = tokens_nz[order][: max_len - 2]
        # Geneformer tokenization uses [CLS]=2 and [SEP]=3.
        return np.concatenate([[2], ranked_tokens, [3]]).astype(np.int64)

    def extract_weighted_mean_embedding(
        self,
        adata: ad.AnnData,
        max_genes_per_cell: int,
    ) -> np.ndarray:
        """
        For each cell, compute an expression-weighted average over Geneformer token embeddings.
        """
        var_idx, tok_ids, _ = self._build_var_mapping(adata)
        if var_idx.size == 0:
            raise ValueError("No genes overlap with Geneformer token mapping")

        X = adata.X
        n_cells = adata.n_obs
        d_model = self.embedding.shape[1]
        out = np.zeros((n_cells, d_model), dtype=np.float32)

        for i in range(n_cells):
            row = X[i]
            if sp.issparse(row):
                dense = row.toarray().ravel()
            else:
                dense = np.asarray(row).ravel()
            expr = dense[var_idx]
            nz = expr > 0
            if not np.any(nz):
                continue
            tok = tok_ids[nz]
            vals = expr[nz].astype(np.float32, copy=False)
            if tok.size > max_genes_per_cell:
                order = np.argsort(-vals)[:max_genes_per_cell]
                tok = tok[order]
                vals = vals[order]
            weight_sum = float(vals.sum())
            if weight_sum <= 0:
                continue
            weights = vals / weight_sum
            out[i] = np.einsum("n,nd->d", weights, self.embedding[tok], optimize=True)
        return out

    def extract_contextual_embedding(
        self,
        adata: ad.AnnData,
        max_genes_per_cell: int,
        batch_size: int,
    ) -> np.ndarray:
        """
        Build per-cell token sequences and pool contextual Geneformer hidden states.
        """
        if self.contextual_model is None or self.contextual_hidden_dim is None:
            raise RuntimeError("Contextual Geneformer model is not initialized")

        var_idx, tok_ids, medians = self._build_var_mapping(adata)
        if var_idx.size == 0:
            raise ValueError("No genes overlap with Geneformer token mapping")

        sequences: List[Optional[np.ndarray]] = []
        X = adata.X
        for i in range(adata.n_obs):
            row = X[i]
            if sp.issparse(row):
                dense = row.toarray().ravel()
            else:
                dense = np.asarray(row).ravel()
            sequences.append(
                self._tokenize_cell(
                    expression_vector=dense,
                    var_indices=var_idx,
                    token_ids=tok_ids,
                    medians=medians,
                    max_len=max_genes_per_cell,
                )
            )

        out = np.zeros((adata.n_obs, self.contextual_hidden_dim), dtype=np.float32)
        pad_id = 0

        with torch.no_grad():
            for start in range(0, adata.n_obs, batch_size):
                end = min(start + batch_size, adata.n_obs)
                batch_seq = sequences[start:end]
                nonempty_local = [j for j, seq in enumerate(batch_seq) if seq is not None]
                if not nonempty_local:
                    continue

                seq_lens = [len(batch_seq[j]) for j in nonempty_local]
                max_len = int(max(seq_lens))
                input_ids = np.full((len(nonempty_local), max_len), pad_id, dtype=np.int64)
                attn = np.zeros((len(nonempty_local), max_len), dtype=np.int64)

                for bi, local_j in enumerate(nonempty_local):
                    seq = batch_seq[local_j]
                    assert seq is not None
                    L = len(seq)
                    input_ids[bi, :L] = seq
                    attn[bi, :L] = 1

                input_ids_t = torch.tensor(input_ids, dtype=torch.long, device=self.device)
                attn_t = torch.tensor(attn, dtype=torch.long, device=self.device)

                outputs = self.contextual_model(input_ids=input_ids_t, attention_mask=attn_t)
                hidden = outputs.last_hidden_state  # (B, L, D)

                valid = (attn_t > 0) & (input_ids_t != 2) & (input_ids_t != 3)
                denom = torch.clamp(valid.sum(dim=1, keepdim=True), min=1).to(hidden.dtype)
                pooled = (hidden * valid.unsqueeze(-1)).sum(dim=1) / denom
                pooled_np = pooled.detach().cpu().numpy().astype(np.float32)

                for bi, local_j in enumerate(nonempty_local):
                    global_idx = start + local_j
                    out[global_idx] = pooled_np[bi]

                if self.device == "mps":
                    torch.mps.empty_cache()

        return out

    def extract_representation(
        self,
        adata: ad.AnnData,
        max_genes_per_cell: int,
        batch_size: int,
    ) -> Tuple[str, np.ndarray]:
        if self.active_mode == "contextual":
            return (
                "geneformer_contextual",
                self.extract_contextual_embedding(
                    adata=adata,
                    max_genes_per_cell=max_genes_per_cell,
                    batch_size=batch_size,
                ),
            )
        return (
            "geneformer_static",
            self.extract_weighted_mean_embedding(
                adata=adata,
                max_genes_per_cell=max_genes_per_cell,
            ),
        )


def _group_split_scores(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int,
    seed: int,
) -> Dict[str, float]:
    """Evaluate donor-held-out classification with repeated group splits."""
    splitter = GroupShuffleSplit(n_splits=n_splits, test_size=0.25, random_state=seed)

    bacc_list: List[float] = []
    f1_list: List[float] = []
    n_train: List[int] = []
    n_test: List[int] = []

    for tr_idx, te_idx in splitter.split(X, y, groups=groups):
        y_tr = y[tr_idx]
        y_te = y[te_idx]
        if np.unique(y_tr).size < 2 or np.unique(y_te).size < 2:
            continue

        model = Pipeline(
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
        model.fit(X[tr_idx], y_tr)
        pred = model.predict(X[te_idx])
        bacc_list.append(float(balanced_accuracy_score(y_te, pred)))
        f1_list.append(float(f1_score(y_te, pred, average="macro")))
        n_train.append(int(tr_idx.size))
        n_test.append(int(te_idx.size))

    if not bacc_list:
        return {
            "n_valid_splits": 0,
            "balanced_accuracy_mean": float("nan"),
            "balanced_accuracy_std": float("nan"),
            "macro_f1_mean": float("nan"),
            "macro_f1_std": float("nan"),
            "mean_train_cells": float("nan"),
            "mean_test_cells": float("nan"),
        }

    return {
        "n_valid_splits": int(len(bacc_list)),
        "balanced_accuracy_mean": float(np.mean(bacc_list)),
        "balanced_accuracy_std": float(np.std(bacc_list)),
        "macro_f1_mean": float(np.mean(f1_list)),
        "macro_f1_std": float(np.std(f1_list)),
        "mean_train_cells": float(np.mean(n_train)),
        "mean_test_cells": float(np.mean(n_test)),
    }


def _manifold_metrics(X: np.ndarray, y_label: np.ndarray, age_numeric: np.ndarray, seed: int) -> Dict[str, float]:
    """Compute compact manifold diagnostics for each representation."""
    n, d = X.shape
    dim = int(max(2, min(32, n - 1, d - 1)))
    if dim < 2:
        return {
            "pca_dim_used": float("nan"),
            "participation_ratio": float("nan"),
            "pc1_explained_ratio": float("nan"),
            "pc5_cumulative_explained_ratio": float("nan"),
            "pc1_age_corr": float("nan"),
            "age_label_silhouette": float("nan"),
        }

    pca = PCA(n_components=dim, svd_solver="randomized", random_state=seed)
    Z = pca.fit_transform(X)
    ev = pca.explained_variance_
    evr = pca.explained_variance_ratio_
    pr = float((ev.sum() ** 2) / np.maximum((ev**2).sum(), 1e-12))

    pc1_age_corr = float("nan")
    valid_age = np.isfinite(age_numeric)
    if valid_age.sum() >= 20:
        v = Z[valid_age, 0]
        a = age_numeric[valid_age]
        if np.std(v) > 0 and np.std(a) > 0:
            pc1_age_corr = float(np.corrcoef(v, a)[0, 1])

    sil = float("nan")
    y_codes, _ = pd.factorize(pd.Series(y_label))
    if np.unique(y_codes).size >= 3:
        try:
            sil = float(silhouette_score(Z[:, : min(10, Z.shape[1])], y_codes, metric="euclidean"))
        except Exception:
            sil = float("nan")

    return {
        "pca_dim_used": float(dim),
        "participation_ratio": pr,
        "pc1_explained_ratio": float(evr[0]),
        "pc5_cumulative_explained_ratio": float(evr[: min(5, evr.size)].sum()),
        "pc1_age_corr": pc1_age_corr,
        "age_label_silhouette": sil,
    }


def _evaluate_representations(
    reps: Dict[str, np.ndarray],
    meta: pd.DataFrame,
    n_splits: int,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run donor-held-out probes and manifold metrics for each representation.
    """
    y_codes, age_classes = pd.factorize(meta["age_label"])
    groups = meta["donor_id"].to_numpy(dtype=object)
    age_numeric = meta["age_numeric"].to_numpy(dtype=np.float64)

    probe_rows: List[Dict[str, Any]] = []
    manifold_rows: List[Dict[str, Any]] = []

    for name, X in reps.items():
        Xn = np.asarray(X, dtype=np.float32)
        if Xn.shape[0] != meta.shape[0]:
            raise ValueError(f"Representation '{name}' row mismatch: {Xn.shape[0]} vs {meta.shape[0]}")

        score = _group_split_scores(
            X=Xn,
            y=y_codes,
            groups=groups,
            n_splits=n_splits,
            seed=seed,
        )
        score.update(
            {
                "representation": name,
                "n_cells": int(Xn.shape[0]),
                "n_features": int(Xn.shape[1]),
                "n_age_classes": int(len(age_classes)),
                "n_donors": int(meta["donor_id"].nunique()),
            }
        )
        probe_rows.append(score)

        mani = _manifold_metrics(Xn, meta["age_label"].to_numpy(dtype=object), age_numeric, seed=seed)
        mani.update(
            {
                "representation": name,
                "n_cells": int(Xn.shape[0]),
                "n_features": int(Xn.shape[1]),
            }
        )
        manifold_rows.append(mani)

    probe_df = pd.DataFrame(probe_rows).sort_values("balanced_accuracy_mean", ascending=False)
    manifold_df = pd.DataFrame(manifold_rows).sort_values("participation_ratio", ascending=False)
    return probe_df, manifold_df


def _permutation_null_for_best(
    reps: Dict[str, np.ndarray],
    probe_df: pd.DataFrame,
    meta: pd.DataFrame,
    n_splits: int,
    n_perm: int,
    seed: int,
) -> pd.DataFrame:
    """
    Compute a lightweight permutation null for the best representation.
    """
    if probe_df.empty:
        return pd.DataFrame()

    best_name = str(probe_df.iloc[0]["representation"])
    X = np.asarray(reps[best_name], dtype=np.float32)
    y_codes, _ = pd.factorize(meta["age_label"])
    groups = meta["donor_id"].to_numpy(dtype=object)

    observed = _group_split_scores(X=X, y=y_codes, groups=groups, n_splits=n_splits, seed=seed)
    observed_metric = float(observed["balanced_accuracy_mean"])

    rng = np.random.default_rng(seed + 17)
    null_scores = []
    for i in range(n_perm):
        y_perm = rng.permutation(y_codes)
        perm_score = _group_split_scores(X=X, y=y_perm, groups=groups, n_splits=n_splits, seed=seed + i + 1000)
        null_scores.append(float(perm_score["balanced_accuracy_mean"]))

    null_arr = np.array(null_scores, dtype=np.float64)
    p_value = float((1 + np.sum(null_arr >= observed_metric)) / (n_perm + 1))

    rows = [
        {
            "representation": best_name,
            "metric": "balanced_accuracy_mean",
            "observed": observed_metric,
            "null_mean": float(np.nanmean(null_arr)),
            "null_std": float(np.nanstd(null_arr)),
            "null_p95": float(np.nanpercentile(null_arr, 95)),
            "p_value_right_tail": p_value,
            "n_permutations": int(n_perm),
        }
    ]
    return pd.DataFrame(rows)


def _device_auto(user_device: str) -> str:
    if user_device != "auto":
        return user_device
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _dataset_id_from_path(path: Path) -> str:
    stem = path.stem
    return re.sub(r"[^a-zA-Z0-9_]+", "_", stem).strip("_")


def _run_one_dataset(
    dataset_path: Path,
    out_dir: Path,
    scgpt_runtime: ScGPTRuntime,
    geneformer_runtime: GeneformerRuntime,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    dataset_id = _dataset_id_from_path(dataset_path)
    dataset_out = out_dir / dataset_id
    dataset_out.mkdir(parents=True, exist_ok=True)

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
    selection.metadata.to_csv(dataset_out / "sampled_obs.csv", index=False)

    adata = _load_subset_anndata(dataset_path, selection.obs_index)
    if adata.n_obs != selection.metadata.shape[0]:
        raise RuntimeError("Loaded subset size does not match metadata rows")

    reps: Dict[str, np.ndarray] = {}

    # Baseline representation from expression only.
    reps["expr_svd"] = _baseline_expression_representation(
        adata=adata,
        n_components=args.baseline_components,
        seed=args.seed,
    )

    # scGPT contextual representations.
    sc_layer_idx: List[int] = []
    if args.scgpt_layers == "default":
        # A compact but informative layer subset for the first pass.
        sc_layer_idx = [0, 3, 6, 9, 11]
    else:
        sc_layer_idx = [int(x.strip()) for x in args.scgpt_layers.split(",") if x.strip()]

    sc_reps = scgpt_runtime.extract_representations(
        adata=adata,
        batch_size=args.scgpt_batch_size,
        max_genes=args.scgpt_max_genes,
        layer_indices=sc_layer_idx,
    )
    reps.update(sc_reps)

    # Geneformer representation (contextual or static depending on runtime mode).
    gf_name, gf_rep = geneformer_runtime.extract_representation(
        adata=adata,
        max_genes_per_cell=args.geneformer_max_genes,
        batch_size=args.geneformer_batch_size,
    )
    reps[gf_name] = gf_rep

    probe_df, manifold_df = _evaluate_representations(
        reps=reps,
        meta=selection.metadata,
        n_splits=args.group_splits,
        seed=args.seed,
    )
    probe_df.to_csv(dataset_out / "probe_metrics.csv", index=False)
    manifold_df.to_csv(dataset_out / "manifold_metrics.csv", index=False)

    null_df = _permutation_null_for_best(
        reps=reps,
        probe_df=probe_df,
        meta=selection.metadata,
        n_splits=args.group_splits,
        n_perm=args.permutation_iters,
        seed=args.seed,
    )
    if not null_df.empty:
        null_df.to_csv(dataset_out / "permutation_null_best.csv", index=False)

    metadata_summary = {
        "dataset_id": dataset_id,
        "dataset_path": str(dataset_path),
        "age_column": cols.age_col,
        "donor_column": cols.donor_col,
        "celltype_column": cols.celltype_col,
        "n_cells_sampled": int(selection.metadata.shape[0]),
        "n_donors_sampled": int(selection.metadata["donor_id"].nunique()),
        "n_age_classes": int(selection.metadata["age_label"].nunique()),
        "n_cell_types_sampled": int(selection.metadata["cell_type"].nunique()),
        "age_class_counts": selection.metadata["age_label"].value_counts().to_dict(),
        "geneformer_mode_active": geneformer_runtime.active_mode,
    }
    (dataset_out / "metadata_summary.json").write_text(
        json.dumps(metadata_summary, indent=2),
        encoding="utf-8",
    )

    best_row = probe_df.iloc[0].to_dict() if not probe_df.empty else {}
    return {
        "dataset_id": dataset_id,
        "dataset_path": str(dataset_path),
        "status": "ok",
        "best_representation": best_row.get("representation", ""),
        "best_balanced_accuracy_mean": best_row.get("balanced_accuracy_mean", float("nan")),
        "best_macro_f1_mean": best_row.get("macro_f1_mean", float("nan")),
        "n_cells_sampled": int(selection.metadata.shape[0]),
        "n_donors_sampled": int(selection.metadata["donor_id"].nunique()),
        "n_age_classes": int(selection.metadata["age_label"].nunique()),
    }


def _discover_core_h5ad(manifest_csv: Path, raw_dir: Path) -> List[Path]:
    manifest = pd.read_csv(manifest_csv)
    core = manifest[manifest["phase"].astype(str) == "core"].copy()
    out: List[Path] = []
    for row in core.itertuples(index=False):
        p = raw_dir / str(row.filename)
        if p.exists():
            out.append(p)
    return out


def _write_checkpoint_report(
    aggregate_df: pd.DataFrame,
    output_md: Path,
) -> None:
    lines = [
        "# Stage-1 Longevity Mechinterp Checkpoint",
        "",
        "This report summarizes donor-held-out age decodability on frozen representations.",
        "",
    ]
    if aggregate_df.empty:
        lines.append("No successful dataset runs.")
    else:
        ok = aggregate_df[aggregate_df["status"] == "ok"].copy()
        if ok.empty:
            lines.append("All runs failed; inspect `stage1_run_aggregate.csv` for errors.")
        else:
            lines.extend(
                [
                    "## Dataset Summary",
                    "",
                    "```text",
                    ok[
                        [
                            "dataset_id",
                            "n_cells_sampled",
                            "n_donors_sampled",
                            "n_age_classes",
                            "best_representation",
                            "best_balanced_accuracy_mean",
                            "best_macro_f1_mean",
                        ]
                    ].to_string(index=False),
                    "```",
                    "",
                ]
            )

            # Gate G1 heuristic: best donor-held-out balanced accuracy above random baseline proxy.
            g1_hits = ok[ok["best_balanced_accuracy_mean"].astype(float) >= 0.40]
            lines.extend(
                [
                    "## G1 Gate Heuristic",
                    "",
                    f"- Datasets with strong donor-held-out age signal (balanced accuracy >= 0.40): {int(g1_hits.shape[0])} / {int(ok.shape[0])}",
                    "",
                    "Interpretation: this is a practical first-pass threshold, not a final publication criterion.",
                    "",
                ]
            )

    output_md.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    default_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Stage-1 execution for longevity mechanistic interpretability on existing core datasets."
    )
    parser.add_argument(
        "--manifest-csv",
        type=Path,
        default=default_root / "implementation" / "data_downloads" / "manifests" / "age_dataset_manifest.csv",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=default_root / "implementation" / "data_downloads" / "raw",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_root / "implementation" / "outputs" / "stage1_longevity_mechinterp",
    )
    parser.add_argument("--max-cells-per-dataset", type=int, default=5000)
    parser.add_argument("--max-cells-per-donor", type=int, default=400)
    parser.add_argument("--min-cells-per-donor", type=int, default=20)
    parser.add_argument("--min-cells-per-age-label", type=int, default=50)
    parser.add_argument("--age-bins", type=int, default=4)
    parser.add_argument("--baseline-components", type=int, default=64)
    parser.add_argument("--scgpt-max-genes", type=int, default=800)
    parser.add_argument("--scgpt-batch-size", type=int, default=8)
    parser.add_argument(
        "--scgpt-layers",
        type=str,
        default="default",
        help="Either 'default' or comma-separated layer indices.",
    )
    parser.add_argument("--geneformer-max-genes", type=int, default=1024)
    parser.add_argument("--geneformer-batch-size", type=int, default=8)
    parser.add_argument(
        "--geneformer-mode",
        type=str,
        default="contextual",
        choices=["contextual", "static", "auto"],
        help="Geneformer representation mode.",
    )
    parser.add_argument("--group-splits", type=int, default=5)
    parser.add_argument("--permutation-iters", type=int, default=25)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = _device_auto(args.device)
    single_cell_root = _ensure_single_cell_src_on_path(Path(__file__))

    scgpt_runtime = ScGPTRuntime(single_cell_root=single_cell_root, device=device)
    geneformer_runtime = GeneformerRuntime(mode=args.geneformer_mode, device=device)

    datasets = _discover_core_h5ad(args.manifest_csv, args.raw_dir)
    if not datasets:
        raise FileNotFoundError("No core dataset files found from manifest/raw dir")

    aggregate_rows: List[Dict[str, Any]] = []
    for ds_path in datasets:
        dataset_id = _dataset_id_from_path(ds_path)
        try:
            row = _run_one_dataset(
                dataset_path=ds_path,
                out_dir=args.output_dir,
                scgpt_runtime=scgpt_runtime,
                geneformer_runtime=geneformer_runtime,
                args=args,
            )
            aggregate_rows.append(row)
            print(f"[ok] {dataset_id} | best={row.get('best_representation')} bacc={row.get('best_balanced_accuracy_mean')}")
        except Exception as exc:
            aggregate_rows.append(
                {
                    "dataset_id": dataset_id,
                    "dataset_path": str(ds_path),
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            print(f"[error] {dataset_id}: {type(exc).__name__}: {exc}")

    aggregate_df = pd.DataFrame(aggregate_rows)
    aggregate_csv = args.output_dir / "stage1_run_aggregate.csv"
    aggregate_df.to_csv(aggregate_csv, index=False)

    report_md = args.output_dir / "stage1_checkpoint_report.md"
    _write_checkpoint_report(aggregate_df, report_md)

    print("[done] stage1 outputs")
    print(f"  - {aggregate_csv}")
    print(f"  - {report_md}")


if __name__ == "__main__":
    main()
