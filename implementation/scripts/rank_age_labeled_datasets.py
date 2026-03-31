#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List

import h5py
import numpy as np
import pandas as pd

# Columns are ordered from highest to lowest preference.
AGE_COLUMN_CANDIDATES = [
    "development_stage",
    "age",
    "donor_age",
    "age_years",
    "age_at_collection",
    "ageAtEnrollment",
]

DONOR_COLUMN_CANDIDATES = [
    "donor_id",
    "donor",
    "individual",
    "subject",
    "participant_id",
    "sample_id",
    "patient_id",
]

CELLTYPE_COLUMN_CANDIDATES = [
    "cell_type",
    "cell_type_ontology_term_id",
    "cell_ontology_class",
    "cell_ontology_term_id",
    "free_annotation",
    "celltype",
]

TISSUE_COLUMN_CANDIDATES = [
    "organ",
    "tissue",
    "tissue_general",
    "donor_tissue",
    "organ_tissue",
]

# Age labels in common atlas-style metadata are often textual.
AGE_REGEXES = [
    re.compile(r"(\d{1,3})\s*[- ]?year[- ]old", re.IGNORECASE),
    re.compile(r"age\s*[:=_-]?\s*(\d{1,3})", re.IGNORECASE),
    re.compile(r"^(\d{1,3})$"),
]


def _decode_value(value: object) -> str:
    """Convert HDF5 scalar values into stable text labels."""
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.generic):
        return str(value.item())
    return str(value)


def _obs_column_names(obs_group: h5py.Group) -> List[str]:
    """List observation columns, excluding AnnData index bookkeeping keys."""
    return [
        key
        for key in obs_group.keys()
        if key not in {"_index", "__categories", "index", "_index_names"}
    ]


def _infer_axis_length(group: h5py.Group | None) -> int | None:
    """Infer row count from an AnnData axis group (`obs` or `var`)."""
    if group is None:
        return None

    if "_index" in group and hasattr(group["_index"], "shape"):
        return int(group["_index"].shape[0])

    for key in group.keys():
        obj = group[key]
        if isinstance(obj, h5py.Group) and "codes" in obj:
            return int(obj["codes"].shape[0])
        if hasattr(obj, "shape") and len(obj.shape) >= 1:
            return int(obj.shape[0])

    return None


def _read_column_counts(obs_group: h5py.Group, column_name: str | None) -> Dict[str, int]:
    """
    Read per-label counts for an obs column.

    Supports both AnnData categorical storage (`categories` + `codes`) and
    direct dataset storage. Returning counts keeps memory usage low and makes
    downstream metrics straightforward.
    """
    if not column_name or column_name not in obs_group:
        return {}

    column_obj = obs_group[column_name]

    # Preferred path: AnnData categorical encoding.
    if isinstance(column_obj, h5py.Group) and {"categories", "codes"}.issubset(column_obj.keys()):
        categories = [_decode_value(v) for v in column_obj["categories"][:]]
        codes = np.asarray(column_obj["codes"][:], dtype=np.int64)
        valid_codes = codes[codes >= 0]
        if valid_codes.size == 0:
            return {}
        unique_codes, counts = np.unique(valid_codes, return_counts=True)
        return {
            categories[int(code)] if int(code) < len(categories) else f"code_{int(code)}": int(count)
            for code, count in zip(unique_codes, counts)
        }

    # Fallback path: direct values in a dense dataset.
    if isinstance(column_obj, h5py.Dataset):
        values = np.asarray(column_obj[()]).reshape(-1)
        labels = np.array([_decode_value(v) for v in values], dtype=object)
        unique_labels, counts = np.unique(labels, return_counts=True)
        return {str(label): int(count) for label, count in zip(unique_labels, counts)}

    return {}


def _find_best_column(columns: Iterable[str], candidates: List[str]) -> str | None:
    """Pick the first candidate present in the dataset (case-insensitive)."""
    lower_to_original = {col.lower(): col for col in columns}
    for candidate in candidates:
        matched = lower_to_original.get(candidate.lower())
        if matched is not None:
            return matched
    return None


def _extract_numeric_ages(age_labels: Iterable[str]) -> List[int]:
    """Extract plausible numeric ages from textual labels."""
    ages: set[int] = set()
    for label in age_labels:
        text = str(label)
        for regex in AGE_REGEXES:
            match = regex.search(text)
            if match is None:
                continue
            value = int(match.group(1))
            if 0 <= value <= 120:
                ages.add(value)
            break
    return sorted(ages)


def _normalized(value: float | int | None, cap: float) -> float:
    """Simple capped normalization helper for scoring."""
    if value is None:
        return 0.0
    return min(float(value), cap) / cap


def _priority_score(
    n_cells: int,
    n_age_levels: int,
    age_span_years: int | None,
    n_donors: int,
    n_cell_types: int,
) -> float:
    """
    Compute a dataset-priority score in [0, 100].

    We prioritize broad age span and donor diversity first, then cell-type
    coverage and practical sample size. This intentionally reflects a
    longevity-mechinterp discovery setup.
    """
    age_span_component = _normalized(age_span_years, cap=60)
    age_levels_component = _normalized(n_age_levels, cap=20)
    donor_component = _normalized(n_donors, cap=30)
    cell_type_component = _normalized(n_cell_types, cap=80)
    cell_count_component = min(math.log10(max(n_cells, 1) + 1) / 6.0, 1.0)

    score_01 = (
        0.35 * age_span_component
        + 0.20 * age_levels_component
        + 0.20 * donor_component
        + 0.15 * cell_type_component
        + 0.10 * cell_count_component
    )

    score = 100.0 * score_01

    # Hard-penalize datasets without real age variation.
    if n_age_levels <= 1:
        score = min(score, 20.0)
    if age_span_years is None or age_span_years <= 0:
        score = min(score, 25.0)

    return round(score, 2)


def _priority_tier(
    score: float,
    n_age_levels: int,
    age_span_years: int | None,
    n_donors: int,
) -> str:
    """Convert numeric score into a coarse experiment-priority tier."""
    span = age_span_years or 0
    if score >= 70 and n_age_levels >= 6 and span >= 15 and n_donors >= 6:
        return "A"
    if score >= 55 and n_age_levels >= 4 and span >= 10 and n_donors >= 4:
        return "B"
    if score >= 35 and n_age_levels >= 2 and span >= 2 and n_donors >= 2:
        return "C"
    return "D"


def _summarize_h5ad(path: Path) -> dict:
    """Extract age-longevity-relevant metadata from a single `.h5ad` file."""
    try:
        with h5py.File(path, "r") as handle:
            obs_group = handle.get("obs")
            var_group = handle.get("var")
            if obs_group is None or not isinstance(obs_group, h5py.Group):
                return {
                    "dataset_path": str(path),
                    "dataset_name": path.name,
                    "error": "missing_obs_group",
                }

            obs_columns = _obs_column_names(obs_group)
            age_column = _find_best_column(obs_columns, AGE_COLUMN_CANDIDATES)
            donor_column = _find_best_column(obs_columns, DONOR_COLUMN_CANDIDATES)
            celltype_column = _find_best_column(obs_columns, CELLTYPE_COLUMN_CANDIDATES)
            tissue_column = _find_best_column(obs_columns, TISSUE_COLUMN_CANDIDATES)

            age_counts = _read_column_counts(obs_group, age_column)
            donor_counts = _read_column_counts(obs_group, donor_column)
            celltype_counts = _read_column_counts(obs_group, celltype_column)
            tissue_counts = _read_column_counts(obs_group, tissue_column)

            numeric_ages = _extract_numeric_ages(age_counts.keys())
            age_min = min(numeric_ages) if numeric_ages else None
            age_max = max(numeric_ages) if numeric_ages else None
            age_span = (age_max - age_min) if (age_min is not None and age_max is not None) else None

            n_cells = _infer_axis_length(obs_group) or 0
            n_genes = _infer_axis_length(var_group) or 0
            n_age_levels = len(age_counts)
            n_donors = len(donor_counts)
            n_cell_types = len(celltype_counts)
            n_tissues = len(tissue_counts)

            score = _priority_score(
                n_cells=n_cells,
                n_age_levels=n_age_levels,
                age_span_years=age_span,
                n_donors=n_donors,
                n_cell_types=n_cell_types,
            )

            notes: List[str] = []
            if n_age_levels <= 1:
                notes.append("no_age_variation")
            if n_donors <= 1:
                notes.append("single_donor")
            if n_cell_types <= 1:
                notes.append("low_celltype_diversity")
            if not notes:
                notes.append("age-ready")

            return {
                "dataset_path": str(path),
                "dataset_name": path.name,
                "n_cells": n_cells,
                "n_genes": n_genes,
                "age_column": age_column,
                "n_age_levels": n_age_levels,
                "age_min_years": age_min,
                "age_max_years": age_max,
                "age_span_years": age_span,
                "donor_column": donor_column,
                "n_donors": n_donors,
                "celltype_column": celltype_column,
                "n_cell_types": n_cell_types,
                "tissue_column": tissue_column,
                "n_tissues": n_tissues,
                "priority_score": score,
                "priority_tier": _priority_tier(score, n_age_levels, age_span, n_donors),
                "notes": ",".join(notes),
                "error": "",
            }
    except Exception as exc:  # pragma: no cover - defensive fallback for broken files
        return {
            "dataset_path": str(path),
            "dataset_name": path.name,
            "error": f"read_error:{type(exc).__name__}:{exc}",
        }


def _discover_h5ad_files(roots: Iterable[Path], recursive: bool = True) -> List[Path]:
    """Collect `.h5ad` files under one or more roots."""
    files: List[Path] = []
    for root in roots:
        if not root.exists():
            continue
        iterator = root.rglob("*.h5ad") if recursive else root.glob("*.h5ad")
        files.extend(path for path in iterator if path.is_file())
    return sorted(set(files))


def _write_markdown_summary(df: pd.DataFrame, output_path: Path, top_k: int) -> None:
    """Write an easy-to-read ranking summary for planning documents."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ranked = df[df["error"] == ""].head(top_k)
    lines = [
        "# Age-Labeled Dataset Ranking",
        "",
        f"Top {len(ranked)} datasets by `priority_score`.",
        "",
        "| Rank | Dataset | Score | Tier | Cells | Age span | Age levels | Donors | Cell types |",
        "|---:|---|---:|:---:|---:|---:|---:|---:|---:|",
    ]

    for rank, (_, row) in enumerate(ranked.iterrows(), start=1):
        span = "na" if pd.isna(row["age_span_years"]) else int(row["age_span_years"])
        lines.append(
            "| {rank} | {name} | {score:.2f} | {tier} | {cells} | {span} | {levels} | {donors} | {celltypes} |".format(
                rank=rank,
                name=row["dataset_name"],
                score=float(row["priority_score"]),
                tier=row["priority_tier"],
                cells=int(row["n_cells"]),
                span=span,
                levels=int(row["n_age_levels"]),
                donors=int(row["n_donors"]),
                celltypes=int(row["n_cell_types"]),
            )
        )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _default_roots() -> List[Path]:
    """Default scan roots for this repository layout."""
    subproject_root = Path(__file__).resolve().parents[2]
    biodyn_work_root = subproject_root.parent
    return [
        biodyn_work_root / "single_cell_mechinterp" / "data" / "raw",
        biodyn_work_root / "single_cell_mechinterp" / "data" / "perturb",
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rank local .h5ad datasets for age/longevity mechanistic interpretability research."
    )
    parser.add_argument(
        "--roots",
        nargs="*",
        default=None,
        help="Root directories to scan for .h5ad files. Defaults to single_cell_mechinterp data roots.",
    )
    parser.add_argument(
        "--non-recursive",
        action="store_true",
        help="Scan only the top level of each root directory.",
    )
    parser.add_argument(
        "--output-csv",
        default=str(Path(__file__).resolve().parents[1] / "outputs" / "age_dataset_ranking.csv"),
        help="Output CSV path for full ranking table.",
    )
    parser.add_argument(
        "--output-md",
        default=str(Path(__file__).resolve().parents[1] / "outputs" / "age_dataset_ranking.md"),
        help="Output Markdown path for top-k summary.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=15,
        help="Top-k datasets to include in Markdown summary.",
    )
    args = parser.parse_args()

    roots = [Path(p).expanduser().resolve() for p in args.roots] if args.roots else _default_roots()
    h5ad_paths = _discover_h5ad_files(roots, recursive=not args.non_recursive)

    if not h5ad_paths:
        raise FileNotFoundError(f"No .h5ad files found under roots: {', '.join(str(r) for r in roots)}")

    rows = [_summarize_h5ad(path) for path in h5ad_paths]
    df = pd.DataFrame(rows)

    # Sort successful reads by score, then append failed reads at the end.
    ok_df = df[df["error"] == ""].sort_values(
        by=["priority_score", "n_cells", "n_age_levels", "n_donors"],
        ascending=[False, False, False, False],
    )
    err_df = df[df["error"] != ""]
    ranked_df = pd.concat([ok_df, err_df], ignore_index=True)

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    ranked_df.to_csv(output_csv, index=False)

    output_md = Path(args.output_md)
    _write_markdown_summary(ok_df, output_md, top_k=max(args.top_k, 1))

    print(f"Scanned files: {len(h5ad_paths)}")
    print(f"CSV written: {output_csv}")
    print(f"Markdown written: {output_md}")

    # Print concise top-5 to terminal for fast feedback loops.
    preview = ok_df.head(5)[
        [
            "dataset_name",
            "priority_score",
            "priority_tier",
            "n_cells",
            "age_span_years",
            "n_age_levels",
            "n_donors",
            "n_cell_types",
        ]
    ]
    if not preview.empty:
        print("\nTop datasets:")
        print(preview.to_string(index=False))


if __name__ == "__main__":
    main()
