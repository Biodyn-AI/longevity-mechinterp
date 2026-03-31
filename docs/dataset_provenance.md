# Dataset Provenance and Access Notes

The manuscript draws on age-labeled human single-cell datasets that were available in the original local project workspace.

## Datasets referenced in the paper
- `AIDA phase 1 v1`: local project asset used as a core cohort.
- `AIDA phase 1 v2`: local project asset used as a core cohort and for the strongest-branch deep-dive analyses.
- `Allen immune atlas`: external reference cohort represented here only through derived summary outputs.
- `Allen aging plasma cells`: external reference subset represented here only through derived summary outputs.
- `Yazar donor cohort`: derived from the published Yazar single-cell eQTL study and represented here only through derived summary outputs.
- `Tabula Sapiens`: used during project planning and data ranking; not redistributed here.

## What is included in this repository
- Processed summary CSV and Markdown artifacts that support the paper figures and tables.
- Manuscript figures and tables generated from those summaries.
- Analysis and summarization scripts.

## What is not included
- Raw `.h5ad` files or other underlying single-cell matrices.
- Large intermediate outputs from the private workspace.
- Dataset files governed by third-party access terms or local-storage constraints.

## Practical interpretation
This repository is designed to make the manuscript and its reported summary analyses transparent and reproducible at the figure/table level, while respecting the access constraints and size limits of the underlying raw datasets.
