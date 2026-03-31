# Longevity Mechinterp

This repository is the public paper companion for the project **"A Donor-Aware Framework for Mechanistic Interpretability of Aging Signals in Single-Cell Foundation Models."** It contains the manuscript, the figure-building code, the stage scripts used in the project, and the small derived summary artifacts required to reproduce the figures and tables.

## What this repository is for
- Document the donor-aware interpretability pipeline used on frozen `scGPT` and `Geneformer` models.
- Provide the processed summary outputs behind the manuscript figures and tables.
- Make the paper build reproducible without redistributing large raw datasets or the full private project workspace.

## Main findings
- Frozen model representations contain detectable age-related signal across five human single-cell datasets.
- Sparse autoencoders recover donor-aware robust features, and cross-model overlap is strongest in inflammation / NF-kappaB programs.
- The clearest positive cell-type-local result appears in AIDA phase 1 v1 monocytes.
- The strongest global Geneformer inflammation branch looks promising under several controls, but it does **not** survive repeated fully composition-matched reruns.

## Repository layout
- `paper/`: manuscript source, compiled manuscript, figures, tables, and the figure-building script.
- `implementation/scripts/`: analysis and summarization scripts used across the staged pipeline.
- `implementation/outputs/`: selected derived CSV/Markdown artifacts needed for the manuscript and reproducibility notes.
- `planning/`: the project research plan.
- `docs/`: dataset provenance and repository-scope notes.

## Quick start
Rebuild the paper figures and tables from the included summary artifacts:

```bash
python paper/build_paper_assets.py
```

Recompile the PDF manuscript:

```bash
cd paper
latexmk -pdf -interaction=nonstopmode manuscript.tex
```

## Scope note
This repository intentionally includes **derived summary outputs only**. It does not redistribute the underlying raw single-cell datasets or the full heavyweight output tree from the private workspace. See `docs/dataset_provenance.md` for dataset notes and access constraints.
