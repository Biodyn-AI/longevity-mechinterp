# Paper

This directory contains the Biogerontology-oriented manuscript draft and all publication assets required to rebuild the figures and tables from the included summary outputs.

## Files
- `manuscript.tex`: LaTeX source for the journal manuscript.
- `manuscript.pdf`: compiled PDF snapshot.
- `manuscript.docx`: Word export for journal submission workflows.
- `refs.bib`: bibliography.
- `figures/`: generated figures used in the manuscript.
- `tables/`: generated summary tables and metrics.
- `build_paper_assets.py`: rebuild figures and tables from the included `implementation/outputs/` subset.

## Rebuild
From the repository root:

```bash
python paper/build_paper_assets.py
cd paper
latexmk -pdf -interaction=nonstopmode manuscript.tex
```

## Note on author metadata
The manuscript in this public snapshot still contains placeholder author information because submission-specific identity fields are finalized separately from the reproducibility package.
