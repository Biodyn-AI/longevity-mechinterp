# Longevity-Age Mechanistic Interpretability Plan (scGPT + Geneformer)

## 1. Core Question
Can frozen single-cell foundation models encode biologically meaningful aging structure in internal representations, and can we map that structure to interpretable mechanisms?

## 2. Working Principles
1. Start with frozen models first (no fine-tuning).
2. Keep donor-aware and cell-type-aware controls mandatory.
3. Treat visualization-only claims as insufficient.
4. Promote hypotheses only after external validation and confound stress tests.

## 3. Data Strategy
## 3.1 Dataset selection
Use `implementation/scripts/rank_age_labeled_datasets.py` to rank candidate datasets by:
- age span
- number of age levels
- donor diversity
- cell-type diversity
- total cells

## 3.2 Initial local priorities (from current assets)
- `tabula_sapiens_immune.h5ad`: broad age range and donor coverage; primary discovery set.
- `tabula_sapiens_immune_subset_20000.h5ad`: fast prototype set.
- `tabula_sapiens_lung.h5ad`: medium-priority cross-tissue replication.
- `krasnow_lung_smartsq2.h5ad`: external-style sanity/replication set.
- De-prioritize single-age datasets for aging signal discovery (can still be used for negative controls).

## 4. Experiment Tracks
## 4.1 Track A: Frozen representation age probes
Goal: test whether age is linearly decodable from hidden states.

Protocol:
1. Extract per-cell embeddings by layer for scGPT and Geneformer.
2. Train linear probes for:
   - age-bin classification
   - age regression (optional)
3. Evaluate with strict splits:
   - donor-held-out
   - within-cell-type
   - tissue-stratified where possible

Success criteria:
- performance above null and confound baselines
- stability across seeds and layers

## 4.2 Track B: Manifold and intrinsic-dimension analysis
Goal: measure compactness and geometry of age-related structure.

Protocol:
1. Compute PCA spectrum, participation ratio, and alternative intrinsic-dimension estimators.
2. Quantify how age aligns to latent axes:
   - correlation with principal axes
   - age-trajectory smoothness in latent space
3. Compare geometry before vs after donor/cell-type balancing.

Success criteria:
- reproducible age-aligned geometry after confound controls
- cross-dataset consistency of top geometry signals

## 4.3 Track C: Sparse Autoencoder (SAE) feature discovery
Goal: find sparse latent features associated with aging programs.

Protocol:
1. Train SAEs on selected layers (start with 2-3 promising layers from Tracks A/B).
2. Score SAE feature activations against age, cell type, and donor.
3. Keep only features that pass donor-aware controls.
4. Annotate features via pathway enrichment (senescence, mTOR, proteostasis, inflammation, mitochondrial stress).

Success criteria:
- sparse features with robust age association not explained by donor or cell composition
- biologically coherent enrichment profiles

## 4.4 Track D: Causal/internal validation
Goal: move from association to mechanism-like evidence.

Protocol:
1. Feature/pathway-level interventions in representation space:
   - activation patching
   - feature ablation
2. Observe changes in age-probe outputs and pathway-level readouts.
3. Validate intervention effects across seeds and datasets.

Success criteria:
- consistent directional effects under controlled interventions
- effects replicate cross-model (scGPT and Geneformer) or are explicitly model-specific

## 5. Cross-Model Alignment (scGPT vs Geneformer)
1. Compare which layers carry strongest age signal in each model.
2. Compare latent directions and SAE feature families across models.
3. Build a consensus table:
   - convergent signals (high confidence)
   - diverging signals (model-specific hypotheses)

## 6. Confound and Failure-Mode Battery (Mandatory)
For every promoted result:
1. Donor leakage checks (donor-held-out, donor-balanced subsampling).
2. Cell-type composition controls (within-cell-type analysis and composition-matched pseudobulk).
3. Batch checks (if batch metadata exists).
4. Label permutation/null controls.
5. Sensitivity to preprocessing and random seed.

Results that fail these checks are reported as exploratory only.

## 7. Decision Gates
- **Gate G1 (Signal existence):** frozen probes show non-trivial age signal with donor-aware validation.
- **Gate G2 (Geometry robustness):** manifold effects remain after confound controls.
- **Gate G3 (Interpretability):** SAE/pathway features are biologically coherent and stable.
- **Gate G4 (Mechanistic support):** intervention effects are reproducible and directional.

Only after G1-G3 should we consider parameter-efficient tuning (LoRA/adapters).

## 8. Fine-Tuning Policy
Default: no fine-tuning.

Escalate to light tuning only if:
1. frozen models fail G1 despite strong data,
2. controls are clean,
3. added complexity is justified by clear gain.

If needed, use:
- linear head first,
- then adapters/LoRA,
- avoid full-model training in early phases.

## 9. Near-Term Execution Plan (first 2 weeks)
1. Run dataset ranking script and lock priority datasets.
2. Implement layerwise embedding extraction for both models on immune subset.
3. Run linear probe baseline matrix (layer x split x metric).
4. Run first manifold analysis on top-3 layers.
5. Start SAE pilot on best single layer per model.
6. Produce first checkpoint report with go/no-go at G1.

## 10. Deliverables
1. Ranked dataset table (`implementation/outputs/age_dataset_ranking.csv`).
2. Baseline probe report (layerwise + confound controls).
3. Manifold dimension/geometry report.
4. SAE feature catalog with pathway annotations.
5. Intervention report with confidence levels and limitations.
