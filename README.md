# Cross-cohort optimal transport maps macrophage plasticity in human atherosclerotic plaques

Analysis code for the manuscript *Cross-Cohort Optimal Transport Maps Macrophage
Plasticity and Competing Routes to Inflammation and Fibrosis in Human
Atherosclerotic Plaques*.

The pipeline reconstructs directed macrophage state-transition networks from
cross-sectional scRNA-seq data by combining pairwise optimal transport (Sinkhorn
divergence) with RNA velocity, across seven publicly available human plaque
cohorts (81,633 monocytes and macrophages).

## Data

All input data are public. Raw FASTQ files were obtained from GEO:

| Cohort | Accession |
| --- | --- |
| Alsaigh | GSE159677 |
| Bashore | GSE253904 |
| Pan | GSE155512 |
| Jaiswal | GSE179159 |
| Pauli | GSE247238 |
| Fernandez | GSE224273 |
| Wirka | GSE131778 |

The annotated AnnData object produced by this pipeline is deposited separately
(see the Data availability statement in the manuscript).

## Repository layout

Directories are numbered in execution order. Each `.py` script has a matching
`.sbatch` or `.sh` submission script for SLURM.

```
01_preprocessing/     Cell Ranger output -> QC -> ambient RNA -> doublets
02_integration/       Scanorama integration, clustering, marker genes
03_optimal_transport/ Pairwise Sinkhorn divergences, bootstrap, LOCO, epsilon sweep
04_velocity/          UniTVelo RNA velocity and directionality assignment
05_gradients/         Target-association gradients and program retention
06_validation/        Anti-circularity analyses in non-HVG gene space
07_clinical/          Patient-level clinical association (Bashore cohort)
08_tf_activity/       Transcription factor activity inference
09_figures/           Main figure generation
```

## Execution order

**1. Preprocessing** (`01_preprocessing/`)

- `run_cellbender.py` — ambient RNA removal per sample (GPU; FPR 0.01, 20,000
  droplets, 150 epochs)
- `run_preprocessing_pipeline.py` — per-sample QC and SOLO doublet detection
  via scvi-tools
- `run_QC_filters.ipynb` — per-dataset threshold selection (Supplementary Table 9)
- `QC_assessment_supplementary.ipynb` — per-cohort UMI medians

**2. Integration and annotation** (`02_integration/`)

- `run_scanorama_v2.py` — Scanorama on the full atlas (5,000 HVGs, 100
  dimensions, 50 nearest neighbours, exact search)
- `find_markers.py` — marker genes on the full atlas, used to select the
  monocyte/macrophage compartment
- Macrophage re-integration — same Scanorama parameters applied to the myeloid
  subset (`submit_macrophage_integration.sbatch`)
- `run_clustree.py` — clustering tree across resolutions 0.1–1.0
- `find_macrophage_markers.py` — marker genes at resolution 0.5

**3. Optimal transport** (`03_optimal_transport/`)

- `ot_pipeline_v8.py --mode {full,loco_alsaigh,loco_bashore,loco_jaiswal}` —
  pairwise Sinkhorn divergences, permutation null, connectivity transformation
- `ot_bootstrap_validation.py` — B = 500 bootstrap confidence intervals
- `epsilon_experiment.py` — sensitivity across epsilon in {0.005, 0.01, 0.025, 0.05}

**4. RNA velocity** (`04_velocity/`)

- `run_unitvelo_pairwise_array_v4.py --type_a A --type_b B` — UniTVelo in
  unified-time mode, one job per meta-cluster pair (SLURM array)
- `run_velocity_permutation.py` — bidirectional asymmetry test; classifies
  transitions as directed at |asymmetry| > 0.5

**5. Target-association gradients** (`05_gradients/`)

- `OT_gradients.py` — per-cell transport weights, library-size correction,
  quartile stratification, permutation testing
- `run_module_scores.py` — program-level retention using module scores over the
  top 100 genes per meta-cluster

**6. Anti-circularity validation** (`06_validation/`)

Uses the 12,003 genes excluded from HVG selection in every cohort, which
contribute nothing to the embedding, the cost matrix or the target-association
scores.

- `run_nonHVG_DEGs.py` — differential expression in non-HVG space
- `run_nonHVG_integration_OT.py` — independent Scanorama integration and OT on
  non-HVG genes
- `run_nonHVG_SRC_all.py`, `run_nonHVG_TGT_all.py`, `run_nonHVG_gradients.py` —
  gradient analyses over all filtered genes
- `nonhvg_target_specificity.ipynb` — target-specificity control comparing
  Scav→Inflam against Scav→Fibro from the same source cells

**7. Clinical association** (`07_clinical/`)

- `clinical_analyses.ipynb` — patient-level aggregation of per-sample OT
  divergences in the Bashore cohort and symptomatic vs asymptomatic comparison

**8. Transcription factor activity** (`08_tf_activity/`)

- decoupleR univariate linear model with the CollecTRI regulatory network,
  comparing Q4 against Q1 of each target-association gradient

**9. Figures** (`09_figures/`)

- `figure_1.ipynb` … `figure_5.ipynb` — main manuscript figures

## Environment

Analyses were run on the UBELIX cluster (University of Bern) under two conda
environments:

- `py311_r_env` — Python 3.11, most analyses
- `unitvelo` — RNA velocity only (requires `TF_USE_LEGACY_KERAS=True`)

Principal packages: Scanpy, Scanorama, POT (Python Optimal Transport), UniTVelo,
scVelo, scvi-tools, CellBender, decoupleR, pyclustree.

## Notes on paths

Scripts contain absolute paths to the compute cluster on which they were run.
They are provided as an exact record of the analyses performed rather than as a
turnkey pipeline; input and output paths need to be adapted before re-running.
Preprocessing scripts additionally reference institutional storage that is not
publicly accessible. The reproducible entry point is the set of raw FASTQ files
listed above.

## Citation

If you use this code, please cite the manuscript.
