#!/usr/bin/env python
import os
import shutil
# This MUST be the first import
os.environ['TF_USE_LEGACY_KERAS'] = 'True'

import scanpy as sc
import scvelo as scv
import unitvelo as utv
from pathlib import Path
import tensorflow as tf
import argparse
import matplotlib.pyplot as plt 
import numpy as np

# --- 1. SET UP ARGUMENT PARSING ---
parser = argparse.ArgumentParser(description='Run UniTVelo for a specific pair of cell types.')
parser.add_argument('--type_a', required=True, help='The starting cell type (IROOT).')
parser.add_argument('--type_b', required=True, help='The target cell type.')
args = parser.parse_args()

print("--- UniTVelo Pairwise Run (META CLUSTERS | TRAINING DATA) ---")

# --- 2. SETUP OUTPUT DIRECTORY (NEW LOCATION) ---
# Changed folder name to avoid overwriting your previous results
output_dir = Path("unitvelo_results_META_VALIDATION_V3")
output_dir.mkdir(parents=True, exist_ok=True)
print(f"Output directory set to: {output_dir.resolve()}")

# --- 3. VISUALIZATION SETTINGS ---
scv.settings.figdir = str(output_dir)
sc.settings.figdir = str(output_dir)

scv.settings.set_figure_params(
    'scvelo', 
    dpi=300, 
    dpi_save=300, 
    format='png', 
    transparent=False, 
    fontsize=14 
)

print("Loading dataset...")
# --- CRITICAL CHANGE 1: Using the TRAINING object (The "Good" one) ---
path_integrated = Path("/storage/homefs/sv24v923/MPI_data/clean_pipeline/macrophage_annotation/Macrophage_Atlas_FINAL_v3_CLEAN_ANNOTATED.h5ad")
adata = sc.read_h5ad(path_integrated)
print(f"Loaded {adata.n_obs} cells.")

# --- CRITICAL CHANGE 2: Force usage of 'cell_type_meta' ---
print("Configuring Meta-Clusters...")
# To this:
if 'cell_type_meta_v3' in adata.obs.columns:
    adata.obs['cell_type'] = adata.obs['cell_type_meta_v3'].copy()
else:
    raise ValueError("ERROR: 'cell_type_meta_v3' column not found! Check your h5ad object.")

# Ensure clean strings
adata.obs['cell_type'] = adata.obs['cell_type'].astype(str).str.strip()

# --- 4. Create Subset ---
TYPE_A = args.type_a.strip()
TYPE_B = args.type_b.strip()

# Safety Check
unique_types = adata.obs['cell_type'].unique()
if TYPE_A not in unique_types:
    raise ValueError(f"ERROR: Cell type '{TYPE_A}' not found in cell_type_meta. Available: {unique_types}")
if TYPE_B not in unique_types:
    raise ValueError(f"ERROR: Cell type '{TYPE_B}' not found in cell_type_meta. Available: {unique_types}")

# Clean filename string
safe_a = TYPE_A.replace(' ', '_').replace('+', 'p').replace('/', '_')
safe_b = TYPE_B.replace(' ', '_').replace('+', 'p').replace('/', '_')
image_subfix = f"_{safe_a}_to_{safe_b}"

print(f"--- Processing Pair: {TYPE_A} -> {TYPE_B} ---")

# Subset
adata_subset = adata[adata.obs['cell_type'].isin([TYPE_A, TYPE_B])].copy()
print(f"Subset size: {adata_subset.n_obs} cells.")

# --- 5. Configure UniTVelo ---
config = utv.config.Configuration()
config.GPU = -1
config.IROOT = TYPE_A 
config.FIT_OPTION = '1'

# Set Unique Temporary Directory
unique_temp_dir = output_dir / f"temp_{safe_a}_vs_{safe_b}"
unique_temp_dir.mkdir(parents=True, exist_ok=True)
config.DATA_PATH = str(unique_temp_dir)

label_key = 'cell_type'

# --- 6. Preprocessing ---
print("--- Starting Preprocessing ---")

# A. Handle Genes
default_n_top_genes = 2000 
if adata_subset.n_vars < default_n_top_genes:
    print(f"Gene count ({adata_subset.n_vars}) is < {default_n_top_genes}. Keeping all genes.")
    adata_subset.var['highly_variable'] = True
else:
    print("Running HVG selection...")
    sc.pp.highly_variable_genes(adata_subset, n_top_genes=default_n_top_genes, subset=False)

# B. Handle Neighbors (Using X_scanorama if available)
if 'X_scanorama' in adata_subset.obsm.keys():
    print("Using X_scanorama for neighbor calculation...")
    sc.pp.neighbors(adata_subset, use_rep='X_scanorama', n_neighbors=30)
else:
    print("WARNING: X_scanorama not found! Falling back to X_pca.")
    if 'X_pca' not in adata_subset.obsm.keys():
        sc.pp.pca(adata_subset)
    sc.pp.neighbors(adata_subset, use_rep='X_pca', n_neighbors=30)

# C. Calculate Moments
print("Calculating moments...")
scv.pp.moments(adata_subset, n_pcs=None, n_neighbors=30)

# --- 7. Run Model ---
print("Running UniTVelo model...")
adata_subset = utv.run_model(adata_subset, label_key, config_file=config)

# --- 8. Save Data ---
save_path_h5ad = output_dir / f"unitvelo_pairwise{image_subfix}.h5ad"
adata_subset.write_h5ad(save_path_h5ad)
print(f"Saved H5AD to: {save_path_h5ad}")

# --- 9. Plotting ---
print("Generating Plots...")

# Ensure Velocity Embedding exists for UMAP
scv.tl.velocity_embedding(adata_subset, basis='umap')

# Plot 1: Latent Time
scv.pl.scatter(
    adata_subset,
    color='latent_time',
    cmap='gnuplot',
    basis='umap',
    title=f'Latent Time: {TYPE_A} -> {TYPE_B}',
    save=f'latent_time{image_subfix}.png', 
    show=False,
    size=120, 
    legend_loc='right margin'
)

# Plot 2: Velocity Stream
try:
    scv.pl.velocity_embedding_stream(
        adata_subset,
        basis='umap',
        color=label_key,
        legend_loc='right margin', 
        title=f'Velocity: {TYPE_A} -> {TYPE_B}',
        save=f'velocity_stream{image_subfix}.png',
        show=False,
        density=1,       
        linewidth=1.5,   
        arrow_size=1.5   
    )
except Exception as e:
    print(f"Warning: Stream plot failed ({e}). Saving Arrow plot instead.")
    scv.pl.velocity_embedding(
        adata_subset,
        basis='umap',
        color=label_key,
        legend_loc='right margin',
        title=f'Velocity Arrows: {TYPE_A} -> {TYPE_B}',
        save=f'velocity_arrow{image_subfix}.png',
        show=False,
        arrow_length=4, 
        arrow_size=4
    )

# --- 10. CLEANUP ---
try:
    if unique_temp_dir.exists():
        shutil.rmtree(unique_temp_dir)
        print(f"Cleaned up temporary folder: {unique_temp_dir}")
except Exception as e:
    print(f"Warning: Could not remove temp folder: {e}")

print(f"--- SUCCESS: Results saved in {output_dir} ---")