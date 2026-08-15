import scanpy as sc
import pandas as pd
import sys
import os
import matplotlib
matplotlib.use('Agg') 

print("--- STARTING MARKER CALCULATION (FAST MODE) ---")

# --- 1. Configuration ---
# CHANGE 1: Use the file that already has the clusters/neighbors
input_h5ad = "/storage/homefs/sv24v923/MPI_data/clean_pipeline/scanorama_integrated_FULL_WITH_CLUSTERS.h5ad"

output_dir = "/storage/homefs/sv24v923/MPI_data/clean_pipeline/results_annotation"
os.makedirs(output_dir, exist_ok=True)

# CHANGE 2: Update Resolution to the one you chose
leiden_resolution = 0.2
target_cluster_key = f"leiden_{str(leiden_resolution).replace('.', '_')}" # e.g. "leiden_0_2"

output_h5ad = f"{output_dir}/FULL_dataset_res{leiden_resolution}_markers.h5ad"
output_markers_csv = f"{output_dir}/markers_FULL_dataset_res{leiden_resolution}.csv"

# --- 2. Load Data ---
try:
    print(f"Loading data from {input_h5ad}...")
    adata = sc.read_h5ad(input_h5ad)
    print(f"Data loaded. Shape: {adata.shape}")
except Exception as e:
    print(f"CRITICAL ERROR: Could not load file. {e}", file=sys.stderr)
    sys.exit(1)

# --- 3. Compute UMAP (Only thing missing) ---
# CHANGE 3: Skip sc.pp.neighbors (Already in file) and sc.tl.leiden (Already in file)
print("Neighbors graph found in file. Skipping calculation.")
print(f"Using existing clusters: {target_cluster_key}")

print("Computing UMAP (needed for plotting later)...")
sc.tl.umap(adata)

# --- 4. Find Marker Genes ---
print(f"Calculating marker genes for {target_cluster_key}...")
print("NOTE: This step calculates statistics for ALL genes across 400k cells.")

sc.tl.rank_genes_groups(
    adata,
    target_cluster_key, # Use "leiden_0_2" directly
    method='wilcoxon',
    use_raw=True,       
    pts=True,           
    n_genes=300         
)
print("Marker calculation complete.")

# --- 5. Save Marker Genes to CSV ---
print(f"Exporting marker stats to {output_markers_csv}...")
markers_df = sc.get.rank_genes_groups_df(adata, key='rank_genes_groups', group=None)
markers_df.to_csv(output_markers_csv, index=False)

# --- 6. Save the Object ---
print(f"Saving final object to {output_h5ad}...")
adata.write_h5ad(output_h5ad, compression="gzip")

print(f"\nWorkflow complete!")
print(f"Analyzed H5AD: {output_h5ad}")
print(f"Markers CSV:   {output_markers_csv}")