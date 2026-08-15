import scanpy as sc
import pandas as pd
import sys
import os
import matplotlib
matplotlib.use('Agg') 

print("--- STARTING MACROPHAGE MARKER CALCULATION (FAST MODE) ---")

# --- 1. Configuration ---

# CHANGE 1: Point to the MACROPHAGE file with clusters (created by your Clustree run)
input_h5ad = "/storage/homefs/sv24v923/MPI_data/clean_pipeline/scanorama_integrated_MACROPHAGES_WITH_CLUSTERS.h5ad"

# CHANGE 2: Save to a specific Macrophage folder
output_dir = "/storage/homefs/sv24v923/MPI_data/clean_pipeline/macrophage_annotation"
os.makedirs(output_dir, exist_ok=True)

# CHANGE 3: Set Resolution to 0.5 (The one we decided on)
leiden_resolution = 0.5
target_cluster_key = f"leiden_{str(leiden_resolution).replace('.', '_')}" # becomes "leiden_0_5"

output_h5ad = f"{output_dir}/MACROPHAGE_subset_res{leiden_resolution}_markers.h5ad"
output_markers_csv = f"{output_dir}/markers_MACROPHAGE_subset_res{leiden_resolution}.csv"

# --- 2. Load Data ---
try:
    print(f"Loading data from {input_h5ad}...")
    adata = sc.read_h5ad(input_h5ad)
    print(f"Data loaded. Shape: {adata.shape}")
except Exception as e:
    print(f"CRITICAL ERROR: Could not load file. {e}", file=sys.stderr)
    sys.exit(1)

# --- 2.1 Filter Noise Clusters (18-24) ---
print("Filtering out noise clusters (18-24)...")
# Generate list ['18', '19', ... '24']
clusters_to_remove = [str(i) for i in range(18, 25)] 

# Keep only cells that are NOT (~) in the remove list
adata = adata[~adata.obs[target_cluster_key].isin(clusters_to_remove)].copy()

print(f"Filtered. New shape: {adata.shape}")

# --- 3. Compute UMAP ---
# Neighbors are already in the file from the Clustree/Re-integration step.
# We just calculate UMAP for plotting later.
print("Neighbors graph found in file. Skipping calculation.")
print(f"Targeting clusters: {target_cluster_key}")

print("Computing UMAP...")
sc.tl.umap(adata)

# --- 4. Find Marker Genes ---
print(f"Calculating marker genes for {target_cluster_key}...")
print("Running Wilcoxon rank-sum test...")

sc.tl.rank_genes_groups(
    adata,
    target_cluster_key, # "leiden_0_5"
    method='wilcoxon',
    use_raw=True,       # Uses Log-Normalized counts
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