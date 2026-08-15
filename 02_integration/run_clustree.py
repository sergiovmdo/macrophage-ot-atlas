import scanpy as sc
import sys
import os
import matplotlib
# CRITICAL: Headless plotting for SLURM
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from pyclustree import clustree 
import time

print("--- STARTING OPTIMIZED PYCLUSTREE WORKFLOW ---")
start_time = time.time()

# --- 1. Configuration ---
input_h5ad = "/storage/homefs/sv24v923/MPI_data/clean_pipeline/macrophages_only/MACROPHAGES_REINTEGRATED.h5ad"
output_plot = "/storage/homefs/sv24v923/MPI_data/clean_pipeline/figures/final_figures/clustree_macrophages_stability.pdf"

# SAFETY SAVE PATH (Saves clusters before plotting)
output_h5ad_safety = "/storage/homefs/sv24v923/MPI_data/clean_pipeline/scanorama_integrated_MACROPHAGES_WITH_CLUSTERS.h5ad"

# Resolutions
resolutions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

# --- 2. Load Data ---
try:
    print(f"Loading data from {input_h5ad}...")
    adata = sc.read_h5ad(input_h5ad)
    print(f"Data loaded successfully. Shape: {adata.shape}")
except Exception as e:
    print(f"CRITICAL ERROR: Could not load file. {e}", file=sys.stderr)
    sys.exit(1)

# --- 3. INTELLIGENT NEIGHBOR CHECK (The Speedup) ---
# We check if neighbors exist and look reasonable.
if 'neighbors' in adata.uns and 'distances' in adata.obsp:
    print("\n[OPTIMIZATION] Found existing neighbor graph in file.")
    print(f"Params used: {adata.uns['neighbors'].get('params', 'Unknown')}")
    print("Skipping sc.pp.neighbors() calculation to save time.")
    print("This preserves exact consistency with your Integration/UMAP.")
else:
    print("\n[WARNING] No neighbors found. Calculating now (This will take time)...")
    # Parameters matches your integration
    sc.pp.neighbors(adata, use_rep="X_scanorama", n_neighbors=50, n_pcs=100)

# --- 4. Run Leiden for all resolutions ---
print(f"\nRunning Leiden clustering for resolutions: {resolutions}...")

cluster_keys = [] 
for res in resolutions:
    # Format key as 'leiden_0_1' 
    key = f"leiden_{str(res).replace('.', '_')}" 
    
    print(f"  ... computing {key} (res={res}) ...")
    # Flavor 'igraph' is often slightly faster/cleaner if installed, defaulting to standard if not
    sc.tl.leiden(adata, resolution=res, key_added=key)
    
    # Ensure it is categorical for pyclustree
    adata.obs[key] = adata.obs[key].astype('category')
    cluster_keys.append(key)

print(f"All clustering complete. Time elapsed: {(time.time() - start_time)/60:.1f} minutes")

# --- SAFETY SAVE ---
# We save here immediately. If plotting fails, you have the data.
print(f"\nSAVING CHECKPOINT to {output_h5ad_safety}...")
adata.write_h5ad(output_h5ad_safety, compression="gzip")
print("Checkpoint saved.")

# --- 5. Generate Clustree Plot ---
print(f"Generating clustree plot...")
plt.figure(figsize=(16, 24)) 

try:
    highest_res_key = cluster_keys[-1]
    
    # FIXED: Using cluster_keys instead of prefix
    graph = clustree(
        adata,
        cluster_keys=cluster_keys,
        edge_prop="count",      
        edge_show=True,
        node_color=highest_res_key, 
        cmap="turbo",           
        node_size_edge=100,     
        show_fraction=False     
    )
    
    # --- 6. Save Plot ---
    print(f"Saving plot to {output_plot}...")
    plt.savefig(output_plot, format='pdf', bbox_inches='tight', dpi=300)
    plt.close()
    
    print("\n" + "="*30)
    print("    --- WORKFLOW COMPLETE ---")
    print("="*30)

except Exception as e:
    print(f"CRITICAL ERROR plotting pyclustree: {e}", file=sys.stderr)
    print("Do not panic. Your clusters are saved in the checkpoint file.")
    sys.exit(1)