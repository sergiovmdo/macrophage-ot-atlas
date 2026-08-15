import matplotlib
# CRITICAL: Headless plotting for SLURM
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import scanpy as sc
import scanorama
import anndata
import sys
import os
import numpy as np
import gc
import scipy.sparse

print(f"Scanpy version: {sc.__version__}")
print(f"Scanorama version: {scanorama.__version__}")

# --- 1. Configuration ---
JOB_ID = "FULL_DATASET_HIGH_PRECISION"

# --- PARAMETERS ---
N_HVGS = 5000           
INTEGRATION_DIMS = 100  
KNN = 50 
COLOR_PALETTE = "Set1"

# Output Paths
OUTPUT_FILE = "scanorama_integrated_FULL_7datasets_HIGH_PRECISION.h5ad"
PLOT_DIR = "/storage/homefs/sv24v923/MPI_data/clean_pipeline/figures/final_figures"

os.makedirs(PLOT_DIR, exist_ok=True)
print(f"Plots will be saved to: {PLOT_DIR}")

dp_base = "/storage/research/igmp_dp_workspace/sergio_vazquez/MPI/Data/Processed_Datasets"
slide_base = "/storage/research/igmp_slide_workspace/GRP Zlobec/Sergio/MPI/Data/Processed_Datasets"

datasets_config = {
    # --- DP Workspace ---
    "alsaigh": f"{dp_base}/alsaigh_run2_clean_with_velocity_FINAL.h5ad",
    "wirka":   f"{dp_base}/wirka_2_clean_with_velocity_FINAL.h5ad",
    "pauli":   f"{dp_base}/pauli_2_clean_with_velocity_FINAL.h5ad",
    
    # --- Slide Workspace ---
    "bashore":   f"{slide_base}/bashore_run2_clean_with_velocity_FINAL.h5ad",
    "jaiswal":   f"{slide_base}/jaiswal_2_clean_with_velocity_FINAL.h5ad",
    "fernandez": f"{slide_base}/fernandez_2_clean_with_velocity_FINAL.h5ad",
    "pan":       f"{slide_base}/pan_clean_with_velocity_FINAL.h5ad"
}

# --- 2. Load Data ---
adatas = {}
common_genes = None

print("--- Loading and Intersecting Datasets ---")

for label, path in datasets_config.items():
    try:
        print(f"Loading {label}...")
        ad = sc.read_h5ad(path)
        ad.var_names_make_unique()
        
        # --- DGE PRESERVATION: RAW COUNTS ---
        # Ensure we have the raw integer counts stash
        if 'counts' not in ad.layers:
            # If no counts layer, assuming .X contains raw counts (standard before processing)
            ad.layers['counts'] = ad.X.copy()
        
        # --- VELOCITY PRESERVATION: SPLICED/UNSPLICED ---
        # We must ensure these layers exist. If missing, fill with 0s to prevent sc.concat from dropping them.
        rows, cols = ad.shape
        if 'spliced' not in ad.layers:
            print(f"WARNING: 'spliced' layer missing in {label}. Filling with 0s to preserve structure.")
            ad.layers['spliced'] = scipy.sparse.csr_matrix((rows, cols), dtype=np.float32)
            
        if 'unspliced' not in ad.layers:
            print(f"WARNING: 'unspliced' layer missing in {label}. Filling with 0s to preserve structure.")
            ad.layers['unspliced'] = scipy.sparse.csr_matrix((rows, cols), dtype=np.float32)

        if common_genes is None:
            common_genes = set(ad.var_names)
        else:
            common_genes = common_genes.intersection(ad.var_names)
            
        adatas[label] = ad
    except Exception as e:
        print(f"CRITICAL ERROR loading {label}: {e}")
        sys.exit(1)

common_genes = list(common_genes)
print(f"\nFound {len(common_genes)} common genes.")

# --- 3. Normalize & Concatenate ---
adatas_list = []
batch_keys = []

print("--- Normalizing and Preparing ---")
for label, ad in adatas.items():
    # Subset to common genes. 
    # NOTE: slicing [:, common_genes] AUTOMATICALLY slices .layers['counts'], 'spliced', 'unspliced' too.
    ad = ad[:, common_genes].copy()
    
    # Normalize X (Log1p) for Integration/Visualization
    # NOTE: This does NOT touch .layers['counts'], so raw data is safe.
    sc.pp.normalize_total(ad, target_sum=1e4)
    sc.pp.log1p(ad)
    
    ad.obs['batch'] = label
    adatas_list.append(ad)
    batch_keys.append(label)

del adatas
gc.collect()

print("--- Creating Master Object ---")
# sc.concat will merge .layers if they are present in all datasets.
# Since we forced spliced/unspliced/counts to exist in loop above, they will be preserved.
adata_full = sc.concat(
    adatas_list, 
    join='outer', # Effectively 'inner' because we already subsetted to common_genes
    label='batch', 
    keys=batch_keys,
    index_unique='-'
)

# Ensure batch is categorical
adata_full.obs['batch'] = adata_full.obs['batch'].astype('category')

# Free the list memory
del adatas_list
gc.collect()

# --- 4. Calculate Robust HVGs ---
print(f"--- Calculating Top {N_HVGS} HVGs (Batch-Aware) ---")

sc.pp.highly_variable_genes(
    adata_full, 
    n_top_genes=N_HVGS, 
    batch_key='batch', 
    flavor='seurat',
    subset=False
)

hvg_genes = adata_full.var_names[adata_full.var['highly_variable']].tolist()
print(f"Selected {len(hvg_genes)} robust highly variable genes.")

# --- 5. Prepare Data for Scanorama ---
print("--- Splitting Data and Densifying for Scanorama ---")
scanorama_input = []
batch_categories = adata_full.obs['batch'].unique()

for b in batch_categories:
    # Slice by batch and by HVG
    # We only need .X for Scanorama, we don't need to carry layers here
    subset = adata_full[adata_full.obs['batch'] == b, hvg_genes].copy()
    
    # CRITICAL FIX: Convert Sparse to Dense to prevent Integer Overflow
    if scipy.sparse.issparse(subset.X):
        subset.X = subset.X.toarray()
        
    scanorama_input.append(subset)
    print(f"Prepared {b}: {subset.shape}")

gc.collect()

# --- 6. Plot BEFORE Integration ---
print("--- Generating 'Before' Plot ---")

adata_temp = adata_full[:, hvg_genes].copy()
sc.pp.scale(adata_temp, max_value=10)
sc.tl.pca(adata_temp, svd_solver='arpack', n_comps=50)
sc.pp.neighbors(adata_temp, n_pcs=50)
sc.tl.umap(adata_temp)

sc.pl.umap(
    adata_temp, 
    color='batch', 
    palette=COLOR_PALETTE, 
    title='Unintegrated (Before)', 
    show=False
)
plt.savefig(f"{PLOT_DIR}/1_umap_before_integration_HIGH_PRECISION.png", bbox_inches='tight', dpi=300)
plt.close()

del adata_temp
gc.collect()

# --- 7. Run Scanorama ---
print(f"--- Running Scanorama (Exact Search) ---")

corrected = scanorama.correct_scanpy(
    scanorama_input,
    return_dimred=True,
    dimred=INTEGRATION_DIMS,
    approx=False,    
    knn=KNN,         
    sigma=15,        
    alpha=0.10,
    verbose=True
)

print("Scanorama complete. Stacking embeddings...")

scanorama_embeddings = []
for ad_corr in corrected:
    scanorama_embeddings.append(ad_corr.obsm['X_scanorama'])

full_embedding = np.concatenate(scanorama_embeddings, axis=0)

if full_embedding.shape[0] != adata_full.shape[0]:
    print(f"CRITICAL: Embedding shape {full_embedding.shape} mismatch with Adata {adata_full.shape}!")
    sys.exit(1)

# Store the integrated embedding
adata_full.obsm['X_scanorama'] = full_embedding

del scanorama_input
del corrected
gc.collect()

# --- 8. Plot AFTER Integration ---
print("--- Generating 'After' Plot ---")

sc.pp.neighbors(adata_full, use_rep='X_scanorama', n_neighbors=KNN)
sc.tl.umap(adata_full)

sc.pl.umap(
    adata_full, 
    color='batch', 
    palette=COLOR_PALETTE, 
    title='Scanorama Integrated (High Precision)', 
    show=False
)
plt.savefig(f"{PLOT_DIR}/2_umap_after_integration_HIGH_PRECISION.png", bbox_inches='tight', dpi=300)
plt.close()

# --- 9. Final DGE Prep & Save ---
print("--- Finalizing Object for DGE/Velocity ---")

# 1. Set .raw to the log-normalized data (Standard Scanpy DGE expects this)
adata_full.raw = adata_full

# 2. Verify layers exist
print("Layers preserved in final object:", adata_full.layers.keys())

# 3. Save
print(f"--- Saving to {OUTPUT_FILE} ---")
adata_full.write_h5ad(OUTPUT_FILE, compression="gzip")

print("Workflow Complete.")