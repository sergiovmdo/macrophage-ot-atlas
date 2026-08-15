import os
import glob
import subprocess
import sys
import warnings
import argparse
import scanpy as sc
import scvi
import anndata as ad
import pandas as pd
import torch
import numpy as np

# --- 1. ARGUMENT PARSING ---
parser = argparse.ArgumentParser(description='Run Single-Cell Preprocessing Pipeline')
parser.add_argument('mode', choices=['dp', 'slide'], help="Storage location: 'dp' or 'slide'")
parser.add_argument('dataset_name', type=str, help="Name of the dataset folder (e.g., wirka_2)")

# Boolean switch: If present, runs CellBender on EVERYTHING
parser.add_argument('--remove_ambient', action='store_true', 
                    help="If set, runs CellBender on ALL samples in the dataset.")

args = parser.parse_args()

TARGET_DATASET = args.dataset_name
PERFORM_AMBIENT = args.remove_ambient

# --- 2. CONFIGURATION ---

if args.mode == 'slide':
    BASE_DIR = "/storage/research/igmp_slide_workspace/GRP Zlobec/Sergio/MPI/Data"
elif args.mode == 'dp':
    BASE_DIR = "/storage/research/igmp_dp_workspace/sergio_vazquez/MPI/Data/"

OUTPUT_ROOT = os.path.join(BASE_DIR, "Processed_Datasets")

# Path to CellBender
CELLBENDER_EXE = os.path.expanduser("~/.conda/envs/cellbender/bin/cellbender")

# --- PRODUCTION SETTINGS ---
TOTAL_DROPLETS = "20000" 
EPOCHS_CB = "150"        
# -------------------------

MIN_GENES = 200
MAX_MITO = 25.0 

# Setup
os.makedirs(OUTPUT_ROOT, exist_ok=True)
sc.set_figure_params(figsize=(4, 4))
warnings.filterwarnings("ignore")

# --- 3. SETUP PATHS ---
DATASET_PATH = os.path.join(BASE_DIR, TARGET_DATASET)

print(f"==================================================")
print(f" PROCESSING DATASET: {TARGET_DATASET}")
print(f" MODE: {args.mode.upper()}")
print(f" AMBIENT REMOVAL: {'ENABLED (ALL SAMPLES)' if PERFORM_AMBIENT else 'DISABLED'}")
print(f" Path: {DATASET_PATH}")
print(f"==================================================")

if not os.path.exists(DATASET_PATH):
    print(f"❌ CRITICAL: Dataset folder not found at {DATASET_PATH}")
    sys.exit(1)

if PERFORM_AMBIENT and not os.path.exists(CELLBENDER_EXE):
    print(f"❌ CRITICAL: CellBender not found at {CELLBENDER_EXE}")
    sys.exit(1)

# --- 4. HELPER FUNCTIONS ---
def run_cellbender(input_h5, output_h5):
    if os.path.exists(output_h5): return
    
    cmd = [
        CELLBENDER_EXE, "remove-background", 
        "--input", input_h5, 
        "--output", output_h5, 
        "--cuda", 
        "--total-droplets-included", TOTAL_DROPLETS, 
        "--fpr", "0.01", 
        "--epochs", EPOCHS_CB
    ]
    
    print(f"    ...Running CellBender (GPU)...")
    
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1) as proc:
        for line in proc.stdout: pass
    if proc.returncode != 0: raise RuntimeError(f"CellBender failed.")

def run_solo(adata):
    scvi.model.SCVI.setup_anndata(adata)
    vae = scvi.model.SCVI(adata)
    vae.train(enable_progress_bar=False)
    solo = scvi.external.SOLO.from_scvi_model(vae)
    solo.train(enable_progress_bar=False)
    adata.obs['doublet_score'] = solo.predict(soft=True)['doublet']
    adata.obs['is_doublet'] = solo.predict(soft=False) == 'doublet'
    return adata

# --- 5. MAIN EXECUTION ---

dataset_out_dir = os.path.join(OUTPUT_ROOT, TARGET_DATASET)
os.makedirs(dataset_out_dir, exist_ok=True)

search_pattern = os.path.join(DATASET_PATH, "*", "outs")
sample_dirs = glob.glob(search_pattern)

if not sample_dirs:
    print(f"No 'outs' folders found in {TARGET_DATASET}. Exiting.")
    sys.exit(0)

dataset_processed_files = []

for outs_dir in sample_dirs:
    sample_dir = os.path.dirname(outs_dir)
    sample_name = os.path.basename(sample_dir)
    
    raw_h5 = os.path.join(outs_dir, "raw_feature_bc_matrix.h5")
    filtered_h5 = os.path.join(outs_dir, "filtered_feature_bc_matrix.h5")
    final_obj_path = os.path.join(dataset_out_dir, f"{sample_name}_processed.h5ad")

    if os.path.exists(final_obj_path):
        print(f"  -> Skipping {sample_name} (Already Done)")
        dataset_processed_files.append(final_obj_path)
        continue

    print(f"  -> Processing Sample: {sample_name}")

    # A. Input Selection
    adata = None
    
    if PERFORM_AMBIENT:
        # --- PATH 1: CellBender on EVERYTHING ---
        print(f"     [Ambient Removal ON] -> Running CellBender")
        if not os.path.exists(raw_h5): 
            print(f"     [Warning] raw_feature_bc_matrix.h5 missing for {sample_name}. Skipping.")
            continue
        
        cb_temp_dir = os.path.join(dataset_out_dir, "cellbender_temp")
        os.makedirs(cb_temp_dir, exist_ok=True)
        cb_outfile = os.path.join(cb_temp_dir, f"{sample_name}_cb_out.h5")
        
        try:
            run_cellbender(raw_h5, cb_outfile)
            cb_filtered_out = cb_outfile.replace(".h5", "_filtered.h5")
            
            # Prefer the filtered output from CB
            if os.path.exists(cb_filtered_out):
                adata = sc.read_10x_h5(cb_filtered_out)
                adata.obs['preprocessing_stage'] = 'CellBender_Corrected'
            elif os.path.exists(cb_outfile):
                adata = sc.read_10x_h5(cb_outfile)
                adata.obs['preprocessing_stage'] = 'CellBender_Corrected'
            else: 
                print("     [Error] CellBender output not found.")
                continue
        except Exception as e:
            print(f"     [Error] CellBender failed: {e}")
            continue

    else:
        # --- PATH 2: Standard CellRanger on EVERYTHING ---
        if os.path.exists(filtered_h5):
            adata = sc.read_10x_h5(filtered_h5)
            adata.obs['preprocessing_stage'] = 'Standard_CellRanger'
        else: 
            print(f"     [Warning] filtered_feature_bc_matrix.h5 missing for {sample_name}. Skipping.")
            continue

    # B. QC & Solo
    adata.var_names_make_unique()
    adata.obs['sample_id'] = sample_name
    adata.obs['dataset_id'] = TARGET_DATASET
    adata.var['mt'] = adata.var_names.str.startswith('MT-')
    sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
    sc.pp.filter_cells(adata, min_genes=MIN_GENES)
    
    # .copy() fix
    adata = adata[adata.obs.pct_counts_mt < MAX_MITO, :].copy()
    
    if adata.n_obs < 50: continue

    try:
        adata = run_solo(adata)
        adata.write(final_obj_path)
        dataset_processed_files.append(final_obj_path)
    except Exception as e:
        print(f"     [Error] Solo failed: {e}")
        continue

# --- 6. CONCATENATE ---
print(f"  -> Concatenating {len(dataset_processed_files)} samples for {TARGET_DATASET}...")

if len(dataset_processed_files) > 0:
    adatas = [sc.read_h5ad(f) for f in dataset_processed_files]
    
    # --- FIX DUPLICATES ---
    print("--- Fixing duplicate gene names in the loaded data ---")
    fixed_count = 0

    for adata_sample in adatas:
        if not adata_sample.var_names.is_unique:
            adata_sample.var_names_make_unique()
            fixed_count += 1

    if fixed_count > 0:
        print(f"Fixed duplicates in {fixed_count} sample(s).")
    
    joint_adata = ad.concat(adatas, join='outer', label="batch_key", index_unique="-")
    
    dataset_merged_path = os.path.join(OUTPUT_ROOT, f"{TARGET_DATASET}_merged_raw.h5ad")
    joint_adata.write(dataset_merged_path)
    print(f"✅ DONE. Saved {TARGET_DATASET} to: {dataset_merged_path}")
else:
    print(f"⚠️ No samples processed for {TARGET_DATASET}.")