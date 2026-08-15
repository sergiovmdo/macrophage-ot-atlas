#!/usr/bin/env python
"""Non-HVG SRC gradient analysis — ALL filtered genes, vectorized."""

import scanpy as sc
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.sparse import issparse
import gc
import time

BASE     = Path("/storage/homefs/sv24v923/MPI_data/clean_pipeline")
ATLAS    = BASE / "macrophage_annotation" / "Macrophage_Atlas_FINAL_v3_CLEAN_ANNOTATED.h5ad"
GRAD_DIR = BASE / "ot_gradients_v3"
DEG_DIR  = BASE / "circularity_validation"
OUT      = BASE / "circularity_validation" / "gradients"
OUT.mkdir(parents=True, exist_ok=True)

CT_COL = "cell_type_meta_v3"
N_PERM = 1000
EXCLUDE_PREFIXES = ("RPL", "RPS", "MT-", "LINC", "MIR", "ENSG")

CT_MAP = {
    "Mono":   "Monocytes",
    "Scav":   "Scavenging / C1q+ Macrophages",
    "Res":    "Resident / Quiescent Macrophages",
    "Inflam": "Inflammatory Macrophages",
    "Foam":   "Lipid-Stressed / Foam Cells",
    "Fibro":  "Fibrotic / Hypoxic Macrophages",
}

TRANSITIONS = [
    ("Mono_to_Scav",    "Mono"),
    ("Mono_to_Res",     "Mono"),
    ("Mono_to_Inflam",  "Mono"),
    ("Mono_to_Foam",    "Mono"),
    ("Mono_to_Fibro",   "Mono"),
    ("Scav_to_Inflam",  "Scav"),
    ("Res_to_Inflam",   "Res"),
    ("Res_to_Fibro",    "Res"),
    ("Foam_to_Fibro",   "Foam"),
    ("Inflam_to_Fibro", "Inflam"),
    ("Fibro_to_Scav",   "Fibro"),
]

print("Loading atlas...")
t0 = time.time()
adata = sc.read_h5ad(ATLAS)
safe_set = set(adata.var_names[adata.var["highly_variable_nbatches"] == 0])
print(f"Atlas: {adata.shape}, Safe genes: {len(safe_set)}, loaded in {time.time()-t0:.0f}s")

all_results = []

for key, src_g in TRANSITIONS:
    t_start = time.time()
    print(f"\n{'='*60}")
    print(f"{key}")

    grad_csv = GRAD_DIR / f"{key}.csv"
    if not grad_csv.exists():
        print(f"  SKIP — no gradient CSV")
        continue
    grad_df = pd.read_csv(grad_csv)
    quartiles = grad_df["ot_quartile_corrected"].values
    n_cells = len(grad_df)

    src_name = CT_MAP[src_g]
    src_idx = np.where(adata.obs[CT_COL].values == src_name)[0]
    if len(src_idx) != n_cells:
        print(f"  WARNING: mismatch {len(src_idx)} vs {n_cells}. Skipping.")
        continue

    # Precompute quartile masks
    q_masks = {q: quartiles == q for q in ["Q1", "Q2", "Q3", "Q4"]}

    src_file = DEG_DIR / f"{key}_src_up_nonHVG_DEGs.csv"
    if not src_file.exists():
        print(f"  No SRC DEG file")
        continue

    src_degs = pd.read_csv(src_file)
    src_named = src_degs[~src_degs["names"].str.startswith(EXCLUDE_PREFIXES)]
    src_filt = src_named[src_named["scores"] < -5].copy()
    genes = [g for g in src_filt["names"].tolist()
             if g in adata.var_names and g in safe_set]

    print(f"  SRC genes to test: {len(genes)}")
    if not genes:
        print("  No genes — skipping")
        continue

    CHUNK = 500
    for chunk_start in range(0, len(genes), CHUNK):
        chunk_genes = genes[chunk_start:chunk_start + CHUNK]
        gene_mask = adata.var_names.isin(chunk_genes)
        X = adata.X[src_idx][:, gene_mask]
        if issparse(X):
            X = X.toarray()
        col_names = list(adata.var_names[gene_mask])

        for gene in chunk_genes:
            if gene not in col_names:
                continue
            col_idx = col_names.index(gene)
            e = X[:, col_idx]

            q1 = e[q_masks["Q1"]].mean()
            q2 = e[q_masks["Q2"]].mean()
            q3 = e[q_masks["Q3"]].mean()
            q4 = e[q_masks["Q4"]].mean()
            lfc = np.log2((q4 + 0.1) / (q1 + 0.1))

            vals = [q1, q2, q3, q4]
            diffs = np.diff(vals)
            if all(d > 0 for d in diffs):
                trend = "up"
            elif all(d < 0 for d in diffs):
                trend = "down"
            else:
                trend = "mixed"

            # Vectorized permutation test
            fc_obs = abs(lfc)
            count = 0
            for _ in range(N_PERM):
                shuf = np.random.permutation(e)
                null_q1 = shuf[q_masks["Q1"]].mean()
                null_q4 = shuf[q_masks["Q4"]].mean()
                fc_null = abs(np.log2((null_q4 + 0.1) / (null_q1 + 0.1)))
                if fc_null >= fc_obs:
                    count += 1
            p = max(count / N_PERM, 1.0 / N_PERM)

            all_results.append({
                "transition": key, "gene": gene, "role": "SRC",
                "Q1": round(float(q1), 4), "Q2": round(float(q2), 4),
                "Q3": round(float(q3), 4), "Q4": round(float(q4), 4),
                "log2FC": round(lfc, 4), "trend": trend, "perm_p": p,
            })

        del X
        gc.collect()
        done = min(chunk_start + CHUNK, len(genes))
        elapsed = time.time() - t_start
        rate = done / elapsed if elapsed > 0 else 0
        eta = (len(genes) - done) / rate if rate > 0 else 0
        print(f"    {done}/{len(genes)} done  [{elapsed:.0f}s elapsed, ~{eta:.0f}s remaining]")

    print(f"  {key} complete in {time.time()-t_start:.0f}s")

results_df = pd.DataFrame(all_results)
results_df.to_csv(OUT / "nonHVG_SRC_allgenes_results.csv", index=False)

# Summary
print(f"\n{'='*60}")
print("SUMMARY — SRC GENES ONLY (all filtered, no cap)")
print(f"{'='*60}")
print(f"{'Transition':20s}  {'Tested':>6}  {'Sig':>5}  {'Down':>5}  {'Sig%':>5}  {'Dn%':>5}")
print("-" * 55)

for key, grp in results_df.groupby("transition"):
    n = len(grp)
    n_sig = (grp["perm_p"] <= 0.001).sum()
    n_down = len(grp[(grp["perm_p"] <= 0.001) & (grp["log2FC"] < 0)])
    print(f"{key:20s}  {n:6d}  {n_sig:5d}  {n_down:5d}  {100*n_sig/n:.0f}%  {100*n_down/n_sig:.0f}%" if n_sig > 0 else f"{key:20s}  {n:6d}  {n_sig:5d}  {n_down:5d}  {100*n_sig/n:.0f}%  n/a")

total = len(results_df)
total_sig = (results_df["perm_p"] <= 0.001).sum()
total_down = len(results_df[(results_df["perm_p"] <= 0.001) & (results_df["log2FC"] < 0)])
expected = total * 0.001
print(f"\n{'OVERALL':20s}  {total:6d}  {total_sig:5d}  {total_down:5d}  {100*total_sig/total:.0f}%  {100*total_down/total_sig:.0f}%")
print(f"Expected by chance: {expected:.1f}")
print(f"Enrichment: {total_sig/expected:.0f}-fold")
print(f"\nTotal runtime: {time.time()-t0:.0f}s")
print(f"Saved to {OUT / 'nonHVG_SRC_allgenes_results.csv'}")