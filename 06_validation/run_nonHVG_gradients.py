#!/usr/bin/env python
"""Step 2: Non-HVG gradient analysis using existing OT commitment quartiles."""

import scanpy as sc
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.sparse import issparse
import gc

BASE     = Path("/storage/homefs/sv24v923/MPI_data/clean_pipeline")
ATLAS    = BASE / "macrophage_annotation" / "Macrophage_Atlas_FINAL_v3_CLEAN_ANNOTATED.h5ad"
GRAD_DIR = BASE / "ot_gradients_v3"
DEG_DIR  = BASE / "circularity_validation"
OUT      = BASE / "circularity_validation" / "gradients"
OUT.mkdir(parents=True, exist_ok=True)

CT_COL  = "cell_type_meta_v3"
N_PERM  = 1000

CT_MAP = {
    "Mono":   "Monocytes",
    "Scav":   "Scavenging / C1q+ Macrophages",
    "Res":    "Resident / Quiescent Macrophages",
    "Inflam": "Inflammatory Macrophages",
    "Foam":   "Lipid-Stressed / Foam Cells",
    "Fibro":  "Fibrotic / Hypoxic Macrophages",
}

TRANSITIONS = [
    ("Mono_to_Scav",    "Mono",   "Scav"),
    ("Mono_to_Res",     "Mono",   "Res"),
    ("Mono_to_Inflam",  "Mono",   "Inflam"),
    ("Mono_to_Foam",    "Mono",   "Foam"),
    ("Mono_to_Fibro",   "Mono",   "Fibro"),
    ("Scav_to_Inflam",  "Scav",   "Inflam"),
    ("Res_to_Inflam",   "Res",    "Inflam"),
    ("Res_to_Fibro",    "Res",    "Fibro"),
    ("Foam_to_Fibro",   "Foam",   "Fibro"),
    ("Inflam_to_Fibro", "Inflam", "Fibro"),
    ("Fibro_to_Scav",   "Fibro",  "Scav"),
]

# ── Load atlas ────────────────────────────────────────────────────
print("Loading atlas...")
adata = sc.read_h5ad(ATLAS)
safe_genes = list(adata.var_names[adata.var["highly_variable_nbatches"] == 0])
print(f"Atlas: {adata.shape}, Safe genes: {len(safe_genes)}")


def get_expr_safe(cell_idx, genes):
    """Extract expression for specific cells and genes from atlas."""
    gene_mask = adata.var_names.isin(genes)
    X = adata.X[cell_idx][:, gene_mask]
    if issparse(X):
        X = X.toarray()
    return pd.DataFrame(X, columns=adata.var_names[gene_mask])


def permutation_test(expr_series, quartiles, n_perm=1000):
    """Permutation test: shuffle quartile labels, recompute Q4/Q1 FC."""
    q_means = expr_series.groupby(quartiles).mean()
    q1_obs = float(q_means.get("Q1", 0))
    q4_obs = float(q_means.get("Q4", 0))
    fc_obs = np.log2((q4_obs + 0.1) / (q1_obs + 0.1))

    null_fcs = np.empty(n_perm)
    for i in range(n_perm):
        shuffled = np.random.permutation(quartiles.values)
        q_null = expr_series.groupby(shuffled).mean()
        q1_n = float(q_null.get("Q1", 0))
        q4_n = float(q_null.get("Q4", 0))
        null_fcs[i] = np.log2((q4_n + 0.1) / (q1_n + 0.1))

    p_val = (np.abs(null_fcs) >= np.abs(fc_obs)).mean()
    return max(p_val, 1.0 / n_perm)


# ── Main loop ─────────────────────────────────────────────────────
all_results = []

for key, src_g, tgt_g in TRANSITIONS:
    print(f"\n{'='*60}")
    print(f"{key}")

    # Load existing gradient CSV for quartile assignments
    grad_csv = GRAD_DIR / f"{key}.csv"
    if not grad_csv.exists():
        print(f"  SKIP — no gradient CSV")
        continue
    grad_df = pd.read_csv(grad_csv)
    quartiles = grad_df["ot_quartile_corrected"]
    n_cells = len(grad_df)
    print(f"  Source cells: {n_cells}")

    # Get the same source cell indices from atlas
    src_name = CT_MAP[src_g]
    src_idx = np.where(adata.obs[CT_COL].values == src_name)[0]

    if len(src_idx) != n_cells:
        print(f"  WARNING: atlas has {len(src_idx)} {src_g} cells, "
              f"gradient CSV has {n_cells}. Skipping.")
        continue

    # Load non-HVG DEGs for this transition
    tgt_deg_file = DEG_DIR / f"{key}_tgt_up_nonHVG_DEGs.csv"
    src_deg_file = DEG_DIR / f"{key}_src_up_nonHVG_DEGs.csv"

    genes_to_test = []
    gene_roles = {}

    if tgt_deg_file.exists():
        tgt_degs = pd.read_csv(tgt_deg_file)
        # Filter: named genes, top 20 by fold change
        tgt_named = tgt_degs[~tgt_degs["names"].str.startswith("ENSG")]
        tgt_top = tgt_named.head(20)["names"].tolist()
        for g in tgt_top:
            gene_roles[g] = "TGT"
        genes_to_test.extend(tgt_top)

    if src_deg_file.exists():
        src_degs = pd.read_csv(src_deg_file)
        src_named = src_degs[~src_degs["names"].str.startswith("ENSG")]
        src_top = src_named.head(20)["names"].tolist()
        for g in src_top:
            gene_roles[g] = "SRC"
        genes_to_test.extend(src_top)

    # Keep only genes actually in atlas
    genes_to_test = [g for g in genes_to_test if g in adata.var_names]
    # Keep only safe (non-HVG) genes — double check
    genes_to_test = [g for g in genes_to_test if g in safe_genes]

    print(f"  Testing {len(genes_to_test)} non-HVG genes "
          f"({sum(1 for g in genes_to_test if gene_roles[g]=='TGT')} tgt, "
          f"{sum(1 for g in genes_to_test if gene_roles[g]=='SRC')} src)")

    if not genes_to_test:
        print("  No genes to test — skipping")
        continue

    # Pull expression
    expr_df = get_expr_safe(src_idx, genes_to_test)

    # Compute gradients + permutation test per gene
    for gene in genes_to_test:
        if gene not in expr_df.columns:
            continue

        expr = expr_df[gene]
        q_means = expr.groupby(quartiles).mean()
        q1 = float(q_means.get("Q1", 0))
        q2 = float(q_means.get("Q2", 0))
        q3 = float(q_means.get("Q3", 0))
        q4 = float(q_means.get("Q4", 0))
        lfc = np.log2((q4 + 0.1) / (q1 + 0.1))

        # Monotonicity check
        vals = [q1, q2, q3, q4]
        diffs = np.diff(vals)
        if all(d > 0 for d in diffs):
            trend = "↑"
        elif all(d < 0 for d in diffs):
            trend = "↓"
        else:
            trend = "mixed"

        # Spearman correlation
        from scipy.stats import spearmanr
        rho, _ = spearmanr([1, 2, 3, 4], vals)

        # Permutation test
        p_val = permutation_test(expr, quartiles, N_PERM)

        role = gene_roles[gene]
        all_results.append({
            "transition": key,
            "gene": gene,
            "role": role,
            "Q1": round(q1, 4),
            "Q2": round(q2, 4),
            "Q3": round(q3, 4),
            "Q4": round(q4, 4),
            "log2FC": round(lfc, 4),
            "spearman_rho": round(rho, 4),
            "perm_p": p_val,
            "trend": trend,
        })

    del expr_df
    gc.collect()
    print(f"  Done — {len([r for r in all_results if r['transition']==key])} genes tested")

# ── Save ──────────────────────────────────────────────────────────
results_df = pd.DataFrame(all_results)
results_df.to_csv(OUT / "nonHVG_gradient_results.csv", index=False)

# Summary
print(f"\n{'='*60}")
print(f"SUMMARY")
print(f"Total genes tested: {len(results_df)}")
sig = results_df[results_df["perm_p"] <= 0.001]
print(f"Significant (p≤0.001): {len(sig)} ({100*len(sig)/len(results_df):.1f}%)")
mono = results_df[results_df["trend"].isin(["↑", "↓"])]
print(f"Monotonic trend: {len(mono)} ({100*len(mono)/len(results_df):.1f}%)")
sig_mono = results_df[(results_df["perm_p"] <= 0.001) & (results_df["trend"].isin(["↑", "↓"]))]
print(f"Significant + monotonic: {len(sig_mono)} ({100*len(sig_mono)/len(results_df):.1f}%)")

# Per-transition summary
print(f"\nPer transition:")
for key, grp in results_df.groupby("transition"):
    n = len(grp)
    n_sig = (grp["perm_p"] <= 0.001).sum()
    n_concordant = 0
    for _, row in grp.iterrows():
        if row["role"] == "TGT" and row["log2FC"] > 0 and row["perm_p"] <= 0.001:
            n_concordant += 1
        elif row["role"] == "SRC" and row["log2FC"] < 0 and row["perm_p"] <= 0.001:
            n_concordant += 1
    print(f"  {key:20s}  genes={n:3d}  sig={n_sig:3d}  concordant={n_concordant:3d}")

print(f"\nResults saved to {OUT / 'nonHVG_gradient_results.csv'}")