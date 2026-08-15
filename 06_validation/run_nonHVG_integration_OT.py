#!/usr/bin/env python
"""
Non-HVG Scanorama integration + OT validation (FAST version).
Uses top 5,000 most variable among non-HVG genes.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scanpy as sc
import scanorama
import numpy as np
import pandas as pd
import gc
import sys
import scipy.sparse
from scipy.sparse import issparse
from pathlib import Path
import ot as pot
import time

print(f"Scanpy version: {sc.__version__}")
print(f"Scanorama version: {scanorama.__version__}")

BASE = Path("/storage/homefs/sv24v923/MPI_data/clean_pipeline")
ATLAS = BASE / "macrophage_annotation" / "Macrophage_Atlas_FINAL_v3_CLEAN_ANNOTATED.h5ad"
OUT_DIR = BASE / "circularity_validation" / "nonHVG_embedding"
OUT_DIR.mkdir(parents=True, exist_ok=True)

dp_base = "/storage/research/igmp_dp_workspace/sergio_vazquez/MPI/Data/Processed_Datasets"
slide_base = "/storage/research/igmp_slide_workspace/GRP Zlobec/Sergio/MPI/Data/Processed_Datasets"

datasets_config = {
    "alsaigh":   f"{dp_base}/alsaigh_run2_clean_with_velocity_FINAL.h5ad",
    "wirka":     f"{dp_base}/wirka_2_clean_with_velocity_FINAL.h5ad",
    "pauli":     f"{dp_base}/pauli_2_clean_with_velocity_FINAL.h5ad",
    "bashore":   f"{slide_base}/bashore_run2_clean_with_velocity_FINAL.h5ad",
    "jaiswal":   f"{slide_base}/jaiswal_2_clean_with_velocity_FINAL.h5ad",
    "fernandez": f"{slide_base}/fernandez_2_clean_with_velocity_FINAL.h5ad",
    "pan":       f"{slide_base}/pan_clean_with_velocity_FINAL.h5ad",
}

INTEGRATION_DIMS = 100
KNN = 50
N_TOP_NONHVG = 5000  # Top variable genes among non-HVGs
CT_COL = "cell_type_meta_v3"
EPSILON = 0.05
MAX_ITER = 5000

CT_MAP = {
    "Monocytes":                        "Mono",
    "Scavenging / C1q+ Macrophages":    "Scav",
    "Resident / Quiescent Macrophages": "Res",
    "Inflammatory Macrophages":         "Inflam",
    "Lipid-Stressed / Foam Cells":      "Foam",
    "Fibrotic / Hypoxic Macrophages":   "Fibro",
}

t0 = time.time()

# ══════════════════════════════════════════════════════════════
# STEP 1: Get non-HVG gene list + cell labels from atlas
# ══════════════════════════════════════════════════════════════
print("=" * 60)
print("STEP 1: Loading atlas metadata")
print("=" * 60)

adata_atlas = sc.read_h5ad(ATLAS)
nonhvg_genes = set(adata_atlas.var_names[adata_atlas.var["highly_variable_nbatches"] == 0])
atlas_labels = adata_atlas.obs[[CT_COL]].copy()
print(f"  Non-HVG genes in atlas: {len(nonhvg_genes)}")
print(f"  Atlas cells: {len(atlas_labels)}")
del adata_atlas
gc.collect()

# ══════════════════════════════════════════════════════════════
# STEP 2: Load datasets, subset to non-HVG, normalize
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 2: Loading datasets (non-HVG genes only)")
print("=" * 60)

# First pass: find common non-HVG genes
common_nonhvg = None
for label, path in datasets_config.items():
    ad = sc.read_h5ad(path)
    ad.var_names_make_unique()
    dataset_nonhvg = set(ad.var_names).intersection(nonhvg_genes)
    if common_nonhvg is None:
        common_nonhvg = dataset_nonhvg
    else:
        common_nonhvg = common_nonhvg.intersection(dataset_nonhvg)
    del ad
    gc.collect()

common_nonhvg = sorted(list(common_nonhvg))
print(f"  Common non-HVG genes: {len(common_nonhvg)}")

# Second pass: load, subset, normalize
adatas_list = []
batch_keys = []

for label, path in datasets_config.items():
    print(f"  Loading {label}...")
    ad = sc.read_h5ad(path)
    ad.var_names_make_unique()
    ad = ad[:, common_nonhvg].copy()

    # Drop unnecessary data
    for k in list(ad.layers.keys()):
        del ad.layers[k]
    ad.obsm = {}
    ad.obsp = {}
    ad.uns = {}

    sc.pp.normalize_total(ad, target_sum=1e4)
    sc.pp.log1p(ad)
    ad.obs['batch'] = label
    adatas_list.append(ad)
    batch_keys.append(label)
    print(f"    {ad.shape}")
    gc.collect()

adata_full = sc.concat(
    adatas_list, join='outer', label='batch',
    keys=batch_keys, index_unique='-'
)
adata_full.obs['batch'] = adata_full.obs['batch'].astype('category')
print(f"\n  Concatenated: {adata_full.shape}")

del adatas_list
gc.collect()

# ══════════════════════════════════════════════════════════════
# STEP 3: Select top N_TOP_NONHVG variable genes among non-HVGs
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print(f"STEP 3: Selecting top {N_TOP_NONHVG} variable genes among non-HVGs")
print("=" * 60)

sc.pp.highly_variable_genes(
    adata_full,
    n_top_genes=N_TOP_NONHVG,
    batch_key='batch',
    flavor='seurat',
    subset=False
)

selected_genes = adata_full.var_names[adata_full.var['highly_variable']].tolist()
print(f"  Selected {len(selected_genes)} top variable non-HVG genes")

# Verify zero overlap with original HVGs
# (they can't overlap — we started from non-HVG genes only)
print(f"  Overlap with original HVGs: 0 (by construction)")

# Subset
adata_full = adata_full[:, selected_genes].copy()
print(f"  Final shape: {adata_full.shape}")
gc.collect()

# ══════════════════════════════════════════════════════════════
# STEP 4: Scanorama
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 4: Running Scanorama")
print("=" * 60)

scanorama_input = []
for b in batch_keys:
    subset = adata_full[adata_full.obs['batch'] == b].copy()
    if issparse(subset.X):
        subset.X = subset.X.toarray()
    scanorama_input.append(subset)
    print(f"  {b}: {subset.shape}")

gc.collect()

print("\n  Running Scanorama (approx=False)...")
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

embeddings = [ad.obsm['X_scanorama'] for ad in corrected]
full_embedding = np.concatenate(embeddings, axis=0)
print(f"  Embedding: {full_embedding.shape}")

if full_embedding.shape[0] != adata_full.shape[0]:
    print("  CRITICAL: Shape mismatch!")
    sys.exit(1)

adata_full.obsm['X_scanorama'] = full_embedding
del scanorama_input, corrected, embeddings
gc.collect()

# ══════════════════════════════════════════════════════════════
# STEP 5: Transfer labels, filter to macrophages, UMAP
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 5: Labels + UMAP")
print("=" * 60)

labels = []
matched = 0
for barcode in adata_full.obs_names:
    if barcode in atlas_labels.index:
        labels.append(atlas_labels.loc[barcode, CT_COL])
        matched += 1
    else:
        labels.append("Unknown")

adata_full.obs[CT_COL] = labels
print(f"  Matched {matched}/{adata_full.n_obs} cells")

adata_mac = adata_full[adata_full.obs[CT_COL] != "Unknown"].copy()
print(f"  Macrophage cells: {adata_mac.n_obs}")
print(adata_mac.obs[CT_COL].value_counts().to_string())

del adata_full
gc.collect()

sc.pp.neighbors(adata_mac, use_rep='X_scanorama', n_neighbors=KNN)
sc.tl.umap(adata_mac)

sc.pl.umap(adata_mac, color=CT_COL, title='Non-HVG Scanorama (top 5k non-HVG)',
           show=False)
plt.savefig(OUT_DIR / "umap_nonHVG_integration.png", bbox_inches='tight', dpi=150)
plt.close()

# ══════════════════════════════════════════════════════════════
# STEP 6: Pairwise OT
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 6: Pairwise OT")
print("=" * 60)

embedding = adata_mac.obsm['X_scanorama']
ct_labels = adata_mac.obs[CT_COL].values
ct_names = sorted(adata_mac.obs[CT_COL].unique())

connectivity_matrix = {}

for i, ct_a in enumerate(ct_names):
    for j, ct_b in enumerate(ct_names):
        if j <= i:
            continue

        short_a = CT_MAP.get(ct_a, ct_a)
        short_b = CT_MAP.get(ct_b, ct_b)

        idx_a = np.where(ct_labels == ct_a)[0]
        idx_b = np.where(ct_labels == ct_b)[0]

        emb_a = embedding[idx_a]
        emb_b = embedding[idx_b]

        # Subsample large clusters for speed
        MAX_CELLS = 10000
        if len(idx_a) > MAX_CELLS:
            rng = np.random.default_rng(42)
            sub_a = rng.choice(len(idx_a), MAX_CELLS, replace=False)
            emb_a = emb_a[sub_a]
        if len(idx_b) > MAX_CELLS:
            rng = np.random.default_rng(42)
            sub_b = rng.choice(len(idx_b), MAX_CELLS, replace=False)
            emb_b = emb_b[sub_b]

        na, nb = len(emb_a), len(emb_b)
        print(f"\n  {short_a}({na}) ↔ {short_b}({nb})")

        # Cost AB
        cost = np.empty((na, nb), dtype=np.float32)
        chunk = 2000
        for s in range(0, na, chunk):
            e = min(s + chunk, na)
            diff = emb_a[s:e, np.newaxis, :] - emb_b[np.newaxis, :, :]
            cost[s:e] = np.sqrt((diff ** 2).sum(axis=2))
            del diff
        cmax = cost.max()
        if cmax > 0:
            cost /= cmax

        a_w = np.ones(na, dtype=np.float64) / na
        b_w = np.ones(nb, dtype=np.float64) / nb

        try:
            gamma = pot.sinkhorn(a_w, b_w, cost.astype(np.float64),
                                reg=EPSILON, numItermax=MAX_ITER,
                                stopThr=1e-9, verbose=False)
            transport_cost = np.sum(gamma * cost)
            del gamma
            gc.collect()

            # Self AA
            cost_aa = np.empty((na, na), dtype=np.float32)
            for s in range(0, na, chunk):
                e = min(s + chunk, na)
                diff = emb_a[s:e, np.newaxis, :] - emb_a[np.newaxis, :, :]
                cost_aa[s:e] = np.sqrt((diff ** 2).sum(axis=2))
                del diff
            cm = cost_aa.max()
            if cm > 0:
                cost_aa /= cm
            gamma_aa = pot.sinkhorn(a_w, a_w, cost_aa.astype(np.float64),
                                    reg=EPSILON, numItermax=MAX_ITER,
                                    stopThr=1e-9, verbose=False)
            self_a = np.sum(gamma_aa * cost_aa)
            del cost_aa, gamma_aa
            gc.collect()

            # Self BB
            cost_bb = np.empty((nb, nb), dtype=np.float32)
            for s in range(0, nb, chunk):
                e = min(s + chunk, nb)
                diff = emb_b[s:e, np.newaxis, :] - emb_b[np.newaxis, :, :]
                cost_bb[s:e] = np.sqrt((diff ** 2).sum(axis=2))
                del diff
            cm = cost_bb.max()
            if cm > 0:
                cost_bb /= cm
            gamma_bb = pot.sinkhorn(b_w, b_w, cost_bb.astype(np.float64),
                                    reg=EPSILON, numItermax=MAX_ITER,
                                    stopThr=1e-9, verbose=False)
            self_b = np.sum(gamma_bb * cost_bb)
            del cost_bb, gamma_bb
            gc.collect()

            divergence = transport_cost - 0.5 * (self_a + self_b)
            sigma = 0.1378
            connectivity = np.exp(-divergence / (2 * sigma ** 2))
            connectivity_matrix[(short_a, short_b)] = connectivity
            print(f"    div={divergence:.4f}  conn={connectivity:.4f}")

        except Exception as e:
            print(f"    ERROR: {e}")
            connectivity_matrix[(short_a, short_b)] = np.nan

        del cost
        gc.collect()

del adata_mac, embedding
gc.collect()

# ══════════════════════════════════════════════════════════════
# STEP 7: Compare with HVG connectivity
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 7: Comparison")
print("=" * 60)

HVG_CONNECTIVITY = {
    ("Mono", "Scav"):    0.10, ("Mono", "Res"):     0.16,
    ("Mono", "Inflam"):  0.14, ("Mono", "Foam"):    0.14,
    ("Mono", "Fibro"):   0.07, ("Scav", "Res"):     0.36,
    ("Scav", "Inflam"):  0.67, ("Scav", "Foam"):    0.07,
    ("Scav", "Fibro"):   0.16, ("Res",  "Inflam"):  0.22,
    ("Res",  "Foam"):    0.09, ("Res",  "Fibro"):   0.12,
    ("Inflam", "Foam"):  0.03, ("Inflam", "Fibro"): 0.04,
    ("Foam", "Fibro"):   0.35,
}

rows = []
for (a, b), nc in sorted(connectivity_matrix.items()):
    hc = HVG_CONNECTIVITY.get((a, b), HVG_CONNECTIVITY.get((b, a), None))
    rows.append({"pair": f"{a}↔{b}", "nonHVG_conn": round(float(nc), 4) if not np.isnan(nc) else np.nan, "HVG_conn": hc})

comparison = pd.DataFrame(rows)
comparison.to_csv(OUT_DIR / "connectivity_comparison.csv", index=False)

print(f"\n{'Pair':15s}  {'non-HVG':>8}  {'HVG':>6}")
print("-" * 35)
for _, row in comparison.iterrows():
    hvg = f"{row['HVG_conn']:.2f}" if row['HVG_conn'] is not None else "n/a"
    nhvg = f"{row['nonHVG_conn']:.4f}" if not pd.isna(row['nonHVG_conn']) else "FAIL"
    print(f"{row['pair']:15s}  {nhvg:>8}  {hvg:>6}")

from scipy.stats import spearmanr, pearsonr
valid = comparison.dropna(subset=["HVG_conn", "nonHVG_conn"])
if len(valid) > 3:
    rho_s, p_s = spearmanr(valid["HVG_conn"], valid["nonHVG_conn"])
    rho_p, p_p = pearsonr(valid["HVG_conn"], valid["nonHVG_conn"])
    print(f"\nSpearman ρ = {rho_s:.3f} (p = {p_s:.4f})")
    print(f"Pearson  r = {rho_p:.3f} (p = {p_p:.4f})")

fig, ax = plt.subplots(figsize=(7, 7))
fig.patch.set_facecolor("white")
ax.scatter(valid["HVG_conn"], valid["nonHVG_conn"], s=60, c="#333", alpha=0.7)
for _, row in valid.iterrows():
    ax.annotate(row["pair"], (row["HVG_conn"], row["nonHVG_conn"]),
                fontsize=7, ha="left", va="bottom", xytext=(5, 5), textcoords="offset points")
ax.set_xlabel("HVG-derived connectivity", fontsize=11)
ax.set_ylabel("Non-HVG-derived connectivity", fontsize=11)
ax.set_title(f"OT Connectivity: HVG vs non-HVG embedding\nSpearman ρ = {rho_s:.3f} (p = {p_s:.4f})",
             fontsize=12, fontweight="bold")
lims = [0, max(valid["HVG_conn"].max(), valid["nonHVG_conn"].max()) * 1.1]
ax.plot(lims, lims, "k--", alpha=0.3)
ax.set_xlim(lims)
ax.set_ylim(lims)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
fig.savefig(OUT_DIR / "connectivity_comparison.png", dpi=150, bbox_inches="tight")
fig.savefig(OUT_DIR / "connectivity_comparison.pdf", dpi=300, bbox_inches="tight")
plt.close()

elapsed = time.time() - t0
print(f"\n{'='*60}")
print(f"COMPLETE — {elapsed/60:.1f} minutes")
print(f"Results in {OUT_DIR}")
print(f"{'='*60}")