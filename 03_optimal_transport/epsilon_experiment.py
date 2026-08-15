#!/usr/bin/env python
"""
Epsilon sensitivity analysis.
Compare Sinkhorn divergence rankings across epsilon values.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scanpy as sc
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr
from itertools import combinations_with_replacement
import ot
import gc
import sys
import time

BASE = Path("/storage/homefs/sv24v923/MPI_data/clean_pipeline")
ATLAS = BASE / "macrophage_annotation" / "Macrophage_Atlas_FINAL_v3_CLEAN_ANNOTATED.h5ad"
OUT = BASE / "circularity_validation" / "epsilon_sensitivity"
OUT.mkdir(parents=True, exist_ok=True)
CT_COL = "cell_type_meta_v3"

EPSILONS = [0.005, 0.01, 0.025, 0.05, 0.1, 0.5]
MAX_CELLS = 5000
SEED = 42

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
# STEP 1: Load atlas
# ══════════════════════════════════════════════════════════════
print("=" * 60)
print("STEP 1: Loading atlas")
print("=" * 60)
sys.stdout.flush()

adata = sc.read_h5ad(ATLAS)
embedding = adata.obsm["X_scanorama"]
ct_labels = adata.obs[CT_COL].values

cell_types = []
emb_dict = {}
for ct_full, ct_short in CT_MAP.items():
    mask = ct_labels == ct_full
    emb_dict[ct_short] = embedding[mask]
    cell_types.append(ct_short)
    print(f"  {ct_short}: {mask.sum()} cells")
sys.stdout.flush()

del adata
gc.collect()

# ══════════════════════════════════════════════════════════════
# STEP 2: OT functions — same as ot_pipeline_v8.py
# ══════════════════════════════════════════════════════════════

def subsample(X, max_n, rng):
    if len(X) <= max_n:
        return X
    return X[rng.choice(len(X), max_n, replace=False)]

def sinkhorn_cost(X_src, X_tgt, epsilon):
    a = np.full(len(X_src), 1.0 / len(X_src), dtype=np.float64)
    b = np.full(len(X_tgt), 1.0 / len(X_tgt), dtype=np.float64)
    C = cdist(X_src, X_tgt, metric="euclidean").astype(np.float64)
    scale = C.max()
    if scale < 1e-9:
        return 0.0
    C /= scale
    T = ot.sinkhorn(a, b, C, reg=epsilon, numItermax=5000,
                    stopThr=1e-9, warn=False)
    return float(np.sum(T * C) * scale)

def to_sinkhorn_divergence(cost_df):
    W = cost_df.to_numpy(dtype=np.float64)
    d = np.diag(W)
    S = W - 0.5 * d[:, None] - 0.5 * d[None, :]
    S[S < 0] = 0.0
    np.fill_diagonal(S, 0.0)
    return pd.DataFrame(S, index=cost_df.index, columns=cost_df.columns)

# ══════════════════════════════════════════════════════════════
# STEP 3: Run OT at each epsilon
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print(f"STEP 2: Running OT at {len(EPSILONS)} epsilon values")
print(f"  Max cells = {MAX_CELLS}, Seed = {SEED}")
print("=" * 60)
sys.stdout.flush()

all_div = {}

for eps in EPSILONS:
    print(f"\n{'─'*50}")
    print(f"ε = {eps}")
    print(f"{'─'*50}")
    sys.stdout.flush()
    t1 = time.time()

    rng = np.random.default_rng(SEED)
    pairs = list(combinations_with_replacement(cell_types, 2))
    n = len(cell_types)
    cost_mat = pd.DataFrame(np.zeros((n, n)), index=cell_types, columns=cell_types)

    for ci, cj in pairs:
        Xi = subsample(emb_dict[ci], MAX_CELLS, rng)
        Xj = subsample(emb_dict[cj], MAX_CELLS, rng)
        cost = sinkhorn_cost(Xi, Xj, eps)
        cost_mat.loc[ci, cj] = cost
        cost_mat.loc[cj, ci] = cost
        if ci != cj:
            print(f"  {ci:6s} ↔ {cj:6s}  cost={cost:.4f}")
            sys.stdout.flush()

    div_mat = to_sinkhorn_divergence(cost_mat)

    # Store divergences only — no connectivity conversion
    div_flat = {}
    for ci in cell_types:
        for cj in cell_types:
            if ci < cj:
                div_flat[f"{ci}↔{cj}"] = div_mat.loc[ci, cj]

    all_div[eps] = div_flat

    del cost_mat, div_mat
    gc.collect()

    ranked = sorted(div_flat.items(), key=lambda x: x[1])
    print(f"\n  Closest 3: {ranked[:3]}")
    print(f"  Farthest 3: {ranked[-3:]}")
    print(f"  Done in {(time.time()-t1)/60:.1f} min")
    sys.stdout.flush()

# ══════════════════════════════════════════════════════════════
# STEP 4: Compare divergence rankings
# ══════════════════════════════════════════════════════════════
print("\n\n" + "=" * 60)
print("STEP 3: Divergence rank comparison")
print("=" * 60)
sys.stdout.flush()

pairs_list = sorted(all_div[EPSILONS[0]].keys())

div_df = pd.DataFrame({"pair": pairs_list})
for eps in EPSILONS:
    div_df[f"eps_{eps}"] = [all_div[eps][p] for p in pairs_list]
div_df.to_csv(OUT / "epsilon_divergence_comparison.csv", index=False)

# Print table
print(f"\nSinkhorn divergence values:")
print(f"{'Pair':15s}", end="")
for eps in EPSILONS:
    print(f"  ε={eps:>6}", end="")
print()
print("-" * (15 + 9 * len(EPSILONS)))
for _, row in div_df.iterrows():
    print(f"{row['pair']:15s}", end="")
    for eps in EPSILONS:
        print(f"  {row[f'eps_{eps}']:7.4f}", end="")
    print()

# Pairwise Spearman on divergences
print(f"\n\nSpearman ρ between epsilon choices (on divergences):")
print(f"{'':>10}", end="")
for eps in EPSILONS:
    print(f"  ε={eps:>6}", end="")
print()
for eps1 in EPSILONS:
    print(f"ε={eps1:>6}  ", end="")
    for eps2 in EPSILONS:
        v1 = div_df[f"eps_{eps1}"].values
        v2 = div_df[f"eps_{eps2}"].values
        rho, _ = spearmanr(v1, v2)
        print(f"  {rho:7.3f}", end="")
    print()

# Rank stability
print(f"\nRank stability (closest = most connected):")
for eps in EPSILONS:
    ranked = div_df.sort_values(f"eps_{eps}")["pair"].tolist()
    print(f"  ε={eps:>6}  Closest: {ranked[:3]}  Farthest: {ranked[-3:]}")

# ══════════════════════════════════════════════════════════════
# STEP 5: Plot
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 4: Plotting")
print("=" * 60)
sys.stdout.flush()

fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(15, 6))
fig.patch.set_facecolor("white")

# Panel A: Divergence values across epsilon
x_pos = np.arange(len(EPSILONS))
for pair in pairs_list:
    vals = [all_div[eps][pair] for eps in EPSILONS]
    ax_a.plot(x_pos, vals, "o-", lw=1.5, markersize=5, alpha=0.6, label=pair)

ax_a.axvline(1, color="#E15759", lw=2, ls="--", alpha=0.5, label="Selected (ε=0.01)")
ax_a.set_xticks(x_pos)
ax_a.set_xticklabels([str(e) for e in EPSILONS], fontsize=9)
ax_a.set_xlabel("ε (regularization parameter)", fontsize=11)
ax_a.set_ylabel("Sinkhorn divergence", fontsize=11)
ax_a.set_title("A   Sinkhorn divergence across ε\n"
               "    Each line = one cluster pair",
               fontsize=11, fontweight="bold", loc="left")
ax_a.legend(fontsize=5.5, ncol=3, loc="upper left")
ax_a.spines["top"].set_visible(False)
ax_a.spines["right"].set_visible(False)

# Panel B: Spearman with reference (ε=0.01)
ref_col = "eps_0.01"
rho_vals = []
for eps in EPSILONS:
    v_ref = div_df[ref_col].values
    v_eps = div_df[f"eps_{eps}"].values
    rho, _ = spearmanr(v_ref, v_eps)
    rho_vals.append(rho)

colors = ["#E15759" if eps == 0.01 else "#4E79A7" for eps in EPSILONS]
ax_b.bar(x_pos, rho_vals, color=colors, alpha=0.85, edgecolor="white")

for i, (rho, eps) in enumerate(zip(rho_vals, EPSILONS)):
    ax_b.text(i, rho + 0.01, f"{rho:.3f}", ha="center", fontsize=9, fontweight="bold")

ax_b.axhline(0.95, color="#2ca02c", ls=":", lw=1, alpha=0.5)
ax_b.text(len(EPSILONS) - 0.3, 0.955, "ρ = 0.95", fontsize=8, color="#2ca02c")

ax_b.set_xticks(x_pos)
ax_b.set_xticklabels([str(e) for e in EPSILONS], fontsize=9)
ax_b.set_xlabel("ε (regularization parameter)", fontsize=11)
ax_b.set_ylabel("Spearman ρ vs ε = 0.01 (selected)", fontsize=11)
ax_b.set_title("B   Rank stability of Sinkhorn divergence\n"
               "    Rankings preserved across ε choices",
               fontsize=11, fontweight="bold", loc="left")
ax_b.set_ylim(0, 1.08)
ax_b.spines["top"].set_visible(False)
ax_b.spines["right"].set_visible(False)

fig.tight_layout()
fig.savefig(OUT / "SuppFig_epsilon_sensitivity.png", dpi=150, bbox_inches="tight")
fig.savefig(OUT / "SuppFig_epsilon_sensitivity.pdf", dpi=300, bbox_inches="tight")
plt.close()

elapsed = time.time() - t0
print(f"\nTotal runtime: {elapsed/60:.1f} min")
print(f"Results saved to {OUT}")
sys.stdout.flush()