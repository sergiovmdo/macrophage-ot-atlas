#!/usr/bin/env python
"""
Velocity directionality via bidirectional asymmetry test.
For each pair (A,B), compute cosine(A→B) and cosine(B→A).
Asymmetry = |cosine(A→B) − cosine(B→A)| measures directionality.
"""

import scanpy as sc
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.neighbors import NearestNeighbors
import sys
import time

BASE = Path("/storage/homefs/sv24v923/MPI_data/clean_pipeline")
V2 = BASE / "unitvelo_results_META_VALIDATION_V2"
V3 = BASE / "unitvelo_results_META_VALIDATION_V3"
OUT = BASE / "velocity_asymmetry_test"
OUT.mkdir(parents=True, exist_ok=True)

K_INTERFACE = 20

CT_FULL_TO_SHORT = {
    "Monocytes":                        "Mono",
    "Scavenging / C1q+ Macrophages":    "Scav",
    "Resident / Quiescent Macrophages": "Res",
    "Inflammatory Macrophages":         "Inflam",
    "Lipid-Stressed / Foam Cells":      "Foam",
    "Fibrotic / Hypoxic Macrophages":   "Fibro",
}

def filename_for(src, tgt):
    def clean(s):
        return s.replace("/", "_").replace(" ", "_").replace("C1q+", "C1qp")
    return f"unitvelo_pairwise_{clean(src)}_to_{clean(tgt)}.h5ad"

# All 15 transitions (using forward direction file)
TRANSITIONS = [
    ("Monocytes",                        "Inflammatory Macrophages",         V3),
    ("Monocytes",                        "Scavenging / C1q+ Macrophages",    V3),
    ("Monocytes",                        "Lipid-Stressed / Foam Cells",      V3),
    ("Monocytes",                        "Resident / Quiescent Macrophages", V3),
    ("Monocytes",                        "Fibrotic / Hypoxic Macrophages",   V3),
    ("Scavenging / C1q+ Macrophages",    "Inflammatory Macrophages",         V2),
    ("Resident / Quiescent Macrophages", "Inflammatory Macrophages",         V2),
    ("Lipid-Stressed / Foam Cells",      "Fibrotic / Hypoxic Macrophages",   V2),
    ("Inflammatory Macrophages",         "Fibrotic / Hypoxic Macrophages",   V2),
    ("Resident / Quiescent Macrophages", "Fibrotic / Hypoxic Macrophages",   V2),
    ("Scavenging / C1q+ Macrophages",    "Fibrotic / Hypoxic Macrophages",   V2),
    ("Scavenging / C1q+ Macrophages",    "Resident / Quiescent Macrophages", V2),
    ("Resident / Quiescent Macrophages", "Lipid-Stressed / Foam Cells",      V2),
    ("Inflammatory Macrophages",         "Lipid-Stressed / Foam Cells",      V2),
    ("Scavenging / C1q+ Macrophages",    "Lipid-Stressed / Foam Cells",      V2),
]


def get_ct_column(adata):
    for c in ["cell_type_meta_v3", "cell_type_meta_v2", "cell_type"]:
        if c in adata.obs.columns:
            return c
    return None


def compute_cosine(adata, src_full, tgt_full):
    """Compute mean cosine of source-cluster velocity at interface vs src→tgt axis."""
    ct_col = get_ct_column(adata)
    if ct_col is None:
        return None

    if "X_umap" not in adata.obsm or "velocity_umap" not in adata.obsm:
        return None

    src_mask = (adata.obs[ct_col] == src_full).values
    tgt_mask = (adata.obs[ct_col] == tgt_full).values

    if src_mask.sum() == 0 or tgt_mask.sum() == 0:
        return None

    umap_coords = adata.obsm["X_umap"]
    velocity_2d = adata.obsm["velocity_umap"]

    src_idx = np.where(src_mask)[0]
    tgt_idx = np.where(tgt_mask)[0]

    # Interface cells: source cells with target neighbors in k-NN
    nn = NearestNeighbors(n_neighbors=K_INTERFACE + 1).fit(umap_coords)
    _, src_neighbors = nn.kneighbors(umap_coords[src_idx])
    src_neighbors = src_neighbors[:, 1:]

    is_target = np.zeros(len(adata), dtype=bool)
    is_target[tgt_idx] = True
    interface_mask = is_target[src_neighbors].any(axis=1)
    interface_src_idx = src_idx[interface_mask]

    n_interface = len(interface_src_idx)
    if n_interface < 10:
        return {"n_interface": n_interface, "cosine": np.nan, "pct_pos": np.nan}

    # Source-to-target axis
    src_centroid = umap_coords[src_idx].mean(axis=0)
    tgt_centroid = umap_coords[tgt_idx].mean(axis=0)
    src_to_tgt = tgt_centroid - src_centroid

    # Per-cell cosines
    interface_velocity = velocity_2d[interface_src_idx]
    vel_norm = np.linalg.norm(interface_velocity, axis=1)
    axis_norm = np.linalg.norm(src_to_tgt)
    valid = (vel_norm > 1e-9) & (axis_norm > 1e-9)

    if valid.sum() == 0:
        return {"n_interface": n_interface, "cosine": np.nan, "pct_pos": np.nan}

    cosines = (interface_velocity[valid] @ src_to_tgt) / (vel_norm[valid] * axis_norm)

    return {
        "n_interface": n_interface,
        "n_valid": int(valid.sum()),
        "cosine": float(cosines.mean()),
        "pct_pos": float((cosines > 0).sum() / len(cosines) * 100),
    }


# ═══════════════════════════════════════════════════════════════
# RUN
# ═══════════════════════════════════════════════════════════════
print("=" * 70)
print("Velocity directionality — bidirectional asymmetry test")
print("=" * 70)
sys.stdout.flush()

t0 = time.time()
results = []

for src_full, tgt_full, data_dir in TRANSITIONS:
    src_short = CT_FULL_TO_SHORT.get(src_full, src_full)
    tgt_short = CT_FULL_TO_SHORT.get(tgt_full, tgt_full)
    label = f"{src_short}↔{tgt_short}"

    print(f"\n{'─'*60}\n{label}\n{'─'*60}")
    sys.stdout.flush()
    t1 = time.time()

    fpath = data_dir / filename_for(src_full, tgt_full)
    if not fpath.exists():
        print(f"  FILE NOT FOUND: {fpath.name}")
        continue

    adata = sc.read_h5ad(fpath)
    print(f"  Loaded: {adata.shape}")
    sys.stdout.flush()

    # Forward: src → tgt
    fwd = compute_cosine(adata, src_full, tgt_full)
    # Reverse: tgt → src (same data, flip the axis)
    rev = compute_cosine(adata, tgt_full, src_full)

    if fwd is None or rev is None:
        print(f"  SKIPPED")
        del adata
        continue

    asymmetry = fwd["cosine"] - rev["cosine"]

    if asymmetry > 0.5:
        direction = f"{src_short}→{tgt_short}"
        classification = "Directed"
    elif asymmetry < -0.5:
        direction = f"{tgt_short}→{src_short}"
        classification = "Reverse-directed"
    else:
        direction = "—"
        classification = "Undirected"

    res = {
        "pair": label,
        "src": src_short,
        "tgt": tgt_short,
        "n_interface_fwd": fwd["n_interface"],
        "n_interface_rev": rev["n_interface"],
        "cosine_fwd": fwd["cosine"],
        "cosine_rev": rev["cosine"],
        "pct_pos_fwd": fwd["pct_pos"],
        "pct_pos_rev": rev["pct_pos"],
        "asymmetry": asymmetry,
        "abs_asymmetry": abs(asymmetry),
        "inferred_direction": direction,
        "classification": classification,
    }
    results.append(res)

    print(f"  Forward  ({src_short}→{tgt_short}): cos={fwd['cosine']:+.3f}  %pos={fwd['pct_pos']:.1f}  n={fwd['n_interface']}")
    print(f"  Reverse  ({tgt_short}→{src_short}): cos={rev['cosine']:+.3f}  %pos={rev['pct_pos']:.1f}  n={rev['n_interface']}")
    print(f"  Asymmetry: {asymmetry:+.3f}")
    print(f"  Classification: {classification}  →  {direction}")
    print(f"  Done in {(time.time()-t1)/60:.1f} min")
    sys.stdout.flush()

    del adata


# ═══════════════════════════════════════════════════════════════
# RESULTS
# ═══════════════════════════════════════════════════════════════
df = pd.DataFrame(results)
df = df.sort_values("abs_asymmetry", ascending=False)
df = df.round(4)

df.to_csv(OUT / "velocity_asymmetry_results.csv", index=False)

print(f"\n\n{'='*90}")
print("FINAL RESULTS — sorted by |asymmetry|")
print(f"{'='*90}")
print(df.to_string(index=False))

print(f"\n\nSummary:")
print(df['classification'].value_counts().to_string())

# Threshold suggestion
sorted_asym = df["abs_asymmetry"].values
print(f"\n|Asymmetry| distribution:")
print(f"  Range: {sorted_asym.min():.3f} to {sorted_asym.max():.3f}")
print(f"  Sorted: {sorted_asym}")

# Look for natural gap in the data
gaps = np.diff(sorted_asym)
biggest_gap_idx = np.argmax(gaps)
gap_threshold = (sorted_asym[biggest_gap_idx] + sorted_asym[biggest_gap_idx + 1]) / 2
print(f"\nLargest gap in |asymmetry|: between {sorted_asym[biggest_gap_idx]:.3f} and {sorted_asym[biggest_gap_idx+1]:.3f}")
print(f"Suggested threshold (gap midpoint): {gap_threshold:.3f}")

print(f"\nTotal: {(time.time()-t0)/60:.1f} min")
print(f"Saved to {OUT / 'velocity_asymmetry_results.csv'}")