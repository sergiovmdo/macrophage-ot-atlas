#!/usr/bin/env python3
"""
Optimal Transport Plasticity Pipeline v8
==========================================
Changes from v7:
  - CELL_TYPE_COL -> cell_type_meta_v3
  - Differentiating / Transitional merged into Monocytes
  - TRANSITIONS_OF_INTEREST updated: 11 transitions (removed 2 Diff transitions)
  - OUTPUT_DIR -> ot_results_v8_{mode}

Usage — submit as 4 separate cluster jobs:
  python ot_pipeline_v8.py --mode full
  python ot_pipeline_v8.py --mode loco_alsaigh
  python ot_pipeline_v8.py --mode loco_bashore
  python ot_pipeline_v8.py --mode loco_jaiswal
"""

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import argparse
import h5py
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import networkx as nx
import ot
import gc
import time
from pathlib import Path
from itertools import combinations_with_replacement
from scipy.spatial.distance import cdist
from joblib import Parallel, delayed

# ──────────────────────────────────────────────────────────────────────────────
# ARGUMENT PARSING
# ──────────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument(
    "--mode",
    choices=["full", "loco_alsaigh", "loco_bashore", "loco_jaiswal"],
    default="full",
    help="full = all cohorts | loco_X = remove cohort X before OT"
)
args = parser.parse_args()
MODE = args.mode

LOCO_REMOVE = {
    "full":         None,
    "loco_alsaigh": "alsaigh",
    "loco_bashore": "bashore",
    "loco_jaiswal": "jaiswal",
}[MODE]

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────
REAL_PATH = Path(
    "/storage/homefs/sv24v923/MPI_data/clean_pipeline/macrophage_annotation/"
    "Macrophage_Atlas_FINAL_v3_CLEAN_ANNOTATED.h5ad"
)

OUTPUT_DIR = Path(
    f"/storage/homefs/sv24v923/MPI_data/clean_pipeline/ot_results_v8_{MODE}_500_bootstrap"
)
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# ── KEY CHANGE 1 ───────────────────────────────────────────────────────────────
CELL_TYPE_COL = "cell_type_meta_v3"   # was cell_type_meta_v2
BATCH_COL     = "batch"
EMBED_KEY     = "X_scanorama"

EPSILON        = 0.01
N_JOBS         = 16
SEED           = 42
Z_THRESHOLD    = 2.0

MAX_CELLS_FULL = 5000
MAX_CELLS_BOOT = 2000
N_BOOTSTRAP    = 500
N_PERMUTATIONS = 50

FIG_SIZE_HEATMAP = (16, 14)
FIG_SIZE_GRAPH   = (14, 14)
GRAPH_TOP_PCT    = 60

# ── KEY CHANGE 2 ───────────────────────────────────────────────────────────────
# Removed:
#   ("Differentiating / Transitional", "Monocytes")          — merged
#   ("Differentiating / Transitional", "Resident / Quiescent Macrophages") — merged
# Remaining: 11 transitions
TRANSITIONS_OF_INTEREST = [
    # Axis A — Inflammatory Reactivation
    ("Scavenging / C1q+ Macrophages",    "Inflammatory Macrophages"),
    ("Resident / Quiescent Macrophages", "Inflammatory Macrophages"),
    ("Monocytes",                        "Inflammatory Macrophages"),
    # Axis B — Monocyte Fate (Differentiating now absorbed into Monocytes)
    ("Monocytes",                        "Resident / Quiescent Macrophages"),
    ("Lipid-Stressed / Foam Cells",      "Monocytes"),
    ("Monocytes",                        "Scavenging / C1q+ Macrophages"),
    # Axis C — Routes to Fibrosis
    ("Lipid-Stressed / Foam Cells",      "Fibrotic / Hypoxic Macrophages"),
    ("Resident / Quiescent Macrophages", "Scavenging / C1q+ Macrophages"),
    ("Scavenging / C1q+ Macrophages",    "Fibrotic / Hypoxic Macrophages"),
    ("Resident / Quiescent Macrophages", "Fibrotic / Hypoxic Macrophages"),
    ("Resident / Quiescent Macrophages", "Lipid-Stressed / Foam Cells"),
]

sns.set_theme(style="whitegrid")
t_total_start = time.time()

print("=" * 70)
print(f"OT PIPELINE v8  —  mode: {MODE}")
print(f"  Cell type column: {CELL_TYPE_COL}")
if LOCO_REMOVE:
    print(f"  Removing cohort: {LOCO_REMOVE}")
print(f"  Transitions of interest: {len(TRANSITIONS_OF_INTEREST)}")
print(f"  Output dir: {OUTPUT_DIR}")
print("=" * 70)

# ──────────────────────────────────────────────────────────────────────────────
# HELPERS  (identical to v7)
# ──────────────────────────────────────────────────────────────────────────────

def load_embeddings_and_labels(h5ad_path, remove_cohort=None):
    print(f"  Loading {h5ad_path.name} via h5py...")
    with h5py.File(h5ad_path, "r") as f:
        X_all = np.array(f["obsm"][EMBED_KEY], dtype=np.float32)
        obs   = f["obs"]

        def read_categorical(obs, col):
            if col in obs and isinstance(obs[col], h5py.Group):
                codes = np.array(obs[col]["codes"])
                cats  = [c.decode("utf-8") if isinstance(c, bytes) else str(c)
                         for c in np.array(obs[col]["categories"])]
            else:
                raw  = np.array(obs[col])
                labs = [l.decode("utf-8") if isinstance(l, bytes) else str(l)
                        for l in raw]
                cats  = sorted(set(labs))
                c2i   = {c: i for i, c in enumerate(cats)}
                codes = np.array([c2i[l] for l in labs])
            return codes, cats

        ct_codes,  ct_cats  = read_categorical(obs, CELL_TYPE_COL)
        bat_codes, bat_cats = read_categorical(obs, BATCH_COL)

    print(f"  → {X_all.shape[0]:,} cells | {X_all.shape[1]}D | "
          f"{len(ct_cats)} meta-clusters | {len(bat_cats)} cohorts")

    if remove_cohort is not None:
        if remove_cohort not in bat_cats:
            raise ValueError(
                f"Cohort '{remove_cohort}' not found. "
                f"Available: {bat_cats}"
            )
        cohort_idx = bat_cats.index(remove_cohort)
        keep       = bat_codes != cohort_idx
        n_removed  = (~keep).sum()
        X_all      = X_all[keep]
        ct_codes   = ct_codes[keep]
        print(f"  → LOCO: removed {n_removed:,} '{remove_cohort}' cells")
        print(f"  → Remaining: {X_all.shape[0]:,} cells")

    present      = set(np.unique(ct_codes))
    ct_cats_keep = [c for i, c in enumerate(ct_cats) if i in present]
    remap        = {old: new for new, old in enumerate(sorted(present))}
    ct_codes_new = np.array([remap[c] for c in ct_codes])

    print(f"\n  Cluster sizes:")
    for i, c in enumerate(ct_cats_keep):
        n    = (ct_codes_new == i).sum()
        pctb = min(MAX_CELLS_BOOT, n) / n * 100
        pctf = min(MAX_CELLS_FULL, n) / n * 100
        flag = "  ⚠️  SMALL" if n < 500 else ""
        print(f"    {c:<40s}  n={n:6,}  "
              f"boot={min(MAX_CELLS_BOOT,n):4d} ({pctb:.0f}%)  "
              f"full={min(MAX_CELLS_FULL,n):4d} ({pctf:.0f}%){flag}")

    return X_all, ct_codes_new, ct_cats_keep


def split_by_labels(X, codes, categories):
    return {ct: X[codes == i] for i, ct in enumerate(categories)}


def subsample(X, max_n, rng, replace=False):
    n = min(max_n, len(X))
    return X[rng.choice(len(X), n, replace=replace)]


def sinkhorn_cost(X_src, X_tgt, epsilon=EPSILON):
    a = np.full(len(X_src), 1.0 / len(X_src), dtype=np.float64)
    b = np.full(len(X_tgt), 1.0 / len(X_tgt), dtype=np.float64)
    C = cdist(X_src, X_tgt, metric="euclidean").astype(np.float64)
    scale = C.max()
    if scale < 1e-9:
        return 0.0
    C /= scale
    T = ot.sinkhorn(a, b, C, reg=epsilon, numItermax=2000, warn=False)
    return float(np.sum(T * C) * scale)


def sinkhorn_cost_with_matrix(X_src, X_tgt, epsilon=EPSILON):
    a = np.full(len(X_src), 1.0 / len(X_src), dtype=np.float64)
    b = np.full(len(X_tgt), 1.0 / len(X_tgt), dtype=np.float64)
    C = cdist(X_src, X_tgt, metric="euclidean").astype(np.float64)
    scale = C.max()
    if scale < 1e-9:
        return 0.0, None
    C /= scale
    T = ot.sinkhorn(a, b, C, reg=epsilon, numItermax=2000, warn=False)
    return float(np.sum(T * C) * scale), T


def compute_pair(embeddings, ct_i, ct_j, seed, max_cells, replace=False):
    rng = np.random.default_rng(seed)
    X_i = subsample(embeddings[ct_i], max_cells, rng, replace=replace)
    X_j = subsample(embeddings[ct_j], max_cells, rng, replace=replace)
    return (ct_i, ct_j, sinkhorn_cost(X_i, X_j))


def build_cost_matrix(embeddings, cell_types, label="",
                      base_seed=SEED, max_cells=MAX_CELLS_BOOT,
                      replace=False):
    pairs   = list(combinations_with_replacement(cell_types, 2))
    rng     = np.random.default_rng(base_seed)
    seeds   = rng.integers(0, 2**31, size=len(pairs))
    results = Parallel(n_jobs=N_JOBS, verbose=0)(
        delayed(compute_pair)(embeddings, ci, cj, int(s),
                              max_cells=max_cells, replace=replace)
        for (ci, cj), s in zip(pairs, seeds)
    )
    n   = len(cell_types)
    mat = pd.DataFrame(np.zeros((n, n)), index=cell_types, columns=cell_types)
    for ci, cj, cost in results:
        mat.loc[ci, cj] = cost
        mat.loc[cj, ci] = cost
    print(f"    [{label}] done")
    return mat


def to_sinkhorn_divergence(cost_df):
    W = cost_df.to_numpy(dtype=np.float64)
    d = np.diag(W)
    S = W - 0.5 * d[:, None] - 0.5 * d[None, :]
    S[S < 0] = 0.0
    np.fill_diagonal(S, 0.0)
    return pd.DataFrame(S, index=cost_df.index, columns=cost_df.columns)


def safe_key(src, tgt):
    return (f"{src}_to_{tgt}"
            .replace(" / ", "_")
            .replace(" ", "_"))


def print_elapsed(t_start, label=""):
    elapsed = (time.time() - t_start) / 60
    print(f"  ⏱  {label}: {elapsed:.1f} min elapsed")


# ──────────────────────────────────────────────────────────────────────────────
# LOAD DATA
# ──────────────────────────────────────────────────────────────────────────────
print("\nLOADING DATA")
print("-" * 70)
t0 = time.time()

X_all, codes_all, cell_types = load_embeddings_and_labels(
    REAL_PATH, remove_cohort=LOCO_REMOVE
)
n_types  = len(cell_types)
real_emb = split_by_labels(X_all, codes_all, cell_types)

# Verify Differentiating is gone and Monocytes are present
print(f"\n  Cell types in {CELL_TYPE_COL}:")
for ct in cell_types:
    print(f"    {ct}")

if any("Differentiating" in ct for ct in cell_types):
    print("\n  ⚠️  WARNING: Differentiating / Transitional still present in "
          f"{CELL_TYPE_COL} — check your labelling before proceeding")
else:
    print("\n  ✓ Differentiating / Transitional not present — merge confirmed")

pd.Series({
    "mode":            MODE,
    "loco_remove":     str(LOCO_REMOVE),
    "h5ad":            REAL_PATH.name,
    "cell_type_col":   CELL_TYPE_COL,
    "embed_key":       EMBED_KEY,
    "n_cells":         int(X_all.shape[0]),
    "n_meta_clusters": n_types,
    "max_cells_full":  MAX_CELLS_FULL,
    "max_cells_boot":  MAX_CELLS_BOOT,
    "n_bootstrap":     N_BOOTSTRAP,
    "n_permutations":  N_PERMUTATIONS,
    "epsilon":         EPSILON,
    "z_threshold":     Z_THRESHOLD,
    "seed":            SEED,
}).to_csv(OUTPUT_DIR / "run_config.csv", header=False)

print_elapsed(t0, "Data loaded")


# ──────────────────────────────────────────────────────────────────────────────
# STEP 1 — FULL-DATA REFERENCE RUN
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print(f"STEP 1: Full-data reference run (max {MAX_CELLS_FULL} cells/cluster)")
print("=" * 70)
t1 = time.time()

full_cost = build_cost_matrix(real_emb, cell_types,
                              label="FULL",
                              max_cells=MAX_CELLS_FULL,
                              replace=False)
full_div = to_sinkhorn_divergence(full_cost)
full_div.to_csv(OUTPUT_DIR / "sinkhorn_divergence_full.csv")
print_elapsed(t1, "Step 1")


# ──────────────────────────────────────────────────────────────────────────────
# STEP 1b — SAVE TRANSPORT MATRICES FOR TRANSITIONS OF INTEREST
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 1b: Saving transport matrices")
print("=" * 70)
t1b = time.time()

TRANSPORT_DIR = OUTPUT_DIR / "transport_matrices"
TRANSPORT_DIR.mkdir(exist_ok=True)

rng_transport = np.random.default_rng(SEED)
transport_log = []

for src, tgt in TRANSITIONS_OF_INTEREST:
    if src not in real_emb or tgt not in real_emb:
        print(f"  SKIP {src} → {tgt}  (cluster not in this run)")
        continue

    X_src = subsample(real_emb[src], MAX_CELLS_FULL, rng_transport)
    X_tgt = subsample(real_emb[tgt], MAX_CELLS_FULL, rng_transport)

    cost, T = sinkhorn_cost_with_matrix(X_src, X_tgt)

    if T is None:
        print(f"  FAILED  {src} → {tgt}")
        continue

    sk = safe_key(src, tgt)
    np.save(TRANSPORT_DIR / f"T_{sk}.npy",           T)
    np.save(TRANSPORT_DIR / f"X_src_{sk}.npy",       X_src)
    np.save(TRANSPORT_DIR / f"X_tgt_{sk}.npy",       X_tgt)
    np.save(TRANSPORT_DIR / f"src_weights_{sk}.npy", T.sum(axis=1))
    np.save(TRANSPORT_DIR / f"tgt_weights_{sk}.npy", T.sum(axis=0))

    src_weights = T.sum(axis=1)
    transport_log.append({
        "source":         src,
        "target":         tgt,
        "n_src_cells":    len(X_src),
        "n_tgt_cells":    len(X_tgt),
        "T_shape":        f"{T.shape[0]}x{T.shape[1]}",
        "cost":           round(cost, 6),
        "src_weight_max": round(float(src_weights.max()), 6),
        "src_weight_min": round(float(src_weights.min()), 6),
    })
    print(f"  OK  {src} → {tgt}  T={T.shape}  cost={cost:.4f}")

pd.DataFrame(transport_log).to_csv(
    TRANSPORT_DIR / "transport_log.csv", index=False
)
print(f"\n  Saved {len(transport_log)} transport matrices")
print_elapsed(t1b, "Step 1b")


# ──────────────────────────────────────────────────────────────────────────────
# STEP 2 — BOOTSTRAP CI
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print(f"STEP 2: Bootstrap CI  (n={MAX_CELLS_BOOT}/cluster × B={N_BOOTSTRAP})")
print("=" * 70)
t2 = time.time()

pairs      = list(combinations_with_replacement(cell_types, 2))
boot_store = {(ci, cj): [] for ci, cj in pairs}

for b in range(N_BOOTSTRAP):
    t_iter    = time.time()
    boot_cost = build_cost_matrix(real_emb, cell_types,
                                  label=f"BOOT-{b+1:03d}/{N_BOOTSTRAP}",
                                  base_seed=SEED + b + 1000,
                                  max_cells=MAX_CELLS_BOOT,
                                  replace=True)
    boot_div = to_sinkhorn_divergence(boot_cost)
    for ci, cj in pairs:
        boot_store[(ci, cj)].append(boot_div.loc[ci, cj])
    elapsed_iter  = (time.time() - t_iter) / 60
    elapsed_total = (time.time() - t2) / 60
    eta = elapsed_total / (b + 1) * (N_BOOTSTRAP - b - 1)
    print(f"  Bootstrap {b+1:3d}/{N_BOOTSTRAP}  "
          f"iter={elapsed_iter:.1f}min  "
          f"elapsed={elapsed_total:.1f}min  "
          f"ETA={eta:.1f}min")

boot_mean  = pd.DataFrame(np.zeros((n_types, n_types)),
                           index=cell_types, columns=cell_types)
boot_std   = pd.DataFrame(np.zeros((n_types, n_types)),
                           index=cell_types, columns=cell_types)
boot_ci_lo = pd.DataFrame(np.zeros((n_types, n_types)),
                           index=cell_types, columns=cell_types)
boot_ci_hi = pd.DataFrame(np.zeros((n_types, n_types)),
                           index=cell_types, columns=cell_types)

for ci, cj in pairs:
    vals = np.array(boot_store[(ci, cj)])
    m, s = np.mean(vals), np.std(vals)
    lo, hi = np.percentile(vals, [2.5, 97.5])
    for mat, val in [(boot_mean, m), (boot_std, s),
                     (boot_ci_lo, lo), (boot_ci_hi, hi)]:
        mat.loc[ci, cj] = val
        mat.loc[cj, ci] = val

boot_mean.to_csv(OUTPUT_DIR  / "bootstrap_mean.csv")
boot_std.to_csv(OUTPUT_DIR   / "bootstrap_std.csv")
boot_ci_lo.to_csv(OUTPUT_DIR / "bootstrap_ci_lo.csv")
boot_ci_hi.to_csv(OUTPUT_DIR / "bootstrap_ci_hi.csv")

print("\n  Consistency check: full-data estimate vs bootstrap 95% CI")
print(f"  {'Pair':55s} {'Full':>8s} {'Boot μ':>8s} {'CI lo':>8s} {'CI hi':>8s} {'In CI?':>6s}")
print("  " + "-" * 95)
all_in_ci = True
for ci in cell_types:
    for cj in cell_types:
        if ci < cj:
            fv    = full_div.loc[ci, cj]
            lo    = boot_ci_lo.loc[ci, cj]
            hi    = boot_ci_hi.loc[ci, cj]
            in_ci = lo <= fv <= hi
            if not in_ci:
                all_in_ci = False
            flag = "✓" if in_ci else "⚠ OUTSIDE CI"
            print(f"  {ci+' ↔ '+cj:55s} "
                  f"{fv:8.4f} "
                  f"{boot_mean.loc[ci,cj]:8.4f} "
                  f"{lo:8.4f} {hi:8.4f}  {flag}")
if all_in_ci:
    print("\n  ✅ All full-data estimates fall within bootstrap 95% CI")
else:
    print("\n  ⚠️  Some estimates outside CI — check embedding stability")

print_elapsed(t2, "Step 2 (bootstrap)")


# ──────────────────────────────────────────────────────────────────────────────
# STEP 3 — NULL MODEL: LABEL PERMUTATION
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print(f"STEP 3: Null model — label permutation × {N_PERMUTATIONS}")
print("=" * 70)
t3 = time.time()

null_store = {(ci, cj): [] for ci, cj in pairs}

for perm in range(N_PERMUTATIONS):
    rng            = np.random.default_rng(SEED + perm + 5000)
    shuffled_codes = codes_all.copy()
    rng.shuffle(shuffled_codes)
    perm_emb  = split_by_labels(X_all, shuffled_codes, cell_types)
    perm_cost = build_cost_matrix(perm_emb, cell_types,
                                  label=f"PERM-{perm+1:02d}/{N_PERMUTATIONS}",
                                  base_seed=SEED + perm + 5000,
                                  max_cells=MAX_CELLS_BOOT,
                                  replace=False)
    perm_div = to_sinkhorn_divergence(perm_cost)
    for ci, cj in pairs:
        null_store[(ci, cj)].append(perm_div.loc[ci, cj])
    del perm_emb
    gc.collect()

null_mean = pd.DataFrame(np.zeros((n_types, n_types)),
                          index=cell_types, columns=cell_types)
null_std  = pd.DataFrame(np.zeros((n_types, n_types)),
                          index=cell_types, columns=cell_types)
for ci, cj in pairs:
    vals = np.array(null_store[(ci, cj)])
    m, s = np.mean(vals), np.std(vals)
    null_mean.loc[ci, cj] = m;  null_mean.loc[cj, ci] = m
    null_std.loc[ci, cj]  = s;  null_std.loc[cj, ci]  = s

null_mean.to_csv(OUTPUT_DIR / "null_mean.csv")
null_std.to_csv(OUTPUT_DIR  / "null_std.csv")
print_elapsed(t3, "Step 3 (null model)")

del X_all, codes_all
gc.collect()


# ──────────────────────────────────────────────────────────────────────────────
# STEP 4 — SIGNIFICANCE TESTING
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 4: Significance testing (z-scores on full-data divergences)")
print("=" * 70)

z_mat   = pd.DataFrame(np.zeros((n_types, n_types)),
                        index=cell_types, columns=cell_types)
sig_mat = pd.DataFrame(np.zeros((n_types, n_types), dtype=bool),
                        index=cell_types, columns=cell_types)

for ci, cj in pairs:
    real_val = full_div.loc[ci, cj]
    null_m   = null_mean.loc[ci, cj]
    null_s   = null_std.loc[ci, cj]
    z = (real_val - null_m) / null_s if (null_s > 1e-9 and ci != cj) else 0.0
    z_mat.loc[ci, cj] = z;  z_mat.loc[cj, ci] = z
    sig = (abs(z) > Z_THRESHOLD) and (ci != cj)
    sig_mat.loc[ci, cj] = sig;  sig_mat.loc[cj, ci] = sig

np.fill_diagonal(z_mat.values,   0.0)
np.fill_diagonal(sig_mat.values, False)

n_sig   = sig_mat.sum().sum() // 2
n_total = n_types * (n_types - 1) // 2
print(f"  Significant pairs: {n_sig} / {n_total} (|z| > {Z_THRESHOLD})")
print(f"\n  {'Pair':55s} {'Full D':>8s} {'Boot μ':>8s} "
      f"{'95% CI':>18s} {'Null μ':>8s} {'z':>8s} {'Sig':>4s}")
print("  " + "-" * 115)

results_rows = []
for ci in cell_types:
    for cj in cell_types:
        if ci < cj:
            z    = z_mat.loc[ci, cj]
            sig  = sig_mat.loc[ci, cj]
            ci95 = f"[{boot_ci_lo.loc[ci,cj]:.3f},{boot_ci_hi.loc[ci,cj]:.3f}]"
            flag = "✓" if sig else ""
            print(f"  {ci+' ↔ '+cj:55s} "
                  f"{full_div.loc[ci,cj]:8.4f} "
                  f"{boot_mean.loc[ci,cj]:8.4f} "
                  f"{ci95:>18s} "
                  f"{null_mean.loc[ci,cj]:8.4f} "
                  f"{z:8.2f}  {flag}")
            results_rows.append({
                "Mode":        MODE,
                "Pair":        f"{ci} ↔ {cj}",
                "Source":      ci,
                "Target":      cj,
                "Full_Div":    full_div.loc[ci, cj],
                "Boot_Mean":   boot_mean.loc[ci, cj],
                "Boot_SD":     boot_std.loc[ci, cj],
                "CI_lo":       boot_ci_lo.loc[ci, cj],
                "CI_hi":       boot_ci_hi.loc[ci, cj],
                "Null_Mean":   null_mean.loc[ci, cj],
                "Null_SD":     null_std.loc[ci, cj],
                "z_score":     z,
                "Significant": sig,
            })

results_df = pd.DataFrame(results_rows).sort_values("z_score", ascending=False)
results_df.to_csv(OUTPUT_DIR / "full_results_with_CI.csv", index=False)
z_mat.to_csv(OUTPUT_DIR   / "z_scores.csv")
sig_mat.to_csv(OUTPUT_DIR / "significant_pairs.csv")

full_div_sig = full_div.copy()
full_div_sig[~sig_mat] = 0.0
full_div_sig.to_csv(OUTPUT_DIR / "sinkhorn_divergence_significant.csv")
print("\n  ✅ Full results with CI saved to full_results_with_CI.csv")


# ──────────────────────────────────────────────────────────────────────────────
# STEP 5 — RANKED CONNECTIONS + CONNECTIVITY SCORES
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 5: Ranked connections + connectivity scores")
print("=" * 70)

sig_vals = full_div_sig.to_numpy()
off_diag = sig_vals[sig_vals > 0]

if len(off_diag) > 0:
    sigma = np.median(off_diag) * 0.5
    conn  = np.exp(-(sig_vals ** 2) / (2 * sigma ** 2))
    np.fill_diagonal(conn, 0)
    conn[~sig_mat.to_numpy()] = 0
    conn_df = pd.DataFrame(conn, index=cell_types, columns=cell_types)
    G = nx.from_pandas_adjacency(conn_df)
    G.remove_edges_from([(u, v) for u, v, d in G.edges(data=True)
                         if d["weight"] < 1e-12])
    if G.number_of_edges() > 0:
        w   = [d["weight"] for _, _, d in G.edges(data=True)]
        thr = np.percentile(w, 100 - GRAPH_TOP_PCT)
        G.remove_edges_from([(u, v) for u, v, d in G.edges(data=True)
                              if d["weight"] < thr])

    edge_rows = []
    for u, v, d in G.edges(data=True):
        edge_rows.append({
            "Mode":         MODE,
            "Source":       u,
            "Target":       v,
            "Full_Div":     full_div.loc[u, v],
            "Boot_Mean":    boot_mean.loc[u, v],
            "CI_lo":        boot_ci_lo.loc[u, v],
            "CI_hi":        boot_ci_hi.loc[u, v],
            "z_score":      z_mat.loc[u, v],
            "Connectivity": d["weight"],
        })
    edges_df = (pd.DataFrame(edge_rows)
                .sort_values("Connectivity", ascending=False)
                .reset_index(drop=True))
    edges_df.to_csv(OUTPUT_DIR / "significant_connections_ranked.csv", index=False)
    print(f"  Saved significant_connections_ranked.csv")
    print(f"\n{edges_df.to_string()}")


# ──────────────────────────────────────────────────────────────────────────────
# STEP 6 — PLOTS
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 6: Plots")
print("=" * 70)

mask_upper   = np.triu(np.ones_like(full_div, dtype=bool), k=1)
title_suffix = f"  [{MODE}  |  {CELL_TYPE_COL}]"

# 6a. Full divergence heatmap
fig, ax = plt.subplots(figsize=FIG_SIZE_HEATMAP)
sns.heatmap(full_div, annot=True, fmt=".3f", cmap="viridis_r",
            linewidths=0.5, mask=mask_upper, square=True,
            cbar_kws={"shrink": 0.5}, ax=ax)
ax.set_title(f"Sinkhorn Divergence — Full Data{title_suffix}", fontsize=14, pad=16)
plt.xticks(rotation=45, ha="right"); plt.yticks(rotation=0)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "heatmap_full.png", dpi=150, bbox_inches="tight")
fig.savefig(OUTPUT_DIR / "heatmap_full.pdf", bbox_inches="tight")
plt.close(fig)

# 6b. Bootstrap mean ± SD
fig, ax = plt.subplots(figsize=FIG_SIZE_HEATMAP)
annot_boot = boot_mean.copy().astype(str)
for ci in cell_types:
    for cj in cell_types:
        annot_boot.loc[ci, cj] = (f"{boot_mean.loc[ci,cj]:.3f}\n"
                                   f"±{boot_std.loc[ci,cj]:.3f}")
sns.heatmap(boot_mean, annot=annot_boot, fmt="", cmap="viridis_r",
            linewidths=0.5, mask=mask_upper, square=True,
            cbar_kws={"shrink": 0.5}, ax=ax)
ax.set_title(f"Bootstrap Mean ± SD (B={N_BOOTSTRAP}){title_suffix}", fontsize=14, pad=16)
plt.xticks(rotation=45, ha="right"); plt.yticks(rotation=0)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "heatmap_bootstrap.png", dpi=150, bbox_inches="tight")
fig.savefig(OUTPUT_DIR / "heatmap_bootstrap.pdf", bbox_inches="tight")
plt.close(fig)

# 6c. Z-score heatmap
fig, ax = plt.subplots(figsize=FIG_SIZE_HEATMAP)
sns.heatmap(z_mat, annot=True, fmt=".1f", cmap="RdBu_r", center=0,
            linewidths=0.5, mask=mask_upper, square=True,
            cbar_kws={"shrink": 0.5, "label": "z-score"}, ax=ax)
ax.set_title(f"Z-scores vs Null{title_suffix}", fontsize=14, pad=16)
plt.xticks(rotation=45, ha="right"); plt.yticks(rotation=0)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "heatmap_zscores.png", dpi=150, bbox_inches="tight")
fig.savefig(OUTPUT_DIR / "heatmap_zscores.pdf", bbox_inches="tight")
plt.close(fig)

# 6d. Divergence + CI
fig, ax = plt.subplots(figsize=FIG_SIZE_HEATMAP)
annot_sig = full_div.copy().astype(str)
for ci in cell_types:
    for cj in cell_types:
        star = "*" if sig_mat.loc[ci, cj] else ""
        annot_sig.loc[ci, cj] = (f"{full_div.loc[ci,cj]:.3f}{star}\n"
                                  f"[{boot_ci_lo.loc[ci,cj]:.3f},"
                                  f"{boot_ci_hi.loc[ci,cj]:.3f}]")
sns.heatmap(full_div, annot=annot_sig, fmt="", cmap="viridis_r",
            linewidths=0.5, mask=mask_upper, square=True,
            cbar_kws={"shrink": 0.5}, ax=ax)
ax.set_title(f"Divergence + 95% CI{title_suffix}", fontsize=14, pad=16)
plt.xticks(rotation=45, ha="right"); plt.yticks(rotation=0)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "heatmap_divergence_CI.png", dpi=150, bbox_inches="tight")
fig.savefig(OUTPUT_DIR / "heatmap_divergence_CI.pdf", bbox_inches="tight")
plt.close(fig)

# 6e. Network graph
if len(off_diag) > 0 and G.number_of_edges() > 0:
    pos = nx.circular_layout(G)
    ew  = [G[u][v]["weight"] * 15 for u, v in G.edges()]
    fig, ax = plt.subplots(figsize=FIG_SIZE_GRAPH)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=3500, node_color="#5B9BD5")
    nx.draw_networkx_edges(G, pos, ax=ax, width=ew, edge_color="grey", alpha=0.6)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=9, font_weight="bold")
    ax.set_title(f"OT Connectivity{title_suffix}", fontsize=16)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "graph_significant.png", dpi=150, bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / "graph_significant.pdf", bbox_inches="tight")
    plt.close(fig)

print("  All plots saved")

print("\n" + "=" * 70)
print_elapsed(t_total_start, "TOTAL PIPELINE")
print(f"DONE — outputs in: {OUTPUT_DIR.resolve()}")
print(f"Transport matrices in: {(OUTPUT_DIR / 'transport_matrices').resolve()}")
print("=" * 70)