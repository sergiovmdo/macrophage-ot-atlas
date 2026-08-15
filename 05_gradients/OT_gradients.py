import numpy as np
import pandas as pd
import ot
import scanpy as sc
from pathlib import Path
from scipy.spatial.distance import cdist
from scipy.sparse import issparse
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LinearRegression

BASE  = Path("/storage/homefs/sv24v923/MPI_data/clean_pipeline")
OUT   = BASE / "ot_gradients_v3"           # new directory — keeps old files safe
OUT.mkdir(parents=True, exist_ok=True)

EPSILON = 0.01
N_DIMS  = 50
CT_COL  = "cell_type_meta_v3"              # <-- v3: no Differentiating

# ── Expanded gene lists for full Supplementary Table 4 coverage ───
GENES = {
    "Mono":   ["FCN1", "LYZ", "VCAN", "S100A9", "SAMSN1", "NAMPT",
               "S100A8", "S100A12", "COTL1", "LST1", "AREG"],
    "Scav":   ["C1QB", "C1QA", "C1QC", "APOE", "SELENOP", "CD81",
               "LGMN", "PLTP", "GPNMB"],
    "Res":    ["RNASE1", "LYVE1", "CFD", "F13A1", "FOLR2", "MRC1",
               "CCL18", "FTL", "MALAT1", "NEAT1", "CRIP1"],
    "Inflam": ["IL1B", "TNF", "CXCL8", "CCL3", "CCL4", "CCL2",
               "HLA-DRA", "ISG15", "CD74", "TNFAIP3", "NFKBIA"],
    "Foam":   ["CSTB", "FABP5", "IFI30", "CTSL", "SPP1", "LIPA",
               "PLIN2", "ABCA1", "ABCG1", "VIM", "S100A11"],
    "Fibro":  ["SPP1", "ANXA2", "FN1", "LGALS1", "MIF", "S100A10",
               "ENO1", "CAPG", "PKM", "VIM", "LGALS3"],
}

CT_MAP = {
    "Mono":   "Monocytes",
    "Scav":   "Scavenging / C1q+ Macrophages",
    "Res":    "Resident / Quiescent Macrophages",
    "Inflam": "Inflammatory Macrophages",
    "Foam":   "Lipid-Stressed / Foam Cells",
    "Fibro":  "Fibrotic / Hypoxic Macrophages",
}

ALL_TRANSITIONS = [
    ("Mono_to_Fibro",   "Mono",   "Fibro"),
    ("Mono_to_Foam",    "Mono",   "Foam"),
    ("Mono_to_Inflam",  "Mono",   "Inflam"),
    ("Mono_to_Res",     "Mono",   "Res"),
    ("Mono_to_Scav",    "Mono",   "Scav"),
    ("Scav_to_Inflam",  "Scav",   "Inflam"),
    ("Scav_to_Fibro",   "Scav",   "Fibro"),
    ("Res_to_Inflam",   "Res",    "Inflam"),
    ("Res_to_Fibro",    "Res",    "Fibro"),
    ("Foam_to_Fibro",   "Foam",   "Fibro"),
    ("Inflam_to_Fibro", "Inflam", "Fibro"),
    ("Fibro_to_Scav",   "Fibro",  "Scav"),
    ("Foam_to_Scav",    "Foam",   "Scav"),
]


# ── Library-size correction ───────────────────────────────────────
def correct_library_size(weights, total_counts):
    """OLS residuals of weights ~ total_counts."""
    X   = total_counts.reshape(-1, 1)
    reg = LinearRegression().fit(X, weights)
    return weights - reg.predict(X)


# ── Load atlas ────────────────────────────────────────────────────
print("Loading atlas...")
adata    = sc.read_h5ad(
    BASE / "macrophage_annotation"
           "/Macrophage_Atlas_FINAL_v3_CLEAN_ANNOTATED.h5ad"
)
X_scano  = adata.obsm["X_scanorama"][:, :N_DIMS].astype(np.float64)
ct_vals  = adata.obs[CT_COL].values
var_list = adata.var_names.tolist()

# Precompute total counts for library-size correction
if "total_counts" in adata.obs.columns:
    total_counts_all = adata.obs["total_counts"].values.astype(np.float64)
else:
    print("  Computing total counts from X...")
    if issparse(adata.X):
        total_counts_all = np.asarray(adata.X.sum(axis=1)).flatten()
    else:
        total_counts_all = adata.X.sum(axis=1)

print(f"Atlas: {adata.n_obs:,} cells x {adata.n_vars:,} genes")
print(f"Embedding: {X_scano.shape}")
print(f"Cell type column: {CT_COL}")
print(f"Unique cell types: {pd.Series(ct_vals).unique().tolist()}")


def get_expr(cell_indices, genes):
    present = [g for g in genes if g in var_list]
    missing = [g for g in genes if g not in var_list]
    if missing:
        print(f"  WARNING not in atlas: {missing}")
    expr = {}
    for gene in present:
        idx_g = var_list.index(gene)
        col   = adata.X[cell_indices, idx_g]
        if hasattr(col, "toarray"):
            col = col.toarray().flatten()
        expr[gene] = np.array(col, dtype=np.float32).flatten()
    return pd.DataFrame(expr)


def compute_fate_scores(src_idx, tgt_idx):
    Xs = X_scano[src_idx]
    Xt = X_scano[tgt_idx]

    print(f"  cost matrix {len(src_idx):,} x {len(tgt_idx):,}...")
    M = cdist(Xs, Xt, metric="sqeuclidean")
    M = M / M.max()

    a = np.ones(len(src_idx)) / len(src_idx)

    k  = min(10, len(tgt_idx) - 1)
    nn = NearestNeighbors(n_neighbors=k, metric="euclidean", n_jobs=1)
    nn.fit(Xt)
    dists, _ = nn.kneighbors(Xt)
    density  = 1.0 / (dists.mean(axis=1) + 1e-10)
    b        = density / density.sum()

    print(f"  sinkhorn epsilon={EPSILON}...")
    T = ot.sinkhorn(a, b, M, reg=EPSILON,
                    numItermax=2000, stopThr=1e-9)

    fate_scores = T.dot(b) * len(src_idx)
    print(f"  scores  min={fate_scores.min():.6f}  "
          f"max={fate_scores.max():.6f}  "
          f"range={fate_scores.max()-fate_scores.min():.6f}")
    return fate_scores


# ── Main loop ─────────────────────────────────────────────────────
print("\n" + "=" * 60)
print(f"  COMPUTING {len(ALL_TRANSITIONS)} TRANSITIONS  (v3 annotation)")
print(f"  embedding=X_scanorama  epsilon={EPSILON}  dims={N_DIMS}")
print("=" * 60)

for key, src_g, tgt_g in ALL_TRANSITIONS:
    out_path = OUT / f"{key}.csv"
    if out_path.exists():
        print(f"\n  SKIP {key} — already done")
        continue

    print(f"\n{'=' * 60}")
    print(f"  {key}")

    src_idx = np.where(ct_vals == CT_MAP[src_g])[0]
    tgt_idx = np.where(ct_vals == CT_MAP[tgt_g])[0]
    print(f"  src={len(src_idx):,}  tgt={len(tgt_idx):,}")

    if len(src_idx) < 100 or len(tgt_idx) < 100:
        print(f"  SKIP — too few cells")
        continue

    try:
        scores = compute_fate_scores(src_idx, tgt_idx)
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback; traceback.print_exc()
        continue

    if scores.max() - scores.min() < 1e-10:
        print(f"  FLAT scores — skipping")
        continue

    # ── Library-size correction ───────────────────────────────────
    tc                 = total_counts_all[src_idx]
    scores_corrected   = correct_library_size(scores, tc)

    from scipy.stats import spearmanr
    rho_before, _ = spearmanr(tc, scores)
    rho_after,  _ = spearmanr(tc, scores_corrected)
    print(f"  Library-size ρ: {rho_before:+.4f} → {rho_after:+.4f}"
          + ("  ⚠ RESIDUAL CONFOUND" if abs(rho_after) > 0.15 else "  ✓"))

    # ── Quartile assignment ───────────────────────────────────────
    ot_quartile           = pd.qcut(scores, q=4,
                                    labels=["Q1","Q2","Q3","Q4"],
                                    duplicates="drop").astype(str)
    ot_quartile_corrected = pd.qcut(scores_corrected, q=4,
                                    labels=["Q1","Q2","Q3","Q4"],
                                    duplicates="drop").astype(str)

    # ── Expression ───────────────────────────────────────────────
    genes   = list(dict.fromkeys(GENES[src_g] + GENES[tgt_g]))
    expr_df = get_expr(src_idx, genes)

    expr_df["ot_weight"]             = scores
    expr_df["ot_quartile"]           = ot_quartile
    expr_df["transition"]            = key
    expr_df["ot_weight_corrected"]   = scores_corrected
    expr_df["ot_quartile_corrected"] = ot_quartile_corrected

    expr_df.to_csv(out_path, index=False)
    print(f"  SAVED {out_path.name}  "
          f"({len(expr_df):,} cells x {len(genes)} genes)")

    # ── Signal check ──────────────────────────────────────────────
    print(f"\n  {'Gene':12s}  {'Q1':>7}  {'Q2':>7}  "
          f"{'Q3':>7}  {'Q4':>7}  {'log2FC':>8}  Role")
    print(f"  {'─' * 60}")
    for role, glist in [("SRC", GENES[src_g]), ("TGT", GENES[tgt_g])]:
        for gene in glist:
            if gene not in expr_df.columns:
                continue
            qm  = expr_df.groupby("ot_quartile_corrected")[gene].mean()
            q1  = float(qm.get("Q1", 0))
            q4  = float(qm.get("Q4", 0))
            lfc = np.log2((q4 + 0.1) / (q1 + 0.1))
            sig = "✅" if abs(lfc) > 1.0 else "—" if abs(lfc) > 0.3 else "⚠️"
            print(f"  {gene:12s}  "
                  f"{float(qm.get('Q1',0)):7.3f}  "
                  f"{float(qm.get('Q2',0)):7.3f}  "
                  f"{float(qm.get('Q3',0)):7.3f}  "
                  f"{q4:7.3f}  {lfc:+8.3f}  {sig} {role}")

print("\n" + "=" * 60)
print("  ALL DONE")
print(f"  Results in: {OUT}")
print("=" * 60)
