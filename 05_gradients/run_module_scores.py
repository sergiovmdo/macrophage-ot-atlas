#!/usr/bin/env python
"""Module score analysis for program retention (Minna's point)."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scanpy as sc
import numpy as np
import pandas as pd
from pathlib import Path
import gc
import sys
import time

BASE = Path("/storage/homefs/sv24v923/MPI_data/clean_pipeline")
ATLAS = BASE / "macrophage_annotation" / "Macrophage_Atlas_FINAL_v3_CLEAN_ANNOTATED.h5ad"
GRAD_DIR = BASE / "ot_gradients_v3"
OUT = BASE / "circularity_validation" / "gradients"
OUT.mkdir(parents=True, exist_ok=True)
CT_COL = "cell_type_meta_v3"

CT_MAP = {
    "Mono":   "Monocytes",
    "Scav":   "Scavenging / C1q+ Macrophages",
    "Res":    "Resident / Quiescent Macrophages",
    "Inflam": "Inflammatory Macrophages",
    "Foam":   "Lipid-Stressed / Foam Cells",
    "Fibro":  "Fibrotic / Hypoxic Macrophages",
}

META_COLORS = {
    "Mono": "#4E79A7", "Scav": "#59A14F", "Res": "#B07AA1",
    "Inflam": "#E15759", "Foam": "#F28E2B", "Fibro": "#8C564B",
}

N_GENES = 100
t0 = time.time()

# ══════════════════════════════════════════════════════════════
# STEP 1: Load atlas, compute DEGs, compute module scores
# ══════════════════════════════════════════════════════════════
print("=" * 60)
print("STEP 1: Load atlas + compute cluster DEGs")
print("=" * 60)
sys.stdout.flush()

adata = sc.read_h5ad(ATLAS)
print(f"  Atlas: {adata.shape}")
sys.stdout.flush()

# DEGs per cluster (each vs rest)
print(f"  Computing DEGs (Wilcoxon, each vs rest)...")
sys.stdout.flush()
sc.tl.rank_genes_groups(adata, groupby=CT_COL, method="wilcoxon", use_raw=False)
print(f"  DEGs done")
sys.stdout.flush()

# Extract top N marker genes per cluster
cluster_programs = {}
for ct_short, ct_full in CT_MAP.items():
    degs = sc.get.rank_genes_groups_df(adata, group=ct_full)
    degs = degs[degs["pvals_adj"] < 0.05].sort_values("scores", ascending=False)
    top_genes = degs.head(N_GENES)["names"].tolist()
    cluster_programs[ct_short] = top_genes
    print(f"  {ct_short}: {len(top_genes)} marker genes")
    # Print top 5 for verification
    print(f"    Top 5: {top_genes[:5]}")
    sys.stdout.flush()

# Save gene lists
for ct_short, genes in cluster_programs.items():
    pd.DataFrame({"gene": genes}).to_csv(OUT / f"module_genes_{ct_short}.csv", index=False)

# Score all cells for all programs
print(f"\n  Computing module scores...")
sys.stdout.flush()
for ct_short, genes in cluster_programs.items():
    sc.tl.score_genes(adata, gene_list=genes, score_name=f"score_{ct_short}")
    print(f"    score_{ct_short} done")
    sys.stdout.flush()

gc.collect()

# ══════════════════════════════════════════════════════════════
# STEP 2: Extract scores per transition along commitment
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 2: Extract module scores along OT commitment")
print("=" * 60)
sys.stdout.flush()

TRANSITIONS = [
    ("Mono_to_Scav",    "Mono→Scav",    "Mono",   "Scav"),
    ("Mono_to_Res",     "Mono→Res",     "Mono",   "Res"),
    ("Mono_to_Inflam",  "Mono→Inflam",  "Mono",   "Inflam"),
    ("Mono_to_Foam",    "Mono→Foam",    "Mono",   "Foam"),
    ("Mono_to_Fibro",   "Mono→Fibro",   "Mono",   "Fibro"),
    ("Scav_to_Inflam",  "Scav→Inflam",  "Scav",   "Inflam"),
    ("Res_to_Inflam",   "Res→Inflam",   "Res",    "Inflam"),
    ("Foam_to_Fibro",   "Foam→Fibro",   "Foam",   "Fibro"),
    ("Inflam_to_Fibro", "Inflam→Fibro", "Inflam", "Fibro"),
    ("Res_to_Fibro",    "Res→Fibro",    "Res",    "Fibro"),
    ("Fibro_to_Scav",   "Fibro→Scav",   "Fibro",  "Scav"),
]

ct_labels = adata.obs[CT_COL].values
results = []

for key, label, src_g, tgt_g in TRANSITIONS:
    grad_csv = GRAD_DIR / f"{key}.csv"
    if not grad_csv.exists():
        print(f"  {key}: no gradient CSV — skip")
        continue
    grad_df = pd.read_csv(grad_csv)
    quartiles = grad_df["ot_quartile_corrected"].values

    src_name = CT_MAP[src_g]
    src_idx = np.where(ct_labels == src_name)[0]
    if len(src_idx) != len(grad_df):
        print(f"  {key}: mismatch {len(src_idx)} vs {len(grad_df)} — skip")
        continue

    src_score = adata.obs[f"score_{src_g}"].values[src_idx]
    tgt_score = adata.obs[f"score_{tgt_g}"].values[src_idx]

    # Quartile means + std
    for q in ["Q1", "Q2", "Q3", "Q4"]:
        mask = quartiles == q
        results.append({
            "transition": key, "label": label,
            "src_g": src_g, "tgt_g": tgt_g,
            "quartile": q, "q_num": int(q[1]),
            "src_score_mean": src_score[mask].mean(),
            "src_score_std": src_score[mask].std(),
            "tgt_score_mean": tgt_score[mask].mean(),
            "tgt_score_std": tgt_score[mask].std(),
            "n_cells": mask.sum(),
        })

    # Summary
    src_q1 = src_score[quartiles == "Q1"].mean()
    src_q4 = src_score[quartiles == "Q4"].mean()
    tgt_q1 = tgt_score[quartiles == "Q1"].mean()
    tgt_q4 = tgt_score[quartiles == "Q4"].mean()
    src_change = "↑" if src_q4 > src_q1 else "↓"
    tgt_change = "↑" if tgt_q4 > tgt_q1 else "↓"

    print(f"  {label:20s}  SRC: {src_q1:.3f}→{src_q4:.3f} {src_change}  "
          f"TGT: {tgt_q1:.3f}→{tgt_q4:.3f} {tgt_change}")
    sys.stdout.flush()

res_df = pd.DataFrame(results)
res_df.to_csv(OUT / "module_score_results.csv", index=False)
print(f"\n  Saved to {OUT / 'module_score_results.csv'}")

# ══════════════════════════════════════════════════════════════
# STEP 3: Plot — by axis
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 3: Plotting")
print("=" * 60)
sys.stdout.flush()

AXES = [
    ("Axis A — Monocyte fate diversification", [
        ("Mono_to_Scav",   "Mono→Scav",   "Mono",  "Scav"),
        ("Mono_to_Res",    "Mono→Res",    "Mono",  "Res"),
        ("Mono_to_Inflam", "Mono→Inflam", "Mono",  "Inflam"),
        ("Mono_to_Foam",   "Mono→Foam",   "Mono",  "Foam"),
        ("Mono_to_Fibro",  "Mono→Fibro",  "Mono",  "Fibro"),
    ]),
    ("Axis B — Inflammatory reactivation", [
        ("Scav_to_Inflam", "Scav→Inflam", "Scav",  "Inflam"),
        ("Res_to_Inflam",  "Res→Inflam",  "Res",   "Inflam"),
    ]),
    ("Axis C — Routes to fibrosis", [
        ("Foam_to_Fibro",   "Foam→Fibro",   "Foam",   "Fibro"),
        ("Inflam_to_Fibro", "Inflam→Fibro", "Inflam", "Fibro"),
        ("Res_to_Fibro",    "Res→Fibro",    "Res",    "Fibro"),
        ("Fibro_to_Scav",   "Fibro→Scav",   "Fibro",  "Scav"),
    ]),
]

def plot_axis(ax, key, label, src_g, tgt_g, res_df):
    grp = res_df[res_df["transition"] == key].sort_values("q_num")
    if len(grp) == 0:
        return

    # Source program (dashed)
    ax.plot(grp["q_num"], grp["src_score_mean"], color=META_COLORS[src_g],
            ls="--", marker="o", lw=2.5, markersize=6, alpha=0.85,
            label=f"{src_g} program")
    ax.fill_between(grp["q_num"],
                     grp["src_score_mean"] - grp["src_score_std"] / np.sqrt(grp["n_cells"]),
                     grp["src_score_mean"] + grp["src_score_std"] / np.sqrt(grp["n_cells"]),
                     color=META_COLORS[src_g], alpha=0.1)

    # Target program (solid)
    ax.plot(grp["q_num"], grp["tgt_score_mean"], color=META_COLORS[tgt_g],
            ls="-", marker="o", lw=2.5, markersize=6, alpha=0.85,
            label=f"{tgt_g} program")
    ax.fill_between(grp["q_num"],
                     grp["tgt_score_mean"] - grp["tgt_score_std"] / np.sqrt(grp["n_cells"]),
                     grp["tgt_score_mean"] + grp["tgt_score_std"] / np.sqrt(grp["n_cells"]),
                     color=META_COLORS[tgt_g], alpha=0.1)

    ax.axhline(0, color="#999", ls=":", lw=0.8)
    ax.set_xticks([1, 2, 3, 4])
    ax.set_xticklabels(["Q1", "Q2", "Q3", "Q4"], fontsize=9)
    ax.set_xlabel("OT commitment quartile", fontsize=9)
    ax.set_title(label, fontsize=10, fontweight="bold")
    ax.legend(fontsize=7, loc="best")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

# Axis A
fig_a, axes_a = plt.subplots(1, 5, figsize=(24, 5), sharey=True)
fig_a.patch.set_facecolor("white")
for i, (key, label, src_g, tgt_g) in enumerate(AXES[0][1]):
    plot_axis(axes_a[i], key, label, src_g, tgt_g, res_df)
    if i == 0:
        axes_a[i].set_ylabel("Module score\n(top 100 DEGs per cluster)", fontsize=10)
fig_a.suptitle("Axis A — Source and target program scores along OT commitment\n"
               "Dashed = source identity | Solid = target program | Shading = SEM",
               fontsize=12, fontweight="bold", y=1.06)
fig_a.tight_layout()
fig_a.savefig(OUT / "SuppFig_module_scores_AxisA.png", dpi=150, bbox_inches="tight")
fig_a.savefig(OUT / "SuppFig_module_scores_AxisA.pdf", dpi=300, bbox_inches="tight")
plt.close()
print("  Axis A saved")

# Axis B
fig_b, axes_b = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
fig_b.patch.set_facecolor("white")
for i, (key, label, src_g, tgt_g) in enumerate(AXES[1][1]):
    plot_axis(axes_b[i], key, label, src_g, tgt_g, res_df)
    if i == 0:
        axes_b[i].set_ylabel("Module score\n(top 100 DEGs per cluster)", fontsize=10)
fig_b.suptitle("Axis B — Source and target program scores along OT commitment\n"
               "Dashed = source identity | Solid = target program | Shading = SEM",
               fontsize=12, fontweight="bold", y=1.06)
fig_b.tight_layout()
fig_b.savefig(OUT / "SuppFig_module_scores_AxisB.png", dpi=150, bbox_inches="tight")
fig_b.savefig(OUT / "SuppFig_module_scores_AxisB.pdf", dpi=300, bbox_inches="tight")
plt.close()
print("  Axis B saved")

# Axis C
fig_c, axes_c = plt.subplots(1, 4, figsize=(20, 5), sharey=True)
fig_c.patch.set_facecolor("white")
for i, (key, label, src_g, tgt_g) in enumerate(AXES[2][1]):
    plot_axis(axes_c[i], key, label, src_g, tgt_g, res_df)
    if i == 0:
        axes_c[i].set_ylabel("Module score\n(top 100 DEGs per cluster)", fontsize=10)
fig_c.suptitle("Axis C — Source and target program scores along OT commitment\n"
               "Dashed = source identity | Solid = target program | Shading = SEM",
               fontsize=12, fontweight="bold", y=1.06)
fig_c.tight_layout()
fig_c.savefig(OUT / "SuppFig_module_scores_AxisC.png", dpi=150, bbox_inches="tight")
fig_c.savefig(OUT / "SuppFig_module_scores_AxisC.pdf", dpi=300, bbox_inches="tight")
plt.close()
print("  Axis C saved")

# Text summary
print(f"\n{'='*70}")
print("SUMMARY — Module score changes Q1→Q4")
print(f"{'='*70}")
print(f"{'Transition':20s}  {'SRC Q1':>7}  {'SRC Q4':>7}  {'SRC Δ':>6}  "
      f"{'TGT Q1':>7}  {'TGT Q4':>7}  {'TGT Δ':>6}  {'Pattern'}")
print("-" * 80)

for key, label, src_g, tgt_g in TRANSITIONS:
    grp = res_df[res_df["transition"] == key]
    if len(grp) == 0:
        continue
    sq1 = grp[grp["quartile"] == "Q1"]["src_score_mean"].values[0]
    sq4 = grp[grp["quartile"] == "Q4"]["src_score_mean"].values[0]
    tq1 = grp[grp["quartile"] == "Q1"]["tgt_score_mean"].values[0]
    tq4 = grp[grp["quartile"] == "Q4"]["tgt_score_mean"].values[0]
    s_dir = "↑" if sq4 > sq1 else "↓"
    t_dir = "↑" if tq4 > tq1 else "↓"

    if s_dir == "↑" and t_dir == "↑":
        pattern = "Layering"
    elif s_dir == "↓" and t_dir == "↑":
        pattern = "Switching/Reconfiguration"
    elif s_dir == "↓" and t_dir == "↓":
        pattern = "Erosion"
    else:
        pattern = "Other"

    print(f"{label:20s}  {sq1:7.3f}  {sq4:7.3f}  {s_dir:>5}  "
          f"{tq1:7.3f}  {tq4:7.3f}  {t_dir:>5}  {pattern}")

elapsed = time.time() - t0
print(f"\nTotal runtime: {elapsed/60:.1f} min")
print(f"Results in {OUT}")
sys.stdout.flush()