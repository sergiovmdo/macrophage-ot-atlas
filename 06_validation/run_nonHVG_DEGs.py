#!/usr/bin/env python
"""Non-HVG DEG analysis for circularity validation. SLURM job."""

import scanpy as sc
import numpy as np
import pandas as pd
from pathlib import Path
import gc

BASE = Path("/storage/homefs/sv24v923/MPI_data/clean_pipeline")
ATLAS = BASE / "macrophage_annotation" / "Macrophage_Atlas_FINAL_v3_CLEAN_ANNOTATED.h5ad"
OUT = BASE / "circularity_validation"
OUT.mkdir(parents=True, exist_ok=True)
CT_COL = "cell_type_meta_v3"

print("Loading atlas...")
adata = sc.read_h5ad(ATLAS)
safe_genes = list(adata.var_names[adata.var["highly_variable_nbatches"] == 0])
print(f"Atlas: {adata.shape}, Safe genes: {len(safe_genes)}")

TRANSITIONS = [
    ("Mono_to_Scav",    "Monocytes",                        "Scavenging / C1q+ Macrophages"),
    ("Mono_to_Res",     "Monocytes",                        "Resident / Quiescent Macrophages"),
    ("Mono_to_Inflam",  "Monocytes",                        "Inflammatory Macrophages"),
    ("Mono_to_Foam",    "Monocytes",                        "Lipid-Stressed / Foam Cells"),
    ("Mono_to_Fibro",   "Monocytes",                        "Fibrotic / Hypoxic Macrophages"),
    ("Scav_to_Inflam",  "Scavenging / C1q+ Macrophages",    "Inflammatory Macrophages"),
    ("Res_to_Inflam",   "Resident / Quiescent Macrophages", "Inflammatory Macrophages"),
    ("Res_to_Fibro",    "Resident / Quiescent Macrophages", "Fibrotic / Hypoxic Macrophages"),
    ("Foam_to_Fibro",   "Lipid-Stressed / Foam Cells",      "Fibrotic / Hypoxic Macrophages"),
    ("Inflam_to_Fibro", "Inflammatory Macrophages",         "Fibrotic / Hypoxic Macrophages"),
    ("Fibro_to_Scav",   "Fibrotic / Hypoxic Macrophages",   "Scavenging / C1q+ Macrophages"),
]

for key, src_name, tgt_name in TRANSITIONS:
    print(f"\n{'='*50}")
    print(f"{key}")

    mask = adata.obs[CT_COL].isin([src_name, tgt_name])
    adata_pair = adata[mask, :][:, safe_genes].copy()
    print(f"  Cells: {adata_pair.n_obs}, Genes: {adata_pair.n_vars}")

    sc.tl.rank_genes_groups(
        adata_pair, groupby=CT_COL, groups=[tgt_name],
        reference=src_name, method="wilcoxon", use_raw=False
    )

    result = sc.get.rank_genes_groups_df(adata_pair, group=tgt_name)
    result = result[result["pvals_adj"] < 0.05].copy()

    tgt_up = result[result["logfoldchanges"] > 0.5].sort_values("logfoldchanges", ascending=False)
    src_up = result[result["logfoldchanges"] < -0.5].sort_values("logfoldchanges", ascending=True)

    print(f"  Target-up: {len(tgt_up)}  |  Source-up: {len(src_up)}")

    # Save full DEG tables
    tgt_up.to_csv(OUT / f"{key}_tgt_up_nonHVG_DEGs.csv", index=False)
    src_up.to_csv(OUT / f"{key}_src_up_nonHVG_DEGs.csv", index=False)

    # Print top named genes
    for label, df in [("TGT", tgt_up), ("SRC", src_up)]:
        named = df[~df["names"].str.startswith("ENSG")]["names"].head(8).tolist()
        print(f"  Top {label}: {named}")

    del adata_pair
    gc.collect()

print(f"\n\nDone — results in {OUT}")