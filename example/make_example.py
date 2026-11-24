#!/usr/bin/env python3
"""Simulate a small SYNTHETIC genotype file (no real individuals) for testing.

Genotypes at the chr22 model SNPs are drawn independently from the ROSMAP
training effect-allele frequencies (train_mean / 2), with ~2% missing calls.
Writes example/example_chr22.vcf.gz (hg38).
"""
import gzip
from pathlib import Path

import numpy as np
import pandas as pd

here = Path(__file__).resolve().parent
w = pd.read_csv(here.parent / "weights" / "weights.tsv.gz", sep="\t")
s = (w[w.chr == 22].drop_duplicates("feature").sort_values("pos_hg38").reset_index(drop=True))
rng = np.random.default_rng(2026)
n = 50
ids = [f"SIM{i:03d}" for i in range(1, n + 1)]
with gzip.open(here / "example_chr22.vcf.gz", "wt") as fh:
    fh.write("##fileformat=VCFv4.2\n##source=predict_5hmc_synthetic_example\n"
             "##contig=<ID=chr22>\n"
             '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n')
    fh.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t" + "\t".join(ids) + "\n")
    for r in s.itertuples():
        # put the effect allele as REF or ALT at random to exercise harmonization
        ea_is_alt = rng.random() < 0.5
        ref, alt = (r.other_allele, r.effect_allele) if ea_is_alt else (r.effect_allele, r.other_allele)
        p = min(max(r.train_mean / 2, 0.01), 0.99)
        h = rng.random((n, 2)) < p                     # True = effect allele
        a = h if ea_is_alt else ~h                     # True = ALT allele
        gt = [f"{int(x)}/{int(y)}" for x, y in a]
        gt = ["./." if rng.random() < 0.02 else g for g in gt]
        fh.write(f"chr22\t{r.pos_hg38}\t.\t{ref}\t{alt}\t.\tPASS\t.\tGT\t" + "\t".join(gt) + "\n")
print(f"wrote {here / 'example_chr22.vcf.gz'}: {len(s)} SNPs x {n} samples")
