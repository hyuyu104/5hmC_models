#!/usr/bin/env python3
"""
predict_5hmc.py -- predict genetically regulated 5hmC peak levels from genotypes.

Applies the ROSMAP elastic-net (EN) 5hmC models (2,059 peaks, hg38-trained) to
any genotype data set readable by PLINK 2 (PLINK1 bed/bim/fam, PLINK2
pgen/pvar/psam, VCF/BCF, BGEN), in hg38 or hg19 coordinates.

For each peak p and SNP j, with training-set standardization stats (m_j, s_j):

    pred_p = intercept_p + sum_j  w_pj * (x_j - m_j) / s_j

x_j is the dosage (0..2) of the model's effect allele. Missing genotypes are
imputed to the training mean (contribute 0); model SNPs absent from the target
data are dropped (also contribute 0). Internally the standardization is folded
into the weights and scoring is done by `plink2 --score`, so memory use stays
small even for biobank-scale cohorts.

Run `python predict_5hmc.py -h` for options. See README.md for a walk-through.
"""
from __future__ import annotations

import argparse
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

__version__ = "1.0.0"

HERE = Path(__file__).resolve().parent
DEFAULT_WEIGHTS = HERE / "weights"
COMP = {"A": "T", "T": "A", "C": "G", "G": "C"}
N_TRAIN = 871  # ROSMAP training sample size (for the training .afreq OBS_CT)

log = logging.getLogger("predict_5hmc")


# -----------------------------------------------------------------------------
# helpers
# -----------------------------------------------------------------------------
def die(msg: str) -> None:
    log.error(msg)
    sys.exit(1)


def parse_chroms(spec: str) -> list[int]:
    out: list[int] = []
    for tok in spec.split(","):
        tok = tok.strip()
        if "-" in tok:
            a, b = tok.split("-")
            out.extend(range(int(a), int(b) + 1))
        elif tok:
            out.append(int(tok))
    bad = [c for c in out if not 1 <= c <= 22]
    if bad:
        die(f"--chr: only autosomes 1-22 are supported (got {bad})")
    return sorted(set(out))


def run_plink(cmd: list[str], out_prefix: Path, allow_empty: bool = False) -> bool:
    """Run plink2; return False (instead of failing) if no variants survived."""
    log.debug("$ %s", " ".join(map(str, cmd)))
    res = subprocess.run(list(map(str, cmd)), stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True)
    if res.returncode != 0:
        txt = res.stdout
        logf = Path(str(out_prefix) + ".log")
        if logf.exists():
            txt += logf.read_text()
        if allow_empty and re.search(r"No variants remaining|0 variants remaining|"
                                     r"No variants loaded|empty", txt, re.I):
            return False
        die(f"plink2 failed (exit {res.returncode}):\n  {' '.join(map(str, cmd))}\n"
            + "\n".join(txt.strip().splitlines()[-15:]))
    return True


def read_plink_table(path: Path) -> pd.DataFrame:
    """Read a plink2 text output (.pvar/.afreq/.sscore) whose header starts with '#'."""
    with open(path) as fh:
        skip = 0
        for line in fh:
            if line.startswith("##"):
                skip += 1
                continue
            break
    df = pd.read_csv(path, sep="\t", skiprows=skip, dtype=str)
    df.columns = [c.lstrip("#") for c in df.columns]
    return df


# -----------------------------------------------------------------------------
# core steps
# -----------------------------------------------------------------------------
def load_weights(wdir: Path, build: str, peaks_file: str | None):
    peaks = pd.read_csv(wdir / "peaks.tsv", sep="\t")
    w = pd.read_csv(wdir / "weights.tsv.gz", sep="\t",
                    dtype={"effect_allele": str, "other_allele": str})
    if peaks_file:
        want = {l.strip() for l in open(peaks_file) if l.strip()}
        unknown = want - set(peaks.peak_id)
        if unknown:
            log.warning("%d requested peaks are not in the model set (e.g. %s)",
                        len(unknown), sorted(unknown)[0])
        peaks = peaks[peaks.peak_id.isin(want)]
        w = w[w.peak_id.isin(want)]
    pos_col = "pos_hg38" if build == "hg38" else "pos_hg19"
    w = w.assign(pos=w[pos_col])
    n_nopos = int(w.pos.isna().sum())
    if n_nopos:
        log.info("%d SNP weights have no %s position and cannot be used", n_nopos, build)
    w["train_sd"] = w.train_sd.where(np.isfinite(w.train_sd) & (w.train_sd > 0), 1.0)
    return peaks.reset_index(drop=True), w


def make_subset(args, geno: str, chroms: list[int], snps: pd.DataFrame,
                tmp: Path, tag: str):
    """Extract model positions (+/-0) from the input into a small pgen; also --freq."""
    bed = tmp / f"{tag}.ranges.bed"
    s = snps[snps.chr.isin(chroms)].dropna(subset=["pos"])
    if s.empty:
        return None
    rng = s[["chr", "pos"]].drop_duplicates().astype(int)
    rows = []
    # write each chromosome under the common naming variants so that inputs
    # labelled '1', 'chr1' or '01' all match
    for c in rng.chr.unique():
        p = rng.pos[rng.chr == c].to_numpy()
        for lab in {str(c), f"chr{c}", f"{c:02d}"}:
            rows.append(pd.DataFrame({"c": lab, "s": p, "e": p, "n": "r"}))
    pd.concat(rows).to_csv(bed, sep="\t", header=False, index=False)

    out = tmp / tag
    cmd = [args.plink2]
    if args.bfile:
        cmd += ["--bfile", geno]
    elif args.pfile:
        cmd += ["--pfile", geno]
    elif args.vcf:
        cmd += ["--vcf", geno] + ([f"dosage={args.vcf_dosage}"] if args.vcf_dosage else [])
        if args.id_delim:
            cmd += ["--id-delim", args.id_delim]
    elif args.bgen:
        cmd += ["--bgen", geno, args.bgen_ref]
        if args.id_delim:
            cmd += ["--id-delim", args.id_delim]
        if args.sample:
            sample = args.sample.replace("{chr}", str(chroms[0])) if "{chr}" in args.sample \
                else args.sample
            cmd += ["--sample", sample]
    cmd += ["--allow-extra-chr", "--chr"] + [",".join(map(str, chroms))]
    cmd += ["--extract", "bed1", bed, "--max-alleles", "2",
            "--set-all-var-ids", "@:#:$r:$a", "--new-id-max-allele-len", "500", "truncate",
            "--rm-dup", "force-first", "--output-chr", "26",
            "--make-pgen", "--freq", "--out", out,
            "--threads", str(args.threads)]
    if args.memory:
        cmd += ["--memory", str(args.memory)]
    if args.keep:
        cmd += ["--keep", args.keep]
    if args.remove:
        cmd += ["--remove", args.remove]
    if not run_plink(cmd, out, allow_empty=True):
        return None
    return out


def harmonize(snps: pd.DataFrame, pvar: pd.DataFrame, afreq: pd.DataFrame,
              palindromic: str) -> pd.DataFrame:
    """Match model SNPs to target variants by position + allele set."""
    pvar = pvar.rename(columns={"CHROM": "tchr", "POS": "pos", "ID": "ID"})
    pvar["pos"] = pvar.pos.astype(int)
    pvar["REF"] = pvar.REF.str.upper()
    pvar["ALT"] = pvar.ALT.str.upper()
    m = snps.merge(pvar[["pos", "ID", "REF", "ALT"]], on="pos", how="inner")
    if m.empty:
        return m.assign(a1=[], status=[])
    ea, oa, r, a = m.effect_allele, m.other_allele, m.REF, m.ALT
    single = ea.str.len().eq(1) & oa.str.len().eq(1) & ea.isin(list(COMP)) & oa.isin(list(COMP))
    ea_c, oa_c = ea.map(COMP), oa.map(COMP)
    pal = single & (ea_c == oa)
    direct = ((ea == r) & (oa == a)) | ((ea == a) & (oa == r))
    flip = single & ~pal & (((ea_c == r) & (oa_c == a)) | ((ea_c == a) & (oa_c == r)))
    m["status"] = np.select([direct & pal, direct, flip], ["palindromic", "direct", "strand_flip"],
                            default="mismatch")
    if palindromic == "drop":
        m.loc[m.status == "palindromic", "status"] = "palindromic_dropped"
    m = m[m.status.isin(["direct", "strand_flip", "palindromic"])].copy()
    m["a1"] = np.where(m.status == "strand_flip", m.effect_allele.map(COMP), m.effect_allele)
    m = m.drop_duplicates("feature", keep="first")
    # allele-frequency diagnostic (target vs training, effect allele)
    if afreq is not None and not afreq.empty:
        f = afreq.set_index("ID").ALT_FREQS.astype(float)
        alt_f = m.ID.map(f)
        m["target_eaf"] = np.where(m.a1 == m.ALT, alt_f, 1 - alt_f)
    else:
        m["target_eaf"] = np.nan
    m["train_eaf"] = m.train_mean / 2
    return m


def score_chrom(args, pfile: Path, c: int, w_c: pd.DataFrame, used: pd.DataFrame,
                peak_ids: list[str], tmp: Path):
    """Build folded score + training-afreq files and run plink2 --score for one chr."""
    u = used.set_index("feature")
    wu = w_c[w_c.feature.isin(u.index)].copy()
    wu["ID"] = wu.feature.map(u.ID)
    wu["w_eff"] = wu.weight / wu.train_sd
    mat = wu.pivot_table(index="ID", columns="peak_id", values="w_eff",
                         aggfunc="first", fill_value=0.0)
    cols = [p for p in peak_ids if p in mat.columns]
    mat = mat[cols]
    a1 = u.reset_index().drop_duplicates("ID").set_index("ID").a1
    score = mat.copy()
    score.insert(0, "A1", a1.reindex(score.index).values)
    score.index.name = "ID"
    sf = tmp / f"chr{c}.score"
    score.to_csv(sf, sep="\t", float_format="%.10g")

    # training allele frequencies -> plink2 mean-imputes missing dosages to the
    # training mean, exactly as in model training/application
    uu = u.reset_index().drop_duplicates("ID")
    alt_freq = np.where(uu.a1 == uu.ALT, uu.train_eaf, 1 - uu.train_eaf)
    af = pd.DataFrame({"#CHROM": c, "ID": uu.ID, "REF": uu.REF, "ALT": uu.ALT,
                       "ALT_FREQS": np.clip(alt_freq, 0, 1), "OBS_CT": 2 * N_TRAIN})
    aff = tmp / f"chr{c}.train.afreq"
    af.to_csv(aff, sep="\t", index=False, float_format="%.10g")

    idf = tmp / f"chr{c}.ids"
    uu.ID.to_csv(idf, index=False, header=False)

    out = tmp / f"chr{c}.score_out"
    cmd = [args.plink2, "--pfile", pfile, "--chr", str(c), "--extract", idf, "--read-freq", aff,
           "--score", sf, "1", "2", "header-read", "cols=maybefid,scoresums",
           "--score-col-nums", f"3-{len(cols) + 2}", "--out", out,
           "--threads", str(args.threads)]
    if args.memory:
        cmd += ["--memory", str(args.memory)]
    run_plink(cmd, out)
    ss = read_plink_table(Path(str(out) + ".sscore"))
    ids = ss[[c_ for c_ in ("FID", "IID") if c_ in ss.columns]]
    sums = ss[[f"{p}_SUM" for p in cols]].astype(float)
    sums.columns = cols
    # intercept adjustment for the folded standardization (used SNPs only)
    wu["m_over_s"] = wu.weight * wu.train_mean / wu.train_sd
    shift = wu.groupby("peak_id").m_over_s.sum().reindex(cols)
    return ids, sums - shift.values


# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="predict_5hmc.py",
        description="Predict 5hmC peak levels from genotypes with the ROSMAP elastic-net models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    g = p.add_argument_group("genotype input (exactly one; '{chr}' in the path = per-chromosome files)")
    x = g.add_mutually_exclusive_group(required=True)
    x.add_argument("--bfile", help="PLINK1 prefix (.bed/.bim/.fam)")
    x.add_argument("--pfile", help="PLINK2 prefix (.pgen/.pvar/.psam)")
    x.add_argument("--vcf", help="VCF/VCF.gz/BCF-as-VCF file")
    x.add_argument("--bgen", help="BGEN v1.2/1.3 file")
    g.add_argument("--sample", help="BGEN .sample file (may contain '{chr}')")
    g.add_argument("--bgen-ref", default="ref-first", choices=["ref-first", "ref-last", "ref-unknown"],
                   help="BGEN allele order (UK Biobank imputed v3: ref-first)")
    g.add_argument("--vcf-dosage", default=None, metavar="FIELD",
                   help="read dosages from this VCF FORMAT field (e.g. DS, HDS); default: GT hard calls")
    g.add_argument("--id-delim", default=None,
                   help="VCF/BGEN sample IDs are FID<delim>IID (default: FID=IID)")

    o = p.add_argument_group("model / matching options")
    o.add_argument("--build", required=True, choices=["hg38", "hg19"],
                   help="genome build of the INPUT genotypes")
    o.add_argument("--out", default="5hmc", help="output prefix")
    o.add_argument("--weights-dir", default=str(DEFAULT_WEIGHTS), help="folder with peaks.tsv + weights.tsv.gz")
    o.add_argument("--peaks", default=None, help="file with peak IDs to predict (one per line); default all")
    o.add_argument("--chr", default="1-22", help="chromosomes to process, e.g. '1-22' or '19,21'")
    o.add_argument("--palindromic", default="keep", choices=["keep", "drop"],
                   help="A/T and C/G SNPs: keep (assume same strand as model) or drop")
    o.add_argument("--min-snps", type=int, default=2,
                   help="minimum number of model SNPs found in the target data to report a peak")
    o.add_argument("--min-frac", type=float, default=0.0,
                   help="minimum fraction of model SNPs found to report a peak (0-1)")

    r = p.add_argument_group("runtime")
    r.add_argument("--plink2", default="plink2", help="path to the plink2 executable")
    r.add_argument("--keep", default=None, help="plink2 --keep sample file")
    r.add_argument("--remove", default=None, help="plink2 --remove sample file")
    r.add_argument("--threads", type=int, default=1)
    r.add_argument("--memory", type=int, default=None, help="plink2 workspace in MiB")
    r.add_argument("--tmpdir", default=None, help="where to put temporary files")
    r.add_argument("--keep-tmp", action="store_true", help="do not delete temporary files")
    r.add_argument("-v", "--verbose", action="store_true")
    r.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S",
                        handlers=[logging.StreamHandler(),
                                  logging.FileHandler(str(out) + ".log", mode="w")])
    log.info("predict_5hmc %s | build=%s", __version__, args.build)
    if shutil.which(args.plink2) is None:
        die(f"plink2 not found ('{args.plink2}'). Install from https://www.cog-genomics.org/plink/2.0/ "
            "or pass --plink2 /path/to/plink2")
    if args.bgen and not args.sample:
        log.warning("--bgen without --sample: sample IDs must be embedded in the BGEN file")
    if not 0 <= args.min_frac <= 1:
        die("--min-frac must be within [0, 1]")

    geno = args.bfile or args.pfile or args.vcf or args.bgen
    chroms = parse_chroms(args.chr)
    peaks, w = load_weights(Path(args.weights_dir), args.build, args.peaks)
    peaks = peaks[peaks.chr.isin(chroms)].reset_index(drop=True)
    w = w[w.chr.isin(chroms)]
    log.info("Models: %d peaks, %d SNP weights (%d unique SNPs) on chr %s",
             len(peaks), len(w), w.feature.nunique(), args.chr)
    snps = (w.drop_duplicates("feature")
             [["feature", "chr", "pos", "effect_allele", "other_allele", "train_mean", "train_sd"]]
             .dropna(subset=["pos"]).astype({"pos": int}))

    tmp = Path(tempfile.mkdtemp(prefix="p5hmc_", dir=args.tmpdir))
    log.debug("temp dir: %s", tmp)
    try:
        # 1. subset genotypes to model positions
        per_chr = "{chr}" in geno
        subsets: dict[int, Path] = {}
        if per_chr:
            for c in chroms:
                f = geno.replace("{chr}", str(c))
                exists = any(Path(f + ext).exists() for ext in ("", ".bed", ".pgen")) \
                    or Path(f).exists()
                if not exists:
                    log.warning("chr%d: input %s not found -- skipped", c, f)
                    continue
                a = argparse.Namespace(**vars(args))
                if args.sample and "{chr}" in args.sample:
                    a.sample = args.sample.replace("{chr}", str(c))
                s = make_subset(a, f, [c], snps, tmp, f"sub_chr{c}")
                if s is not None:
                    subsets[c] = s
        else:
            s = make_subset(args, geno, chroms, snps, tmp, "sub_all")
            if s is not None:
                subsets = {c: s for c in chroms}

        # 2-3. harmonize + score per chromosome
        all_used, preds, id_frame, chr_seen = [], [], None, set()
        for c in chroms:
            w_c = w[w.chr == c]
            if c not in subsets or w_c.empty:
                continue
            pv = read_plink_table(Path(str(subsets[c]) + ".pvar"))
            pv = pv[pv.CHROM == str(c)]
            if pv.empty:
                continue
            chr_seen.add(c)
            af_path = Path(str(subsets[c]) + ".afreq")
            af = read_plink_table(af_path) if af_path.exists() else None
            used = harmonize(snps[snps.chr == c], pv, af, args.palindromic)
            if used.empty:
                log.info("chr%-2d: 0 model SNPs matched", c)
                continue
            used["chr"] = c
            all_used.append(used)
            # peak coverage on this chromosome
            pk_c = peaks.peak_id[peaks.chr == c].tolist()
            cov = w_c.assign(found=w_c.feature.isin(used.feature)).groupby("peak_id").found
            n_found, n_tot = cov.sum(), cov.size()
            ok = [p for p in pk_c if n_found.get(p, 0) >= args.min_snps
                  and n_found.get(p, 0) / n_tot.get(p, 1) >= args.min_frac]
            log.info("chr%-2d: %5d / %5d model SNPs matched (%d strand-flipped, %d palindromic); "
                     "%d / %d peaks pass coverage filters", c, len(used),
                     int((snps.chr == c).sum()), int((used.status == "strand_flip").sum()),
                     int((used.status == "palindromic").sum()), len(ok), len(pk_c))
            if not ok:
                continue
            ids, pr = score_chrom(args, subsets[c], c, w_c[w_c.peak_id.isin(ok)], used, ok, tmp)
            ic = peaks.set_index("peak_id").intercept.reindex(pr.columns)
            pr = pr + ic.values
            if id_frame is None:
                id_frame = ids
            elif not ids["IID"].equals(id_frame["IID"]):
                pr.index = ids["IID"].values
                pr = pr.reindex(id_frame["IID"].values)
                if pr.isna().all(axis=1).any():
                    log.warning("chr%d: some samples missing relative to the first chromosome", c)
            preds.append(pr.reset_index(drop=True))

        if not preds:
            die("No peak could be predicted -- check --build, chromosome naming and input paths.")

        # 4. outputs
        pred = pd.concat([id_frame.reset_index(drop=True)] + preds, axis=1)
        order = [p for p in peaks.peak_id if p in pred.columns]
        pred = pred[list(id_frame.columns) + order]
        pred.to_csv(str(out) + ".predicted.tsv.gz", sep="\t", index=False, float_format="%.8g")

        used_all = pd.concat(all_used) if all_used else pd.DataFrame(columns=["feature"])
        wf = w.assign(found=w.feature.isin(used_all.feature),
                      absw=w.weight.abs())
        cov = wf.groupby("peak_id").agg(n_snps_model=("feature", "size"),
                                        n_snps_used=("found", "sum"))
        cov["frac_snps_used"] = cov.n_snps_used / cov.n_snps_model
        cov["frac_abs_weight_used"] = (wf[wf.found].groupby("peak_id").absw.sum()
                                       / wf.groupby("peak_id").absw.sum()).reindex(cov.index).fillna(0)
        cov = peaks[["peak_id", "chr", "start", "end", "nearest_gene", "cv_r2"]] \
            .merge(cov, left_on="peak_id", right_index=True, how="left")
        cov["predicted"] = cov.peak_id.isin(order)
        cov.to_csv(str(out) + ".coverage.tsv", sep="\t", index=False, float_format="%.4g")

        if not used_all.empty:
            used_all[["feature", "chr", "pos", "ID", "REF", "ALT", "effect_allele", "a1", "status",
                      "train_eaf", "target_eaf"]] \
                .rename(columns={"pos": f"pos_{args.build}", "ID": "target_id",
                                 "a1": "effect_allele_target_strand"}) \
                .to_csv(str(out) + ".snps.tsv.gz", sep="\t", index=False, float_format="%.4g")
            d = (used_all.target_eaf - used_all.train_eaf).abs()
            n_off = int((d > 0.2).sum())
            if n_off:
                log.warning("%d matched SNPs differ in effect-allele frequency by >0.2 from training "
                            "(see %s.snps.tsv.gz; possible strand/build/ancestry issue)", n_off, out)

        n_exp = int(snps.chr.isin(chr_seen).sum())
        if n_exp and len(used_all) / n_exp < 0.5:
            log.warning("Only %d / %d (%.0f%%) model SNPs were found on the processed chromosomes -- "
                        "check that --build matches the input data (hg38 vs hg19).",
                        len(used_all), n_exp, 100 * len(used_all) / n_exp)
        log.info("Predicted %d / %d peaks for %d samples", len(order), len(peaks), len(pred))
        log.info("Wrote %s.predicted.tsv.gz, %s.coverage.tsv, %s.snps.tsv.gz, %s.log", out, out, out, out)
    finally:
        if args.keep_tmp:
            log.info("temporary files kept in %s", tmp)
        else:
            shutil.rmtree(tmp, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
