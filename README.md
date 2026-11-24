# Genetic prediction models for 5hmC

5-hydroxymethylcytosine (5hmC) is an epigenetic mark on DNA, often referred as the sixth DNA base. Since direct 5hmC profiling is costly, we train elastic models to predict 5hmC levels using genotype data. The models were trained on 871 ROSMAP participants, who have both genotype data and brain 5hmC measurements.

This repository contains the elastic net model weights for 2,059 brain 5hmC peaks. We also provide a command line interface so that users can input genotypes in GRCh38/hg38 and get the corresponding 5hmC predictions.

---

## Quick start

```bash
git clone <this-repo-url> 5hmC_models && cd 5hmC_models
pip install -r requirements.txt          # numpy, pandas (Python >= 3.8)
# plink2 must be on PATH (or pass --plink2): https://www.cog-genomics.org/plink/2.0/

# try it on the bundled synthetic data (50 simulated individuals, chr22)
python predict_5hmc.py --vcf example/example_chr22.vcf.gz --build hg38 --chr 22 --out demo/example

# predict for your cohort
python predict_5hmc.py --bfile mycohort --build hg38 --out results/mycohort
```

## How to use

### 1. Inputs

You need any genotype format that **PLINK 2** can read, in **hg38 or hg19**.
Tell the tool which build with `--build`. Chromosomes can be labelled `1`, `chr1` or `01`.

| Input | Flag | Notes |
|---|---|---|
| PLINK1 binary | `--bfile PREFIX` | |
| PLINK2 binary | `--pfile PREFIX` | |
| VCF / VCF.gz | `--vcf FILE` | hard calls (`GT`) by default; add `--vcf-dosage DS` to use imputed dosages |
| BGEN | `--bgen FILE --sample FILE` | `--bgen-ref ref-first` (default) |

If your data are split by chromosome, put `{chr}` in the path, e.g.
`--vcf 'imputed/chr{chr}.dose.vcf.gz'` or `--bgen 'ukb_imp_chr{chr}_v3.bgen' --sample 'ukb_chr{chr}.sample'`.
Quote the path so your shell leaves the braces alone.

### 2. Examples

```bash
# hg38 PLINK files, all chromosomes, 8 threads
python predict_5hmc.py --bfile data/cohort --build hg38 --threads 8 --out out/cohort

# imputed VCFs split by chromosome, using dosages, subset of samples
python predict_5hmc.py --vcf 'imp/chr{chr}.dose.vcf.gz' --vcf-dosage DS \
    --keep samples.txt --build hg38 --out out/cohort

# UK Biobank imputed v3 (hg19 BGEN)
python predict_5hmc.py --bgen 'ukb_imp_chr{chr}_v3.bgen' --sample 'ukb_imp_chr{chr}.sample' \
    --build hg19 --threads 16 --memory 32000 --out out/ukb

# only some peaks / chromosomes; require >= 80% of each model's SNPs
python predict_5hmc.py --pfile data/cohort --build hg38 --peaks my_peaks.txt --chr 19 \
    --min-frac 0.8 --out out/apoe_region
```

Run `python predict_5hmc.py -h` to see every option.

### 3. Outputs

| File | Content |
|---|---|
| `<out>.predicted.tsv.gz` | Predicted 5hmC: one row per sample (`FID` if present, `IID`), one column per peak |
| `<out>.coverage.tsv` | Per peak: location, nearest gene, CV-R², `n_snps_model`, `n_snps_used`, `frac_snps_used`, `frac_abs_weight_used`, `predicted` |
| `<out>.snps.tsv.gz` | Every matched model SNP: target variant, alleles, match `status` (`direct` / `strand_flip` / `palindromic`), effect-allele frequency in training vs. your data |
| `<out>.log` | Run log |

### 4. What the tool does

1. **Extract**: uses plink2 to pull only the model SNP positions (about 72k) out of your data.
2. **Harmonize**: matches each model SNP by position and by its unordered allele pair
   (effect allele, other allele). It recovers strand-flipped SNPs.
   - Ambiguous A/T and C/G SNPs are assumed to be on the model strand (`--palindromic keep`, as in the paper).
     Use `--palindromic drop` to exclude them.
   - Multi-allelic sites and allele mismatches are dropped.
3. **Score**: for each peak *p*:

   &nbsp;&nbsp;&nbsp;&nbsp;**pred<sub>p</sub> = intercept<sub>p</sub> + Σ<sub>j</sub> w<sub>pj</sub> · (x<sub>j</sub> − m<sub>j</sub>) / s<sub>j</sub>**

   - *x<sub>j</sub>* is the dosage (0–2) of the effect allele.
   - *m<sub>j</sub>* and *s<sub>j</sub>* are the training mean and SD.
   - Missing genotypes and model SNPs absent from your data contribute 0, the same as imputing the training mean.
   - Computed with `plink2 --score`, so memory use stays low even for biobank-scale data.
4. **Filter**: reports a peak only if at least `--min-snps` (default 2) of its SNPs were found,
   and at least a `--min-frac` (default 0) fraction of them.

### 5. Interpreting predictions

- **Scale.** Predictions are on the scale of covariate-residualized 5hmC peak intensity.
  The residualization was against sex, education, age at death and PMI, within ROS and MAP.
  Only relative differences between individuals are meaningful.
  Standardize each peak in your cohort before an association test.
- **Accuracy.** Expected accuracy is the training CV-R² (`cv_r2` in `weights/peaks.tsv`;
  range 0.10–0.78, median 0.17). Expect lower accuracy in cohorts that differ from ROSMAP.
- **Coverage.** Check `<out>.coverage.tsv`. Peaks that lose a large part of their weight
  (`frac_abs_weight_used` well below 1) are predicted less accurately.
  The paper used `--min-snps 2` with no fraction filter.
- **Ancestry.** ROSMAP is mostly of European ancestry, and accuracy in other ancestries is
  not established.
  The log warns if many SNPs have very different allele frequencies from training
  (more than 0.2 apart), which can point to a strand, build or ancestry problem.
- **Covariates.** In association tests, adjust for the usual covariates
  (genotype PCs, age, sex, batch).

---

## Model files (`weights/`)

**`peaks.tsv`**: one row per peak (2,059 rows)

| Column | Description |
|---|---|
| `peak_id` | Peak ID `C<chr>P<index>` (the index is sequential, not a position) |
| `chr`, `start`, `end` | Peak location, hg38 (BED-style, 0-based start) |
| `nearest_gene`, `nearest_gene_id` | Nearest gene annotation (GENCODE) |
| `cv_r2` | 5-fold out-of-fold CV-R² in ROSMAP (selection threshold ≥ 0.10) |
| `n_snps` | Number of SNPs with non-zero weight |
| `intercept` | Model intercept |
| `alpha`, `lambda` | Elastic-net mixing parameter (0.5) and selected λ (λ.min) |

**`weights.tsv.gz`**: one row per peak × SNP (78,714 rows; 71,899 unique SNPs)

| Column | Description |
|---|---|
| `peak_id` | Peak ID |
| `feature` | SNP ID used in training, `CHR:POS_EFFECTALLELE` (hg38) |
| `chr`, `pos_hg38`, `pos_hg19` | Position. `pos_hg19` comes from UCSC liftOver; it is empty for the 52 SNPs that do not map 1:1 |
| `effect_allele`, `other_allele` | Effect (counted) allele and the other allele |
| `weight` | Elastic-net coefficient on the standardized dosage |
| `train_mean`, `train_sd` | Training mean and SD of the effect-allele dosage (the training effect-allele frequency is `train_mean / 2`) |

To apply the models without the CLI (for example in R), use the formula in step 4.3
directly with these columns.

## Requirements

- Python ≥ 3.8 with `numpy` and `pandas`
- [PLINK 2.0](https://www.cog-genomics.org/plink/2.0/) (tested with v2.0.0-a.6.13)