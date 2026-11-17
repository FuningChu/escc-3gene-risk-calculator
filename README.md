# ESCC 3-Gene Prognostic Risk Calculator

Streamlit tool for the simplified cross-platform signature (ZNF34, LUC7L2, ZNF710) described in:

Chu F, Bai B, Wang Z, Zhao K, Wang J, Qu J. A Privacy-Preserving Generative AI Framework for Prognostic Risk Stratification in Esophageal Squamous Cell Carcinoma. *Computational and Structural Biotechnology Journal* (manuscript under revision).

This app is the public web server for reviewers and readers. It implements only the downstream 3-gene risk score. It does not host full TCGA/GEO matrices, CTGAN weights, or synthetic samples. Official source data remain at [UCSC Xena](https://xenabrowser.net/) and [GEO GSE53625](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE53625).

## Official formula (locked)

1. Z-score each of the three genes (population SD, divisor *n*).
2. Risk score = equal-weight mean of the three Z-scores.
3. High risk if score ≥ the median; otherwise low risk.

For a new cohort (batch CSV), Z-scores and the median are computed within that file, as in the paper. For a single sample, the app applies the locked GSE53625 reference mean, SD, and median cutoff (n=179 tumors: GSE53624 + GSE53622).

The example CSV (`data/example_GSE53625_3gene.csv`) contains only the three gene-level values plus public overall-survival time and event, derived from the GEO series matrices, so reviewers can regenerate a Kaplan–Meier curve with one click.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Put this folder in a **public** GitHub repository.
2. At [share.streamlit.io](https://share.streamlit.io), deploy `app.py`.
3. In the app settings, set visibility to **Public**.
4. Open the URL in a private/incognito window (not signed in) to confirm that reviewers can use it without logging in.
5. In the manuscript Availability section, give this public URL, and keep the GitHub repository public for the source.

## Input format

Required CSV columns: `ZNF34`, `LUC7L2`, `ZNF710` (quantile-normalized log2 intensities).

Optional: `sample_id`, `os_months`, `death` (1 = death, 0 = censored).

## Research use only

This tool is not a medical device and is not intended for diagnosis, treatment, or clinical decision-making.
