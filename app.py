#!/usr/bin/env python3
"""ESCC 3-gene prognostic risk calculator (Streamlit).

Locked to the simplified cross-platform signature described for Figure 5:
Z-score each of ZNF34, LUC7L2 and ZNF710, take the equal-weight mean,
and split at the median (high risk if score >= median).
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import chi2

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
REF = json.loads((DATA / "reference.json").read_text())
EXAMPLE = DATA / "example_GSE53625_3gene.csv"
GENES = list(REF["signature_genes"])
REF_MEAN = {g: float(REF["genes"][g]["mean"]) for g in GENES}
REF_SD = {g: float(REF["genes"][g]["sd"]) for g in GENES}
REF_CUTOFF = float(REF["score_median_cutoff"])

CITATION = (
    "Chu F, Bai B, Wang Z, Zhao K, Wang J, Qu J. A Privacy-Preserving "
    "Generative AI Framework for Prognostic Risk Stratification in "
    "Esophageal Squamous Cell Carcinoma. Computational and Structural "
    "Biotechnology Journal. Manuscript under revision."
)


def zscore_series(s: pd.Series, mean: float | None = None, sd: float | None = None) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    mu = float(s.mean()) if mean is None else mean
    sigma = float(s.std(ddof=0)) if sd is None else sd
    if sigma == 0 or not np.isfinite(sigma):
        return s * np.nan
    return (s - mu) / sigma


def score_with_reference(df: pd.DataFrame) -> pd.Series:
    z = pd.DataFrame({g: zscore_series(df[g], REF_MEAN[g], REF_SD[g]) for g in GENES})
    return z[GENES].mean(axis=1)


def score_within_cohort(df: pd.DataFrame) -> tuple[pd.Series, float]:
    z = pd.DataFrame({g: zscore_series(df[g]) for g in GENES})
    score = z[GENES].mean(axis=1)
    cutoff = float(score.median())
    return score, cutoff


def classify(score: pd.Series, cutoff: float) -> pd.Series:
    out = pd.Series(np.where(score >= cutoff, "High risk", "Low risk"), index=score.index)
    out[score.isna()] = np.nan
    return out


def logrank_p(t1, t0, e1, e0) -> float:
    t1 = np.asarray(t1, dtype=float)
    e1 = np.asarray(e1, dtype=int)
    t0 = np.asarray(t0, dtype=float)
    e0 = np.asarray(e0, dtype=int)
    times = np.unique(np.concatenate([t1[e1 == 1], t0[e0 == 1]]))
    o_e = 0.0
    var = 0.0
    for t in times:
        n1 = np.sum(t1 >= t)
        n0 = np.sum(t0 >= t)
        n = n1 + n0
        if n <= 1:
            continue
        d1 = np.sum((t1 == t) & (e1 == 1))
        d0 = np.sum((t0 == t) & (e0 == 1))
        d = d1 + d0
        if d == 0:
            continue
        ev = d * n1 / n
        o_e += d1 - ev
        var += d * (n - d) / (n - 1) * (n1 / n) * (n0 / n)
    if var <= 0:
        return float("nan")
    return float(chi2.sf(o_e**2 / var, 1))


def km_curve(time, event):
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=int)
    order = np.argsort(time)
    time, event = time[order], event[order]
    ts = [0.0]
    ss = [1.0]
    s = 1.0
    for t in np.unique(time):
        at_risk = np.sum(time >= t)
        d = np.sum((time == t) & (event == 1))
        if at_risk > 0 and d > 0:
            s *= 1.0 - d / at_risk
        ts.append(float(t))
        ss.append(float(s))
    return np.array(ts), np.array(ss)


def plot_km(time, event, group, p_value: float):
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    colors = {"High risk": "#d62728", "Low risk": "#2ca02c"}
    for label in ("High risk", "Low risk"):
        mask = group == label
        if mask.sum() < 2:
            continue
        t, s = km_curve(time[mask], event[mask])
        ax.step(t, s, where="post", color=colors[label], lw=2, label=f"{label} (n={int(mask.sum())})")
        ax.plot(t, s, linestyle="None", color=colors[label])
    ax.set_xlabel("Survival time (months)")
    ax.set_ylabel("Overall survival")
    ax.set_ylim(0, 1.02)
    ax.set_xlim(left=0)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    if np.isfinite(p_value):
        ax.text(0.55, 0.22, f"Log-rank P = {p_value:.3f}", transform=ax.transAxes)
    fig.tight_layout()
    return fig


def plot_reference_hist(user_score: float | None, scores: pd.Series | None = None):
    if scores is None:
        ex = pd.read_csv(EXAMPLE)
        scores = score_with_reference(ex)
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    ax.hist(scores.dropna(), bins=20, color="#9ecae1", edgecolor="white")
    ax.axvline(REF_CUTOFF, color="#333", ls="--", lw=1.5, label=f"Locked cutoff ({REF_CUTOFF:.4f})")
    if user_score is not None and np.isfinite(user_score):
        ax.axvline(user_score, color="#d62728", lw=2, label=f"This sample ({user_score:.3f})")
    ax.set_xlabel("3-gene risk score")
    ax.set_ylabel("Patients (GSE53625, n=179)")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return fig


def find_column(df: pd.DataFrame, options: list[str]):
    lower = {c.lower().strip(): c for c in df.columns}
    for name in options:
        if name.lower() in lower:
            return lower[name.lower()]
    return None


st.set_page_config(page_title="ESCC 3-Gene Risk Calculator", layout="centered")
st.title("ESCC 3-Gene Prognostic Risk Calculator")
st.caption(
    "Simplified cross-platform signature (ZNF34, LUC7L2, ZNF710) from a "
    "privacy-preserving generative AI framework for esophageal squamous cell carcinoma."
)
st.info(
    "**Official formula (locked to the paper).** Each gene is converted to a Z-score. "
    "The risk score is the equal-weight mean of the three Z-scores. "
    "Patients with a score **≥ the median** are classified as high risk. "
    "The threshold is not user-adjustable. For research use only — not a medical device."
)

tab_single, tab_batch, tab_methods = st.tabs(["Single sample", "Batch / example", "Methods"])

with tab_single:
    st.write(
        "Enter quantile-normalized log2 intensities for the three genes "
        "(the same scale as the GSE53625 microarray). "
        "The app applies the locked GSE53625 reference mean and standard deviation."
    )
    c1, c2, c3 = st.columns(3)
    with c1:
        v_znf34 = st.number_input("ZNF34", value=float("nan"), format="%.6f", key="s_znf34")
    with c2:
        v_luc = st.number_input("LUC7L2", value=float("nan"), format="%.6f", key="s_luc")
    with c3:
        v_znf710 = st.number_input("ZNF710", value=float("nan"), format="%.6f", key="s_znf710")

    already_z = st.checkbox("My values are already Z-scored within my own cohort")

    if st.button("Calculate risk", type="primary"):
        vals = {"ZNF34": v_znf34, "LUC7L2": v_luc, "ZNF710": v_znf710}
        if not all(np.isfinite(v) for v in vals.values()):
            st.error("Please enter all three gene values.")
        else:
            if already_z:
                score = float(np.mean(list(vals.values())))
            else:
                z = {g: (vals[g] - REF_MEAN[g]) / REF_SD[g] for g in GENES}
                score = float(np.mean(list(z.values())))
            label = "High risk" if score >= REF_CUTOFF else "Low risk"
            if label == "High risk":
                st.error(f"**{label}**  ·  score = {score:.4f}  ·  cutoff = {REF_CUTOFF:.4f}")
            else:
                st.success(f"**{label}**  ·  score = {score:.4f}  ·  cutoff = {REF_CUTOFF:.4f}")
            st.pyplot(plot_reference_hist(score), clear_figure=True)
            if already_z:
                st.caption(
                    "Because the inputs were already Z-scored, the score is their unweighted mean, "
                    "compared with the GSE53625 median. For a new cohort of several patients, "
                    "use the Batch tab and the paper method instead."
                )

with tab_batch:
    st.write(
        "Upload a CSV with columns **ZNF34**, **LUC7L2** and **ZNF710**. "
        "Optional columns: **sample_id**, **os_months**, **death** (1 = death, 0 = censored). "
        "You can also load the public GSE53625 example (n=179 tumors) to run the same formula used in the paper."
    )
    mode = st.radio(
        "Scoring mode",
        [
            "Paper method: Z-score and median within this file (recommended for a new cohort)",
            "Apply locked GSE53625 reference mean, SD and cutoff (for a single sample or direct comparison)",
        ],
        index=0,
    )
    use_example = st.button("Load GSE53625 example (n=179)")
    uploaded = st.file_uploader("Or upload a CSV", type=["csv"])

    df = None
    if use_example:
        df = pd.read_csv(EXAMPLE)
        st.session_state["batch_df"] = df
        st.session_state["batch_source"] = "example"
    elif uploaded is not None:
        df = pd.read_csv(uploaded)
        st.session_state["batch_df"] = df
        st.session_state["batch_source"] = "upload"
    elif "batch_df" in st.session_state:
        df = st.session_state["batch_df"]

    if df is not None:
        colmap = {}
        for g in GENES:
            c = find_column(df, [g])
            if c is None:
                st.error(f"Missing required column: {g}")
                st.stop()
            colmap[g] = c
        work = pd.DataFrame({g: pd.to_numeric(df[colmap[g]], errors="coerce") for g in GENES})
        sid = find_column(df, ["sample_id", "geo", "id", "sample", "patient"])
        work["sample_id"] = df[sid].astype(str) if sid else np.arange(1, len(df) + 1).astype(str)
        tcol = find_column(df, ["os_months", "time", "os", "survival", "survival_months"])
        ecol = find_column(df, ["death", "event", "status", "os_event"])
        if tcol:
            work["os_months"] = pd.to_numeric(df[tcol], errors="coerce")
        if ecol:
            raw = df[ecol]
            if raw.dtype == object:
                work["death"] = raw.astype(str).str.lower().isin(["1", "yes", "dead", "deceased", "event", "true"]).astype(int)
                # numeric strings already covered; also keep numeric
            try:
                work["death"] = pd.to_numeric(df[ecol], errors="coerce")
            except Exception:
                pass

        paper_mode = mode.startswith("Paper method")
        if paper_mode:
            if len(work) < 2:
                st.error("Within-cohort Z-score needs at least two samples. Use the locked reference mode or the Single sample tab.")
                st.stop()
            work["risk_score"], cutoff = score_within_cohort(work)
        else:
            work["risk_score"] = score_with_reference(work)
            cutoff = REF_CUTOFF
        work["risk_group"] = classify(work["risk_score"], cutoff)
        work["cutoff_used"] = cutoff

        n_high = int((work["risk_group"] == "High risk").sum())
        n_low = int((work["risk_group"] == "Low risk").sum())
        st.write(f"n = {len(work)}  ·  High risk = {n_high}  ·  Low risk = {n_low}  ·  cutoff = {cutoff:.4f}")
        st.dataframe(work, use_container_width=True, hide_index=True)

        st.download_button(
            "Download results CSV",
            work.to_csv(index=False).encode("utf-8"),
            file_name="ESCC_3gene_risk_results.csv",
            mime="text/csv",
        )

        if {"os_months", "death"}.issubset(work.columns):
            sub = work.dropna(subset=["os_months", "death", "risk_group"])
            high = sub["risk_group"] == "High risk"
            low = sub["risk_group"] == "Low risk"
            if high.sum() >= 2 and low.sum() >= 2:
                p = logrank_p(
                    sub.loc[high, "os_months"],
                    sub.loc[low, "os_months"],
                    sub.loc[high, "death"],
                    sub.loc[low, "death"],
                )
                st.pyplot(plot_km(sub["os_months"].values, sub["death"].values, sub["risk_group"].values, p), clear_figure=True)
                if st.session_state.get("batch_source") == "example":
                    st.caption(
                        "Public GSE53625 tumor samples (GSE53624 + GSE53622, n=179), scored with the paper's "
                        "stated formula. Compare the curve shape with Figure 5 of the manuscript."
                    )

        st.pyplot(plot_reference_hist(None, work["risk_score"]), clear_figure=True)

with tab_methods:
    st.markdown(
        f"""
**Signature.** Conserved protein-coding core drivers shared by RNA-seq and the
Agilent-038314 microarray: **ZNF34**, **LUC7L2**, **ZNF710**.
Platform-specific lncRNAs without microarray probes (including AC007128.1) are not used.

**Score.** For gene \\(g\\) in a cohort,

\\[z_g = (x_g - \\bar{{x}}_g) / s_g, \\quad \\text{{score}} = (z_{{\\mathrm{{ZNF34}}}} + z_{{\\mathrm{{LUC7L2}}}} + z_{{\\mathrm{{ZNF710}}}})/3\\]

where \\(s_g\\) is the population standard deviation (denominator \\(n\\), not \\(n-1\\)).
High risk if score \\(\\ge\\) the cohort median. This matches the external-validation
procedure in the manuscript. Gain-weighted scores are not used for official classification.

**Single-sample reference.** When only one (or a few) samples are scored against
the locked cutoff, \\(\\bar{{x}}_g\\) and \\(s_g\\) are taken from the GSE53625 SuperSeries
tumor set (n=179; GSE53624 n=119 + GSE53622 n=60):

| Gene | Reference mean | Reference SD |
| --- | --- | --- |
| ZNF34 | {REF_MEAN['ZNF34']:.8f} | {REF_SD['ZNF34']:.8f} |
| LUC7L2 | {REF_MEAN['LUC7L2']:.8f} | {REF_SD['LUC7L2']:.8f} |
| ZNF710 | {REF_MEAN['ZNF710']:.8f} | {REF_SD['ZNF710']:.8f} |

Locked cutoff (median score on this reference cohort): **{REF_CUTOFF:.8f}**.
Input values should be quantile-normalized log2 intensities, as released in the GEO series matrices.

**Data access (official sources, not redistributed here).**
- Training RNA-seq: [TCGA-ESCC via UCSC Xena](https://xenabrowser.net/)
- External validation: [GEO GSE53625](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE53625) (SuperSeries of [GSE53624](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE53624) and [GSE53622](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE53622)), platform [GPL18109](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GPL18109)

This calculator implements only the downstream 3-gene signature. It does not train or host the PCA–CTGAN model, synthetic transcripts, or full expression matrices.

**Citation.** {CITATION}

**Disclaimer.** For research use only. This tool is not a medical device and is not intended for diagnosis, treatment, or clinical decision-making.
"""
    )

st.markdown("---")
st.caption("For research use only. Not for clinical use.")
