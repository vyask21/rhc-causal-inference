"""
RHC Causal Inference — Streamlit App
Part 2B Stage 1: Interactive visualization of propensity scores and population results.
"""
import json, os
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import streamlit as st

PROJECT = os.environ.get("RHC_PROJECT", "/home/node/.openclaw/projects/rhc-causal-inference")

# ── Load data ──
@st.cache_resource
def load_resources():
    with open(os.path.join(PROJECT, "results.json")) as f:
        results = json.load(f)
    prop_model = joblib.load(os.path.join(PROJECT, "models", "propensity_model.joblib"))
    ps_scores = pd.read_csv(os.path.join(PROJECT, "data", "interim", "propensity_scores.csv"))
    analysis_pq = pd.read_parquet(os.path.join(PROJECT, "data", "interim", "analysis.parquet"))
    smds = pd.read_csv(os.path.join(PROJECT, "data", "interim", "smds.csv"))
    return results, prop_model, ps_scores, analysis_pq, smds

results, prop_model, ps_scores, analysis_pq, smds = load_resources()

# Coerce bool columns for consistency
for c in analysis_pq.select_dtypes(include=["bool"]).columns:
    analysis_pq[c] = analysis_pq[c].astype(float)

# Feature column names for the propensity model (66 features, same order the model was fit on)
confounder_cols = [c for c in analysis_pq.columns if c not in ("treatment", "dth30")]

def _medians_modes():
    """Compute column-wise medians (numeric) and modes (bool/categorical) from analysis data."""
    defaults = {}
    for c in confounder_cols:
        vals = analysis_pq[c]
        if vals.dtype == "object":
            defaults[c] = vals.mode().iloc[0] if len(vals.mode()) > 0 else vals.iloc[0]
        else:
            defaults[c] = float(vals.median())
    return defaults

BASELINE = _medians_modes()

st.set_page_config(page_title="RHC Causal Inference", page_icon="🫀", layout="wide")

# ──────────────────────────────────────────────────────────────
# TITLE
# ──────────────────────────────────────────────────────────────
st.title("Right Heart Catheterization: Causal Effects on 30-Day Mortality")
st.caption("Based on the SUPPORT study, Connors et al. 1996 (JAMA 276:889-897)")

# ──────────────────────────────────────────────────────────────
# SECTION 1 — Population-level results (static)
# ──────────────────────────────────────────────────────────────
st.header("Population-Level Results")
st.info(
    "These estimates represent the average treatment effect across all 5735 patients in the dataset. "
    "They are NOT specific to any individual patient."
)

col_a, col_b = st.columns([1, 1])

with col_a:
    estimators = [
        ("Naive (unadjusted)", results["naive_estimate"], results["naive_ci"][0], results["naive_ci"][1]),
        ("IPW (stabilized)", results["ipw_estimate"], results["ipw_ci"][0], results["ipw_ci"][1]),
        ("AIPW (doubly robust)", results["aipw_estimate"], results["aipw_ci"][0], results["aipw_ci"][1]),
    ]
    rows = []
    for name, est, lo, hi in estimators:
        rows.append({
            "Method": name,
            "ATE on mortality": f"{est:+.4f}",
            "95% CI": f"[{lo:+.4f}, {hi:+.4f}]",
        })
    st.table(pd.DataFrame(rows))
    st.caption("Positive ATE means RHC is associated with higher 30-day mortality.")

with col_b:
    # Forest plot
    fig, ax = plt.subplots(figsize=(7, 3))
    colors = ["steelblue", "forestgreen", "darkred"]
    for i, (name, est, lo, hi) in enumerate(estimators):
        ax.errorbar(est, i,
                    xerr=[[est - lo], [hi - est]],
                    fmt="o", color=colors[i], capsize=5, markersize=8, label=name)
    ax.axvline(0, color="gray", linestyle="-", alpha=0.4)
    ax.set_yticks(range(3))
    ax.set_yticklabels([r[0] for r in estimators])
    ax.set_xlabel("Effect on 30-day mortality (ATE)")
    ax.set_title("Causal Effect of RHC on 30-Day Mortality")
    ax.invert_yaxis()
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    st.pyplot(fig)

# Benchmarks
st.subheader("Comparison to Published Estimates")
bench_rows = [
    {"Method": "AIPW (this analysis)", "Effect on mortality": f"+{results['aipw_estimate']:.4f}", "Source": "This analysis"},
    {"Method": "Hirano & Imbens (2001)", "Effect on mortality": "+0.053 to +0.062", "Source": "Converted from survival scale"},
    {"Method": "Crump et al. (2009)", "Effect on mortality": "+0.059 to +0.060", "Source": "Converted from survival scale"},
]
st.table(pd.DataFrame(bench_rows))
st.caption(
    "Published estimates were reported on a survival scale (negative = RHC harmful). "
    "Signs flipped here for comparison on a mortality scale (positive = RHC harmful)."
)

# E-value
ev = results["e_value_details"]
st.subheader("E-Value Sensitivity")
st.write(
    f"E-value (point): **{results['e_value_point']:.2f}**. "
    f"An unmeasured confounder would need a risk ratio of at least {results['e_value_point']:.2f} "
    f"with both treatment assignment and outcome, conditional on measured covariates, "
    f"to explain away the observed effect."
)

# ──────────────────────────────────────────────────────────────
# SECTION 2 — Explore propensity and overlap (interactive)
# ──────────────────────────────────────────────────────────────
st.header("Explore Propensity and Overlap")
st.info(
    "Adjust the sliders to define a hypothetical patient. "
    "The app computes their predicted propensity score (probability of receiving RHC) "
    "and shows where it falls on the actual overlap distribution. "
    "This is NOT a personalized treatment effect estimate."
)

# Determine top covariates by |SMD| for the sliders (top 8)
top_vars = smds.nlargest(8, "smd_before")["variable"].tolist()

# Map user-friendly names to model feature names, with value ranges and defaults
slider_specs = [
    # (feature_name, display_name, min, max, default, step, type)
    ("aps1", "APACHE severity score", 0.0, 80.0, BASELINE.get("aps1", 50.0), 1.0, "float"),
    ("meanbp1", "Mean blood pressure (mmHg)", 10.0, 180.0, BASELINE.get("meanbp1", 70.0), 1.0, "float"),
    ("pafi1", "PaO2/FIO2 ratio", 10.0, 600.0, BASELINE.get("pafi1", 200.0), 5.0, "float"),
    ("crea1", "Creatinine (mg/dL)", 0.0, 10.0, BASELINE.get("crea1", 1.5), 0.1, "float"),
    ("wtkilo1", "Weight (kg)", 20.0, 200.0, BASELINE.get("wtkilo1", 70.0), 1.0, "float"),
    ("cat1_MOSF w/Sepsis", "Primary dx: MOSF w/Sepsis", False, True, BASELINE.get("cat1_MOSF w/Sepsis", False), None, "bool"),
    ("neuro_Yes", "Neurological diagnosis", False, True, BASELINE.get("neuro_Yes", False), None, "bool"),
    ("card_Yes", "Cardiovascular diagnosis", False, True, BASELINE.get("card_Yes", False), None, "bool"),
]

col_sl1, col_sl2 = st.columns(2)

user_features = dict(BASELINE)  # start with baseline values

with col_sl1:
    for feat, label, lo, hi, default, step, dtype in slider_specs[:6]:
        if dtype == "bool":
            val = st.toggle(label, value=bool(default), key=feat)
            user_features[feat] = val
        else:
            val = st.slider(label, min_value=float(lo), max_value=float(hi),
                           value=float(default), step=float(step), key=feat)
            user_features[feat] = val

with col_sl2:
    for feat, label, lo, hi, default, step, dtype in slider_specs[6:]:
        if dtype == "bool":
            val = st.toggle(label, value=bool(default), key=feat)
            user_features[feat] = val
        else:
            val = st.slider(label, min_value=float(lo), max_value=float(hi),
                           value=float(default), step=float(step), key=feat)
            user_features[feat] = val

user_features["age2"] = user_features.get("age", BASELINE.get("age", 60.0)) ** 2

# Build feature vector in the same order as the model
feature_order = [c for c in confounder_cols if c != "age2"]
if "age2" in [c for c in confounder_cols]:
    feature_order.append("age2")
elif "age" in confounder_cols:
    age_idx = [i for i, c in enumerate(confounder_cols) if c == "age"]
    if age_idx:
        feature_order.insert(age_idx[0] + 1, "age2")

# Create a DataFrame with all features
feat_vec = []
feat_names_ordered = []
for c in confounder_cols:
    if c == "age2":
        feat_names_ordered.append(c)
        feat_vec.append(float(user_features.get("age2", 60.0 ** 2)))
    else:
        feat_names_ordered.append(c)
        val = user_features.get(c, BASELINE.get(c, 0))
        feat_vec.append(float(val) if not isinstance(val, bool) else (1.0 if val else 0.0))

X_pred = np.array(feat_vec).reshape(1, -1)

# Predict propensity for hypothetical patient
try:
    ps_hyp = prop_model.predict_proba(X_pred)[0][1]
except Exception as e:
    st.error(f"Could not compute propensity score: {e}")
    ps_hyp = None

if ps_hyp is not None:
    st.subheader("Hypothetical Patient Propensity Score")

    # Classification
    if ps_hyp < 0.05 or ps_hyp > 0.95:
        overlap_label = "poor overlap"
        overlap_msg = (
            "Poor overlap: this patient profile falls in a sparsely supported region "
            "of the propensity score distribution (near 0 or 1). "
            "The causal estimate is less reliable for patients with similar profiles."
        )
        color = "red"
    elif ps_hyp < 0.15 or ps_hyp > 0.85:
        overlap_label = "moderate overlap"
        overlap_msg = (
            "Moderate overlap: this profile is in a moderate region. "
            "The estimate is somewhat reliable but with higher uncertainty at the edges."
        )
        color = "orange"
    else:
        overlap_label = "well-supported"
        overlap_msg = (
            "Well-supported: this profile falls in the well-overlapped region of the "
            "propensity score distribution. The treatment effect estimate is most "
            "reliable for patients with similar profiles."
        )
        color = "green"

    st.metric(
        "Predicted propensity of receiving RHC",
        f"{ps_hyp:.3f}",
        delta=f"Overlap: {overlap_label}",
    )
    st.info(overlap_msg)

    # Overlap plot with hypothetical patient marked
    fig, ax = plt.subplots(figsize=(8, 4))
    t_mask = ps_scores["treatment"] == 1
    c_mask = ps_scores["treatment"] == 0
    ax.hist(ps_scores.loc[c_mask, "propensity_score"], bins=50, alpha=0.4,
            label="Control (No RHC)", density=True, color="steelblue")
    ax.hist(ps_scores.loc[t_mask, "propensity_score"], bins=50, alpha=0.4,
            label="Treated (RHC)", density=True, color="coral")
    ax.axvline(ps_hyp, color=color, linewidth=2, linestyle="--",
               label=f"Hypothetical patient (PS={ps_hyp:.3f})")
    ax.axvline(0.5, color="gray", linestyle=":", alpha=0.3)
    ax.axvline(0.01, color="black", linestyle=":", alpha=0.2, label="Trim boundaries (0.01 / 0.99)")
    ax.axvline(0.99, color="black", linestyle=":", alpha=0.2)
    ax.set_xlabel("Propensity Score")
    ax.set_ylabel("Density")
    ax.set_title("Propensity Score Distribution with Hypothetical Patient")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    st.pyplot(fig)

    # Population reminder
    st.warning(
        "The population-level treatment effects shown above (Naive, IPW, AIPW) are fixed "
        "estimates from the full dataset. They do NOT change based on the patient profile "
        "you adjusted. Only the propensity score changes."
    )

# ──────────────────────────────────────────────────────────────
# SECTION 3 — Methods and citations
# ──────────────────────────────────────────────────────────────
st.header("Methods")
st.write(
    "This analysis uses three causal inference methods to estimate the effect "
    "of Right Heart Catheterization on 30-day mortality in critically ill ICU patients."
)
st.markdown("""
- **Propensity score model**: Logistic regression on 65 pre-treatment confounders, including age and age-squared.
- **Inverse probability weighting (IPW)**: Stabilized weights, propensity scores trimmed to the [0.01, 0.99] range.
- **Augmented IPW (AIPW)**: Doubly robust estimator combining propensity scores with outcome regression models. Uses separate logistic regression outcome models for the treated and control arms.
- **Bootstrap confidence intervals**: 1000 bootstrap resamples with a fixed seed (42).
- **E-value**: Sensitivity analysis per VanderWeele and Ding (2017), measuring how strong an unmeasured confounder would need to be to explain away the observed effect.
""")

st.header("Citations")
st.markdown("""
- Connors Jr, A. F., et al. (1996). The effectiveness of right heart catheterization in the initial care of critically ill patients. *JAMA*, 276(11), 889-897.
- Hirano, K., & Imbens, G. W. (2001). Estimation of causal effects using propensity score stratification.
- Crump, R. K., Hotz, V. J., Imbens, G. W., & Mitnik, O. A. (2009). Dealing with limited overlap in estimation of average treatment effects. *Biometrika*, 96(1), 187-199.
- VanderWeele, T. J., & Ding, P. (2017). Sensitivity analysis in observational research: Introducing the E-value. *Annals of Internal Medicine*, 167(4), 268-274.
- Rosenbaum, P. R. (2012). *Observational Studies* (2nd ed.). Springer.
""")

# ──────────────────────────────────────────────────────────────
# FOOTER
# ──────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "Code: [github.com/atreus-01/rhc-causal-inference](https://github.com/atreus-01/rhc-causal-inference) | "
    "Hosted on HuggingFace Spaces (free tier). Sleeps after inactivity, ~30s cold start on wake."
)
