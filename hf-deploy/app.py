"""
RHC Causal Inference — Streamlit App (HF Spaces)
Part 2B Stage 2: Flat-path version for HuggingFace Spaces deployment.
"""
import json, os
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import streamlit as st

# ── Flat paths for HF Spaces (all files in same dir as app.py) ──
HERE = os.path.dirname(os.path.abspath(__file__))

# ── Load data ──
@st.cache_resource
def load_resources():
    with open(os.path.join(HERE, "results.json")) as f:
        results = json.load(f)
    prop_model = joblib.load(os.path.join(HERE, "propensity_model.joblib"))
    overlap = pd.read_csv(os.path.join(HERE, "overlap_data.csv"))
    return results, prop_model, overlap

results, prop_model, overlap = load_resources()

# Feature column names for the propensity model (from the saved model metadata)
# These are the columns at the time the model was fit
PROP_FEATURE_COUNT = getattr(prop_model, 'n_features_in_', 66)

# Pre-compute baseline covariate values from the propensity score distribution
# For the interactive panel, we use approximate medians from literature/knowledge
# since we don't ship full patient-level data to HF Spaces
BASELINE = {
    "aps1": 50.0,
    "meanbp1": 70.0,
    "pafi1": 200.0,
    "crea1": 1.5,
    "wtkilo1": 70.0,
    "cat1_MOSF w/Sepsis": False,
    "neuro_Yes": False,
    "card_Yes": False,
    "age": 60.0,
    "age2": 3600.0,
    # All other confounders default to 0 (binary one-hot) or median (continuous)
    # The propensity model will use whatever values we provide
}

st.set_page_config(page_title="RHC Causal Inference", page_icon="\U0001fac0", layout="wide")

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

# User-friendly slider specs for top covariates by SMD
slider_specs = [
    ("aps1", "APACHE severity score", 0.0, 80.0, 50.0, 1.0, "float"),
    ("meanbp1", "Mean blood pressure (mmHg)", 10.0, 180.0, 70.0, 1.0, "float"),
    ("pafi1", "PaO2/FIO2 ratio", 10.0, 600.0, 200.0, 5.0, "float"),
    ("crea1", "Creatinine (mg/dL)", 0.0, 10.0, 1.5, 0.1, "float"),
    ("wtkilo1", "Weight (kg)", 20.0, 200.0, 70.0, 1.0, "float"),
    ("cat1_MOSF w/Sepsis", "Primary dx: MOSF w/Sepsis", False, True, False, None, "bool"),
    ("neuro_Yes", "Neurological diagnosis", False, True, False, None, "bool"),
    ("card_Yes", "Cardiovascular diagnosis", False, True, False, None, "bool"),
]

# Build full feature vector with all 66 features
ALL_FEATURES = [
    'age', 'age2', 'alb1', 'amihx', 'aps1', 'bili1', 'ca_No', 'ca_Yes',
    'card_Yes', 'cardiohx', 'cat1_CHF', 'cat1_COPD', 'cat1_Cirrhosis',
    'cat1_Colon Cancer', 'cat1_Coma', 'cat1_Lung Cancer', 'cat1_MOSF w/Malignancy',
    'cat1_MOSF w/Sepsis', 'chfhx', 'chrpulhx', 'crea1', 'das2d3pc', 'dementhx',
    'dnr1_Yes', 'edu', 'gastr_Yes', 'gibledhx', 'hema1', 'hema_Yes', 'hrt1',
    'immunhx', 'income_$25-$50k', 'income_> $50k', 'income_Under $11k',
    'liverhx', 'malighx', 'meanbp1', 'meta_Yes', 'neuro_Yes', 'ninsclas_Medicare',
    'ninsclas_Medicare & Medicaid', 'ninsclas_No insurance', 'ninsclas_Private',
    'ninsclas_Private & Medicare', 'ortho_Yes', 'paco21', 'pafi1', 'ph1',
    'pot1', 'psychhx', 'race_other', 'race_white', 'renal_Yes', 'renalhx',
    'resp1', 'resp_Yes', 'scoma1', 'seps_Yes', 'sex_Male', 'sod1', 'surv2md1',
    'temp1', 'transhx', 'trauma_Yes', 'wblc1', 'wtkilo1'
]

# Default all features to 0 (continuous features below will be overridden)
user_features = {f: 0.0 for f in ALL_FEATURES}

# Set continuous defaults at population medians (approximate from literature)
user_features.update({
    "age": 60.0, "age2": 3600.0,
    "aps1": 50.0, "scoma1": 14.0, "meanbp1": 70.0, "wblc1": 10000.0,
    "hrt1": 90.0, "resp1": 22.0, "temp1": 37.5, "pafi1": 200.0,
    "paco21": 40.0, "ph1": 7.35, "alb1": 3.0, "hema1": 36.0,
    "bili1": 1.5, "crea1": 1.5, "sod1": 140.0, "pot1": 4.0,
    "wtkilo1": 70.0, "edu": 12.0, "surv2md1": 0.5, "das2d3pc": 40.0,
})

col_sl1, col_sl2 = st.columns(2)

with col_sl1:
    for feat, label, lo, hi, default, step, dtype in slider_specs[:6]:
        if dtype == "bool":
            val = st.toggle(label, value=bool(default), key=feat)
        else:
            val = st.slider(label, min_value=float(lo), max_value=float(hi),
                           value=float(default), step=float(step), key=feat)
        user_features[feat] = float(val) if not isinstance(val, bool) else (1.0 if val else 0.0)

with col_sl2:
    for feat, label, lo, hi, default, step, dtype in slider_specs[6:]:
        if dtype == "bool":
            val = st.toggle(label, value=bool(default), key=feat)
        else:
            val = st.slider(label, min_value=float(lo), max_value=float(hi),
                           value=float(default), step=float(step), key=feat)
        user_features[feat] = float(val) if not isinstance(val, bool) else (1.0 if val else 0.0)

# Update age2 from age
if "age" in user_features:
    user_features["age2"] = user_features["age"] ** 2

# Build feature vector in exact model order
feat_vec = np.array([user_features.get(f, 0.0) for f in ALL_FEATURES]).reshape(1, -1)

# Predict propensity for hypothetical patient
try:
    ps_hyp = float(prop_model.predict_proba(feat_vec)[0][1])
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
    t_mask = overlap["treatment"] == 1
    c_mask = overlap["treatment"] == 0
    ax.hist(overlap.loc[c_mask, "propensity_score"], bins=50, alpha=0.4,
            label="Control (No RHC)", density=True, color="steelblue")
    ax.hist(overlap.loc[t_mask, "propensity_score"], bins=50, alpha=0.4,
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
