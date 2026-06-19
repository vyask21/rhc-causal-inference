# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # RHC Causal Inference — Evaluation
#
# Causal effect of Right Heart Catheterization (RHC) within 24h of ICU admission on 30-day mortality.
# SUPPORT dataset (Connors et al. 1996, JAMA 276:889-897).

# %% [markdown]
# ## 1. Data and Sample

# %%
import pandas as pd
import json

pq = pd.read_parquet('/home/node/.openclaw/projects/rhc-causal-inference/data/interim/analysis.parquet')
n = len(pq)
treated = pq['treatment'].sum()
control = (pq['treatment'] == 0).sum()
rate_t = pq.loc[pq['treatment'] == 1, 'dth30'].mean()
rate_c = pq.loc[pq['treatment'] == 0, 'dth30'].mean()

print(f"Sample: n={n}, treated={treated}, control={control}")
print(f"30-day mortality: treated={rate_t:.4f}, control={rate_c:.4f}")
print(f"Naive difference: {rate_t - rate_c:+.4f}")

# %% [markdown]
# ## 2. Propensity Score Model
#
# Logistic regression on all confounders plus age^2. Propensity scores estimated once, used for IPW and AIPW.

# %%
ps = pd.read_csv('/home/node/.openclaw/projects/rhc-causal-inference/data/interim/propensity_scores.csv')
from sklearn.metrics import roc_auc_score
auc = roc_auc_score(ps['treatment'], ps['propensity_score'])
print(f"Propensity model AUC: {auc:.4f}")
print(f"Below 0.05: {(ps['propensity_score'] < 0.05).sum()} ({(ps['propensity_score'] < 0.05).mean():.4f})")
print(f"Above 0.95: {(ps['propensity_score'] > 0.95).sum()} ({(ps['propensity_score'] > 0.95).mean():.4f})")

# %% [markdown]
# ## 3. Overlap
# ![Overlap](figs/overlap.png)

# %% [markdown]
# ## 4. Covariate Balance (Love Plot)
# ![Love Plot](figs/love.png)

# %%
smds = pd.read_csv('/home/node/.openclaw/projects/rhc-causal-inference/data/interim/smds.csv')
print("Top 10 SMDs (before IPW):")
print(smds.head(10)[['variable', 'smd_before', 'smd_after']].to_string(index=False))

# %% [markdown]
# ## 5. Causal Estimates
#
# All estimates are on the dth30 scale (1 = died within 30 days). Positive = RHC increases mortality.

# %%
with open('/home/node/.openclaw/projects/rhc-causal-inference/results.json') as f:
    r = json.load(f)

print(f"Naive:  {r['naive_estimate']:+.6f}  95%CI [{r['naive_ci'][0]:+.6f}, {r['naive_ci'][1]:+.6f}]")
print(f"IPW:    {r['ipw_estimate']:+.6f}  95%CI [{r['ipw_ci'][0]:+.6f}, {r['ipw_ci'][1]:+.6f}]")
print(f"AIPW:   {r['aipw_estimate']:+.6f}  95%CI [{r['aipw_ci'][0]:+.6f}, {r['aipw_ci'][1]:+.6f}]")
print(f"\nBootstrap: seed={r['bootstrap_seed']}, reps={r['bootstrap_reps']}")

# %% [markdown]
# ## 6. Forest Plot
# ![Forest Plot](figs/estimates_forest.png)

# %% [markdown]
# ## 7. Comparison to Published Benchmarks
#
# Published estimates are on a survival scale (negative = RHC harmful). Converted to mortality scale (positive = harmful) for comparison.

# %%
comparison = pd.DataFrame({
    'Method': ['Naive (unadjusted)', 'IPW (stabilized)', 'AIPW (doubly robust)',
               'Hirano & Imbens (2001)', 'Crump et al. (2009)'],
    'ATE on mortality': [r['naive_estimate'], r['ipw_estimate'], r['aipw_estimate'],
                         '+0.053 to +0.062', '+0.059 to +0.060'],
    'Source': ['This analysis', 'This analysis', 'This analysis',
               'Hirano & Imbens 2001 (survival scale, sign flipped)',
               'Crump, Hotz, Imbens & Mitnik 2009 (survival scale, sign flipped)']
})
print(comparison.to_string(index=False))

# %% [markdown]
# ## 8. E-value Sensitivity Analysis
#
# VanderWeele and Ding (2017). Approximated from ATE on probability scale using baseline control rate.

# %%
print(f"E-value (point estimate): {r['e_value_point']:.4f}")
print(f"E-value (CI lower bound): {r['e_value_ci_lower']:.4f}")
print(f"RR (point): {r['e_value_details']['rr_point']:.4f}")
print(f"Baseline control rate: {r['e_value_details']['baseline_control_rate']:.4f}")
print(f"\nAn unmeasured confounder would need a risk ratio of at least "
      f"{r['e_value_point']:.2f} with both treatment and outcome, "
      f"conditional on measured covariates, to explain away the observed effect.")

# %% [markdown]
# ## 9. Interpretation
#
# All three estimators agree on the direction: RHC is associated with increased 30-day mortality.
# The naive estimate (+0.074) is inflated by confounding (sicker patients receive RHC).
# After adjustment, IPW (+0.056) and AIPW (+0.059) converge closely, consistent with
# published estimates of +0.053 to +0.062 (Hirano & Imbens 2001) and +0.059 to +0.060
# (Crump et al. 2009). The AIPW estimate sits squarely in the published range.

# %% [markdown]
# ## References
#
# - Connors Jr, A. F., et al. (1996). The effectiveness of right heart catheterization in the initial care of critically ill patients. *JAMA*, 276(11), 889-897.
# - Hirano, K., & Imbens, G. W. (2001). Estimation of causal effects using propensity score stratification.
# - Crump, R. K., Hotz, V. J., Imbens, G. W., & Mitnik, O. A. (2009). Dealing with limited overlap in estimation of average treatment effects. *Biometrika*, 96(1), 187-199.
# - VanderWeele, T. J., & Ding, P. (2017). Sensitivity analysis in observational research: Introducing the E-value. *Annals of Internal Medicine*, 167(4), 268-274.
# - Rosenbaum, P. R. (2012). *Observational Studies*. Springer.
