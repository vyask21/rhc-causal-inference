"""
RHC Causal Inference — Part 2A: Estimation
Stages 2-6 in one script. Prints ONLY summary numbers. Saves all artifacts.
"""
import warnings
warnings.filterwarnings("ignore")

import json, math, os, joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Paths ──
PROJECT = "/home/node/.openclaw/projects/rhc-causal-inference"
PARQUET = f"{PROJECT}/data/interim/analysis.parquet"
PS_CSV   = f"{PROJECT}/data/interim/propensity_scores.csv"
MODEL    = f"{PROJECT}/models/propensity_model.joblib"
RESULTS  = f"{PROJECT}/results.json"
LOGDIR   = f"{PROJECT}/logs"
FIGDIR   = f"{PROJECT}/notebooks/figs"
os.makedirs(LOGDIR, exist_ok=True)
os.makedirs(FIGDIR, exist_ok=True)

BOOTSTRAP_SEED = 42
N_BOOT = 1000
PS_LO, PS_HI = 0.01, 0.99

# ── Load data ──
pq = pd.read_parquet(PARQUET)
ps_data = pd.read_csv(PS_CSV)

# Coerce bool columns to float64 (one-hot columns may be bool)
bool_cols = pq.select_dtypes(include=["bool"]).columns.tolist()
for c in bool_cols:
    pq[c] = pq[c].astype(float)

# Confounders = all columns except treatment and dth30
confounder_cols = [c for c in pq.columns if c not in ("treatment", "dth30")]
X = pq[confounder_cols].values  # numpy array
y = pq["treatment"].values.astype(float)
dth = pq["dth30"].values.astype(float)

ps = ps_data["propensity_score"].values.astype(float)

n = len(pq)
n_treat = int(y.sum())
n_ctrl = n - n_treat
rate_treat = dth[y == 1].mean()
rate_ctrl = dth[y == 0].mean()

# ── Repropensity AUC from scores on disk ──
from sklearn.metrics import roc_auc_score
auc = roc_auc_score(y, ps)

# ── (a) Naive ATE ──
ate_naive = rate_treat - rate_ctrl

# ── (b) IPW ATE ──
ps_clipped = np.clip(ps, PS_LO, PS_HI)
prop_treat = y.mean()
sw = np.where(y == 1, prop_treat / ps_clipped, (1 - prop_treat) / (1 - ps_clipped))
ate_ipw = (np.sum(sw * y * dth) / np.sum(sw * y)) - (np.sum(sw * (1 - y) * dth) / np.sum(sw * (1 - y)))

# ── (c) AIPW ATE (doubly robust) ──
from sklearn.linear_model import LogisticRegression

prop_model = joblib.load(MODEL)

# Outcome models per arm
mt = LogisticRegression(max_iter=5000, solver="lbfgs")
mt.fit(X[y == 1], dth[y == 1])
mc = LogisticRegression(max_iter=5000, solver="lbfgs")
mc.fit(X[y == 0], dth[y == 0])

mu1 = mt.predict_proba(X)[:, 1]
mu0 = mc.predict_proba(X)[:, 1]

phi_aipw = (mu1 - mu0
            + y * (dth - mu1) / ps_clipped
            - (1 - y) * (dth - mu0) / (1 - ps_clipped))
ate_aipw = np.mean(phi_aipw)

# ── Weighted SMDs (before vs after IPW) for love plot ──
smd_vars = confounder_cols
smd_before, smd_after = [], []
for col in smd_vars:
    vals = pq[col].values.astype(float)
    t_mask = y == 1
    c_mask = y == 0
    mt_v, mc_v = vals[t_mask].mean(), vals[c_mask].mean()
    st_v, sc_v = vals[t_mask].std(ddof=1), vals[c_mask].std(ddof=1)
    nt_v, nc_v = t_mask.sum(), c_mask.sum()
    sp_v = np.sqrt(((nt_v - 1) * st_v**2 + (nc_v - 1) * sc_v**2) / (nt_v + nc_v - 2))
    smd_before.append((mt_v - mc_v) / sp_v if sp_v > 0 else 0.0)

    wt_t, wt_c = sw[t_mask], sw[c_mask]
    mt_w = np.average(vals[t_mask], weights=wt_t)
    mc_w = np.average(vals[c_mask], weights=wt_c)
    vt_w = np.average((vals[t_mask] - mt_w) ** 2, weights=wt_t)
    vc_w = np.average((vals[c_mask] - mc_w) ** 2, weights=wt_c)
    sp_w = np.sqrt((vt_w + vc_w) / 2)
    smd_after.append((mt_w - mc_w) / sp_w if sp_w > 0 else 0.0)

smd_sort = np.argsort(np.abs(smd_before))[::-1]
sorted_vars = [smd_vars[i] for i in smd_sort]
sorted_before = [smd_before[i] for i in smd_sort]
sorted_after = [smd_after[i] for i in smd_sort]

# ── Bootstrap CIs (1000 reps, influence-function based) ──
# AIPW is asymptotically linear: phi_i is the influence function.
# Bootstrap = resample phi_aipw values and take mean. No model refits needed.
# IPW: resample the weighted formula (propensity scores fixed at nuisance estimate).
# Naive: resample group mean difference.
# This is computationally trivial (vectorized) and statistically valid for
# asymptotic CIs of asymptotically linear estimators.
rng = np.random.RandomState(BOOTSTRAP_SEED)

# Precompute per-unit IPW contribution for fast bootstrap
ipw_unit = sw * (y * dth / np.sum(sw * y) - (1 - y) * dth / np.sum(sw * (1 - y)))
# Recompute properly: the IPW estimator is a function of weighted means,
# so for bootstrap we resample indices and recompute weights with fixed ps.

def fast_bootstrap(seed):
    rng_bs = np.random.RandomState(seed)
    naive_arr = np.empty(N_BOOT)
    ipw_arr = np.empty(N_BOOT)
    aipw_arr = np.empty(N_BOOT)

    for b in range(N_BOOT):
        idx = rng_bs.randint(0, n, size=n)
        yb, db, pb = y[idx], dth[idx], ps_clipped[idx]

        # Naive
        naive_arr[b] = db[yb == 1].mean() - db[yb == 0].mean()

        # IPW (fixed propensity scores, recompute weights on bootstrap sample)
        ptb = yb.mean()
        swb = np.where(yb == 1, ptb / pb, (1 - ptb) / (1 - pb))
        ipw_arr[b] = (np.sum(swb * yb * db) / np.sum(swb * yb)
                      - np.sum(swb * (1 - yb) * db) / np.sum(swb * (1 - yb)))

        # AIPW via influence function: bootstrap mean of resampled phi values
        aipw_arr[b] = np.mean(phi_aipw[idx])

    return naive_arr, ipw_arr, aipw_arr

naive_bs, ipw_bs, aipw_bs = fast_bootstrap(BOOTSTRAP_SEED)

def pct_ci(arr, alpha=0.05):
    return tuple(np.percentile(arr, [100 * alpha / 2, 100 * (1 - alpha / 2)]))

ci_naive_lo, ci_naive_hi = pct_ci(naive_bs)
ci_ipw_lo, ci_ipw_hi = pct_ci(ipw_bs)
ci_aipw_lo, ci_aipw_hi = pct_ci(aipw_bs)

# ── E-value ──
rr_point = (rate_ctrl + ate_aipw) / rate_ctrl
e_val_pt = rr_point + math.sqrt(rr_point * (rr_point - 1)) if rr_point > 1 else 1.0

rr_ci_lo = (rate_ctrl + rate_ctrl if ci_aipw_lo <= 0 else (rate_ctrl + ci_aipw_lo)) / rate_ctrl
# If CI lower <= 0, E-value is 1 (null crossed)
e_val_ci = rr_ci_lo + math.sqrt(rr_ci_lo * (rr_ci_lo - 1)) if rr_ci_lo > 1 else 1.0

# ── Print summary (stdout only) ──
print("=== RHC Part 2A Estimation Summary ===")
print(f"n={n}, treated={n_treat}, control={n_ctrl}")
print(f"dth30 rate: treated={rate_treat:.4f}, control={rate_ctrl:.4f}")
print(f"Propensity AUC: {auc:.4f}")
print()
print("Estimates (ATE on dth30, positive = RHC increases mortality):")
print(f"  Naive:  {ate_naive:+.6f}  95%CI [{ci_naive_lo:+.6f}, {ci_naive_hi:+.6f}]")
print(f"  IPW:    {ate_ipw:+.6f}  95%CI [{ci_ipw_lo:+.6f}, {ci_ipw_hi:+.6f}]")
print(f"  AIPW:   {ate_aipw:+.6f}  95%CI [{ci_aipw_lo:+.6f}, {ci_aipw_hi:+.6f}]")
print(f"Bootstrap: seed={BOOTSTRAP_SEED}, reps={N_BOOT} (all three, influence-function based for AIPW)")
print()
print(f"E-value (point): {e_val_pt:.4f} (RR={rr_point:.4f})")
print(f"E-value (CI lower {ci_aipw_lo:+.6f}): {e_val_ci:.4f}")
print()
print(f"Benchmark gate: AIPW positive={ate_aipw > 0}, |AIPW|={abs(ate_aipw):.6f}, in [0.03,0.08]={0.03<=abs(ate_aipw)<=0.08}")

# ── Save figures ──

# Forest plot
fig, ax = plt.subplots(figsize=(6, 3))
colors = ["steelblue", "forestgreen", "darkred"]
labels = ["Naive", "IPW", "AIPW"]
estimates = [ate_naive, ate_ipw, ate_aipw]
cis_lo = [ci_naive_lo, ci_ipw_lo, ci_aipw_lo]
cis_hi = [ci_naive_hi, ci_ipw_hi, ci_aipw_hi]
for i, (lo, est, hi) in enumerate(zip(cis_lo, estimates, cis_hi)):
    ax.errorbar(est, i,
                xerr=[[est - lo], [hi - est]],
                fmt="o", color=colors[i], capsize=4, markersize=8)
ax.axvline(0, color="gray", linestyle="-", alpha=0.5)
ax.set_yticks(range(3))
ax.set_yticklabels(labels)
ax.set_xlabel("ATE on 30-day mortality (dth30)")
ax.set_title("Causal Effect of RHC on 30-day Mortality")
ax.invert_yaxis()
fig.tight_layout()
fig.savefig(f"{FIGDIR}/estimates_forest.png", dpi=150)
plt.close(fig)

# Love plot
y_pos = np.arange(len(sorted_vars))
fig_h = max(6, len(sorted_vars) * 0.3)
fig, ax = plt.subplots(figsize=(8, fig_h))
ax.vlines(x=0, ymin=-0.5, ymax=len(sorted_vars) - 0.5, color="gray", alpha=0.3)
ax.vlines(x=[-0.1, 0.1], ymin=-0.5, ymax=len(sorted_vars) - 0.5, color="red", linestyle="--", alpha=0.5)
ax.barh(y_pos - 0.15, sorted_before, height=0.3, color="steelblue", label="Before IPW", alpha=0.7)
ax.barh(y_pos + 0.15, sorted_after, height=0.3, color="coral", label="After IPW", alpha=0.7)
ax.set_yticks(y_pos)
ax.set_yticklabels(sorted_vars, fontsize=8)
ax.set_xlabel("Standardized Mean Difference")
ax.set_title("Love Plot: Covariate Balance Before/After IPW")
ax.legend(loc="lower right")
fig.tight_layout()
fig.savefig(f"{FIGDIR}/love.png", dpi=150)
plt.close(fig)

# ── Save results.json ──
final = {
    "naive_estimate": round(ate_naive, 6),
    "naive_ci": [round(ci_naive_lo, 6), round(ci_naive_hi, 6)],
    "ipw_estimate": round(ate_ipw, 6),
    "ipw_ci": [round(ci_ipw_lo, 6), round(ci_ipw_hi, 6)],
    "aipw_estimate": round(ate_aipw, 6),
    "aipw_ci": [round(ci_aipw_lo, 6), round(ci_aipw_hi, 6)],
    "e_value_point": round(e_val_pt, 4),
    "e_value_ci_lower": round(e_val_ci, 4),
    "e_value_details": {
        "rr_point": round(rr_point, 4),
        "rr_ci_lower": round(rr_ci_lo, 4),
        "baseline_control_rate": round(rate_ctrl, 4),
        "method": "VanderWeele and Ding (2017) E-value for risk ratio, approximated from ATE on probability scale"
    },
    "propensity_auc": round(auc, 4),
    "bootstrap_seed": BOOTSTRAP_SEED,
    "bootstrap_reps": N_BOOT,
    "propensity_trim": [PS_LO, PS_HI],
    "outcome_model": "logistic_regression_per_arm",
    "propensity_model": "logistic_regression_all_confounders_plus_age2",
    "project": "rhc-causal-inference",
    "part": "2A"
}
with open(RESULTS, "w") as f:
    json.dump(final, f, indent=2)

# Save SMDs as CSV for the notebook
smd_df = pd.DataFrame({"variable": sorted_vars, "smd_before": sorted_before, "smd_after": sorted_after})
smd_df.to_csv(f"{PROJECT}/data/interim/smds.csv", index=False)

print()
print(f"Saved: {RESULTS}")
print(f"Saved: {FIGDIR}/estimates_forest.png, {FIGDIR}/love.png")
print(f"Saved: data/interim/smds.csv")
print("DONE")
