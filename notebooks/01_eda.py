# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# %%
df = pd.read_csv('/home/node/.openclaw/projects/rhc-causal-inference/data/raw/rhc.csv')
print(f"Loaded {len(df)} rows")

# %%
# SMD calculation function
def compute_smd(df, col, treatment_col='swang1'):
    """Compute standardized mean difference between treated and control."""
    treated = df[df[treatment_col] == 'RHC'][col]
    control = df[df[treatment_col] == 'No RHC'][col]
    
    # Handle binary/categorical vs continuous
    if df[col].dtype == 'object' or df[col].nunique() <= 2:
        # For binary/proportion: use pooled proportion formula
        p_treated = treated.mean() if len(treated) > 0 else 0
        p_control = control.mean() if len(control) > 0 else 0
        # Pooled proportion for 2-group SMD
        p_pooled = (p_treated * len(treated) + p_control * len(control)) / (len(treated) + len(control))
        if p_pooled * (1 - p_pooled) == 0:
            return 0.0
        smd = (p_treated - p_control) / np.sqrt(p_pooled * (1 - p_pooled))
    else:
        # Continuous: use pooled std
        mean_t = treated.mean()
        mean_c = control.mean()
        std_t = treated.std()
        std_c = control.std()
        n_t, n_c = len(treated), len(control)
        if n_t + n_c <= 2:
            return 0.0
        pooled_std = np.sqrt(((n_t - 1) * std_t**2 + (n_c - 1) * std_c**2) / (n_t + n_c - 2))
        if pooled_std == 0:
            return 0.0
        smd = (mean_t - mean_c) / pooled_std
    
    return smd

# Confounders from CODEBOOK (pre-treatment columns)
confounders = [
    # Demographics
    'age', 'sex', 'race', 'edu', 'income', 'ninsclas',
    # Comorbidities
    'cardiohx', 'chfhx', 'dementhx', 'psychhx', 'chrpulhx', 'renalhx', 
    'liverhx', 'gibledhx', 'malighx', 'immunhx', 'transhx', 'amihx', 'ca',
    # Primary diagnosis
    'cat1',
    # Admission diagnosis categories
    'resp', 'card', 'neuro', 'gastr', 'renal', 'meta', 'hema', 'seps', 'trauma', 'ortho',
    # Disease severity
    'aps1', 'scoma1', 'meanbp1', 'wblc1', 'hrt1', 'resp1', 'temp1', 
    'pafi1', 'paco21', 'ph1', 'alb1', 'hema1', 'bili1', 'crea1', 
    'sod1', 'pot1', 'wtkilo1',
    # Functional status
    'das2d3pc',
    # Other
    'surv2md1', 'dnr1'
]

# Filter to existing columns
confounders = [c for c in confounders if c in df.columns]
print(f"Computing SMD for {len(confounders)} confounders")

# Compute SMD for each confounder
smd_results = []
for col in confounders:
    try:
        smd = compute_smd(df, col)
        smd_results.append({'variable': col, 'SMD': smd, '|SMD|': abs(smd)})
    except Exception as e:
        print(f"Error computing SMD for {col}: {e}")

smd_df = pd.DataFrame(smd_results).sort_values('|SMD|', ascending=False)
print("\n=== SMD Table (sorted by |SMD| descending) ===")
print(smd_df.to_string(index=False))

# %%
# dth30 rate per arm
crosstab = pd.crosstab(df['swang1'], df['dth30'], normalize='index')
print("\n=== dth30 Rate per Arm ===")
print(crosstab)
print(f"\nTreated (RHC) dth30 rate: {(df[df['swang1']=='RHC']['dth30']=='Yes').mean():.4f}")
print(f"Control (No RHC) dth30 rate: {(df[df['swang1']=='No RHC']['dth30']=='Yes').mean():.4f}")

# %%
# Missingness summary
missing_pct = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
missing_df = missing_pct[missing_pct > 0].reset_index()
missing_df.columns = ['variable', 'missing_pct']
print("\n=== Missingness Summary ===")
print(missing_df.to_string(index=False))

# Verify only cat2, adld3p, urin1 have >50% missing
high_missing = missing_df[missing_df['missing_pct'] > 50]
print(f"\n>50% missing columns: {high_missing['variable'].tolist()}")
print(f"cat2, adld3p, urin1 are ONLY >50% missing: {set(high_missing['variable'].tolist()) == {'cat2', 'adld3p', 'urin1'}}")

# %%
# Visualization: SMD plot (top imbalanced covariates)
top_n = 20
top_smd = smd_df.head(top_n)

fig, ax = plt.subplots(figsize=(10, 8))
colors = ['red' if abs(s) >= 0.2 else 'steelblue' for s in top_smd['SMD']]
ax.barh(range(len(top_smd)), top_smd['SMD'], color=colors)
ax.set_yticks(range(len(top_smd)))
ax.set_yticklabels(top_smd['variable'])
ax.axvline(x=0.2, color='red', linestyle='--', linewidth=1.5, label='SMD=0.2 threshold')
ax.axvline(x=-0.2, color='red', linestyle='--', linewidth=1.5)
ax.set_xlabel('Standardized Mean Difference')
ax.set_title('Pre-treatment Covariate Imbalance (SMD)')
ax.invert_yaxis()
ax.legend()
plt.tight_layout()
plt.savefig('/home/node/.openclaw/projects/rhc-causal-inference/notebooks/figs/smd_pre.png', dpi=150)
print("Saved smd_pre.png")

# %%
# Visualization: outcome by arm
fig, ax = plt.subplots(figsize=(6, 5))
rates = [
    (df[df['swang1']=='RHC']['dth30']=='Yes').mean(),
    (df[df['swang1']=='No RHC']['dth30']=='Yes').mean()
]
bars = ax.bar(['RHC (Treated)', 'No RHC (Control)'], rates, color=['coral', 'steelblue'])
ax.set_ylabel('30-Day Mortality Rate')
ax.set_title('Unadjusted 30-Day Mortality by Treatment Arm')
ax.set_ylim(0, max(rates) * 1.2)
for bar, rate in zip(bars, rates):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
            f'{rate:.1%}', ha='center', va='bottom', fontsize=12)
plt.tight_layout()
plt.savefig('/home/node/.openclaw/projects/rhc-causal-inference/notebooks/figs/outcome_by_arm.png', dpi=150)
print("Saved outcome_by_arm.png")