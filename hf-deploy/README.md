---
title: RHC Causal Inference
emoji: "\U0001fac0"
colorFrom: red
colorTo: blue
sdk: streamlit
app_file: app.py
pinned: false
license: mit
short_description: RHC causal inference on ICU mortality
---

# RHC Causal Inference

Interactive visualization of a causal inference analysis estimating the effect of Right Heart Catheterization (RHC) on 30-day mortality in critically ill ICU patients.

Based on the SUPPORT study, Connors et al. 1996 (JAMA 276:889-897).

## Data

The SUPPORT/RHC dataset is publicly available with no data use agreement required.
Source: [hbiostat.org/data/repo/rhc.html](https://hbiostat.org/data/repo/rhc.html)

## Methods

Propensity score model (logistic regression on 65 confounders), stabilized inverse probability weighting, augmented IPW (doubly robust), bootstrap confidence intervals, E-value sensitivity analysis.

## Source Code

[github.com/vyask21/rhc-causal-inference](https://github.com/vyask21/rhc-causal-inference)
