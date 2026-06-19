# RHC Causal Inference

Causal inference analysis of Right Heart Catheterization (RHC) using the SUPPORT study data to estimate the effect of RHC on 30-day mortality in critically ill ICU patients.

## Background

Should critically ill patients who present with respiratory symptoms and a high probability of a cardiopulmonary diagnosis receive Right Heart Catheterization within the first 24 hours of ICU admission? This analysis estimates the causal effect of that clinical decision on 30-day mortality.

A direct comparison of mortality rates between treated and untreated patients is misleading. Patients who receive RHC are systematically different from those who do not. They tend to be sicker at baseline, with higher APACHE scores, lower blood pressure, and worse oxygenation levels. This is confounding by indication: clinicians are more likely to order invasive monitoring for patients they perceive to be at higher risk. Without adjustment, the naive estimate attributes this baseline risk imbalance to the treatment itself.

## Data

The dataset comes from the SUPPORT study (Connors et al. 1996), available as a public resource with no data use agreement required.

- **Source:** [hbiostat.org/data/repo/rhc.html](https://hbiostat.org/data/repo/rhc.html)
- **Original publication:** Connors Jr et al., "The effectiveness of right heart catheterization in the initial care of critically ill patients," *JAMA* 276(11):889-897, 1996.
- **Sample:** n = 5735 ICU patients (2184 received RHC, 3551 did not)
- **Outcome:** Death within 30 days (dth30)
- **Features:** 65 pre-treatment covariates including demographics, comorbidities, admission diagnoses, physiological measures, and functional status

## Methods

The analysis follows a standard causal inference workflow: propensity score estimation, overlap assessment, and three estimators with bootstrap confidence intervals.

1. **Propensity score model:** Logistic regression on all 65 pre-treatment confounders, with age and age-squared included as features.
2. **Estimators:**
   - Naive: unadjusted difference in 30-day mortality between treated and control arms.
   - IPW: stabilized inverse probability weighting, with propensity scores trimmed to [0.01, 0.99].
   - AIPW: augmented inverse probability weighting (doubly robust), combining propensity scores with per-arm outcome regression models (logistic regression).
3. **Bootstrap confidence intervals:** 1000 resamples with a fixed seed (42). AIPW uses influence-function resampling (no model refitting), IPW resamples the weighted formula with fixed propensity scores.
4. **E-value sensitivity analysis:** VanderWeele and Ding (2017), measuring how strong an unmeasured confounder would need to be to explain away the observed effect.

## Results

The naive estimate shows RHC patients have a 30-day mortality rate of 38.0% versus 30.6% for controls, a difference of +0.074. After adjustment for confounding, all estimators converge on a positive effect in the same direction: RHC is associated with increased mortality.

| Method | ATE on 30-day mortality | 95% CI |
|--------|------------------------|--------|
| Naive (unadjusted) | +0.074 | [0.048, 0.099] |
| IPW (stabilized) | +0.056 | [0.022, 0.088] |
| AIPW (doubly robust) | +0.059 | [0.033, 0.087] |

The doubly robust AIPW estimate of +0.059 (95% CI: 0.033 to 0.087) is the preferred result. It indicates that RHC within 24 hours of ICU admission is associated with approximately a 6 percentage point increase in 30-day mortality, after adjusting for observed confounders.

These results are consistent with published estimates. Hirano and Imbens (2001) reported regression-adjusted estimates of -0.053 to -0.062 on the survival scale, and Crump, Hotz, Imbens and Mitnik (2009) reported -0.059 to -0.060 on the same scale. Converted to the mortality scale used here (where an increase in mortality corresponds to a decrease in survival), both published ranges correspond to approximately +0.053 to +0.062, which brackets our AIPW estimate of +0.059.

The propensity model achieved an AUC of 0.7855, indicating moderate-to-strong confounding. The overlap diagnostics show 3.6% of patients below a propensity score of 0.05 and 0.2% above 0.95, suggesting limited but not prohibitive overlap issues. The E-value for the AIPW point estimate is 1.67, meaning an unmeasured confounder would need a risk ratio of at least 1.67 with both treatment and outcome, conditional on measured covariates, to explain away the observed association.

## Repository Structure

```
rhc-causal-inference/
├── data/
│   ├── raw/              # raw CSV from hbiostat.org
│   ├── interim/          # analysis.parquet, propensity scores, SMDs
│   └── processed/        # (future)
├── notebooks/
│   ├── 01_eda.py         # exploratory analysis: SMDs, missingness, plots
│   ├── 02_evaluation.py  # evaluation: estimates, benchmarks, citations
│   ├── figs/             # smd_pre.png, overlap.png, love.png, estimates_forest.png
│   └── 01_eda.html       # executed EDA output
├── src/
│   └── run_estimation.py # standalone estimation script (naive, IPW, AIPW, bootstrap, E-value)
├── models/
│   └── propensity_model.joblib  # fitted logistic propensity model
├── app/                  # (upcoming) streaming delivery interface
├── hf-deploy/            # (upcoming) HuggingFace Spaces deployment
├── results.json          # all estimates, CIs, E-values, configuration
└── requirements.txt      # pinned venv dependencies
```

## How to Reproduce

```bash
# 1. Clone and enter the project
git clone https://github.com/atreus-01/rhc-causal-inference.git
cd rhc-causal-inference

# 2. Set up the virtual environment
python3 -m virtualenv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 3. Run the estimation script (produces figures and results.json)
python src/run_estimation.py

# 4. Run the EDA notebook (convert and execute)
jupytext --to notebook notebooks/01_eda.py -o notebooks/01_eda.ipynb
jupyter nbconvert --to html --execute notebooks/01_eda.ipynb

# 5. Run the evaluation notebook
jupytext --to notebook notebooks/02_evaluation.py -o notebooks/02_evaluation.ipynb
jupyter nbconvert --to html --execute notebooks/02_evaluation.ipynb
```

All results are reproducible with a fixed random seed (42) for bootstrap resampling.

## References

- Connors Jr, A. F., Speroff, T., Dawson, N. V., Thomas, C., Harrell Jr, F. E., Wagner, D., Desbiens, N., Goldman, L., bellamy, A., Fulkerson Jr, W. J., Clark, W., Goldman, R. K., Lynn, J., & Knaus, W. A. (1996). The effectiveness of right heart catheterization in the initial care of critically ill patients. *Journal of the American Medical Association*, 276(11), 889-897.
- Hirano, K., & Imbens, G. W. (2001). Estimation of causal effects using propensity score stratification.
- Crump, R. K., Hotz, V. J., Imbens, G. W., & Mitnik, O. A. (2009). Dealing with limited overlap in estimation of average treatment effects. *Biometrika*, 96(1), 187-199.
- VanderWeele, T. J., & Ding, P. (2017). Sensitivity analysis in observational research: Introducing the E-value. *Annals of Internal Medicine*, 167(4), 268-274.
- Rosenbaum, P. R. (2012). *Observational Studies* (2nd ed.). Springer.
