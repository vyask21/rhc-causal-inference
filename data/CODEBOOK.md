# RHC Dataset Codebook

## Overview

The Right Heart Catheterization (RHC) dataset is from Connors et al. (1996), "The effectiveness of RHC in the initial care of critically ill patients," published in JAMA 276:889-897. This dataset pertains to day 1 of hospitalization from the SUPPORT study.

## Treatment Variable

| Variable | Description |
|----------|-------------|
| **swang1** | Right Heart Catheterization (RHC) on day 1 |

**Coding:**
- `RHC` = Received Right Heart Catheterization (treatment group)
- `No RHC` = Did not receive Right Heart Catheterization (control group)

## Outcome Variable

| Variable | Description |
|----------|-------------|
| **dth30** | Death within 30 days |

**Coding:**
- `Yes` = Died within 30 days
- `No` = Survived at least 30 days (or follow-up ended)

**Alternative outcome columns:**
- `death`: Death at any time up to 180 days (`Yes`/`No`)
- `t3d30`: Days to death within 30-day follow-up (integer, 1-30; value 30 = alive at 30 days)

## Sample Sizes

| Group | Count |
|-------|-------|
| **Total n** | 5735 |
| **Treated (RHC)** | 2184 |
| **Control (No RHC)** | 3551 |

## Confounders (Baseline/Pre-treatment Variables)

These variables were measured at baseline (day 1, before the RHC decision) and can be used as confounders:

### Demographics

| Variable | Type | Description | Missing |
|----------|------|-------------|---------|
| age | continuous | Age (years) | 0 (0.00%) |
| sex | categorical | Sex (Male/Female) | 0 (0.00%) |
| race | categorical | Race (white/black/other) | 0 (0.00%) |
| edu | continuous | Years of education | 0 (0.00%) |
| income | categorical | Income category | 0 (0.00%) |
| ninsclas | categorical | Medical insurance category | 0 (0.00%) |

### Comorbidities (binary: 1=present, 0=absent)

| Variable | Description | Missing |
|----------|-------------|---------|
| cardiohx | Acute MI, Peripheral Vascular Disease, Severe CV Symptoms (NYHA-III), Very Severe CV Symptoms (NYHA-IV) | 0 (0.00%) |
| chfhx | Congestive Heart Failure | 0 (0.00%) |
| dementhx | Dementia, Stroke or Cerebral Infarct, Parkinson's Disease | 0 (0.00%) |
| psychhx | Psychiatric History, Active Psychosis or Severe Depression | 0 (0.00%) |
| chrpulhx | Chronic Pulmonary Disease, Severe Pulmonary Disease, Very Severe Pulmonary Disease | 0 (0.00%) |
| renalhx | Chronic Renal Disease, Chronic Hemodialysis or Peritoneal Dialysis | 0 (0.00%) |
| liverhx | Cirrhosis, Hepatic Failure | 0 (0.00%) |
| gibledhx | Upper GI Bleeding | 0 (0.00%) |
| malighx | Solid Tumor, Metastatic Disease, Chronic Leukemia/Myeloma, Acute Leukemia, Lymphoma | 0 (0.00%) |
| immunhx | Immunosuppression, Organ Transplant, HIV, Diabetes Mellitus With/Without End Organ Damage, Connective Tissue Disease | 0 (0.00%) |
| transhx | Transfer (>24 hours) from Another Hospital | 0 (0.00%) |
| amihx | Definite Myocardial Infarction | 0 (0.00%) |
| ca | Cancer (category) | 0 (0.00%) |

### Primary Diagnosis

| Variable | Description | Missing |
|----------|-------------|---------|
| cat1 | Primary disease category (ARF, COPD, MOSF w/Sepsis, etc.) | 0 (0.00%) |
| cat2 | Secondary disease category | 4535 (79.08%) ⚠️ |

### Admission Diagnosis Categories (binary: present/absent)

| Variable | Description |
|----------|-------------|
| resp | Respiratory Diagnosis |
| card | Cardiovascular Diagnosis |
| neuro | Neurological Diagnosis |
| gastr | Gastrointestinal Diagnosis |
| renal | Renal Diagnosis |
| meta | Metabolic Diagnosis |
| hema | Hematologic Diagnosis |
| seps | Sepsis Diagnosis |
| trauma | Trauma Diagnosis |
| ortho | Orthopedic Diagnosis |

### Disease Severity / Physiological Measures

| Variable | Description | Missing |
|----------|-------------|---------|
| aps1 | APACHE score | 0 (0.00%) |
| scoma1 | Glasgow Coma Score | 0 (0.00%) |
| meanbp1 | Mean blood pressure | 0 (0.00%) |
| wblc1 | WBC count | 0 (0.00%) |
| hrt1 | Heart rate | 0 (0.00%) |
| resp1 | Respiratory rate | 0 (0.00%) |
| temp1 | Temperature | 0 (0.00%) |
| pafi1 | PaO2/FIO2 ratio | 0 (0.00%) |
| paco21 | PaCO2 | 0 (0.00%) |
| ph1 | PH | 0 (0.00%) |
| alb1 | Albumin | 0 (0.00%) |
| hema1 | Hematocrit | 0 (0.00%) |
| bili1 | Bilirubin | 0 (0.00%) |
| crea1 | Creatinine | 0 (0.00%) |
| sod1 | Sodium | 0 (0.00%) |
| pot1 | Potassium | 0 (0.00%) |
| wtkilo1 | Weight (kg) | 0 (0.00%) |

### Functional Status

| Variable | Description | Missing |
|----------|-------------|---------|
| adld3p | ADL (Activities of Daily Living) score | 4296 (74.91%) ⚠️ |
| das2d3pc | Duke Activity Status Index | 0 (0.00%) |

### Laboratory / Other

| Variable | Description | Missing |
|----------|-------------|---------|
| surv2md1 | Support model estimate of probability of surviving 2 months | 0 (0.00%) |
| urin1 | Urine output | 3028 (52.80%) ⚠️ |
| dnr1 | DNR status on day 1 | 0 (0.00%) |

## POST-TREATMENT Variables

**These columns are measured AFTER the RHC decision or are outcome-derived. They must NEVER be used as confounders:**

| Variable | Description |
|----------|-------------|
| dth30 | 30-day mortality (outcome) |
| t3d30 | Days to death within 30-day follow-up (outcome) |
| death | Death at any time up to 180 days (outcome) |
| sadmdte | Study Admission Date (date) |
| dschdte | Hospital Discharge Date (date) |
| dthdte | Date of Death (date) |
| lstctdte | Date of Last Contact (date) |

## Columns with >50% Missing ⚠️

| Variable | Missing Count | Missing % |
|----------|--------------|-----------|
| cat2 (Secondary disease category) | 4535 | 79.08% |
| adld3p (ADL score) | 4296 | 74.91% |
| urin1 (Urine output) | 3028 | 52.80% |

## Data Source

- URL: https://hbiostat.org/data/repo/rhc.csv
- Original source: SUPPORT study (Connors et al., 1996, JAMA)