# Week 1 Task: Data Acquisition, Cleaning, and Preprocessing

## Overview
This project demonstrates a complete data-cleaning and preprocessing workflow on the
Titanic passenger dataset (891 rows, 15 original columns), sourced from a public
GitHub mirror (`mwaskom/seaborn-data`).

## Contents
| File | Description |
|---|---|
| `01_explore.py` | Initial data exploration: structure, dtypes, missing values, duplicates, redundant-column checks |
| `02_clean_preprocess.py` | Full cleaning pipeline: missing-value handling, outlier treatment, feature engineering, encoding, scaling, chart generation |
| `titanic.csv` | Raw, unmodified dataset |
| `titanic_cleaned.csv` | Cleaned dataset (891 rows x 17 columns, 0 missing values) |
| `titanic_model_ready.csv` | Cleaned + encoded + scaled dataset, ready for modeling (891 rows x 22 columns) |
| `Week1_Data_Cleaning_Report.docx` | Full written report with narrative, code snippets, and charts |

## How to run
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
python 01_explore.py
python 02_clean_preprocess.py
```

## Summary of work
- Identified and resolved missing values in `deck` (77.2%), `age` (19.9%), and
  `embarked` (0.2%) using approaches suited to each column's missingness level
  (binary flag, group-median imputation, mode imputation respectively).
- Detected and treated fare outliers using the 1.5xIQR rule (capping/winsorization
  instead of deletion, to preserve all 891 records).
- Removed 3 fully redundant columns (`alive`, `class`, `embark_town`).
- Investigated and documented a decision not to drop 107 "duplicate" rows, since the
  dataset lacks a unique passenger identifier.
- Engineered new features (`family_size`, `is_alone`, `has_cabin_record`,
  `zero_fare_flag`) and produced a fully encoded/scaled model-ready dataset.

Full rationale for every decision is documented in `Week1_Data_Cleaning_Report.docx`.
