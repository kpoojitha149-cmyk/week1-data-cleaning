"""
Week 1 Task: Data Acquisition, Cleaning, and Preprocessing
Dataset: Titanic passenger data (seaborn-data mirror of the classic Kaggle Titanic set)
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

sns.set_theme(style="whitegrid")
plt.rcParams['figure.dpi'] = 150

RAW_PATH = 'titanic.csv'
df = pd.read_csv(RAW_PATH)
raw_shape = df.shape

# ------------------------------------------------------------------
# 1. MISSINGNESS CHART (before cleaning)
# ------------------------------------------------------------------
miss = df.isnull().sum().sort_values(ascending=False)
miss = miss[miss > 0]
plt.figure(figsize=(7, 4))
sns.barplot(x=miss.values, y=miss.index, hue=miss.index, palette='rocket', legend=False)
plt.title('Missing Values by Column (Raw Data)')
plt.xlabel('Number of missing rows (out of 891)')
plt.ylabel('Column')
for i, v in enumerate(miss.values):
    plt.text(v + 5, i, f"{v} ({v/len(df)*100:.1f}%)", va='center', fontsize=9)
plt.tight_layout()
plt.savefig('chart_missingness.png')
plt.close()

# ------------------------------------------------------------------
# 2. DROP FULLY REDUNDANT COLUMNS
#    - alive is a string mirror of survived (0/1)
#    - class is a string mirror of pclass (1/2/3)
#    - embark_town is a string mirror of embarked (S/C/Q)
#    These were confirmed via crosstab in the exploration step (01_explore.py)
# ------------------------------------------------------------------
redundant_cols = ['alive', 'class', 'embark_town']
df = df.drop(columns=redundant_cols)

# ------------------------------------------------------------------
# 3. HANDLE 'deck' - 77% missing, too sparse to impute reliably.
#    Convert into a low-information-loss binary feature instead of
#    imputing 688 unknown values, which would fabricate data.
# ------------------------------------------------------------------
df['has_cabin_record'] = df['deck'].notna().astype(int)
df = df.drop(columns=['deck'])

# ------------------------------------------------------------------
# 4. HANDLE 'embarked' - only 2 missing (0.22%). Impute with the mode,
#    a safe choice given the tiny share of affected rows.
# ------------------------------------------------------------------
embarked_mode = df['embarked'].mode()[0]
df['embarked'] = df['embarked'].fillna(embarked_mode)

# ------------------------------------------------------------------
# 5. HANDLE 'age' - ~20% missing. A single global median would ignore
#    the fact that age varies systematically with sex and class
#    (e.g. 1st class passengers skew older). Impute with the median
#    age of each (pclass, sex) group instead of a single global value.
# ------------------------------------------------------------------
age_before = df['age'].copy()
df['age'] = df.groupby(['pclass', 'sex'])['age'].transform(lambda s: s.fillna(s.median()))
# fallback for any group that was itself entirely NaN (not the case here, but defensive)
df['age'] = df['age'].fillna(df['age'].median())

plt.figure(figsize=(7, 4))
sns.kdeplot(age_before.dropna(), label='Before imputation', fill=True, alpha=0.4)
sns.kdeplot(df['age'], label='After group-median imputation', fill=True, alpha=0.4)
plt.title('Age Distribution Before vs. After Imputation')
plt.xlabel('Age (years)')
plt.legend()
plt.tight_layout()
plt.savefig('chart_age_imputation.png')
plt.close()

# ------------------------------------------------------------------
# 6. ERRONEOUS / SPECIAL-CASE VALUES: fare == 0
#    15 rows have fare = 0.00, all adult males embarked at Southampton.
#    This matches known Titanic records for crew members and employees
#    of the shipping line who travelled free of charge - a legitimate
#    special case, not a data-entry error, so the rows are kept but
#    flagged rather than deleted or overwritten.
# ------------------------------------------------------------------
df['zero_fare_flag'] = (df['fare'] == 0).astype(int)

# ------------------------------------------------------------------
# 7. OUTLIERS IN 'fare' - detect with IQR rule, visualize, then treat
#    with winsorization (capping) rather than deletion, since removing
#    ~13% of rows (116/891) would discard real, legitimate passengers
#    (first-class fares were genuinely much higher).
# ------------------------------------------------------------------
q1, q3 = df['fare'].quantile([0.25, 0.75])
iqr = q3 - q1
lower_bound, upper_bound = q1 - 1.5 * iqr, q3 + 1.5 * iqr
n_outliers = ((df['fare'] < lower_bound) | (df['fare'] > upper_bound)).sum()

fare_before = df['fare'].copy()

plt.figure(figsize=(7, 4))
plt.boxplot([fare_before], vert=False, labels=['Fare (raw)'])
plt.title(f'Fare Distribution - Boxplot Before Capping\n({n_outliers} points flagged by 1.5xIQR rule)')
plt.xlabel('Fare (£)')
plt.tight_layout()
plt.savefig('chart_fare_boxplot_before.png')
plt.close()

df['fare_capped'] = df['fare'].clip(lower=max(lower_bound, 0), upper=upper_bound)
# keep original fare for reference, log-transform the capped version to reduce right-skew
df['fare_log'] = np.log1p(df['fare_capped'])

plt.figure(figsize=(7, 4))
plt.boxplot([df['fare_capped']], vert=False, labels=['Fare (capped)'])
plt.title('Fare Distribution - Boxplot After IQR Capping')
plt.xlabel('Fare (£)')
plt.tight_layout()
plt.savefig('chart_fare_boxplot_after.png')
plt.close()

plt.figure(figsize=(8, 4))
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
sns.histplot(fare_before, bins=40, ax=axes[0], color='indianred')
axes[0].set_title('Fare - Raw (right-skewed)')
sns.histplot(df['fare_log'], bins=40, ax=axes[1], color='seagreen')
axes[1].set_title('Fare - Capped + Log-Transformed')
plt.tight_layout()
plt.savefig('chart_fare_transform.png')
plt.close()

# ------------------------------------------------------------------
# 8. DUPLICATE ROWS
#    df.duplicated() flags 107 fully-identical rows. This dataset has
#    no passenger ID/name column, so identical combinations of the
#    remaining 15 attributes (e.g. two 3rd-class men, same age, same
#    fare, same port) can arise by coincidence among genuinely
#    different passengers travelling in the same class with common
#    ages and standard fares. Historical passenger counts (891 in
#    this sample) are consistent with keeping all rows, so these are
#    treated as coincidental matches, not true duplicates, and are NOT
#    dropped. This decision is documented rather than silently applied.
# ------------------------------------------------------------------
n_flagged_duplicates = df.duplicated().sum()

# ------------------------------------------------------------------
# 9. CONSISTENCY CHECKS / TYPE FIXES
# ------------------------------------------------------------------
df['sex'] = df['sex'].str.lower().str.strip()
df['embarked'] = df['embarked'].str.upper().str.strip()
df['adult_male'] = df['adult_male'].astype(int)
df['alone'] = df['alone'].astype(int)
assert df['age'].min() > 0, "Found non-positive age"
assert df['fare_capped'].min() >= 0, "Found negative fare"
assert set(df['pclass'].unique()) == {1, 2, 3}

# ------------------------------------------------------------------
# 10. FEATURE ENGINEERING for downstream modeling readiness
# ------------------------------------------------------------------
df['family_size'] = df['sibsp'] + df['parch'] + 1
df['is_alone'] = (df['family_size'] == 1).astype(int)

# ------------------------------------------------------------------
# 11. ENCODING categorical variables
# ------------------------------------------------------------------
df_model = pd.get_dummies(df, columns=['sex', 'embarked', 'who'], drop_first=True)

# ------------------------------------------------------------------
# 12. SCALING numeric features (for distance-based / gradient-based models)
# ------------------------------------------------------------------
numeric_cols_to_scale = ['age', 'fare_log', 'family_size']
scaler = StandardScaler()
df_model[[c + '_scaled' for c in numeric_cols_to_scale]] = scaler.fit_transform(df_model[numeric_cols_to_scale])

# ------------------------------------------------------------------
# CORRELATION HEATMAP (final numeric feature set)
# ------------------------------------------------------------------
corr_cols = ['survived', 'pclass', 'age', 'fare_log', 'family_size', 'has_cabin_record', 'adult_male', 'alone']
plt.figure(figsize=(7, 6))
sns.heatmap(df_model[corr_cols].corr(), annot=True, fmt='.2f', cmap='coolwarm', center=0)
plt.title('Correlation Heatmap - Cleaned Numeric Features')
plt.tight_layout()
plt.savefig('chart_correlation_heatmap.png')
plt.close()

# ------------------------------------------------------------------
# SAVE OUTPUTS
# ------------------------------------------------------------------
df.to_csv('titanic_cleaned.csv', index=False)
df_model.to_csv('titanic_model_ready.csv', index=False)

# ------------------------------------------------------------------
# SUMMARY REPORT PRINTED TO STDOUT (captured for the Word document)
# ------------------------------------------------------------------
print("RAW SHAPE:", raw_shape)
print("CLEANED SHAPE (pre-encoding):", df.shape)
print("MODEL-READY SHAPE (post-encoding):", df_model.shape)
print("\nMissing values remaining in cleaned df:\n", df.isnull().sum().sum())
print("\nFlagged (coincidental) duplicate rows kept:", n_flagged_duplicates)
print("\nFare outliers (1.5xIQR) found:", n_outliers, "| bounds:", round(lower_bound,2), round(upper_bound,2))
print("Zero-fare rows flagged:", df['zero_fare_flag'].sum())
print("Rows with no cabin/deck record:", (df['has_cabin_record']==0).sum())
print("\nFinal dtypes:\n", df.dtypes)
print("\nSample of cleaned data:\n", df.head())
print("\nModel-ready columns:\n", list(df_model.columns))
