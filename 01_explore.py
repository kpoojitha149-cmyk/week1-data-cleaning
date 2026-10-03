import pandas as pd
import numpy as np

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 120)

df = pd.read_csv('titanic.csv')

print("="*70)
print("SHAPE:", df.shape)
print("="*70)
print("\nDTYPES:\n", df.dtypes)
print("="*70)
print("\nHEAD:\n", df.head())
print("="*70)
print("\nDESCRIBE (numeric):\n", df.describe())
print("="*70)
print("\nDESCRIBE (categorical):\n", df.describe(include='object'))
print("="*70)
print("\nMISSING VALUES (count):\n", df.isnull().sum())
print("\nMISSING VALUES (%):\n", (df.isnull().sum() / len(df) * 100).round(2))
print("="*70)
print("\nDUPLICATE ROWS:", df.duplicated().sum())
print("="*70)
print("\nUNIQUE VALUES PER COLUMN:\n", df.nunique())
print("="*70)
# Check redundant columns
print("\nsurvived vs alive crosstab:\n", pd.crosstab(df['survived'], df['alive']))
print("\npclass vs class crosstab:\n", pd.crosstab(df['pclass'], df['class']))
print("\nsex vs who crosstab:\n", pd.crosstab(df['sex'], df['who']))
print("\nembarked vs embark_town crosstab:\n", pd.crosstab(df['embarked'], df['embark_town']))
