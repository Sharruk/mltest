import pandas as pd
import numpy as np
import os

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

csv_path = r"c:\Directory\coding\3122247001061\mltest\ml\demo\machine_rul.csv"
df = pd.read_csv(csv_path)

print("=" * 60)
print("1. df.head():")
print(df.head())

print("\n" + "=" * 60)
print("2. df.shape:")
print(df.shape)

print("\n" + "=" * 60)
print("3. df.info():")
df.info()

print("\n" + "=" * 60)
print("4. df.describe():")
print(df.describe())

print("\n" + "=" * 60)
print("5. df.isnull().sum():")
print(df.isnull().sum())

print("\n" + "=" * 60)
print("6. df.dtypes:")
print(df.dtypes)

print("\n" + "=" * 60)
print("7. df.nunique():")
print(df.nunique())

print("\n" + "=" * 60)
print("8. df.duplicated().sum():")
print(df.duplicated().sum())

print("\n" + "=" * 60)
print("9. Additional Verifications:")
# 1. Observations count
assert df.shape[0] == 1000, f"Expected 1000 rows, got {df.shape[0]}"
print("[PASS] Exactly 1,000 observations verified.")

# 2. Record ID unique
assert df['Record ID'].nunique() == 1000, "Record ID is not unique"
print("[PASS] Record ID contains exactly 1,000 unique values.")

# 3. Target is numeric and has no missing values
assert pd.api.types.is_numeric_dtype(df['Remaining Useful Life Hours']), "Target is not numeric"
assert df['Remaining Useful Life Hours'].isnull().sum() == 0, "Target has missing values"
assert (df['Remaining Useful Life Hours'] > 0).all(), "Target contains non-positive values"
print("[PASS] Remaining Useful Life Hours is numeric, positive, and has 0 missing values.")

# 4. Numerical predictors contain some missing values
num_cols = ['Operating Hours', 'Temperature', 'Vibration', 'Pressure', 'Rotational Speed', 'Load', 'Maintenance Count', 'Ambient Temperature']
num_nulls = df[num_cols].isnull().sum()
assert num_nulls.sum() > 0, "Numerical predictors must contain some missing values"
print(f"[PASS] Numerical predictors missing values: {dict(num_nulls[num_nulls > 0])}")

# 5. Categorical predictors contain some missing values
cat_cols = ['Machine Type', 'Shift']
cat_nulls = df[cat_cols].isnull().sum()
assert cat_nulls.sum() > 0, "Categorical predictors must contain some missing values"
print(f"[PASS] Categorical predictors missing values: {dict(cat_nulls[cat_nulls > 0])}")

# 6. Correlated pairs check
corr = df[num_cols].corr()
print("\nCorrelated pairs (|r| >= 0.4):")
corr_pairs = []
for i in range(len(num_cols)):
    for j in range(i + 1, len(num_cols)):
        r = corr.iloc[i, j]
        if abs(r) >= 0.4:
            corr_pairs.append((num_cols[i], num_cols[j], round(r, 3)))
            print(f"  - {num_cols[i]} & {num_cols[j]}: r = {r:.3f}")
assert len(corr_pairs) > 0, "No correlated pairs found"
print("[PASS] Correlated numerical pairs verified.")

# 7. Outliers check (IQR method)
print("\nOutlier counts (1.5 * IQR above Q3 or below Q1):")
for col in ['Temperature', 'Vibration', 'Pressure']:
    series = df[col].dropna()
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    outliers = series[(series < q1 - 1.5 * iqr) | (series > q3 + 1.5 * iqr)]
    print(f"  - {col}: {len(outliers)} outliers detected (min: {series.min():.2f}, max: {series.max():.2f})")
    assert len(outliers) > 0, f"Expected outliers in {col}"
print("[PASS] Mild outliers verified.")

print("\nAll verifications passed successfully!")
