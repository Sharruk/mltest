# Practical Machine Learning Lab: Data Cleaning & Preprocessing Guide

Welcome to your practical data cleaning preparation guide. This module is designed for your **Machine Learning Lab Exam** to help you inspect, diagnose, clean, and preprocess any raw tabular CSV dataset independently using your **Machine Learning Algorithms Cheatsheet (Page 1)**.

---

## 📁 Directory Files

| File | Description |
| :--- | :--- |
| [`loan_applications_dirty.csv`](file:///c:/Directory/coding/3122247001061/mltest/data_cleaning_practice/loan_applications_dirty.csv) | The intentionally messy synthetic dataset containing realistic data quality flaws (120 rows). |
| [`01_data_cleaning.ipynb`](file:///c:/Directory/coding/3122247001061/mltest/data_cleaning_practice/01_data_cleaning.ipynb) | The step-by-step Jupyter Notebook implementing all 10 cleaning & preprocessing sections with cheatsheet references. |
| [`loan_applications_cleaned.csv`](file:///c:/Directory/coding/3122247001061/mltest/data_cleaning_practice/loan_applications_cleaned.csv) | The final, validated clean dataset exported by the notebook (116 valid rows, 0 nulls, standardized dtypes). |
| [`README.md`](file:///c:/Directory/coding/3122247001061/mltest/data_cleaning_practice/README.md) | This guide and exam execution instructions. |

---

## 🚀 How to Run the Notebook with Your `ml-lab` Conda Environment

Your machine has the `ml-lab` Conda environment configured inside Ubuntu WSL. Follow these simple steps:

### Option A: From WSL Terminal (Recommended)
1. Open your terminal (PowerShell or Windows Terminal) and enter WSL:
   ```bash
   wsl -d Ubuntu
   ```
2. Activate your `ml-lab` Conda environment:
   ```bash
   conda activate ml-lab
   ```
   *(Or if conda is not in PATH: `source ~/miniconda3/bin/activate ml-lab`)*

3. Navigate to this practice folder:
   ```bash
   cd /mnt/c/Directory/coding/3122247001061/mltest/data_cleaning_practice
   ```

4. Launch Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
   *(Or launch JupyterLab with `jupyter lab`)*

5. Click on `01_data_cleaning.ipynb` and run the cells from top to bottom by pressing `Shift + Enter`.

### Option B: From VS Code / Antigravity IDE Directly
1. Open [`01_data_cleaning.ipynb`](file:///c:/Directory/coding/3122247001061/mltest/data_cleaning_practice/01_data_cleaning.ipynb) in the editor.
2. In the top-right corner of the notebook editor, click **Select Kernel**.
3. Choose **Python Environments...** -> Select **`ml-lab` (`/home/sharruk/miniconda3/envs/ml-lab/bin/python`)** or your local Python 3 kernel.
4. Click **Run All** or execute each cell sequentially.

---

## 🗺️ Cheatsheet Translation Map (Page 1)

During your lab exam, match the data problem you observe to the corresponding section of your printed cheatsheet:

| Observed Problem in Lab Exam | Cheatsheet Section (Page 1) | Cheatsheet Syntax to Use |
| :--- | :--- | :--- |
| **Inspect rows, shape & column types** | *Pandas / IO* | `df.head()`, `df.shape`, `df.info()`, `df.describe()`, `df.dtypes` |
| **Identify numeric vs. string columns** | *Feature Type Identification* | `df.select_dtypes(include=['int64','float64']).columns` |
| **Duplicate rows in dataset** | *Pandas / IO* | `df.duplicated().sum()`, `df = df.drop_duplicates().reset_index(drop=True)` |
| **Check missing values** | *Nulls & Masks* | `any(df["col"].isna())`, `df["col"].isnull().sum()` |
| **Fill missing numerical values** | *Fill / Impute* | `df["col"].fillna(value=df["col"].median(), inplace=True)` |
| **Fill with Scikit-Learn** | *Fill / Impute* | `imp = SimpleImputer(missing_values=np.nan, strategy="median")` |
| **Domain rule / Invalid data correction** | *Transformations* | `df["col"] = np.where(<condition>, <true_val>, <false_val>)` |
| **Detect extreme outliers** | *Outlier Analysis and Removal* | `Q1 = df["col"].quantile(0.25)`, `Q3 = df["col"].quantile(0.75)`, `IQR = Q3 - Q1`, `lower_bound = Q1 - 1.5 * IQR`, `upper_bound = Q3 + 1.5 * IQR` |
| **Encode unordered categories** | *Scaling & Encoding* | `X_oh = OneHotEncoder(handle_unknown="ignore").fit_transform(X_cat).toarray()` |
| **Encode ordered categories** | *Scaling & Encoding* | `X_ord = OrdinalEncoder(categories=[<ordered_list>]).fit_transform(X_cat)` |
| **Scale features (Normal data)** | *Scaling & Encoding* | `X_std = StandardScaler().fit_transform(X)` |
| **Scale features with extreme outliers** | *Scaling & Encoding* | `X_robust = RobustScaler().fit_transform(X)` |
| **Save cleaned data** | *Pandas / IO* | `df.to_csv("cleaned_data.csv", index=False)` |

---

## ⚠️ 5 Golden Rules for Your Practical Lab Exam

1. **Never Impute Target Labels:**  
   If the target column (e.g. `Loan_Approved`) has missing values, **drop** those rows (`df.dropna(subset=['target'])`). Never guess or fill target values, as this creates artificial ground truth.
2. **Do Not Blindly Assume the Last Column is the Target:**  
   In many exam datasets, the last column might be a timestamp (`Application_Date`) or an extra metadata column. Always inspect `df.columns` and specify your target column explicitly.
3. **Investigate Duplicate IDs Before Dropping:**  
   Duplicate full rows (`df.duplicated()`) are errors that should be dropped. However, a repeated ID with different dates or amounts may be a legitimate customer re-application.
4. **Inspect Coercion Errors:**  
   When converting strings to numbers with `pd.to_numeric(..., errors="coerce")`, inspect which rows turned to `NaN` before overwriting so you can identify invalid text like `"UNKNOWN"`.
5. **Detecting Outliers != Removing Outliers:**  
   Outlier detection with IQR identifies extreme observations. If the high value is legitimate real-world data (such as a high income of $450,000 for a senior executive), keep it in your dataset unless the exam explicitly tells you to drop it.
