# ============================================================
# DECISION TREE CLASSIFIER - COMPLETE ML WORKFLOW
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import ttest_rel, wilcoxon

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

# 1. LOAD DATA
DATA_PATH = "student_performance.csv"  # CHANGE
TARGET = "Result"                       # CHANGE

df = pd.read_csv(DATA_PATH)
df = df.drop_duplicates().dropna(subset=[TARGET]).copy()

print("Shape:", df.shape)
print(df.head())
print(df.info())
print("Missing values:\n", df.isnull().sum())
print("Target distribution:\n", df[TARGET].value_counts())

# 2. EDA - TWO PLOTS
df[TARGET].value_counts().plot(kind="bar")
plt.title("Target Distribution")
plt.show()

num_eda = df.drop(columns=[TARGET]).select_dtypes(include=np.number)
if num_eda.shape[1] >= 2:
    sns.heatmap(num_eda.corr(), cmap="coolwarm")
    plt.title("Feature Correlation")
    plt.show()

# 3. FEATURES AND TARGET
X = df.drop(columns=[TARGET])
y = df[TARGET]

num_cols = X.select_dtypes(include=np.number).columns.tolist()
cat_cols = X.select_dtypes(exclude=np.number).columns.tolist()

# 4. TRAIN-TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 5. PREPROCESSING
try:
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
except TypeError:
    encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

num_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

cat_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", encoder)
])

preprocessor = ColumnTransformer([
    ("num", num_pipe, num_cols),
    ("cat", cat_pipe, cat_cols)
], sparse_threshold=0.0)

# 6. MODEL, GRID SEARCH AND PCA COMPARISON
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

param_grid = {
    "model__criterion": ["gini", "entropy"],
    "model__max_depth": [None, 3, 5, 10],
    "model__min_samples_split": [2, 5],
    "model__min_samples_leaf": [1, 2]
}

fold_results = {}
results = []

for use_pca in [False, True]:

    steps = [("preprocessor", preprocessor)]

    if use_pca:
        steps.extend([
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=0.95))
        ])

    steps.append(("model", DecisionTreeClassifier(random_state=42)))
    pipe = Pipeline(steps)

    search = GridSearchCV(
        pipe, param_grid, cv=cv,
        scoring="accuracy", n_jobs=-1
    )
    search.fit(X_train, y_train)

    best_model = search.best_estimator_
    y_pred = best_model.predict(X_test)
    idx = search.best_index_

    fold_scores = np.array([
        search.cv_results_[f"split{i}_test_score"][idx]
        for i in range(5)
    ])

    setting = "With PCA" if use_pca else "Without PCA"
    fold_results[setting] = fold_scores

    print("\nDECISION TREE:", setting)
    print("Best parameters:", search.best_params_)
    print("Mean 5-fold CV accuracy:", search.best_score_)
    print("Test accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(
        y_test, y_pred, average="weighted", zero_division=0
    ))
    print("Recall:", recall_score(
        y_test, y_pred, average="weighted", zero_division=0
    ))
    print("F1-score:", f1_score(
        y_test, y_pred, average="weighted", zero_division=0
    ))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred, zero_division=0))

    if use_pca:
        print("Explained variance:",
              best_model.named_steps["pca"].explained_variance_ratio_.sum())

    results.append({
        "Setting": setting,
        "Best Parameters": search.best_params_,
        "CV Accuracy": search.best_score_,
        "Test Accuracy": accuracy_score(y_test, y_pred)
    })

print("\nRESULT COMPARISON")
print(pd.DataFrame(results))

# 7. STATISTICAL SIGNIFICANCE: WITH PCA VS WITHOUT PCA
a = fold_results["Without PCA"]
b = fold_results["With PCA"]

if not np.allclose(a, b):
    print("\nPaired t-test p-value:", ttest_rel(a, b).pvalue)

    try:
        print("Wilcoxon p-value:", wilcoxon(a, b).pvalue)
    except ValueError:
        print("Wilcoxon test could not be calculated.")
else:
    print("Fold scores are identical; no difference was observed.")



# ============================================================
# DECISION TREE REGRESSOR - COMPLETE ML WORKFLOW
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import ttest_rel, wilcoxon

from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeRegressor

from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error
)

# 1. LOAD DATA
DATA_PATH = "student_performance.csv"  # CHANGE
TARGET = "Final_Score"                  # CHANGE

df = pd.read_csv(DATA_PATH)
df = df.drop_duplicates().copy()
df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
df = df.dropna(subset=[TARGET])

print("Shape:", df.shape)
print(df.head())
print(df.info())
print("Missing values:\n", df.isnull().sum())
print(df.describe())

# 2. EDA - TWO PLOTS
df[TARGET].hist(bins=15)
plt.title("Target Distribution")
plt.xlabel(TARGET)
plt.show()

num_eda = df.drop(columns=[TARGET]).select_dtypes(include=np.number)
if num_eda.shape[1] >= 2:
    sns.heatmap(num_eda.corr(), cmap="coolwarm")
    plt.title("Feature Correlation")
    plt.show()

# 3. FEATURES AND TARGET
X = df.drop(columns=[TARGET])
y = df[TARGET]

num_cols = X.select_dtypes(include=np.number).columns.tolist()
cat_cols = X.select_dtypes(exclude=np.number).columns.tolist()

# 4. TRAIN-TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5. PREPROCESSING
try:
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
except TypeError:
    encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

num_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

cat_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", encoder)
])

preprocessor = ColumnTransformer([
    ("num", num_pipe, num_cols),
    ("cat", cat_pipe, cat_cols)
], sparse_threshold=0.0)

# 6. MODEL, GRID SEARCH AND PCA COMPARISON
cv = KFold(n_splits=5, shuffle=True, random_state=42)

param_grid = {
    "model__criterion": ["squared_error", "absolute_error"],
    "model__max_depth": [None, 3, 5, 10],
    "model__min_samples_split": [2, 5],
    "model__min_samples_leaf": [1, 2]
}

fold_results = {}
results = []

for use_pca in [False, True]:

    steps = [("preprocessor", preprocessor)]

    if use_pca:
        steps.extend([
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=0.95))
        ])

    steps.append(("model", DecisionTreeRegressor(random_state=42)))
    pipe = Pipeline(steps)

    search = GridSearchCV(
        pipe, param_grid, cv=cv,
        scoring="r2", n_jobs=-1
    )
    search.fit(X_train, y_train)

    best_model = search.best_estimator_
    y_pred = best_model.predict(X_test)
    idx = search.best_index_

    fold_scores = np.array([
        search.cv_results_[f"split{i}_test_score"][idx]
        for i in range(5)
    ])

    setting = "With PCA" if use_pca else "Without PCA"
    fold_results[setting] = fold_scores

    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    print("\nDECISION TREE REGRESSOR:", setting)
    print("Best parameters:", search.best_params_)
    print("Mean 5-fold CV R2:", search.best_score_)
    print("Test R2:", r2)
    print("MAE:", mae)
    print("MSE:", mse)
    print("RMSE:", np.sqrt(mse))

    # Adjusted R2
    p = best_model.named_steps["preprocessor"].transform(X_test).shape[1]
    if use_pca:
        p = best_model.named_steps["pca"].n_components_

    n = len(y_test)
    if n > p + 1:
        adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
        print("Adjusted R2:", adjusted_r2)

    if use_pca:
        print("Explained variance:",
              best_model.named_steps["pca"].explained_variance_ratio_.sum())

    results.append({
        "Setting": setting,
        "Best Parameters": search.best_params_,
        "CV R2": search.best_score_,
        "Test R2": r2,
        "MAE": mae,
        "MSE": mse,
        "RMSE": np.sqrt(mse)
    })

    # Regression plots
    plt.scatter(y_test, y_pred)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(f"Actual vs Predicted - {setting}")
    plt.show()

    residuals = np.asarray(y_test) - np.asarray(y_pred)
    plt.scatter(y_pred, residuals)
    plt.axhline(y=0, linestyle="--")
    plt.xlabel("Predicted")
    plt.ylabel("Residual")
    plt.title(f"Residual Plot - {setting}")
    plt.show()

print("\nRESULT COMPARISON")
print(pd.DataFrame(results))

# 7. STATISTICAL SIGNIFICANCE: WITH PCA VS WITHOUT PCA
a = fold_results["Without PCA"]
b = fold_results["With PCA"]

if not np.allclose(a, b):
    print("\nPaired t-test p-value:", ttest_rel(a, b).pvalue)

    try:
        print("Wilcoxon p-value:", wilcoxon(a, b).pvalue)
    except ValueError:
        print("Wilcoxon test could not be calculated.")
else:
    print("Fold scores are identical; no difference was observed.")




# ============================================================
# K-MEANS CLUSTERING - COMPLETE WORKFLOW
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from sklearn.metrics import (
    silhouette_score, calinski_harabasz_score,
    davies_bouldin_score, adjusted_rand_score,
    normalized_mutual_info_score
)

# 1. LOAD DATA
DATA_PATH = "student_performance.csv"  # CHANGE

# Set to a real class-label column if provided; otherwise None.
LABEL_COLUMN = None  # Example: "Result"

df = pd.read_csv(DATA_PATH)
df = df.drop_duplicates().copy()

print("Shape:", df.shape)
print(df.head())
print(df.info())
print("Missing values:\n", df.isnull().sum())
print(df.describe(include="all"))

# 2. SEPARATE FEATURES AND OPTIONAL TRUE LABELS
y_true = df[LABEL_COLUMN].copy() if LABEL_COLUMN in df.columns else None

X = df.drop(
    columns=[LABEL_COLUMN] if LABEL_COLUMN in df.columns else []
)

# CHANGE: drop identifier columns if they are not useful features.
X = X.drop(columns=["Student_ID"], errors="ignore")

num_cols = X.select_dtypes(include=np.number).columns.tolist()
cat_cols = X.select_dtypes(exclude=np.number).columns.tolist()

print("Numerical columns:", num_cols)
print("Categorical columns:", cat_cols)

# 3. EDA - TWO PLOTS
if y_true is not None:
    y_true.value_counts().plot(kind="bar")
    plt.title("Ground-Truth Label Distribution")
    plt.show()
elif num_cols:
    X[num_cols[0]].hist(bins=15)
    plt.title("Feature Distribution")
    plt.xlabel(num_cols[0])
    plt.show()

if len(num_cols) >= 2:
    sns.heatmap(X[num_cols].corr(), cmap="coolwarm")
    plt.title("Feature Correlation")
    plt.show()

# 4. PREPROCESSING
try:
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
except TypeError:
    encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

num_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

cat_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", encoder)
])

preprocessor = ColumnTransformer([
    ("num", num_pipe, num_cols),
    ("cat", cat_pipe, cat_cols)
], sparse_threshold=0.0)

X_processed = preprocessor.fit_transform(X)

# Scaling is useful because K-Means uses distances.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_processed)

# 5. ELBOW METHOD + SILHOUETTE
k_values = range(2, min(8, len(X_scaled) - 1) + 1)
inertia_values = []
silhouette_values = []

for k in k_values:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels_k = km.fit_predict(X_scaled)

    inertia_values.append(km.inertia_)
    silhouette_values.append(silhouette_score(X_scaled, labels_k))

plt.plot(list(k_values), inertia_values, marker="o")
plt.title("Elbow Method")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia / WCSS")
plt.show()

plt.plot(list(k_values), silhouette_values, marker="o")
plt.title("Silhouette Score vs K")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Silhouette Score")
plt.show()

# 6. FINAL K-MEANS
K = 3  # CHANGE: choose based on elbow/silhouette results

kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
labels = kmeans.fit_predict(X_scaled)

print("Inertia / WCSS:", kmeans.inertia_)
print("Cluster labels:", labels)
print("Cluster centres:\n", kmeans.cluster_centers_)

# 7. CLUSTERING METRICS
print("Silhouette Score:", silhouette_score(X_scaled, labels))
print("Calinski-Harabasz Score:",
      calinski_harabasz_score(X_scaled, labels))
print("Davies-Bouldin Score:",
      davies_bouldin_score(X_scaled, labels))

# External metrics only if actual labels are available.
if y_true is not None:
    print("Adjusted Rand Index:",
          adjusted_rand_score(y_true, labels))
    print("Normalized Mutual Information:",
          normalized_mutual_info_score(y_true, labels))

# 8. PCA VISUALIZATION
if X_scaled.shape[1] >= 2:
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    print("PCA explained variance:",
          pca.explained_variance_ratio_.sum())

    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels)
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title("K-Means Clusters Visualized with PCA")
    plt.show()

# Statistical significance:
# A single K-Means fit does not itself provide a statistical
# significance test. A test requires a defined comparison,
# suitable repeated results, and an appropriate test design.


