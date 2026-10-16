source /opt/anaconda3/bin/activate
anaconda-navigator


# ML Lab Practical — Assignments 1–9
## Simple, commented code with cheatsheet references

**Use:** Copy and run one assignment at a time in a separate notebook cell. Change the filename and target column to match the question. The filenames/columns below are examples; inspect `df.head()` and `df.columns` before running.

---

# Assignment 1 — EDA and Data Preprocessing
**Sir-style question:** Load a dataset, inspect it, handle missing values, identify outliers and visualise data.  
**Cheatsheet:** Page 1, Section 1 (Pandas/IO, Nulls, Imputation, Outliers, Feature Types); Page 3, Section 5 (Visualisation).

```python
# ASSIGNMENT 1: EDA AND PREPROCESSING
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# CHEATSHEET P1: Pandas / IO
# CHANGE filename to the dataset given by Sir
df = pd.read_csv("Diabetes_prediction.csv")

print(df.head())             # First 5 rows
print(df.shape)              # Rows and columns
df.info()                    # Data types and non-null counts
print(df.describe())         # Statistics
print(df.isnull().sum())     # Missing values

# CHEATSHEET P1: Fill / Impute
num_cols = df.select_dtypes(include=np.number).columns
for col in num_cols:
    df[col] = df[col].fillna(df[col].median())

# CHEATSHEET P1: Outlier Analysis and Removal
col = num_cols[0]            # CHANGE: select the column asked about
Q1 = df[col].quantile(0.25)
Q3 = df[col].quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR
print("Outliers:", ((df[col] < lower) | (df[col] > upper)).sum())

# CHEATSHEET P3: Histograms & Boxplots
sns.histplot(data=df, x=col, kde=True)
plt.title("Distribution of " + col)
plt.show()

sns.heatmap(df.select_dtypes(include=np.number).corr(),
            annot=True, cmap="coolwarm")
plt.show()

# OPTIONAL: remove outliers ONLY if asked
# df = df[(df[col] >= lower) & (df[col] <= upper)]
```

---

# Assignment 2 — Naive Bayes and KNN
**Sir-style question:** Build Naive Bayes and KNN classifiers, tune KNN with GridSearchCV, and compare accuracy.  
**Cheatsheet:** Page 1 (Split); Page 2, Section 2 (Naive Bayes/KNN); Page 2, Section 4 (GridSearchCV); Page 3, Section 5 (metrics).

This code assumes the common SMS spam dataset: `v1` = label, `v2` = message. Change names if yours differ.

```python
# ==========================================
# EXPERIMENT 2: NAIVE BAYES & KNN
# ==========================================

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay

# 1. Load Data
df = pd.read_csv("spam.csv", encoding="latin-1")
# Assume target is 'label' or 'class' and features are numeric
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# 2. Train-Test Split & Scaling
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. Naive Bayes Variants (Cheatsheet Sec 2)
models = {
    "Gaussian NB": GaussianNB(),
    "Bernoulli NB": BernoulliNB()
}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    print(f"--- {name} Accuracy: {accuracy_score(y_test, y_pred):.4f} ---")

# 4. KNN with GridSearchCV (Cheatsheet Sec 9 Param Grid)
knn = KNeighborsClassifier()
knn_params = {
    'n_neighbors': [3, 5, 7, 9],
    'algorithm': ['kd_tree', 'ball_tree']
}

grid_knn = GridSearchCV(knn, knn_params, cv=5, scoring='accuracy', n_jobs=-1)
grid_knn.fit(X_train_scaled, y_train)

best_knn = grid_knn.best_estimator_
y_pred_knn = best_knn.predict(X_test_scaled)

print("\nBest KNN Params:", grid_knn.best_params_)
print("Best KNN CV Score:", grid_knn.best_score_)
print("KNN Test Accuracy:", accuracy_score(y_test, y_pred_knn))
print("\nClassification Report:\n", classification_report(y_test, y_pred_knn))

# 5. Confusion Matrix (Cheatsheet Sec 5)
ConfusionMatrixDisplay.from_predictions(y_test, y_pred_knn)
```

**Note:** If the dataset is already numeric, TF-IDF is unnecessary. Choose preprocessing according to the data type.

---

# Assignment 3 — Regression Models
**Sir-style question:** Predict `loan_amount` using Linear Regression, Ridge, Lasso and ElasticNet. Compare cross-validation and regression metrics.  
**Cheatsheet:** Page 1 (Split/Scaling); Page 2, Section 2 (Linear models); Page 2, Section 4 (CV); Page 3, Section 5 (Regression metrics).

```python
# ASSIGNMENT 3: REGRESSION
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# CHEATSHEET P1: Pandas / IO
df = pd.read_csv("loan_approval_dataset.csv")

# CHANGE: target column must exist
y = df["loan_amount"]
X = df.select_dtypes(include=np.number).drop(
    columns=["loan_amount", "loan_id"], errors="ignore"
)

# CHEATSHEET P1: Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# CHEATSHEET P1: Scaling (fit on train, transform test)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# CHEATSHEET P2: Linear models
models = {
    "Linear": LinearRegression(),
    "Ridge": Ridge(alpha=1.0),
    "Lasso": Lasso(alpha=0.1),
    "ElasticNet": ElasticNet(alpha=0.1)
}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # CHEATSHEET P2: Cross-validation
    cv = cross_val_score(model, X_train, y_train, cv=5, scoring="r2")

    # CHEATSHEET P3: Regression Metrics
    print("\\nMODEL:", name)
    print("Mean 5-Fold CV R2:", np.mean(cv))
    print("Test R2:", r2_score(y_test, y_pred))
    print("Test MAE:", mean_absolute_error(y_test, y_pred))
    print("Test MSE:", mean_squared_error(y_test, y_pred))
```

R² is generally better higher; MAE and MSE are better lower. Encode any categorical features needed for prediction rather than silently dropping them.

---

# Assignment 4 — Logistic Regression and SVM
**Sir-style question:** Train Logistic Regression and SVM with different kernels, compare accuracy and plot ROC.  
**Cheatsheet:** Page 2, Section 2 (Logistic Regression/SVM); Page 3, Section 5 (metrics/ROC).

Assumes the same SMS spam CSV columns as Assignment 2.

```python
# ASSIGNMENT 4: LOGISTIC REGRESSION AND SVM
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, RocCurveDisplay

# CHEATSHEET P1: Pandas / IO
df = pd.read_csv("spam.csv", encoding="latin-1")
X = df["v2"].fillna("")       # CHANGE column names if needed
y = df["v1"]

# Convert raw text to numbers
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X)

# CHEATSHEET P1: Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# CHEATSHEET P2: Linear & Logistic
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred))

# CHEATSHEET P2: Naive Bayes, KNN, SVM
for kernel in ["linear", "rbf", "poly"]:
    svm = SVC(kernel=kernel)
    svm.fit(X_train, y_train)
    y_pred = svm.predict(X_test)
    print(kernel, "Accuracy:", accuracy_score(y_test, y_pred))

# CHEATSHEET P3: ROC Curve
RocCurveDisplay.from_estimator(lr, X_test, y_test)
plt.title("Logistic Regression ROC Curve")
plt.show()
```

---

# Assignment 5 — Decision Tree and Random Forest
**Sir-style question:** Tune a Decision Tree with GridSearchCV, compare Random Forest accuracy and plot feature importance.  
**Cheatsheet:** Page 1 (Split); Page 2, Section 2 (Decision Tree/Random Forest); Page 2, Section 4 (GridSearchCV); Page 3, Section 5 (Feature Importance); Page 4 (parameter grid).

```python
# ASSIGNMENT 5: DECISION TREE AND RANDOM FOREST
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# CHEATSHEET P1: Pandas / IO
df = pd.read_csv("breast_cancer.csv")

# CHANGE if target is not the last column
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# CHEATSHEET P1: Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# CHEATSHEET P2: Decision Tree
dt = DecisionTreeClassifier(random_state=42)

# CHEATSHEET P4: Decision Tree Parameter Grid
params = {
    "criterion": ["gini", "entropy"],
    "max_depth": [3, 5, 7, None],
    "min_samples_split": [2, 5, 10]
}

# CHEATSHEET P2: GridSearchCV
grid = GridSearchCV(dt, params, cv=5, scoring="accuracy", n_jobs=-1)
grid.fit(X_train, y_train)
best_dt = grid.best_estimator_
dt_pred = best_dt.predict(X_test)

print("Best DT Parameters:", grid.best_params_)
print("DT CV Accuracy:", grid.best_score_)
print("DT Test Accuracy:", accuracy_score(y_test, dt_pred))

# CHEATSHEET P2: Bagging & RandomForest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
print("RF Test Accuracy:", accuracy_score(y_test, rf_pred))

# CHEATSHEET P3: Feature Importance
importance = pd.Series(rf.feature_importances_, index=X.columns).nlargest(10)
importance.plot(kind="barh")
plt.title("Top 10 Important Features")
plt.show()
```

---

# Assignment 6 — Ensemble Methods
**Sir-style question:** Implement Bagging, AdaBoost, Gradient Boosting and Stacking, then compare accuracy.  
**Cheatsheet:** Page 2, Section 2 (Bagging/Boosting/Stacking); Page 3, Section 5 (metrics).

```python
# ASSIGNMENT 6: ENSEMBLE METHODS
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import (
    BaggingClassifier, AdaBoostClassifier,
    GradientBoostingClassifier, StackingClassifier
)
from sklearn.metrics import accuracy_score

# CHEATSHEET P1: Pandas / IO
df = pd.read_csv("breast_cancer.csv")

# CHANGE if target is not the last column
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# CHEATSHEET P1: Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# CHEATSHEET P2: Bagging
bag = BaggingClassifier(
    estimator=DecisionTreeClassifier(max_depth=3),
    n_estimators=50, random_state=42
)
bag.fit(X_train, y_train)

# CHEATSHEET P2: Boosting
ada = AdaBoostClassifier(n_estimators=50, random_state=42)
ada.fit(X_train, y_train)

gb = GradientBoostingClassifier(n_estimators=50, random_state=42)
gb.fit(X_train, y_train)

# CHEATSHEET P2: Stacking
base_models = [
    ("knn", KNeighborsClassifier(n_neighbors=5)),
    ("dt", DecisionTreeClassifier(max_depth=3))
]
stack = StackingClassifier(
    estimators=base_models,
    final_estimator=LogisticRegression(),
    cv=5
)
stack.fit(X_train, y_train)

# CHEATSHEET P2: General + P3: Classification Metrics
models = {
    "Bagging": bag,
    "AdaBoost": ada,
    "Gradient Boosting": gb,
    "Stacking": stack
}
for name, model in models.items():
    pred = model.predict(X_test)
    print(name, "Accuracy:", accuracy_score(y_test, pred))
```

---

# Assignment 7 — PCA and Model Evaluation
**Sir-style question:** Compare Logistic Regression with and without PCA, report explained variance and plot principal components.  
**Cheatsheet:** Page 1 (Split/Scaling); Page 2, Section 2 (Logistic Regression); Page 3, Section 5 (metrics/visualisation); Page 4, Section 6 (PCA).

```python
# ASSIGNMENT 7: PCA COMPARISON
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# CHEATSHEET P1: Pandas / IO
df = pd.read_csv("breast_cancer.csv")

# CHANGE if target is not the last column
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# CHEATSHEET P1: Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# CHEATSHEET P1: Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 1. Model WITHOUT PCA — CHEATSHEET P2: Logistic Regression
model1 = LogisticRegression(max_iter=1000)
model1.fit(X_train, y_train)
pred1 = model1.predict(X_test)
print("Accuracy WITHOUT PCA:", accuracy_score(y_test, pred1))

# 2. PCA — CHEATSHEET P4: Principal Component Analysis
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

print("Variance per component:", pca.explained_variance_ratio_)
print("Total variance retained:", np.sum(pca.explained_variance_ratio_))

# 3. Model WITH PCA
model2 = LogisticRegression(max_iter=1000)
model2.fit(X_train_pca, y_train)
pred2 = model2.predict(X_test_pca)
print("Accuracy WITH PCA:", accuracy_score(y_test, pred2))

# CHEATSHEET P3: Dimensionality Reduction
plt.scatter(
    X_train_pca[:, 0], X_train_pca[:, 1],
    c=pd.factorize(y_train)[0]
)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA Visualisation")
plt.show()
```

---

# Assignment 8 — Clustering Algorithms
**Sir-style question:** Apply K-Means, use the elbow method to examine the number of clusters and evaluate results. Also includes DBSCAN and Agglomerative Clustering.  
**Cheatsheet:** Page 1 (Pandas/Scaling); Page 2, Section 3 (K-Means); Page 3, Section 5 (clustering metrics).

```python
# ASSIGNMENT 8: CLUSTERING
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score

# CHEATSHEET P1: Pandas / IO
df = pd.read_csv("iris.csv")

# Numeric features only; remove target if present
X = df.select_dtypes(include="number")
X = X.drop(columns=["target"], errors="ignore")

# CHEATSHEET P1: Scaling
X_scaled = StandardScaler().fit_transform(X)

# 1. Elbow Method — CHEATSHEET P2: K-Means / inertia_
wcss = []
for k in range(2, 9):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    wcss.append(km.inertia_)

# CHEATSHEET P3: Plotting
plt.plot(range(2, 9), wcss, marker="o")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.title("Elbow Method")
plt.show()

# 2. K-Means — CHEATSHEET P2: K-Means
km = KMeans(n_clusters=3, random_state=42, n_init=10)
labels_km = km.fit_predict(X_scaled)
print("K-Means Silhouette Score:", silhouette_score(X_scaled, labels_km))

# 3. DBSCAN — additional import from sklearn.cluster
db = DBSCAN(eps=0.8, min_samples=3)
labels_db = db.fit_predict(X_scaled)
non_noise = labels_db != -1
n_clusters = len(set(labels_db[non_noise]))

if n_clusters >= 2 and non_noise.sum() > n_clusters:
    print("DBSCAN Silhouette:",
          silhouette_score(X_scaled[non_noise], labels_db[non_noise]))
else:
    print("DBSCAN did not produce enough non-noise clusters for silhouette.")

# 4. Agglomerative Clustering — additional import from sklearn.cluster
agg = AgglomerativeClustering(n_clusters=3)
labels_agg = agg.fit_predict(X_scaled)
print("Agglomerative Silhouette:", silhouette_score(X_scaled, labels_agg))
```

---

# Assignment 9 — Perceptron and MLP
**Sir-style question:** Train Perceptron and MLP classifiers, compare accuracy and plot MLP training loss.  
**Cheatsheet:** Page 1 (Split/Scaling); Page 2, Section 2 (general model training); Page 3, Section 5 (metrics/visualisation).

**Scope note:** Your instruction document says not to prioritise MLP, so practise the earlier assignments first.

```python
# ASSIGNMENT 9: PERCEPTRON VS MLP
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report

# CHEATSHEET P1: Pandas / IO
df = pd.read_csv("digits.csv")

# CHANGE if target is not the last column
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# CHEATSHEET P1: Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# CHEATSHEET P1: Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 1. Perceptron — additional import from sklearn.linear_model
p = Perceptron(max_iter=100, random_state=42)
p.fit(X_train, y_train)
pred_p = p.predict(X_test)
print("Perceptron Accuracy:", accuracy_score(y_test, pred_p))

# 2. MLP — additional import from sklearn.neural_network
mlp = MLPClassifier(
    hidden_layer_sizes=(64, 32), max_iter=200, random_state=42
)
mlp.fit(X_train, y_train)
pred_mlp = mlp.predict(X_test)
print("MLP Accuracy:", accuracy_score(y_test, pred_mlp))
print(classification_report(y_test, pred_mlp))

# CHEATSHEET P3: Training Dynamics
plt.plot(mlp.loss_curve_)
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("MLP Training Loss")
plt.show()
```

---

# Quick Cheatsheet Page Finder

- **Page 1:** Load data, inspect columns, missing values, imputation, outliers, encoding, scaling, train/test split.
- **Page 2, Section 2:** Classification/regression models, Decision Tree, KNN, Naive Bayes, SVM, Random Forest, Boosting and Stacking.
- **Page 2, Section 3:** K-Means.
- **Page 2, Section 4:** K-Fold, cross-validation, GridSearchCV and RandomizedSearchCV.
- **Page 3, Section 5:** Classification, regression and clustering metrics; ROC curves, confusion matrices and plots.
- **Page 4, Section 6:** PCA.
- **Page 4, Section 7:** Statistical significance tests (Friedman and one-way ANOVA).
- **Page 4, Section 8:** Valid metric strings for scikit-learn.
- **Page 4, Section 9:** NumPy and Pandas quick reference.

## Three exam rules

1. Inspect first: `df.head()`, `df.columns`, `df.info()`, `df.isnull().sum()`.
2. Check the target column; do not assume it is the last column.
3. Avoid data leakage: fit scalers and PCA on training data only; transform test data with the fitted objects.

**Final note:** These are learning templates. Verify filenames, column names, data types and preprocessing before running. Text datasets need text vectorisation; categorical features need encoding; numeric missing values need suitable imputation.
