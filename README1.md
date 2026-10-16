# ============================================
# KNN CLASSIFICATION: COMPLETE ML WORKFLOW
# ============================================

import numpy as np
import pandas as pd

from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    GridSearchCV,
    StratifiedKFold
)

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# 1. LOAD DATA
df = pd.read_csv("student_performance.csv")

print("Shape:", df.shape)
print(df.head())
print(df.isnull().sum())

# 2. SEPARATE FEATURES AND TARGET
X = df.drop(columns=["Result"])
y = df["Result"]

# 3. IDENTIFY NUMERICAL AND CATEGORICAL COLUMNS
num_cols = X.select_dtypes(include=np.number).columns.tolist()
cat_cols = X.select_dtypes(exclude=np.number).columns.tolist()

print("Numerical columns:", num_cols)
print("Categorical columns:", cat_cols)

# 4. SPLIT DATA BEFORE LEARNED PREPROCESSING
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 5. NUMERICAL PREPROCESSING
num_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# 6. CATEGORICAL PREPROCESSING
cat_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

# 7. COMBINE PREPROCESSING
preprocessor = ColumnTransformer([
    ("num", num_pipe, num_cols),
    ("cat", cat_pipe, cat_cols)
])

# 8. BUILD COMPLETE PIPELINE
knn_pipe = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", KNeighborsClassifier())
])

# 9. CROSS-VALIDATION BASELINE
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

cv_scores = cross_val_score(
    knn_pipe,
    X_train,
    y_train,
    cv=cv,
    scoring="accuracy",
    n_jobs=-1
)

print("\nCross-validation scores:", cv_scores)
print("Mean CV accuracy:", cv_scores.mean())

# 10. HYPERPARAMETER TUNING
param_grid = {
    "classifier__n_neighbors": [3, 5, 7, 9],
    "classifier__weights": ["uniform", "distance"],
    "classifier__metric": ["euclidean", "manhattan"]
}

search = GridSearchCV(
    estimator=knn_pipe,
    param_grid=param_grid,
    cv=cv,
    scoring="accuracy",
    n_jobs=-1,
    refit=True
)

search.fit(X_train, y_train)

print("\nBest parameters:", search.best_params_)
print("Best CV accuracy:", search.best_score_)

# 11. PREDICTION ON TEST DATA
best_model = search.best_estimator_
y_pred = best_model.predict(X_test)

# 12. FINAL EVALUATION
print("\nTest accuracy:", accuracy_score(y_test, y_pred))

print("\nConfusion matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification report:")
print(classification_report(y_test, y_pred))

