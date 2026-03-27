# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 19:52:25 2026

@author: Ahsan
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from xgboost import XGBClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

# 1. Load and Clean Data
df = pd.read_csv(r"C:\Users\Ahsan\Downloads\Task2  predicting customer\dataset.csv")
df.drop(columns=["customerID"], errors='ignore', inplace=True)

# Handle TotalCharges (convert strings to NaN, then we'll impute in the pipeline)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors='coerce')
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# 2. Feature Selection
X = df.drop("Churn", axis=1)
y = df["Churn"]

# Define specific columns to treat correctly
num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
cat_cols = [col for col in X.columns if col not in num_cols]

# 3. Train-Test Split (with Stratify to maintain churn ratio)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Building the Preprocessing Layers
num_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

cat_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", num_transformer, num_cols),
    ("cat", cat_transformer, cat_cols)
])

# 5. The Full Production Pipeline (includes SMOTE and XGBoost)
# Note: Using ImbPipeline allows SMOTE to work inside the cross-validation
full_pipeline = ImbPipeline(steps=[
    ("preprocessor", preprocessor),
    ("smote", SMOTE(random_state=42)),
    ("model", XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42))
])

# 6. Hyperparameter Tuning (Optimized for XGBoost)
param_grid = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [3, 5, 7],
    "model__learning_rate": [0.01, 0.1, 0.2],
    "model__subsample": [0.8, 1.0]
}

print("Starting Grid Search... this may take a minute.")
grid_search = GridSearchCV(
    full_pipeline, 
    param_grid, 
    cv=5, 
    scoring="f1",  # Optimizing for F1 to balance catching churners vs false alarms
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

# 7. Evaluation
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)[:, 1]

print("\n--- Model Performance ---")
print(f"Best Params: {grid_search.best_params_}")
print(classification_report(y_test, y_pred))
print(f"ROC-AUC Score: {roc_auc_score(y_test, y_proba):.4f}")

# 8. Export the Complete Pipeline
joblib.dump(best_model, "churn_pipeline_v2.pkl")
print("\nPipeline exported as 'churn_pipeline_v2.pkl'")
