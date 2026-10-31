"""Employee attrition model (logistic regression) with honest evaluation.

Gender and Age are deliberately NOT used as features: acting on a prediction
that uses protected characteristics would be discriminatory. They are kept
in the data only so the fairness audit can check the outcomes by group.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_score,
    recall_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC = ["MonthlyIncome", "YearsAtCompany", "YearsSinceLastPromotion",
           "DistanceFromHome", "JobSatisfaction", "WorkLifeBalance"]
CATEGORICAL = ["OverTime", "Department"]
FEATURES = NUMERIC + CATEGORICAL
AUDIT_COLUMNS = ["Gender", "Age"]
TARGET = "Attrition"
REQUIRED = FEATURES + [TARGET]


def load_employees(source) -> tuple[pd.DataFrame, list[str]]:
    df = pd.read_csv(source)
    df.columns = [c.strip() for c in df.columns]
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))
    keep = REQUIRED + [c for c in AUDIT_COLUMNS if c in df.columns]
    df = df[keep].copy()
    warnings: list[str] = []

    df[TARGET] = df[TARGET].astype(str).str.strip().str.lower().map(
        {"yes": 1, "no": 0, "1": 1, "0": 0, "true": 1, "false": 0}
    )
    n0 = len(df)
    for c in NUMERIC:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=NUMERIC + CATEGORICAL + [TARGET])
    if len(df) < n0:
        warnings.append(f"{n0 - len(df)} row(s) with missing or invalid values were dropped.")
    df[TARGET] = df[TARGET].astype(int)
    return df.reset_index(drop=True), warnings


def _pipeline(seed: int) -> Pipeline:
    pre = ColumnTransformer(
        [
            ("num", StandardScaler(), NUMERIC),
            ("cat", OneHotEncoder(handle_unknown="ignore", drop="if_binary"), CATEGORICAL),
        ]
    )
    clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed)
    return Pipeline([("prep", pre), ("clf", clf)])


def train_and_evaluate(df: pd.DataFrame, test_size: float = 0.25, seed: int = 42) -> dict:
    """Train on a stratified split and report held-out metrics."""
    if df[TARGET].nunique() < 2:
        raise ValueError("The target has only one class; cannot train a model.")
    if df[TARGET].value_counts().min() < 10:
        raise ValueError("Need at least 10 examples of each class.")

    X, y = df[FEATURES], df[TARGET]
    X_tr, X_te, y_tr, y_te, idx_tr, idx_te = train_test_split(
        X, y, df.index, test_size=test_size, random_state=seed, stratify=y
    )
    model = _pipeline(seed).fit(X_tr, y_tr)
    proba = model.predict_proba(X_te)[:, 1]
    pred = (proba >= 0.5).astype(int)

    cv = cross_val_score(
        _pipeline(seed), X, y, cv=StratifiedKFold(5, shuffle=True, random_state=seed), scoring="roc_auc"
    )
    base_acc = max(y_te.mean(), 1 - y_te.mean())  # always predict the majority class

    names = model.named_steps["prep"].get_feature_names_out()
    coefs = model.named_steps["clf"].coef_[0]
    importance = (
        pd.DataFrame({"Feature": [n.split("__", 1)[1] for n in names],
                      "Coefficient": coefs.round(3),
                      "Odds ratio": np.exp(coefs).round(2)})
        .sort_values("Coefficient", key=np.abs, ascending=False)
        .reset_index(drop=True)
    )
    tn, fp, fn, tp = confusion_matrix(y_te, pred, labels=[0, 1]).ravel()

    return {
        "model": model,
        "metrics": {
            "Accuracy": round(accuracy_score(y_te, pred), 3),
            "Baseline accuracy (always 'stays')": round(base_acc, 3),
            "Precision": round(precision_score(y_te, pred, zero_division=0), 3),
            "Recall": round(recall_score(y_te, pred, zero_division=0), 3),
            "F1": round(f1_score(y_te, pred, zero_division=0), 3),
            "ROC AUC (test)": round(roc_auc_score(y_te, proba), 3),
            "ROC AUC (5-fold CV mean)": round(float(cv.mean()), 3),
            "ROC AUC (5-fold CV std)": round(float(cv.std()), 3),
        },
        "confusion": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)},
        "importance": importance,
        "test_index": idx_te,
        "y_test": y_te.to_numpy(),
        "y_pred": pred,
        "n_train": len(X_tr),
        "n_test": len(X_te),
        "attrition_rate": round(float(y.mean()), 3),
    }
