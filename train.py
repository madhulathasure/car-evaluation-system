import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)

from src.config import (PROCESSED_DATA, TARGET_COL, MODEL_PATH, METRICS_PATH,
                        COMPARISON_PATH, CONFUSION_PATH, OUTPUT_DIR,
                        RANDOM_STATE, CV_FOLDS)
from src.preprocess import get_train_test


def evaluate(y_true, y_pred, name):
    return {
        "model": name,
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision_macro": round(precision_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "recall_macro": round(recall_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "f1_macro": round(f1_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "f1_weighted": round(f1_score(y_true, y_pred, average="weighted", zero_division=0), 4),
    }


def train():
    print("Loading train/test split...")
    X_train, X_test, y_train, y_test = get_train_test()

    models = {
        "DecisionTree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "RandomForest": RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                                random_state=RANDOM_STATE, n_jobs=-1),
        "KNN": KNeighborsClassifier(n_neighbors=7),
        "NaiveBayes": GaussianNB(),
        "SVM": SVC(kernel="rbf", C=10, gamma="scale", class_weight="balanced"),
        "LogisticRegression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    }

    results, fitted = [], {}
    cms = {}

    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        fitted[name] = model
        cms[name] = confusion_matrix(y_test, y_pred)
        metrics = evaluate(y_test, y_pred, name)
        results.append(metrics)
        print(f"  Acc={metrics['accuracy']}  F1(macro)={metrics['f1_macro']}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    pd.DataFrame(results).sort_values("f1_macro", ascending=False)\
        .to_csv(COMPARISON_PATH, index=False)

    # Pick best by macro-F1 (handles imbalance better than accuracy)
    best_name = max(results, key=lambda r: r["f1_macro"])["model"]
    print(f"\nBest model (pre-tuning): {best_name}")

    # ---- Hyperparameter tuning ----
    print(f"Tuning {best_name} with GridSearchCV...")
    if best_name == "RandomForest":
        param_grid = {"n_estimators": [100, 200, 300],
                      "max_depth": [None, 10, 20],
                      "min_samples_split": [2, 5]}
        gs = GridSearchCV(RandomForestClassifier(class_weight="balanced",
                                                 random_state=RANDOM_STATE, n_jobs=-1),
                          param_grid, cv=CV_FOLDS, scoring="f1_macro", n_jobs=-1)
    elif best_name == "SVM":
        param_grid = {"C": [1, 10, 100], "gamma": ["scale", 0.01, 0.1]}
        gs = GridSearchCV(SVC(kernel="rbf", class_weight="balanced"),
                          param_grid, cv=CV_FOLDS, scoring="f1_macro", n_jobs=-1)
    elif best_name == "DecisionTree":
        param_grid = {"max_depth": [None, 5, 10, 20], "min_samples_split": [2, 5, 10]}
        gs = GridSearchCV(DecisionTreeClassifier(random_state=RANDOM_STATE),
                          param_grid, cv=CV_FOLDS, scoring="f1_macro", n_jobs=-1)
    else:
        gs = None

    if gs is not None:
        gs.fit(X_train, y_train)
        best_model = gs.best_estimator_
        print(f"  Best params: {gs.best_params_}")
    else:
        best_model = fitted[best_name]

    # Final evaluation
    y_pred = best_model.predict(X_test)
    final_metrics = evaluate(y_test, y_pred, f"{best_name}_tuned")
    print("\nFinal classification report:\n", classification_report(y_test, y_pred))

    # 5-fold CV on full data (sanity check)
    X_all = pd.read_csv(PROCESSED_DATA).drop(columns=[TARGET_COL])
    y_all = pd.read_csv(PROCESSED_DATA)[TARGET_COL]
    cv = cross_val_score(best_model, X_all, y_all, cv=CV_FOLDS, scoring="f1_macro")

    # Save everything
    joblib.dump(best_model, MODEL_PATH)
    with open(METRICS_PATH, "w") as f:
        json.dump({
            "best_model": best_name,
            "final_test_metrics": final_metrics,
            "cv_f1_macro_mean": round(float(cv.mean()), 4),
            "cv_f1_macro_std": round(float(cv.std()), 4),
            "all_models": results,
        }, f, indent=2)

    # Confusion matrix plot
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, (name, cm) in zip(axes.flatten(), cms.items()):
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_title(name)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
    plt.tight_layout()
    plt.savefig(CONFUSION_PATH)
    plt.close()

    print(f"\nSaved model to {MODEL_PATH}")
    print(f"CV F1-macro = {cv.mean():.4f} ± {cv.std():.4f}")


if __name__ == "__main__":
    train()