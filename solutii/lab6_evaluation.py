#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import pickle

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


def load_test_data(data_dir=None):
    if data_dir:
        x_test_path = os.path.join(data_dir, "X_test.npy")
        y_test_path = os.path.join(data_dir, "y_test.npy")
        if os.path.exists(x_test_path) and os.path.exists(y_test_path):
            return np.load(x_test_path), np.load(y_test_path), f"loaded from {data_dir}"

    if os.path.exists("X_test.npy") and os.path.exists("y_test.npy"):
        return np.load("X_test.npy"), np.load("y_test.npy"), "loaded from current folder"

    X, y = make_classification(
        n_samples=10000,
        n_features=38,
        n_informative=20,
        n_redundant=8,
        n_classes=2,
        random_state=42,
    )
    X = StandardScaler().fit_transform(X)
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_test, y_test, "generated synthetic dataset"


def load_classic_models(models_dir):
    models = {}
    dt_path = os.path.join(models_dir, "decision_tree_model.pkl")
    rf_path = os.path.join(models_dir, "random_forest_model.pkl")
    knn_path = os.path.join(models_dir, "knn_model.pkl")

    try:
        with open(dt_path, "rb") as f:
            models["Decision Tree"] = pickle.load(f)
        with open(rf_path, "rb") as f:
            models["Random Forest"] = pickle.load(f)
        with open(knn_path, "rb") as f:
            models["KNN"] = pickle.load(f)
    except FileNotFoundError:
        pass

    return models


def create_fallback_classic_models(X_test, y_test):
    print("[!] Classical models not found. Training fallback models on synthetic split...")
    X_train, _, y_train, _ = train_test_split(
        np.vstack([X_test, X_test]),
        np.hstack([y_test, y_test]),
        test_size=0.5,
        random_state=42,
    )

    return {
        "Decision Tree": DecisionTreeClassifier(max_depth=10, random_state=42).fit(X_train, y_train),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train),
        "KNN": KNeighborsClassifier(n_neighbors=5).fit(X_train, y_train),
    }


def load_deep_models(models_dir):
    models = {}
    mlp_path = os.path.join(models_dir, "mlp_model.keras")
    lstm_path = os.path.join(models_dir, "lstm_model.keras")

    try:
        import tensorflow as tf
    except Exception:
        print("[!] TensorFlow unavailable, skipping deep models.")
        return models

    if os.path.exists(mlp_path):
        models["MLP"] = tf.keras.models.load_model(mlp_path)
    if os.path.exists(lstm_path):
        models["LSTM"] = tf.keras.models.load_model(lstm_path)

    if not models:
        print("[!] Deep learning models not found.")
    return models


def predict_for_model(model_name, model, X_test):
    if model_name == "MLP":
        return (model.predict(X_test, verbose=0) > 0.5).astype(int).flatten()
    if model_name == "LSTM":
        X_lstm = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
        return (model.predict(X_lstm, verbose=0) > 0.5).astype(int).flatten()
    return model.predict(X_test)


def get_probabilities(model_name, model, X_test):
    if model_name == "MLP":
        return model.predict(X_test, verbose=0).flatten()
    if model_name == "LSTM":
        X_lstm = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
        return model.predict(X_lstm, verbose=0).flatten()
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X_test)[:, 1]
    return None


def plot_confusion_matrices(y_test, predictions, out_path):
    n_models = len(predictions)
    cols = 3
    rows = (n_models + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = np.array(axes).reshape(-1)

    for i, (name, y_pred) in enumerate(predictions.items()):
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Normal", "Attack"],
            yticklabels=["Normal", "Attack"],
            ax=axes[i],
        )
        axes[i].set_title(name)
        axes[i].set_xlabel("Predicted")
        axes[i].set_ylabel("Actual")

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_metrics_summary(results_df, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
    x = np.arange(len(results_df))
    width = 0.2
    colors = ["steelblue", "forestgreen", "coral", "purple"]

    for i, metric in enumerate(metrics):
        axes[0].bar(x + i * width, results_df[metric], width, label=metric, color=colors[i])

    axes[0].set_xlabel("Model")
    axes[0].set_ylabel("Score")
    axes[0].set_title("Metrics Comparison")
    axes[0].set_xticks(x + width * 1.5)
    axes[0].set_xticklabels(results_df["Model"], rotation=45, ha="right")
    axes[0].set_ylim([0.7, 1.0])
    axes[0].legend()

    sorted_df = results_df.sort_values("F1-Score", ascending=True)
    colors_bar = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(sorted_df)))
    axes[1].barh(sorted_df["Model"], sorted_df["F1-Score"], color=colors_bar)
    axes[1].set_xlabel("F1-Score")
    axes[1].set_title("F1 Ranking")
    axes[1].set_xlim([0.7, 1.0])
    for i, val in enumerate(sorted_df["F1-Score"]):
        axes[1].text(val + 0.01, i, f"{val:.4f}", va="center")

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_roc_curves(models, X_test, y_test, out_path):
    plt.figure(figsize=(10, 8))
    plotted = False
    for name, model in models.items():
        try:
            y_proba = get_probabilities(name, model, X_test)
            if y_proba is None:
                continue
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, linewidth=2, label=f"{name} (AUC = {roc_auc:.4f})")
            plotted = True
        except Exception as exc:
            print(f"[!] ROC skipped for {name}: {exc}")

    if not plotted:
        return False

    plt.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random (AUC = 0.5)")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    return True


def plot_fp_fn(results_df, out_path):
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(results_df))
    width = 0.35

    ax.bar(
        x - width / 2,
        results_df["False Positives"],
        width,
        label="False Positives",
        color="orange",
    )
    ax.bar(
        x + width / 2,
        results_df["False Negatives"],
        width,
        label="False Negatives",
        color="red",
    )

    ax.set_xlabel("Model")
    ax.set_ylabel("Error count")
    ax.set_title("False Positives vs False Negatives")
    ax.set_xticks(x)
    ax.set_xticklabels(results_df["Model"], rotation=45, ha="right")
    ax.legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Lab 6 - Evaluation and model comparison")
    parser.add_argument("--data_dir", default=None, help="Folder with X_test.npy and y_test.npy")
    parser.add_argument("--ml_models_dir", default="solutii/lab4_out", help="Folder with classic ML models")
    parser.add_argument("--dl_models_dir", default="solutii/lab5_out", help="Folder with deep learning models")
    parser.add_argument("--outdir", default="lab6_out", help="Folder output")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")

    X_test, y_test, source = load_test_data(args.data_dir)
    print(f"[+] Test data source: {source}")
    print(f"[+] Test set shape: {X_test.shape}")

    models = load_classic_models(args.ml_models_dir)
    if not models:
        models = create_fallback_classic_models(X_test, y_test)

    models.update(load_deep_models(args.dl_models_dir))
    print(f"[+] Loaded models: {list(models.keys())}")

    predictions = {}
    for name, model in models.items():
        predictions[name] = predict_for_model(name, model, X_test)
        print(f"    - {name}: {len(predictions[name])} predictions")

    plot_confusion_matrices(
        y_test,
        predictions,
        os.path.join(args.outdir, "confusion_matrices_all_models.png"),
    )

    results = []
    for name, y_pred in predictions.items():
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        results.append(
            {
                "Model": name,
                "Accuracy": accuracy_score(y_test, y_pred),
                "Precision": precision_score(y_test, y_pred),
                "Recall": recall_score(y_test, y_pred),
                "F1-Score": f1_score(y_test, y_pred),
                "True Positives": tp,
                "True Negatives": tn,
                "False Positives": fp,
                "False Negatives": fn,
            }
        )

    results_df = pd.DataFrame(results)
    print("\n[+] Comparative results:")
    print(results_df.to_string(index=False))

    plot_metrics_summary(results_df, os.path.join(args.outdir, "metrics_comparison.png"))
    plot_fp_fn(results_df, os.path.join(args.outdir, "fp_vs_fn.png"))

    roc_path = os.path.join(args.outdir, "roc_curves.png")
    if plot_roc_curves(models, X_test, y_test, roc_path):
        print(f"[+] ROC plot saved: {roc_path}")

    fp_fn_df = results_df[["Model", "False Positives", "False Negatives", "Recall"]].sort_values(
        "False Negatives"
    )
    best_model = fp_fn_df.iloc[0]["Model"]
    print(f"\n[+] Model with lowest False Negatives: {best_model}")

    results_df.to_csv(os.path.join(args.outdir, "final_comparison_results.csv"), index=False)
    print(f"[+] Saved outputs in: {os.path.abspath(args.outdir)}")


if __name__ == "__main__":
    main()
