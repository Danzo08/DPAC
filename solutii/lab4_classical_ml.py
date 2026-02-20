#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import pickle
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix
)
from sklearn.model_selection import cross_val_score


# ============================================================
# Helper functions
# ============================================================

def evaluate_model(model, X_test, y_test, model_name="Model"):
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"\n{'='*50}")
    print(f"Rezultate {model_name}")
    print(f"{'='*50}")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")

    return {
        "model": model_name,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1
    }


def get_confusion(y_true, y_pred):
    return confusion_matrix(y_true, y_pred)


def plot_confusions_tile(cms, titles, output_path):
    cols = 3
    rows = (len(cms) + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = np.array(axes).reshape(-1)

    for i, (cm, title) in enumerate(zip(cms, titles)):
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Normal", "Attack"],
            yticklabels=["Normal", "Attack"],
            ax=axes[i]
        )
        axes[i].set_title(title)
        axes[i].set_xlabel("Predicted")
        axes[i].set_ylabel("Actual")

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


# ============================================================
# Main
# ============================================================

def main():
    ap = argparse.ArgumentParser(description="Lab 4 - Machine Learning Clasic (IDS)")

    ap.add_argument(
        "--data_dir",
        required=True,
        help="Folder cu X_train.npy, X_test.npy, y_train.npy, y_test.npy"
    )

    ap.add_argument(
        "--outdir",
        default="lab4_out",
        help="Folder unde se salveaza toate output-urile"
    )

    args = ap.parse_args()

    # Creeaza folder output daca nu exista
    os.makedirs(args.outdir, exist_ok=True)

    print("[+] Încărcare date preprocesate...")

    X_train = np.load(os.path.join(args.data_dir, "X_train.npy"))
    X_test  = np.load(os.path.join(args.data_dir, "X_test.npy"))
    y_train = np.load(os.path.join(args.data_dir, "y_train.npy"))
    y_test  = np.load(os.path.join(args.data_dir, "y_test.npy"))

    results = []
    conf_mats = []
    conf_titles = []

    # ========================================================
    # Decision Tree
    # ========================================================

    print("\n[+] Antrenare Decision Tree...")
    dt_model = DecisionTreeClassifier(
        max_depth=10,
        min_samples_split=10,
        criterion="gini",
        random_state=42
    )
    dt_model.fit(X_train, y_train)

    dt_res = evaluate_model(dt_model, X_test, y_test, "Decision Tree")
    results.append(dt_res)

    y_pred_dt = dt_model.predict(X_test)
    conf_mats.append(get_confusion(y_test, y_pred_dt))
    conf_titles.append("Decision Tree")

    plt.figure(figsize=(20, 10))
    plot_tree(
        dt_model,
        max_depth=3,
        filled=True,
        class_names=["Normal", "Attack"],
        fontsize=10
    )
    plt.title("Decision Tree (primele 3 nivele)")
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "decision_tree_structure.png"), dpi=300)
    plt.close()

    # ========================================================
    # Random Forest
    # ========================================================

    print("\n[+] Antrenare Random Forest...")
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        min_samples_split=5,
        max_features="sqrt",
        n_jobs=-1,
        random_state=42
    )
    rf_model.fit(X_train, y_train)

    rf_res = evaluate_model(rf_model, X_test, y_test, "Random Forest")
    results.append(rf_res)

    y_pred_rf = rf_model.predict(X_test)
    conf_mats.append(get_confusion(y_test, y_pred_rf))
    conf_titles.append("Random Forest")

    # # -----------------------------
 	# # Ex. 95: Feature importance
    # # -----------------------------
    # feature_importance = pd.DataFrame({
    # "feature": [f"feature_{i}" for i in range(X_train.shape[1])],
    # "importance": rf_model.feature_importances_
    # }).sort_values(by="importance", ascending=False)

    # print("\nTop 10 Feature Importance - Random Forest")
    # print(feature_importance.head(10))

    # plt.figure(figsize=(10, 5))
    # plt.bar(
    # feature_importance["feature"].head(10),
    # feature_importance["importance"].head(10)
    # )
    # plt.xticks(rotation=45)
    # plt.title("Top 10 Feature Importance - Random Forest")
    # plt.xlabel("Feature")
    # plt.ylabel("Importance")
    # plt.tight_layout()
    # plt.savefig(os.path.join(args.outdir, "rf_feature_importance.png"), dpi=300)
    # plt.close()


    # ========================================================
    # KNN
    # ========================================================

    print("\n[+] Antrenare KNN...")
    knn_model = KNeighborsClassifier(
        n_neighbors=3,
        metric="euclidean",
        weights="distance",
        n_jobs=-1
    )
    knn_model.fit(X_train, y_train)

    knn_res = evaluate_model(knn_model, X_test, y_test, "KNN (K=5)")
    results.append(knn_res)

    y_pred_knn = knn_model.predict(X_test)
    conf_mats.append(get_confusion(y_test, y_pred_knn))
    conf_titles.append("KNN (K=5)")

        # ========================================================
    # Exercițiu suplimentar – Vizualizare KNN simplificată
    # ========================================================

    print("\n[+] KNN distanțe euclidiene (vizualizare)...")

    # Nume de "features" pentru exemplul didactic
    feat_x_name = "src_bytes"
    feat_y_name = "dst_bytes"

    # 1) Set de date simplu (20 puncte)
    X_normal = np.array([
        [1, 2], [1.5, 2.2], [2, 1.8], [1.2, 1.5], [2.2, 2.5],
        [1.8, 1.2], [2.5, 1.7], [1.1, 2.4], [2.3, 2.1], [1.7, 2.6]
    ])
    X_attack = np.array([
        [6, 2], [6.5, 2.2], [7, 1.8], [6.2, 1.5], [7.2, 2.5],
        [6.8, 1.2], [7.5, 1.7], [6.1, 2.4], [7.3, 2.1], [6.7, 2.6]
    ])

    X_simple = np.vstack((X_normal, X_attack))
    y_simple = np.array([0]*len(X_normal) + [1]*len(X_attack))  # 0=Normal, 1=Atac

    # 2) KNN (K=5)
    K = 5
    knn = KNeighborsClassifier(n_neighbors=K, metric="euclidean")
    knn.fit(X_simple, y_simple)

    # 3) Alegem un punct nou (query)
    x_query = np.array([2.8, 2.0])  
    pred = int(knn.predict([x_query])[0])
    pred_label = "Normal" if pred == 0 else "Atac"

    # 4) Distanțe euclidiene la toate punctele:
    # d = sqrt((x1-x2)^2 + (y1-y2)^2)
    diff = X_simple - x_query
    distances = np.sqrt(np.sum(diff**2, axis=1))

    # 5) Indicii celor mai apropiați K vecini
    nn_idx = np.argsort(distances)[:K]
    nn_points = X_simple[nn_idx]
    nn_dist = distances[nn_idx]
    nn_labels = y_simple[nn_idx]

    # 6) Plot: puncte + query + linii către vecini + distanțe
    plt.figure(figsize=(10, 7))

    # puncte dataset
    plt.scatter(X_normal[:, 0], X_normal[:, 1], s=120, edgecolor="black", label="Normal")
    plt.scatter(X_attack[:, 0], X_attack[:, 1], s=120, edgecolor="black", label="Atac")

    # query point
    plt.scatter(x_query[0], x_query[1], s=220, marker="X", edgecolor="black", linewidths=2,
                label=f"Punct nou (pred: {pred_label})")

    # linii către K vecini + etichete distanță
    for p, d in zip(nn_points, nn_dist):
        # linia
        plt.plot([x_query[0], p[0]], [x_query[1], p[1]], linewidth=2)

        # text distanță la mijlocul segmentului
        mid_x = (x_query[0] + p[0]) / 2
        mid_y = (x_query[1] + p[1]) / 2
        plt.text(mid_x, mid_y, f"{d:.2f}", fontsize=10, ha="center", va="center")

    # evidențiem vecinii (cerc)
    plt.scatter(nn_points[:, 0], nn_points[:, 1], s=260, facecolors="none",
                edgecolors="black", linewidths=2, label=f"Cei {K} vecini")

    plt.title(f"KNN (K={K}) – Distanțe Euclidiene către cei mai apropiați vecini\nPredicție pentru punctul nou: {pred_label}")
    plt.xlabel(f"{feat_x_name}")
    plt.ylabel(f"{feat_y_name}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(os.path.join(args.outdir, "knn_distances_demo.png"), dpi=300)
    plt.close()

    print("Grafic salvat: knn_distances_demo.png")
    print("Vecinii (K=5) și distanțele lor:")
    for i, (p, d, lab) in enumerate(zip(nn_points, nn_dist, nn_labels), start=1):
        lab_txt = "Normal" if lab == 0 else "Atac"
        print(f"  {i}. punct={p}  dist={d:.4f}  clasa={lab_txt}")

    # ========================================================
    # Comparare modele
    # ========================================================

    results_df = pd.DataFrame(results)
    print("\nCOMPARAȚIE MODELE")
    print(results_df.to_string(index=False))

    # ========================================================
    # Cross-validation
    # ========================================================

    print("\n[+] Cross-validation (5-fold, F1)...")

    models = {
        "Decision Tree": DecisionTreeClassifier(max_depth=10, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1),
        "KNN": KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
    }

    for name, model in models.items():
        scores = cross_val_score(model, X_train, y_train, cv=5, scoring="f1")
        print(f"{name}: mean={scores.mean():.4f}, std={scores.std():.4f}")

    # ========================================================
    # Confusion Matrices
    # ========================================================

    plot_confusions_tile(
        conf_mats,
        conf_titles,
        os.path.join(args.outdir, "confusion_matrices_all_models.png")
    )

    # ========================================================
    # Salvare modele
    # ========================================================

    with open(os.path.join(args.outdir, "decision_tree_model.pkl"), "wb") as f:
        pickle.dump(dt_model, f)

    with open(os.path.join(args.outdir, "random_forest_model.pkl"), "wb") as f:
        pickle.dump(rf_model, f)

    with open(os.path.join(args.outdir, "knn_model.pkl"), "wb") as f:
        pickle.dump(knn_model, f)

    results_df.to_csv(
        os.path.join(args.outdir, "classical_ml_results.csv"),
        index=False
    )

    print("\nModele și rezultate salvate în:", args.outdir)
    print("Cel mai bun model (F1):",
          results_df.loc[results_df["f1"].idxmax(), "model"])


if __name__ == "__main__":
    main()
