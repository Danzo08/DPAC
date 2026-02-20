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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam


def load_or_generate_data(data_dir=None):
    if data_dir:
        x_train_path = os.path.join(data_dir, "X_train.npy")
        x_test_path = os.path.join(data_dir, "X_test.npy")
        y_train_path = os.path.join(data_dir, "y_train.npy")
        y_test_path = os.path.join(data_dir, "y_test.npy")
        if all(os.path.exists(p) for p in [x_train_path, x_test_path, y_train_path, y_test_path]):
            return (
                np.load(x_train_path),
                np.load(x_test_path),
                np.load(y_train_path),
                np.load(y_test_path),
                f"loaded from {data_dir}",
            )

    if all(os.path.exists(p) for p in ["X_train.npy", "X_test.npy", "y_train.npy", "y_test.npy"]):
        return (
            np.load("X_train.npy"),
            np.load("X_test.npy"),
            np.load("y_train.npy"),
            np.load("y_test.npy"),
            "loaded from current folder",
        )

    X, y = make_classification(
        n_samples=20000,
        n_features=38,
        n_informative=25,
        n_redundant=8,
        n_classes=2,
        weights=[0.5, 0.5],
        random_state=42,
    )
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )
    return X_train, X_test, y_train, y_test, "generated synthetic dataset"


def create_mlp_model(input_dim):
    model = Sequential(
        [
            Dense(128, activation="relu", input_shape=(input_dim,)),
            BatchNormalization(),
            Dropout(0.3),
            Dense(64, activation="relu"),
            BatchNormalization(),
            Dropout(0.3),
            Dense(32, activation="relu"),
            BatchNormalization(),
            Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(optimizer=Adam(learning_rate=0.001), loss="binary_crossentropy", metrics=["accuracy"])
    return model


def create_lstm_model(input_shape):
    model = Sequential(
        [
            LSTM(64, input_shape=input_shape, return_sequences=True),
            Dropout(0.3),
            LSTM(32, return_sequences=False),
            Dropout(0.3),
            Dense(32, activation="relu"),
            Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(optimizer=Adam(learning_rate=0.001), loss="binary_crossentropy", metrics=["accuracy"])
    return model


def plot_learning_curves(history, output_path, title_prefix):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history.history["loss"], label="Train Loss")
    axes[0].plot(history.history["val_loss"], label="Validation Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title(f"{title_prefix} - Loss")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(history.history["accuracy"], label="Train Accuracy")
    axes[1].plot(history.history["val_accuracy"], label="Validation Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title(f"{title_prefix} - Accuracy")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def save_confusion_matrix(y_true, y_pred, output_path, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Normal", "Attack"],
        yticklabels=["Normal", "Attack"],
    )
    plt.title(title)
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    return cm


def evaluate_predictions(y_true, y_pred, model_name):
    return {
        "Model": model_name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1-Score": f1_score(y_true, y_pred),
    }


def plot_model_comparison(df, output_path):
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(df))
    width = 0.2
    metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
    colors = ["steelblue", "forestgreen", "coral", "purple"]

    for i, metric in enumerate(metrics):
        ax.bar(x + i * width, df[metric], width, label=metric, color=colors[i])

    ax.set_xlabel("Model")
    ax.set_ylabel("Score")
    ax.set_title("Deep Learning Models Comparison")
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(df["Model"])
    ax.set_ylim([0.8, 1.0])
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Lab 5 - Deep Learning (MLP + LSTM)")
    parser.add_argument(
        "--data_dir",
        default=None,
        help="Folder cu X_train.npy, X_test.npy, y_train.npy, y_test.npy (optional)",
    )
    parser.add_argument("--outdir", default="lab5_out", help="Folder output")
    parser.add_argument("--epochs_mlp", type=int, default=100, help="Numar epoci pentru MLP")
    parser.add_argument("--epochs_lstm", type=int, default=50, help="Numar epoci pentru LSTM")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    np.random.seed(42)
    tf.random.set_seed(42)
    plt.style.use("seaborn-v0_8-whitegrid")

    X_train, X_test, y_train, y_test, source = load_or_generate_data(args.data_dir)
    print(f"[+] Data source: {source}")
    print(f"[+] Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"[+] Features: {X_train.shape[1]}")

    n_features = X_train.shape[1]

    print("\n[+] Training MLP...")
    mlp_model = create_mlp_model(n_features)
    mlp_callbacks = [
        EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True, verbose=1),
        ModelCheckpoint(
            os.path.join(args.outdir, "best_mlp_model.keras"),
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
    ]
    mlp_history = mlp_model.fit(
        X_train,
        y_train,
        epochs=args.epochs_mlp,
        batch_size=args.batch_size,
        validation_split=0.2,
        callbacks=mlp_callbacks,
        verbose=1,
    )
    plot_learning_curves(
        mlp_history,
        os.path.join(args.outdir, "mlp_learning_curves.png"),
        "MLP",
    )

    y_pred_mlp = (mlp_model.predict(X_test) > 0.5).astype(int).flatten()
    mlp_metrics = evaluate_predictions(y_test, y_pred_mlp, "MLP")
    save_confusion_matrix(
        y_test,
        y_pred_mlp,
        os.path.join(args.outdir, "mlp_confusion_matrix.png"),
        "MLP - Confusion Matrix",
    )
    print(
        f"MLP -> Accuracy={mlp_metrics['Accuracy']:.4f}, Precision={mlp_metrics['Precision']:.4f}, "
        f"Recall={mlp_metrics['Recall']:.4f}, F1={mlp_metrics['F1-Score']:.4f}"
    )

    print("\n[+] Training LSTM...")
    X_train_lstm = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_test_lstm = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
    lstm_model = create_lstm_model((1, n_features))
    lstm_history = lstm_model.fit(
        X_train_lstm,
        y_train,
        epochs=args.epochs_lstm,
        batch_size=args.batch_size,
        validation_split=0.2,
        callbacks=[EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True, verbose=1)],
        verbose=1,
    )
    plot_learning_curves(
        lstm_history,
        os.path.join(args.outdir, "lstm_learning_curves.png"),
        "LSTM",
    )

    y_pred_lstm = (lstm_model.predict(X_test_lstm) > 0.5).astype(int).flatten()
    lstm_metrics = evaluate_predictions(y_test, y_pred_lstm, "LSTM")
    save_confusion_matrix(
        y_test,
        y_pred_lstm,
        os.path.join(args.outdir, "lstm_confusion_matrix.png"),
        "LSTM - Confusion Matrix",
    )
    print(
        f"LSTM -> Accuracy={lstm_metrics['Accuracy']:.4f}, Precision={lstm_metrics['Precision']:.4f}, "
        f"Recall={lstm_metrics['Recall']:.4f}, F1={lstm_metrics['F1-Score']:.4f}"
    )

    results_df = pd.DataFrame([mlp_metrics, lstm_metrics])
    print("\n[+] Comparison:")
    print(results_df.to_string(index=False))
    plot_model_comparison(results_df, os.path.join(args.outdir, "deep_learning_comparison.png"))

    mlp_model.save(os.path.join(args.outdir, "mlp_model.keras"))
    lstm_model.save(os.path.join(args.outdir, "lstm_model.keras"))
    results_df.to_csv(os.path.join(args.outdir, "deep_learning_results.csv"), index=False)

    with open(os.path.join(args.outdir, "mlp_history.pkl"), "wb") as f:
        pickle.dump(mlp_history.history, f)

    with open(os.path.join(args.outdir, "lstm_history.pkl"), "wb") as f:
        pickle.dump(lstm_history.history, f)

    print(f"\n[+] Saved outputs in: {os.path.abspath(args.outdir)}")


if __name__ == "__main__":
    main()
