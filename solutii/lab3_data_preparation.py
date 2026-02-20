#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import pickle

import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import arff
import matplotlib.pyplot as plt
import seaborn as sns



COLUMN_NAMES = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
    'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
    'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
    'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
    'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
    'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
    'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
    'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'label', 'difficulty'
]

ATTACK_MAPPING = {
    'normal': 'normal',
    # DoS
    'neptune': 'DoS', 'smurf': 'DoS', 'back': 'DoS', 'teardrop': 'DoS',
    'pod': 'DoS', 'land': 'DoS',
    # Probe
    'satan': 'Probe', 'ipsweep': 'Probe', 'nmap': 'Probe', 'portsweep': 'Probe',
    # R2L
    'warezclient': 'R2L', 'guess_passwd': 'R2L', 'warezmaster': 'R2L',
    'imap': 'R2L', 'ftp_write': 'R2L',
    # U2R
    'buffer_overflow': 'U2R', 'rootkit': 'U2R', 'loadmodule': 'U2R', 'perl': 'U2R'
}


def create_sample_dataset(n_samples=10000):
    np.random.seed(42)
    data = {
        'duration': np.random.exponential(100, n_samples),
        'protocol_type': np.random.choice(['tcp', 'udp', 'icmp'], n_samples, p=[0.7, 0.2, 0.1]),
        'service': np.random.choice(['http', 'ftp', 'smtp', 'ssh', 'dns', 'other'], n_samples),
        'flag': np.random.choice(['SF', 'S0', 'REJ', 'RSTR', 'SH'], n_samples),
        'src_bytes': np.random.exponential(1000, n_samples),
        'dst_bytes': np.random.exponential(2000, n_samples),
        'land': np.random.choice([0, 1], n_samples, p=[0.99, 0.01]),
        'wrong_fragment': np.random.choice([0, 1, 2, 3], n_samples, p=[0.95, 0.03, 0.01, 0.01]),
        'urgent': np.random.choice([0, 1], n_samples, p=[0.999, 0.001]),
        'hot': np.random.poisson(0.5, n_samples),
        'num_failed_logins': np.random.choice([0, 1, 2, 3], n_samples, p=[0.9, 0.07, 0.02, 0.01]),
        'logged_in': np.random.choice([0, 1], n_samples, p=[0.3, 0.7]),
        'num_compromised': np.random.poisson(0.1, n_samples),
        'root_shell': np.random.choice([0, 1], n_samples, p=[0.99, 0.01]),
        'su_attempted': np.random.choice([0, 1], n_samples, p=[0.995, 0.005]),
        'num_root': np.random.poisson(0.1, n_samples),
        'num_file_creations': np.random.poisson(0.2, n_samples),
        'num_shells': np.random.choice([0, 1], n_samples, p=[0.99, 0.01]),
        'num_access_files': np.random.poisson(0.1, n_samples),
        'num_outbound_cmds': np.zeros(n_samples),
        'is_host_login': np.random.choice([0, 1], n_samples, p=[0.999, 0.001]),
        'is_guest_login': np.random.choice([0, 1], n_samples, p=[0.99, 0.01]),
        'count': np.random.poisson(50, n_samples),
        'srv_count': np.random.poisson(30, n_samples),
        'serror_rate': np.random.beta(0.5, 5, n_samples),
        'srv_serror_rate': np.random.beta(0.5, 5, n_samples),
        'rerror_rate': np.random.beta(0.5, 10, n_samples),
        'srv_rerror_rate': np.random.beta(0.5, 10, n_samples),
        'same_srv_rate': np.random.beta(5, 1, n_samples),
        'diff_srv_rate': np.random.beta(1, 5, n_samples),
        'srv_diff_host_rate': np.random.beta(1, 5, n_samples),
        'dst_host_count': np.random.poisson(100, n_samples),
        'dst_host_srv_count': np.random.poisson(50, n_samples),
        'dst_host_same_srv_rate': np.random.beta(5, 1, n_samples),
        'dst_host_diff_srv_rate': np.random.beta(1, 5, n_samples),
        'dst_host_same_src_port_rate': np.random.beta(2, 3, n_samples),
        'dst_host_srv_diff_host_rate': np.random.beta(1, 5, n_samples),
        'dst_host_serror_rate': np.random.beta(0.5, 5, n_samples),
        'dst_host_srv_serror_rate': np.random.beta(0.5, 5, n_samples),
        'dst_host_rerror_rate': np.random.beta(0.5, 10, n_samples),
        'dst_host_srv_rerror_rate': np.random.beta(0.5, 10, n_samples),
        'label': np.random.choice(
            ['normal', 'neptune', 'satan', 'ipsweep', 'portsweep', 'smurf', 'nmap', 'back', 'teardrop', 'warezclient'],
            n_samples,
            p=[0.5, 0.15, 0.05, 0.05, 0.05, 0.05, 0.05, 0.03, 0.04, 0.03]
        ),
        'difficulty': np.random.randint(1, 22, n_samples)
    }
    df = pd.DataFrame(data)
    return df


def load_nsl_kdd(path: str):
    # .txt = CSV fără header, separat prin virgule
    if path.lower().endswith(".txt") or path.lower().endswith(".csv"):
        return pd.read_csv(path, header=None, names=COLUMN_NAMES)

    # .arff = ARFF
    if path.lower().endswith(".arff"):
        import arff
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            dataset = arff.load(f)
        df = pd.DataFrame(dataset["data"], columns=[a[0] for a in dataset["attributes"]])
        return df

    raise ValueError("Format necunoscut. Folosește .txt/.csv sau .arff")



def basic_eda(df: pd.DataFrame):
    print(f"[+] Dataset: {df.shape[0]} rânduri x {df.shape[1]} coloane")

    # lipsă
    missing = df.isnull().sum()
    total_missing = int(missing.sum())
    print(f"[+] Missing total: {total_missing}")
    if total_missing:
        print(missing[missing > 0].sort_values(ascending=False).head(20))

    # distribuție label
    print("\n[+] Distribuție label (top 10):")
    print(df['label'].value_counts().head(10))

    # categorii atac
    df_tmp = df.copy()
    df_tmp['attack_category'] = df_tmp['label'].map(ATTACK_MAPPING).fillna('other')
    print("\n[+] Distribuție attack_category:")
    print(df_tmp['attack_category'].value_counts())

    print("\n[+] Distribuție protocol_type:")
    print(df['protocol_type'].value_counts())

def preprocess(df: pd.DataFrame, scale_mode="standard"):
    # target
    df = df.copy()
    df['attack_category'] = df['label'].map(ATTACK_MAPPING).fillna('other')
    df['is_attack'] = (df['label'] != 'normal').astype(int)

    y = df['is_attack']
    X = df.drop(columns=['label', 'difficulty', 'attack_category', 'is_attack'])

    # categorice
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

    X_enc = X.copy()
    label_encoders = {}

    for col in categorical_cols:
        le = LabelEncoder()
        X_enc[col] = le.fit_transform(X_enc[col].astype(str))
        label_encoders[col] = le

    # scalare
    if scale_mode == "minmax":
        scaler = MinMaxScaler()
    else:
        scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X_enc)

    return X_enc, X_scaled, y, scaler, label_encoders


def main():
    ap = argparse.ArgumentParser(description="Lab 3 - Pregătire/curățare date NSL-KDD (local, VS Code)")
    ap.add_argument("--input", help="Calea către NSL-KDD (ex: KDDTrain+.txt). Dacă lipsește, folosește dataset sintetic.")
    ap.add_argument("--outdir", default="lab3_out", help="Folder output")
    ap.add_argument("--samples", type=int, default=10000, help="Nr samples pentru dataset sintetic")
    ap.add_argument("--scale", choices=["standard", "minmax"], default="standard", help="Tip scalare")
    ap.add_argument("--test_size", type=float, default=0.2, help="Procent test (ex: 0.2)")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    if args.input:
        df = load_nsl_kdd(args.input)
    else:
        df = create_sample_dataset(args.samples)

    basic_eda(df)

    # Detectare coloane cu valori constante
    # constant_cols = [col for col in df.columns if df[col].nunique() <= 1]

    # print("\n[+] Features cu valori constante:")
    # print(constant_cols)

    X_enc, X_scaled, y, scaler, label_encoders = preprocess(df, scale_mode=args.scale)

    # One-Hot Encoding (doar pentru verificare exercițiu)
    # X_tmp = df.drop(columns=['label', 'difficulty'])
    # X_onehot = pd.get_dummies(X_tmp, columns=['protocol_type', 'service', 'flag'])
    # print("\n[+] Nr coloane după One-Hot:",X_onehot.shape[1])
    # print("protocol_type:", len([c for c in X_onehot.columns if c.startswith("protocol_type_")]))
    # print("service:", len([c for c in X_onehot.columns if c.startswith("service_")]))
    # print("flag:", len([c for c in X_onehot.columns if c.startswith("flag_")]))

    print("\n[+] Verificare scalare:")
    print(f"    Mean: {X_scaled.mean():.2f}")
    print(f"    Std:  {X_scaled.std():.2f}")

    # Alegem o coloană numerică
    feature_name = "count"

    # Luăm varianta înainte de scalare pentru analiză
    feature_values = X_enc[feature_name]

    print(f"\n[+] Statistici pentru {feature_name}:")
    print(feature_values.describe())

    feature_log = np.log1p(feature_values)  # log(1+x)

    plt.figure(figsize=(8,5))
    sns.histplot(feature_values, bins=50, kde=True, kde_kws={"bw_adjust":0.5})

    plt.title(f"Distribuția log pentru {feature_name}")
    plt.xlabel("log(10 + count)")
    plt.ylabel("Frecvență")

    plt.tight_layout()
    plt.show()

    # split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled,
        y,
        test_size=args.test_size,
        random_state=42,
        stratify=y
    )

    print("\n[+] Split:")
    print(f"    Train: {X_train.shape}  Test: {X_test.shape}")
    print(f"    Atacuri (%): {y.mean()*100:.1f}")

    # save npy
    np.save(os.path.join(args.outdir, "X_train.npy"), X_train)
    np.save(os.path.join(args.outdir, "X_test.npy"), X_test)
    np.save(os.path.join(args.outdir, "y_train.npy"), y_train.to_numpy())
    np.save(os.path.join(args.outdir, "y_test.npy"), y_test.to_numpy())

    # save scaler + encoders
    with open(os.path.join(args.outdir, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)

    with open(os.path.join(args.outdir, "label_encoders.pkl"), "wb") as f:
        pickle.dump(label_encoders, f)

    # save CSV-uri (scaled)
    cols = X_enc.columns.tolist()
    train_df = pd.DataFrame(X_train, columns=cols)
    train_df["target"] = y_train.to_numpy()
    train_df.to_csv(os.path.join(args.outdir, "train_processed.csv"), index=False)

    test_df = pd.DataFrame(X_test, columns=cols)
    test_df["target"] = y_test.to_numpy()
    test_df.to_csv(os.path.join(args.outdir, "test_processed.csv"), index=False)

    print("\n[+] Fișiere create în", os.path.abspath(args.outdir))
    print("    - X_train.npy date pentru antrenare, X_test.npy date pentru testare, y_train.npy etichete pentru antrenare, y_test.npy etichete pentru testare")
    print("    - scaler.pkl, label_encoders.pkl")
    print("    - train_processed.csv, test_processed.csv")

if __name__ == "__main__":
    raise SystemExit(main())
