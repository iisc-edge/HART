import os
import time
import argparse
import warnings
import random
import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

from momentfm import MOMENTPipeline

from har_dataset_loader import (
    load_dataset,
    get_input_columns,
    get_label_column,
    get_subject_column,
    DATASET_CONFIGS,
)

warnings.filterwarnings("ignore")

# ============================================================
# Argument Parser
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description="MOMENT LOSO HAR Pipeline")

    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        choices=sorted(DATASET_CONFIGS.keys()),
        help="Dataset name"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="moment_results",
        help="Directory to save results"
    )

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seq_len", type=int, default=300)

    return parser.parse_args()


# ============================================================
# Dataset Class
# ============================================================

class ClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, X, y, seq_len=300):
        self.data = X
        self.labels = y
        self.seq_len = seq_len
        self.scaler = StandardScaler()

        n, c, t = self.data.shape
        flat = self.data.reshape(n, -1)
        flat = self.scaler.fit_transform(flat)
        self.data = flat.reshape(n, c, t)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        timeseries = self.data[idx]
        label = int(self.labels[idx])

        pad_len = max(self.seq_len - timeseries.shape[1], 0)
        if pad_len > 0:
            timeseries = np.pad(timeseries, ((0, 0), (pad_len, 0)))

        mask = np.ones(self.seq_len, dtype=np.int64)
        if pad_len > 0:
            mask[:pad_len] = 0

        return timeseries.astype(np.float32), mask, label


# ============================================================
# Convert dataframe to numpy
# ============================================================

def convert_df_to_numpy(df, signal_cols):
    n_cases = len(df)
    n_channels = len(signal_cols)
    n_timepoints = len(df.iloc[0][signal_cols[0]])

    X = np.zeros((n_cases, n_channels, n_timepoints))

    for i in range(n_cases):
        for j, col in enumerate(signal_cols):
            X[i, j, :] = df.iloc[i][col].values

    y = df["__encoded_label__"].values
    return X, y


# ============================================================
# Train + Evaluate
# ============================================================

def train_and_evaluate(train_dataset, test_dataset, args, subject_id, n_classes):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = MOMENTPipeline.from_pretrained(
        "AutonLab/MOMENT-1-large",
        model_kwargs={
            'task_name': 'classification',
            'n_channels': train_dataset.data.shape[1],
            'num_class': n_classes,
            'freeze_encoder': False,
            'freeze_embedder': False,
            'reduction': 'mean',
        },
    )

    model.init()
    model.to(device).float()

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=len(train_loader) * args.epochs,
        eta_min=1e-7
    )

    start_time = time.time()

    # ---------------- TRAIN ----------------
    for epoch in range(args.epochs):
        model.train()
        for batch_x, batch_masks, batch_labels in tqdm(
            train_loader,
            desc=f"Subject {subject_id} | Epoch {epoch+1}/{args.epochs}"
        ):
            batch_x = batch_x.to(device)
            batch_masks = batch_masks.to(device)
            batch_labels = batch_labels.to(device)

            output = model(x_enc=batch_x, input_mask=batch_masks, reduction='mean')
            loss = criterion(output.logits, batch_labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(model.parameters(), 5.0)
            optimizer.step()
            scheduler.step()

    # ---------------- EVAL ----------------
    model.eval()
    preds, labels = [], []

    with torch.no_grad():
        for batch_x, batch_masks, batch_labels in test_loader:
            batch_x = batch_x.to(device)
            batch_masks = batch_masks.to(device)
            batch_labels = batch_labels.to(device)

            output = model(x_enc=batch_x, input_mask=batch_masks, reduction='mean')
            _, pred = torch.max(output.logits, 1)

            preds.append(pred.cpu().numpy())
            labels.append(batch_labels.cpu().numpy())

    preds = np.concatenate(preds)
    labels = np.concatenate(labels)

    elapsed = time.time() - start_time

    return {
        "subject_id": subject_id,
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, average='macro', zero_division=0),
        "recall": recall_score(labels, preds, average='macro', zero_division=0),
        "f1": f1_score(labels, preds, average='macro', zero_division=0),
        "time_sec": elapsed
    }, preds


# ============================================================
# Main LOSO
# ============================================================

def main():

    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(
        args.output_dir,
        f"moment_{args.dataset_name}.csv"
    )

    # ── Load Dataset ───────────────────────────────────────
    df_cleaned = load_dataset(args.dataset_name)
    input_columns = get_input_columns(args.dataset_name)
    label_column = get_label_column(args.dataset_name)
    subject_column = get_subject_column(args.dataset_name)

    print(f"Loaded {args.dataset_name}: "
          f"{len(df_cleaned)} windows | "
          f"{df_cleaned[subject_column].nunique()} subjects | "
          f"{df_cleaned[label_column].nunique()} classes")

    # ── Keep common activities ─────────────────────────────
    per_subject_labels = df_cleaned.groupby(subject_column)[label_column].apply(set)
    common_labels = set.intersection(*per_subject_labels)
    before = len(df_cleaned)
    df_cleaned = df_cleaned[df_cleaned[label_column].isin(common_labels)].reset_index(drop=True)
    print(f"Common activities across all subjects ({len(common_labels)}): {sorted(common_labels)}")
    print(f"Dropped {before - len(df_cleaned)} windows ({before} → {len(df_cleaned)})")

    # ── Encode labels globally ─────────────────────────────
    label_map = {
        l: i for i, l in enumerate(sorted(df_cleaned[label_column].unique()))
    }
    df_cleaned["__encoded_label__"] = df_cleaned[label_column].map(label_map)

    # ── LOSO ───────────────────────────────────────────────
    results = []
    prediction_rows = []

    for subj in sorted(df_cleaned[subject_column].unique()):

        print(f"\n🚀 Subject {subj}")

        train_df = df_cleaned[df_cleaned[subject_column] != subj]
        test_df = df_cleaned[df_cleaned[subject_column] == subj]

        X_train, y_train = convert_df_to_numpy(train_df, input_columns)
        X_test, y_test = convert_df_to_numpy(test_df, input_columns)

        train_dataset = ClassificationDataset(X_train, y_train, args.seq_len)
        test_dataset = ClassificationDataset(X_test, y_test, args.seq_len)

        metrics, preds = train_and_evaluate(
            train_dataset,
            test_dataset,
            args,
            subj,
            n_classes=len(label_map)
        )

        results.append(metrics)

        # ---- Store predictions per window ----
        for idx, pred_label in enumerate(preds):
            prediction_rows.append({
                "subject_id": subj,
                "window_id": test_df.iloc[idx]["window_id"],
                "true_label": test_df.iloc[idx][label_column],
                "predicted_label": list(label_map.keys())[pred_label]
            })

        pd.DataFrame(results).to_csv(output_path, index=False)
        pred_path = os.path.join(args.output_dir,f"moment_{args.dataset_name}_predictions.csv")
        # break

    pd.DataFrame(prediction_rows).to_csv(pred_path, index=False)

    print(f"\nPredictions saved to: {pred_path}")
        # pd.DataFrame(results).to_csv(output_path, index=False)

    print("\n✅ LOSO Complete")
    print(pd.DataFrame(results))


if __name__ == "__main__":
    main()