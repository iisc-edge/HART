import os
import argparse
import random
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from har_dataset_loader import (
    load_dataset,
    get_input_columns,
    get_label_column,
    get_subject_column,
    DATASET_CONFIGS,
)

# ============================================================
# Argument Parser
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description="HarNet10 LOSO with Late Fusion")

    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        choices=sorted(DATASET_CONFIGS.keys()),
    )

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--seq_len", type=int, default=300)

    return parser.parse_args()


# ============================================================
# Utilities
# ============================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

# import re

# def group_into_triplets(columns):
#     """
#     Groups signal columns into 3-axis sensor blocks.
#     Works across all datasets in DATASET_CONFIGS.

#     Returns:
#         List[List[str]]  # each sublist contains 3 axis columns
#     """

#     axis_pattern = re.compile(r"(.*?)[_\-]?([xyz])$", re.IGNORECASE)

#     groups = {}

#     for col in columns:
#         match = axis_pattern.match(col.lower())
#         if match:
#             base = match.group(1)
#             axis = match.group(2)

#             groups.setdefault(base, {})[axis] = col

#     triplets = []
#     for base, axes in groups.items():
#         if all(a in axes for a in ["x", "y", "z"]):
#             triplets.append([
#                 axes["x"],
#                 axes["y"],
#                 axes["z"],
#             ])

#     return triplets
def group_into_triplets(columns):
    """
    Robust 3-axis grouping without regex.
    Supports:
        shank_acc_x
        x_acc
        T_xacc
        T_xgyro
        Acc_x
        AG-X
    """

    groups = {}

    for col in columns:
        col_lower = col.lower()

        axis = None
        base = None

        # -----------------------------
        # Case 1: axis at end (_x, -x)
        # -----------------------------
        if col_lower.endswith(("x", "y", "z")):
            if col_lower[-1] in ["x", "y", "z"]:
                axis = col_lower[-1]
                base = col_lower[:-1]

                if base.endswith(("_", "-")):
                    base = base[:-1]

        # -----------------------------
        # Case 2: axis at start (x_acc)
        # -----------------------------
        if axis is None and col_lower.startswith(("x_", "y_", "z_")):
            axis = col_lower[0]
            base = col_lower[2:]  # remove 'x_'

        # -----------------------------
        # Case 3: DSADS style (T_xacc)
        # detect "_xacc", "_xgyro", "_xmag"
        # -----------------------------
        if axis is None:
            parts = col_lower.split("_")
            if len(parts) == 2:
                prefix, suffix = parts
                if suffix and suffix[0] in ["x", "y", "z"]:
                    axis = suffix[0]
                    base = prefix + "_" + suffix[1:]  # remove axis

        if axis and base:
            groups.setdefault(base, {})[axis] = col

    triplets = []
    for base, axes in groups.items():
        if all(a in axes for a in ["x", "y", "z"]):
            triplets.append([
                axes["x"],
                axes["y"],
                axes["z"],
            ])

    return triplets


def df_to_tensor(df, signal_cols, seq_len):
    n = len(df)
    c = len(signal_cols)
    t = len(df.iloc[0][signal_cols[0]])

    X = np.zeros((n, c, t), dtype=np.float32)

    for i in range(n):
        for j, col in enumerate(signal_cols):
            X[i, j, :] = df.iloc[i][col]

    X = torch.tensor(X)
    X = F.adaptive_avg_pool1d(X, seq_len)
    return X


# ============================================================
# Main
# ============================================================

def main():

    args = parse_args()
    set_seed(42)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # --------------------------------------------------------
    # Load Dataset
    # --------------------------------------------------------

    df_cleaned = load_dataset(args.dataset_name)
    input_columns = get_input_columns(args.dataset_name)
    label_column = get_label_column(args.dataset_name)
    subject_column = get_subject_column(args.dataset_name)

    print(f"Loaded {args.dataset_name}: "
          f"{len(df_cleaned)} windows | "
          f"{df_cleaned[subject_column].nunique()} subjects | "
          f"{df_cleaned[label_column].nunique()} classes")

    # --------------------------------------------------------
    # Keep Common Activities
    # --------------------------------------------------------

    per_subject_labels = df_cleaned.groupby(subject_column)[label_column].apply(set)
    common_labels = set.intersection(*per_subject_labels)

    before = len(df_cleaned)
    df_cleaned = df_cleaned[df_cleaned[label_column].isin(common_labels)].reset_index(drop=True)

    print(f"Common activities ({len(common_labels)}): {sorted(common_labels)}")
    print(f"Dropped {before - len(df_cleaned)} windows ({before} → {len(df_cleaned)})")

    # --------------------------------------------------------
    # Group Channels into 3-axis Sensors
    # --------------------------------------------------------

    sensor_groups = group_into_triplets(input_columns)

    print(f"Detected sensor groups:")
    for g in sensor_groups:
        print(g)

    # --------------------------------------------------------
    # Label Encoding
    # --------------------------------------------------------

    all_classes = sorted(df_cleaned[label_column].unique())
    label_map = {label: idx for idx, label in enumerate(all_classes)}
    num_classes = len(all_classes)

    df_cleaned["target"] = df_cleaned[label_column].map(label_map)

    # --------------------------------------------------------
    # LOSO
    # --------------------------------------------------------

    subjects = sorted(df_cleaned[subject_column].unique())
    all_results = []

    for sid in subjects:

        print(f"\n🚀 LOSO Subject {sid}")

        train_df = df_cleaned[df_cleaned[subject_column] != sid]
        test_df = df_cleaned[df_cleaned[subject_column] == sid]

        y_train = torch.tensor(train_df["target"].values, dtype=torch.long)
        y_test = torch.tensor(test_df["target"].values, dtype=torch.long)

        # ----------------------------------------------------
        # Load Model
        # ----------------------------------------------------

        repo = "OxWearables/ssl-wearables"
        model = torch.hub.load(
            repo,
            "harnet10",
            class_num=num_classes,
            pretrained=True
        )

        for param in model.feature_extractor.parameters():
            param.requires_grad = False

        model.to(device)

        optimizer = torch.optim.Adam(model.classifier.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        # ----------------------------------------------------
        # Training (iterate over sensor groups)
        # ----------------------------------------------------

        model.train()

        for epoch in range(args.epochs):
            for group in sensor_groups:

                X_train = df_to_tensor(train_df, group, args.seq_len)
                train_loader = DataLoader(
                    TensorDataset(X_train, y_train),
                    batch_size=args.batch_size,
                    shuffle=True
                )

                for xb, yb in train_loader:
                    xb, yb = xb.to(device), yb.to(device)

                    outputs = model(xb)
                    loss = criterion(outputs, yb)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

        # ----------------------------------------------------
        # Evaluation with Late Fusion
        # ----------------------------------------------------

        model.eval()
        preds_all = []

        with torch.no_grad():

            for group in sensor_groups:

                X_test = df_to_tensor(test_df, group, args.seq_len)
                test_loader = DataLoader(
                    TensorDataset(X_test, y_test),
                    batch_size=args.batch_size,
                    shuffle=False
                )

                logits_group = []

                for xb, _ in test_loader:
                    xb = xb.to(device)
                    outputs = model(xb)
                    logits_group.append(outputs.cpu())

                logits_group = torch.cat(logits_group)
                preds_all.append(logits_group)

            # ---- Late Fusion ----
            logits_final = sum(preds_all) / len(preds_all)
            y_pred = torch.argmax(logits_final, dim=1).numpy()

        y_true = y_test.numpy()

        metrics = {
            "dataset": args.dataset_name,
            "subject_id": sid,
            "accuracy": accuracy_score(y_true, y_pred),
            "f1_macro": f1_score(y_true, y_pred, average="macro"),
            "precision_macro": precision_score(y_true, y_pred, average="macro"),
            "recall_macro": recall_score(y_true, y_pred, average="macro"),
        }

        all_results.append(metrics)

        print(f"Subject {sid} | Acc: {metrics['accuracy']:.4f} | F1: {metrics['f1_macro']:.4f}")
        # break

    # --------------------------------------------------------
    # Save Results
    # --------------------------------------------------------

    results_df = pd.DataFrame(all_results)

    output_dir = f"./HAR_finetune_results/"
    os.makedirs(output_dir, exist_ok=True)

    results_df.to_csv(os.path.join(output_dir, f"{args.dataset_name}.csv"), index=False)

    print("\n✅ LOSO complete.")
    print(results_df)


if __name__ == "__main__":
    main()