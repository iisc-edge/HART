import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from har_dataset_loader import (
    load_dataset,
    get_input_columns,
    get_label_column,
    get_subject_column,
)
from types import SimpleNamespace
from models.UniTS import Model
# ============================================================
# Utility: Convert df_cleaned → numpy
# ============================================================

def convert_df_to_numpy(df, input_columns):
    """
    Converts df_cleaned (one row per window) to:
        X : [N, T, C]
        y : [N]
    """
    X = []
    for _, row in df.iterrows():
        window = np.stack([row[col].values for col in input_columns], axis=1)
        X.append(window)

    X = np.stack(X)  # [N, T, C]
    y = df["__encoded_label__"].values.astype(np.int64)
    return X, y



def build_units_args(input_channels, seq_len):
    return SimpleNamespace(
        # Model dimension
        d_model=128,
        n_heads=8,
        e_layers=2,

        # Prompt
        prompt_num=2,

        # Patch
        patch_len=16,
        stride=16,

        # Dropout
        dropout=0.1,

        # Pretraining params (required even if not used)
        right_prob=0.5,
        min_mask_ratio=0.1,
        max_mask_ratio=0.5,
    )





# ============================================================
# Simple HAR Classifier
# ============================================================

# class HARClassifier(nn.Module):
#     def __init__(self, input_channels, seq_len, num_classes):
#         super().__init__()

#         self.net = nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(input_channels * seq_len, 256),
#             nn.ReLU(),
#             nn.Dropout(0.2),
#             nn.Linear(256, num_classes)
#         )

#     def forward(self, x):
#         return self.net(x)


# ============================================================
# Training Loop
# ============================================================

def train_one_subject(X_train, y_train, X_test, y_test, test_meta_df, args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Normalize (fit on train only)
    N, T, C = X_train.shape
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train.reshape(-1, C)).reshape(N, T, C)

    Nt = X_test.shape[0]
    X_test = scaler.transform(X_test.reshape(-1, C)).reshape(Nt, T, C)

    # Build UniTS model config
    num_classes = len(np.unique(y_train))
    model_args = build_units_args(C, T)

    configs_list = [
        (
            "har_classification",
            {
                "dataset": args.dataset_name,
                "task_name": "classification",
                "enc_in": C,
                "seq_len": T,
                "num_class": num_classes,
            }
        )
    ]

    model = Model(model_args, configs_list, pretrain=False).to(device)

    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long)
    )
    test_ds = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.long)
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # -------------------- Train --------------------
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)

            optimizer.zero_grad()

            out = model(
                xb,
                None,
                task_id=0,
                task_name="classification"
            )

            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}: Loss {total_loss/len(train_loader):.4f}")

    # -------------------- Evaluate --------------------
    model.eval()
    preds = []
    trues = []

    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)

            out = model(
                xb,
                None,
                task_id=0,
                task_name="classification"
            )

            pred = torch.argmax(out, dim=1)
            preds.append(pred.cpu())
            trues.append(yb)

    preds = torch.cat(preds).numpy()
    trues = torch.cat(trues).numpy()

    acc = accuracy_score(trues, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        trues, preds, average="macro", zero_division=0
    )

    pred_df = test_meta_df.copy()
    pred_df["predicted_activity_id"] = preds

    return acc, precision, recall, f1, pred_df


# ============================================================
# Main
# ============================================================

def main(args):

    df_cleaned = load_dataset(args.dataset_name)

    input_columns = get_input_columns(args.dataset_name)
    label_column = get_label_column(args.dataset_name)
    subject_column = get_subject_column(args.dataset_name)

    print(f"\nLoaded {args.dataset_name}: "
          f"{len(df_cleaned)} windows | "
          f"{df_cleaned[subject_column].nunique()} subjects | "
          f"{df_cleaned[label_column].nunique()} classes")

    # --------------------------------------------------------
    # Keep common activities across subjects
    # --------------------------------------------------------

    per_subject_labels = df_cleaned.groupby(subject_column)[label_column].apply(set)
    common_labels = set.intersection(*per_subject_labels)

    before = len(df_cleaned)
    df_cleaned = df_cleaned[df_cleaned[label_column].isin(common_labels)].reset_index(drop=True)

    print(f"Common activities ({len(common_labels)}): {sorted(common_labels)}")
    print(f"Dropped {before - len(df_cleaned)} windows")

    # --------------------------------------------------------
    # Encode labels
    # --------------------------------------------------------

    label_map = {l: i for i, l in enumerate(sorted(df_cleaned[label_column].unique()))}
    df_cleaned["__encoded_label__"] = df_cleaned[label_column].map(label_map)

    # --------------------------------------------------------
    # LOSO
    # --------------------------------------------------------

    subjects = sorted(df_cleaned[subject_column].unique())

    results_rows = []
    all_predictions = []

    for subj in subjects:

        print(f"\n🚀 LOSO Subject {subj}")

        train_df = df_cleaned[df_cleaned[subject_column] != subj]
        test_df = df_cleaned[df_cleaned[subject_column] == subj]

        X_train, y_train = convert_df_to_numpy(train_df, input_columns)
        X_test, y_test = convert_df_to_numpy(test_df, input_columns)

        acc, prec, rec, f1, pred_df = train_one_subject(
            X_train,
            y_train,
            X_test,
            y_test,
            test_df[[subject_column, "window_id", label_column]].copy(),
            args
        )

        print(
            f"Subject {subj} | "
            f"Acc: {acc:.4f} | "
            f"Prec: {prec:.4f} | "
            f"Rec: {rec:.4f} | "
            f"F1: {f1:.4f}"
        )

        # Store per-subject metrics
        results_rows.append({
            "subject_id": subj,
            "accuracy": acc,
            "precision_macro": prec,
            "recall_macro": rec,
            "f1_macro": f1
        })

        all_predictions.append(pred_df)

# --------------------------------------------------------
# Save predictions
# --------------------------------------------------------

        predictions_df = pd.concat(all_predictions).reset_index(drop=True)

        predictions_df = predictions_df.rename(columns={
            label_column: "activity_id"
        })

        predictions_df = predictions_df[
            [subject_column, "window_id", "activity_id", "predicted_activity_id"]
        ]

        predictions_df.to_csv(f"./results/{args.dataset_name}_predictions.csv", index=False)

        # --------------------------------------------------------
        # Save results
        # --------------------------------------------------------

        results_df = pd.DataFrame(results_rows)
        results_df.to_csv(f"./results/{args.dataset_name}_results.csv", index=False)

        # --------------------------------------------------------
        # Print average
        # --------------------------------------------------------

        print("\n══════════════════════════════════")
        print(f"Average Accuracy     : {results_df['accuracy'].mean():.4f}")
        print(f"Average Precision    : {results_df['precision_macro'].mean():.4f}")
        print(f"Average Recall       : {results_df['recall_macro'].mean():.4f}")
        print(f"Average F1 (macro)   : {results_df['f1_macro'].mean():.4f}")
        print("══════════════════════════════════")

        print(f"\nSaved:")
        print(f" - {args.dataset_name}_predictions.csv")
        print(f" - {args.dataset_name}_results.csv")
# ============================================================
# Entry
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)

    args = parser.parse_args()

    main(args)