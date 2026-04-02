import os
import argparse
import warnings
import numpy as np
import pandas as pd

from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from sktime.classification.kernel_based import Arsenal, RocketClassifier
from sktime.classification.distance_based import KNeighborsTimeSeriesClassifier

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
    parser = argparse.ArgumentParser(description="sktime LOSO HAR Baselines")

    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        help=f"Dataset name. One of: {sorted(DATASET_CONFIGS.keys())}"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="Directory to store output CSV"
    )

    return parser.parse_args()


# ============================================================
# Metrics
# ============================================================

def compute_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "precision_weighted": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_weighted": recall_score(y_true, y_pred, average="weighted", zero_division=0),
    }


# ============================================================
# Classifier Factory
# ============================================================

def get_classifiers():
    return {
        "DummyUniform": DummyClassifier(strategy="uniform"),
        # "Rocket": RocketClassifier(rocket_transform="rocket", n_jobs=-1),
        # "MiniRocket": RocketClassifier(rocket_transform="minirocket", n_jobs=-1),
        # "Arsenal": Arsenal(n_jobs=-1),
        "KNN_Euclidean": KNeighborsTimeSeriesClassifier(
            algorithm="brute",
            distance="dtw",
            n_jobs=-1,
        ),
    }


# ============================================================
# LOSO Evaluation
# ============================================================

def loso_evaluation(df, input_columns, label_column, subject_column):

    if "window_id" not in df.columns:
        df = df.copy()
        df = StandardScaler().fit(df)
               
        df["window_id"] = np.arange(len(df))

    X_all = df[input_columns]
    y_all = df[label_column]
    subjects_all = df[subject_column]
    window_ids = df["window_id"]

    classifiers = get_classifiers()

    metrics_results = []
    prediction_results = []

    for sid in subjects_all.unique():

        print(f"\nLOSO Subject {sid}")

        train_mask = subjects_all != sid
        test_mask = subjects_all == sid

        X_train, X_test = X_all[train_mask], X_all[test_mask]
        y_train, y_test = y_all[train_mask], y_all[test_mask]
        test_window_ids = window_ids[test_mask]

        for clf_name, clf in classifiers.items():

            print(f"  Training {clf_name}...")

            try:
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)

                # ---- Store metrics ----
                metrics = compute_metrics(y_test, y_pred)

                metrics_results.append({
                    "subject_id": sid,
                    "classifier": clf_name,
                    **metrics
                })

                # ---- Store predictions ----
                for wid, true_label, pred_label in zip(
                    test_window_ids, y_test, y_pred
                ):
                    prediction_results.append({
                        "subject_id": sid,
                        "window_id": wid,
                        "classifier": clf_name,
                        "true_label": true_label,
                        "predicted_label": pred_label
                    })

            except Exception as e:
                print(f"    {clf_name} failed: {e}")

    return (
        pd.DataFrame(metrics_results),
        pd.DataFrame(prediction_results),
    )
# def loso_evaluation(df, input_columns, label_column, subject_column):

#     X_all = df[input_columns]
#     y_all = df[label_column]
#     subjects_all = df[subject_column]

#     classifiers = get_classifiers()
#     results = []

#     for sid in subjects_all.unique():

#         print(f"\nLOSO Subject {sid}")

#         train_mask = subjects_all != sid
#         test_mask = subjects_all == sid

#         X_train, X_test = X_all[train_mask], X_all[test_mask]
#         y_train, y_test = y_all[train_mask], y_all[test_mask]

#         for clf_name, clf in classifiers.items():

#             print(f"  Training {clf_name}...")

#             try:
#                 clf.fit(X_train, y_train)
#                 y_pred = clf.predict(X_test)

#                 metrics = compute_metrics(y_test, y_pred)

#                 results.append({
#                     "subject_id": sid,
#                     "classifier": clf_name,
#                     **metrics
#                 })

#             except Exception as e:
#                 print(f"    {clf_name} failed: {e}")

#     return pd.DataFrame(results)


# ============================================================
# Main
# ============================================================

def main():

    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(
        args.output_dir,
        f"sktime_{args.dataset_name}.csv"
    )

    # ── Load dataset via unified loader ─────────────────────
    input_columns = get_input_columns(args.dataset_name)
    label_column = get_label_column(args.dataset_name)
    subject_column = get_subject_column(args.dataset_name)

    df_cleaned = load_dataset(args.dataset_name)

    print(f"Loaded '{args.dataset_name}': "
          f"{len(df_cleaned)} windows | "
          f"{df_cleaned[subject_column].nunique()} subjects | "
          f"{df_cleaned[label_column].nunique()} classes")

    per_subject_labels = df_cleaned.groupby(subject_column)[label_column].apply(set)
    common_labels = set.intersection(*per_subject_labels)
    before = len(df_cleaned)
    df_cleaned = df_cleaned[df_cleaned[label_column].isin(common_labels)].reset_index(drop=True)
    print(f"Common activities across all subjects ({len(common_labels)}): {sorted(common_labels)}")
    print(f"Dropped {before - len(df_cleaned)} windows ({before} → {len(df_cleaned)})")

    # ── Run LOSO ───────────────────────────────────────────
    # results_df = loso_evaluation(
    #     df_cleaned,
    #     input_columns,
    #     label_column,
    #     subject_column
    # )

    # results_df.to_csv(output_path, index=False)
    metrics_df, predictions_df = loso_evaluation(
        df_cleaned,
        input_columns,
        label_column,
        subject_column
    )

    metrics_df.to_csv(output_path, index=False)

    pred_path = os.path.join(
        args.output_dir,
        f"sktime_{args.dataset_name}_predictions.csv"
    )
    predictions_df.to_csv(pred_path, index=False)

    print(f"\nMetrics saved to: {output_path}")
    print(f"Predictions saved to: {pred_path}")

    # print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()