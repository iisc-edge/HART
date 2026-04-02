"""
HAR_dataloader.py
-----------------
Single script that knows the structure of all HAR test datasets.
Loads any dataset, applies sliding windows with configurable overlap,
and returns a df_cleaned DataFrame where each row is one window.

Usage:
    from HAR_dataloader import load_dataset, DATASET_CONFIGS, TEST_SET_DIR

    df_cleaned = load_dataset("daphnet")
    df_cleaned = load_dataset("harth")
    df_cleaned = load_dataset("uci_har")

    # or load all datasets at once:
    from HAR_dataloader import load_all_datasets
    datasets = load_all_datasets()   # dict: name -> df_cleaned
"""

import os
from typing import Optional
import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# Path to the test_set folder  
# Override at call-time via load_dataset(..., test_set_dir="/your/path")
# ─────────────────────────────────────────────────────────────────────────────
TEST_SET_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "test_set"
)

# ─────────────────────────────────────────────────────────────────────────────
# Config for each dataset
#   file       : CSV filename inside TEST_SET_DIR
#   signal_cols: ordered list of sensor/signal columns to use as input features
#   subject_col: column identifying the subject/participant
#   label_col  : column identifying the activity label
#   total_rows : approximate number of data rows (excluding header) for reference
# ─────────────────────────────────────────────────────────────────────────────
DATASET_CONFIGS = {

    # "adl": {
    #     "file": "adl_30hz_clean_test.csv",
    #     "dir": "/media/user/DATA21/Time_Series_Foundation_Models/HAR/HAR_New/left_out",
    #     "signal_cols": ["Acc_x", "Acc_y", "Acc_z"],
    #     "subject_col": "subject_id",
    #     "label_col": "activity_id",
    #     "total_rows": 50400,
    # },

    "capture24": {
        "file": "capture24_30Hz_test_downsampled.csv",
        "signal_cols": ["acc_wrist_x", "acc_wrist_y", "acc_wrist_z"],
        "subject_col": "subject_id",
        "label_col": "activity_id",
        "total_rows": 8679196,
    },

    "daphnet": {
        "file": "daphnet_30Hz_test.csv",
        "signal_cols": [
            "shank_acc_x", "shank_acc_y", "shank_acc_z",
            "thigh_acc_x", "thigh_acc_y", "thigh_acc_z",
            "trunk_acc_x", "trunk_acc_y", "trunk_acc_z",
        ],
        "subject_col": "subject_id",
        "label_col": "activity_id",
        "total_rows": 439200,
    },

    "dsads": {
        "file": "dsads_30Hz_test.csv",
        "signal_cols": [
            "T_xacc",  "T_yacc",  "T_zacc",  "T_xgyro",  "T_ygyro",  "T_zgyro",  "T_xmag",  "T_ymag",  "T_zmag",
            "RA_xacc", "RA_yacc", "RA_zacc", "RA_xgyro", "RA_ygyro", "RA_zgyro", "RA_xmag", "RA_ymag", "RA_zmag",
            "LA_xacc", "LA_yacc", "LA_zacc", "LA_xgyro", "LA_ygyro", "LA_zgyro", "LA_xmag", "LA_ymag", "LA_zmag",
            "RL_xacc", "RL_yacc", "RL_zacc", "RL_xgyro", "RL_ygyro", "RL_zgyro", "RL_xmag", "RL_ymag", "RL_zmag",
            "LL_xacc", "LL_yacc", "LL_zacc", "LL_xgyro", "LL_ygyro", "LL_zgyro", "LL_xmag", "LL_ymag", "LL_zmag",
        ],
        "subject_col": "subject_id",
        "label_col": "activity_id",
        "total_rows": 342000,
    },

    "harsense": {
        "file": "harsense_30Hz_test.csv",
        "signal_cols": ["AG-X", "AG-Y", "AG-Z", "Acc-X", "Acc-Y", "Acc-Z", "RR-X", "RR-Y", "RR-Z"],
        "subject_col": "subject_id",
        "label_col": "activity_id",
        "total_rows": 41700,
    },

    "harth": {
        "file": "harth_30Hz_test.csv",
        "signal_cols": ["back_x", "back_y", "back_z", "thigh_x", "thigh_y", "thigh_z"],
        "subject_col": "subject_id",
        "label_col": "activity_id",
        "total_rows": 716700,
    },

    "hhar": {
        "file": "hhar_30Hz_test.csv",
        "signal_cols": ["x_acc", "y_acc", "z_acc", "x_gyro", "y_gyro", "z_gyro"],
        "subject_col": "subject_id",
        "label_col": "activity_id",
        "total_rows": 245100,
    },

    "hugadb": {
        "file": "hugadb_30Hz_test.csv",
        "signal_cols": [
            "RF_acc_x", "RF_acc_y", "RF_acc_z", "RF_gyro_x", "RF_gyro_y", "RF_gyro_z",
            "RS_acc_x", "RS_acc_y", "RS_acc_z", "RS_gyro_x", "RS_gyro_y", "RS_gyro_z",
            "RT_acc_x", "RT_acc_y", "RT_acc_z", "RT_gyro_x", "RT_gyro_y", "RT_gyro_z",
            "LF_acc_x", "LF_acc_y", "LF_acc_z", "LF_gyro_x", "LF_gyro_y", "LF_gyro_z",
            "LS_acc_x", "LS_acc_y", "LS_acc_z", "LS_gyro_x", "LS_gyro_y", "LS_gyro_z",
            "LT_acc_x", "LT_acc_y", "LT_acc_z", "LT_gyro_x", "LT_gyro_y", "LT_gyro_z",
        ],
        "subject_col": "subject_id",
        "label_col": "activity_id",
        "total_rows": 287400,
    },

    "ku_har": {
        "file": "ku_har_30Hz_test.csv",
        "signal_cols": ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"],
        "subject_col": "subject_id",
        "label_col": "activity_id",
        "total_rows": 109800,
    },

    "mhealth": {
        "file": "mhealth_30Hz_test.csv",
        "signal_cols": [
            "chest_acc_x",   "chest_acc_y",   "chest_acc_z",
            "lankle_acc_x",  "lankle_acc_y",  "lankle_acc_z",
            "lankle_gyro_x", "lankle_gyro_y", "lankle_gyro_z",
            "lankle_mag_x",  "lankle_mag_y",  "lankle_mag_z",
            "rarm_acc_x",    "rarm_acc_y",    "rarm_acc_z",
            "rarm_gyro_x",   "rarm_gyro_y",   "rarm_gyro_z",
            "rarm_mag_x",    "rarm_mag_y",    "rarm_mag_z",
        ],
        "subject_col": "subject_id",
        "label_col": "activity_id",
        "total_rows": 361200,
    },

    "motion_sense": {
        "file": "motion_sense_30Hz_test.csv",
        "signal_cols": ["accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z"],
        "subject_col": "subject_id",
        "label_col": "activity_id",
        "total_rows": 176100,
    },

    "opportunity": {
        "file": "opportunity_30Hz_test.csv",
        "signal_cols": [
            "RKN_upper_acc_x",    "RKN_upper_acc_y",    "RKN_upper_acc_z",
            "HIP_acc_x",          "HIP_acc_y",          "HIP_acc_z",
            "LUA_upper_acc_x",    "LUA_upper_acc_y",    "LUA_upper_acc_z",
            "RUA_lower_acc_x",    "RUA_lower_acc_y",    "RUA_lower_acc_z",
            "LH_acc_x",           "LH_acc_y",           "LH_acc_z",
            "BACK_acc_x",         "BACK_acc_y",         "BACK_acc_z",
            "RKN_lower_acc_x",    "RKN_lower_acc_y",    "RKN_lower_acc_z",
            "RWR_acc_x",          "RWR_acc_y",          "RWR_acc_z",
            "RUA_upper_acc_x",    "RUA_upper_acc_y",    "RUA_upper_acc_z",
            "LUA_lower_acc_x",    "LUA_lower_acc_y",    "LUA_lower_acc_z",
            "LWR_acc_x",          "LWR_acc_y",          "LWR_acc_z",
            "RH_acc_x",           "RH_acc_y",           "RH_acc_z",
            "IMU_BACK_acc_x",     "IMU_BACK_acc_y",     "IMU_BACK_acc_z",
            "IMU_BACK_gyro_x",    "IMU_BACK_gyro_y",    "IMU_BACK_gyro_z",
            "IMU_BACK_mag_x",     "IMU_BACK_mag_y",     "IMU_BACK_mag_z",
            "IMU_RUA_acc_x",      "IMU_RUA_acc_y",      "IMU_RUA_acc_z",
            "IMU_RUA_gyro_x",     "IMU_RUA_gyro_y",     "IMU_RUA_gyro_z",
            "IMU_RUA_mag_x",      "IMU_RUA_mag_y",      "IMU_RUA_mag_z",
            "IMU_RLA_acc_x",      "IMU_RLA_acc_y",      "IMU_RLA_acc_z",
            "IMU_RLA_gyro_x",     "IMU_RLA_gyro_y",     "IMU_RLA_gyro_z",
            "IMU_RLA_mag_x",      "IMU_RLA_mag_y",      "IMU_RLA_mag_z",
            "IMU_LUA_acc_x",      "IMU_LUA_acc_y",      "IMU_LUA_acc_z",
            "IMU_LUA_gyro_x",     "IMU_LUA_gyro_y",     "IMU_LUA_gyro_z",
            "IMU_LUA_mag_x",      "IMU_LUA_mag_y",      "IMU_LUA_mag_z",
            "IMU_LLA_acc_x",      "IMU_LLA_acc_y",      "IMU_LLA_acc_z",
            "IMU_LLA_gyro_x",     "IMU_LLA_gyro_y",     "IMU_LLA_gyro_z",
            "IMU_LLA_mag_x",      "IMU_LLA_mag_y",      "IMU_LLA_mag_z",
            "IMU_LSHOE_nav_acc_x","IMU_LSHOE_nav_acc_y","IMU_LSHOE_nav_acc_z",
            "IMU_LSHOE_body_acc_x","IMU_LSHOE_body_acc_y","IMU_LSHOE_body_acc_z",
            "IMU_RSHOE_nav_acc_x","IMU_RSHOE_nav_acc_y","IMU_RSHOE_nav_acc_z",
            "IMU_RSHOE_body_acc_x","IMU_RSHOE_body_acc_y","IMU_RSHOE_body_acc_z",
        ],
        "subject_col": "subject_id",
        "label_col": "activity_id",
        "total_rows": 244800,
    },

    "realworld": {
        "file": "realworld_30hz_clean_test.csv",
        "signal_cols": ["Acc_x", "Acc_y", "Acc_z"],
        "subject_col": "subject_id",
        "label_col": "activity_id",
        "total_rows": 1383000,
    },

    "uci_har": {
        "file": "uci_har_30Hz_test.csv",
        "signal_cols": [
            "total_acc_z", "body_gyro_y", "body_acc_y",
            "total_acc_x", "body_gyro_z", "total_acc_y",
            "body_gyro_x", "body_acc_x",  "body_acc_z",
        ],
        "subject_col": "subject_id",
        "label_col": "activity_id",
        "total_rows": 61800,
    },

    "usc_had": {
        "file": "usc_had_30Hz_test.csv",
        "signal_cols": ["Ax", "Ay", "Az", "GyroX", "GyroY", "GyroZ"],
        "subject_col": "subject_id",
        "label_col": "activity_id",
        "total_rows": 314400,
    },

    "w_har": {
        "file": "w_har_30Hz_test.csv",
        "signal_cols": ["Ax", "Ay", "Az", "GyroX", "GyroY", "GyroZ"],
        "subject_col": "subject_id",
        "label_col": "activity_id",
        "total_rows": 39000,
    },

    "wisdm_19_phone": {
        "file": "wisdm_19_phone_30Hz_test.csv",
        "signal_cols": [
            "accel_phone_x", "accel_phone_y", "accel_phone_z",
            "gyro_phone_x",  "gyro_phone_y",  "gyro_phone_z",
        ],
        "subject_col": "subject_id",
        "label_col": "activity_id",
        "total_rows": 404700,
    },

    "wisdm_19_watch": {
        "file": "wisdm_19_watch_30Hz_test.csv",
        "signal_cols": [
            "accel_watch_x", "accel_watch_y", "accel_watch_z",
            "gyro_watch_x",  "gyro_watch_y",  "gyro_watch_z",
        ],
        "subject_col": "subject_id",
        "label_col": "activity_id",
        "total_rows": 488400,
    },

    "wisdm": {
        "file": "wisdm_30Hz_test.csv",
        "signal_cols": ["accel_x", "accel_y", "accel_z"],
        "subject_col": "subject_id",
        "label_col": "activity_id",
        "total_rows": 204300,
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Core windowing function — supports both non-overlapping and overlapping modes
# ─────────────────────────────────────────────────────────────────────────────

def load_dataset(
    dataset_name: str,
    window_length: int = 120,
    hop_length: Optional[int] = None,
    test_set_dir: str = TEST_SET_DIR,
    drop_incomplete_windows: bool = True,
) -> pd.DataFrame:
    """
    Load a HAR test dataset, apply sliding windows, and return a df_cleaned
    DataFrame where each row is one window.

    Windowing is performed **per subject** to avoid cross-subject contamination.

    Each row contains:
        - one column per signal (as a pd.Series of length window_length)
        - subject_id   : subject identifier
        - activity_id  : activity label (majority vote within the window)
        - window_id    : global window index (1-based)

    Parameters
    ----------
    dataset_name : str
        Key from DATASET_CONFIGS (e.g. "daphnet", "harth", "uci_har").
    window_length : int
        Number of raw samples per window. Default 120 (= 4 s @ 30 Hz).
    hop_length : int or None
        Step between consecutive window starts (in raw samples).
        None (default) → non-overlapping (hop = window_length).
        hop_length=30 with window_length=120 → 75 % overlap.
    test_set_dir : str
        Path to the folder containing the CSV files.
    drop_incomplete_windows : bool
        Kept for API compatibility; only full windows are always produced
        when using per-subject overlapping windowing.

    Returns
    -------
    pd.DataFrame
        df_cleaned — one row per window, ready to pass into
        prepare_dataset() / ClassificationDFDataset.
    """
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. "
            f"Available: {sorted(DATASET_CONFIGS.keys())}"
        )

    cfg = DATASET_CONFIGS[dataset_name]
    csv_path = os.path.join(cfg.get("dir", test_set_dir), cfg["file"])  # optional per-dataset dir override
    signal_cols = cfg["signal_cols"]
    subject_col = cfg["subject_col"]
    label_col   = cfg["label_col"]

    print(f"[{dataset_name}] Loading {csv_path} ...")
    df = pd.read_csv(csv_path, on_bad_lines='skip', encoding='latin1')
    # Drop rows where any signal column or label/subject is NaN (caused by skipped bad lines)
    essential_cols = signal_cols + [subject_col, label_col]
    df = df.dropna(subset=essential_cols).reset_index(drop=True)
    print(f"[{dataset_name}] Raw shape: {df.shape}")

    # Effective hop: None → non-overlapping
    hop = hop_length if hop_length is not None else window_length

    # ── Per-subject overlapping windowing ─────────────────────────────────
    # Windowing within each subject avoids cross-subject contamination and
    # supports any hop_length < window_length for data augmentation.
    all_signals  = {col: [] for col in signal_cols}
    all_subjects = []
    all_labels   = []
    all_win_ids  = []
    win_counter  = 0

    for subj, sdf in df.groupby(subject_col, sort=True):
        vals = sdf[signal_cols].to_numpy()   # (n, num_signals)
        labs = sdf[label_col].to_numpy()     # (n,)
        n    = len(vals)
        if n < window_length:
            continue  # skip subjects with insufficient data

        starts  = np.arange(0, n - window_length + 1, hop)
        n_wins  = len(starts)
        idx     = starts[:, None] + np.arange(window_length)  # (n_wins, window_length)

        sig_wins = vals[idx]   # (n_wins, window_length, num_signals)
        lbl_wins = labs[idx]   # (n_wins, window_length)

        # Majority-vote label per window (vectorised via pandas)
        lbl_maj = (
            pd.DataFrame(lbl_wins)
            .apply(lambda r: r.value_counts().index[0], axis=1)
            .values
        )

        for i, col in enumerate(signal_cols):
            all_signals[col].extend(
                [pd.Series(sig_wins[w, :, i]) for w in range(n_wins)]
            )
        all_subjects.extend([subj] * n_wins)
        all_labels.extend(lbl_maj.tolist())
        new_ids = list(range(win_counter + 1, win_counter + 1 + n_wins))
        all_win_ids.extend(new_ids)
        win_counter += n_wins

    n_windows = win_counter
    print(f"[{dataset_name}] Windows: {n_windows} "
          f"(window_length={window_length}, hop={hop})")

    df_cleaned = pd.DataFrame({
        **all_signals,
        subject_col: all_subjects,
        label_col:   all_labels,
        "window_id": all_win_ids,
    })
    print(f"[{dataset_name}] df_cleaned shape: {df_cleaned.shape}  "
          f"(subjects={df_cleaned[subject_col].nunique()}, "
          f"classes={df_cleaned[label_col].nunique()})")
    return df_cleaned


def load_all_datasets(
    window_length: int = 300,
    hop_length: Optional[int] = None,
    test_set_dir: str = TEST_SET_DIR,
    drop_incomplete_windows: bool = True,
) -> dict:
    """
    Load and window all 18 datasets. Returns a dict mapping dataset name -> df_cleaned.
    """
    results = {}
    for name in DATASET_CONFIGS:
        try:
            results[name] = load_dataset(
                name,
                window_length=window_length,
                hop_length=hop_length,
                test_set_dir=test_set_dir,
                drop_incomplete_windows=drop_incomplete_windows,
            )
        except Exception as e:
            print(f"[{name}] ERROR: {e}")
    return results


def get_input_columns(dataset_name: str) -> list:
    """Return the signal column names for a given dataset."""
    return DATASET_CONFIGS[dataset_name]["signal_cols"]


def get_label_column(dataset_name: str) -> str:
    """Return the label column name for a given dataset."""
    return DATASET_CONFIGS[dataset_name]["label_col"]


def get_subject_column(dataset_name: str) -> str:
    """Return the subject column name for a given dataset."""
    return DATASET_CONFIGS[dataset_name]["subject_col"]


# ─────────────────────────────────────────────────────────────────────────────
# Quick test when run directly
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Available datasets:", sorted(DATASET_CONFIGS.keys()))
    print()

    # Test with daphnet (same as the notebook)
    df_cleaned = load_dataset("daphnet")
    print(df_cleaned[["subject_id", "activity_id", "window_id"]].head(10))
    print()
    print("Signal columns:", get_input_columns("daphnet"))
