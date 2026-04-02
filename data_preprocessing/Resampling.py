#!/usr/bin/env python3
"""
Resample multiple HAR datasets to a unified sampling rate using polyphase FIR.

Usage
-----
python resample_to_30hz.py \
    --input_dir Original_dataset \
    --output_dir convert_30Hz_poly \
    --target_fs 30
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from fractions import Fraction
from scipy.signal import resample_poly
import dask.dataframe as dd


# ─────────────────────────────────────────────────────────────
# Polyphase FIR Resampler
# ─────────────────────────────────────────────────────────────

def resample_imu_poly(x: np.ndarray, orig_fs: float, target_fs: float, axis: int = 0):
    """
    Polyphase FIR resampling with anti-alias filtering.

    Parameters
    ----------
    x : ndarray [T, C]
    orig_fs : float
    target_fs : float
    axis : int

    Returns
    -------
    ndarray
    """

    if orig_fs == target_fs:
        return x.astype(np.float32)

    x_in = np.asarray(x, dtype=np.float32)

    # Forward-fill NaNs
    if np.isnan(x_in).any():

        arr = x_in if axis == 0 else np.swapaxes(x_in, axis, 0)
        arr = arr.copy()

        for c in range(arr.shape[1]):

            col = arr[:, c]
            mask = np.isnan(col)

            if mask.any():
                idx = np.where(~mask, np.arange(len(col)), 0)
                np.maximum.accumulate(idx, out=idx)
                col[:] = col[idx]

        x_in = arr if axis == 0 else np.swapaxes(arr, 0, axis)

    frac = Fraction(target_fs, orig_fs).limit_denominator(1000)

    y = resample_poly(
        x_in,
        up=frac.numerator,
        down=frac.denominator,
        axis=axis
    )

    return y.astype(np.float32)


# ─────────────────────────────────────────────────────────────
# Dataset Configuration
# ─────────────────────────────────────────────────────────────

META_COLS = {"timestamp", 'activity_name', "Trial", "Scenario"}

DATASET_CONFIGS = {

    "w_har": {"file": "w_har.csv", "orig_fs": 250, "has_sess": True},
    "pamap2": {"file": "pamap2.csv", "orig_fs": 100, "has_sess": True},
    "har70": {"file": "har70.csv", "orig_fs": 50, "has_sess": True},
    "uci_har": {"file": "uci_har.csv", "orig_fs": 50, "has_sess": True},
    "usc_had": {"file": "usc_had.csv", "orig_fs": 100, "has_sess": True},
    "dsads": {"file": "dsads.csv", "orig_fs": 25, "has_sess": True},
    "ku_har": {"file": "ku_har.csv", "orig_fs": 100, "has_sess": False},
    "falldet": {"file": "falldet.csv", "orig_fs": 50, "has_sess": True},
    "daphnet": {"file": "daphnet.csv", "orig_fs": 64, "has_sess": True},
    "mhealth": {"file": "mhealth.csv", "orig_fs": 50, "has_sess": True},
    "wisdm": {"file": "wisdm.csv", "orig_fs": 20, "has_sess": True},
    "real_world": {"file": "real_world.csv", "orig_fs": 50, "has_sess": True},
    "har_sense": {"file": "har_sense.csv", "orig_fs": 50, "has_sess": True},
    "motion_sense": {"file": "motion_sense.csv", "orig_fs": 50, "has_sess": True},
    "wear": {"file": "wear.csv", "orig_fs": 50, "has_sess": True},
    "real_life_har": {"file": "real_life_har.csv", "orig_fs": 20, "has_sess": True},
    "utd_mhad": {"file": "utd_mhad.csv", "orig_fs": 50, "has_sess": True},
    "hang_time": {"file": "hang_time.csv", "orig_fs": 50, "has_sess": True},
    "opportunity": {"file": "opportunity.csv", "orig_fs": 30, "has_sess": True},
    "uca_ehar": {"file": "uca_ehar.csv", "orig_fs": 25, "has_sess": True},
    "hugadb": {"file": "hugadb.csv", "orig_fs": 60, "has_sess": True},
    "up_fall": {"file": "up_fall.csv", "orig_fs": 20, "has_sess": True},
    "wisdm_19_phone": {"file": "wisdm_19_phone.csv", "orig_fs": 20, "has_sess": True},
    "sad": {"file": "sad.csv", "orig_fs": 50, "has_sess": True},
    "wisdm_19_watch": {"file": "wisdm_19_watch.csv", "orig_fs": 20, "has_sess": True},
    "gotov": {"file": "gotov.csv", "orig_fs": 50, "has_sess": True},
    "uma_fall": {"file": "uma_fall.csv", "orig_fs": 200, "has_sess": True},
    "capture24": {"file": "capture24.csv", "orig_fs": 100, "has_sess": True}
}


# ─────────────────────────────────────────────────────────────
# Dataset Resampler
# ─────────────────────────────────────────────────────────────

def resample_dataset(name, cfg, input_dir, output_dir, target_fs):

    
    fpath = input_dir / cfg["file"]

    if not fpath.exists():
        raise FileNotFoundError(f"{fpath} not found")

    orig_fs = cfg["orig_fs"]
    has_sess = cfg.get("has_sess", True)

    print(f"\nProcessing {name}")

    # df = pd.read_csv(fpath, on_bad_lines="skip", encoding="latin1",low_memory=True)
    df = dd.read_csv(fpath, on_bad_lines="skip", encoding="latin1", blocksize="1024MB")

    sig_cols = [c for c in df.columns if c not in META_COLS]

    if has_sess and "session_id" in df.columns:
        group_keys = ["subject_id", "activity_id", "session_id"]
    else:
        group_keys = ["subject_id", "activity_id"]

    min_samples_orig = orig_fs
    min_samples_target = target_fs

    out_chunks = []
    skipped = 0

    for keys, segment in df.groupby(group_keys, sort=False):

        segment = segment.reset_index(drop=True)

        if len(segment) < min_samples_orig:
            skipped += 1
            continue

        X = segment[sig_cols].values.astype(np.float32)

        X_res = resample_imu_poly(X, orig_fs, target_fs)

        if len(X_res) < min_samples_target:
            skipped += 1
            continue

        chunk = pd.DataFrame(X_res, columns=sig_cols)

        if isinstance(keys, tuple):
            for k, v in zip(group_keys, keys):
                chunk[k] = v
        else:
            chunk[group_keys[0]] = keys

        out_chunks.append(chunk)

    if not out_chunks:
        raise RuntimeError("All segments skipped")

    out = pd.concat(out_chunks, ignore_index=True)

    out_path = output_dir / f"{name}_{target_fs}Hz.csv"

    out.to_csv(out_path, index=False)

    ratio = len(out) / len(df)

    print(
        f"{orig_fs}Hz → {target_fs}Hz | "
        f"{len(df):,} → {len(out):,} rows | "
        f"ratio {ratio:.4f} | "
        f"skipped {skipped} segments"
    )


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing original datasets"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save resampled datasets"
    )

    parser.add_argument(
        "--target_fs",
        type=int,
        default=30,
        help="Target sampling frequency"
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    target_fs = args.target_fs

    output_dir.mkdir(parents=True, exist_ok=True)

    print("\nResampling datasets")
    print(f"Input  : {input_dir.resolve()}")
    print(f"Output : {output_dir.resolve()}")
    print(f"Target : {target_fs} Hz")

    for name, cfg in DATASET_CONFIGS.items():

        if cfg["file"] == "ku_har.csv":
            continue

        try:
            resample_dataset(name, cfg, input_dir, output_dir, target_fs)

        except Exception as e:

            print(f"[{name}] ERROR: {e}")

    print("\nAll datasets processed.")


if __name__ == "__main__":
    main()