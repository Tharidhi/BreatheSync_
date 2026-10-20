"""Build the per-minute feature table from the raw Apnea-ECG records.

Usage:  python src/build_dataset.py
Output: data/features.csv  (one row per minute: record, minute, label, features...)
"""
import time
import numpy as np
import pandas as pd
import wfdb

from config import RAW_DIR, FEATURES_CSV, LEARNING_RECORDS, MIN_SAMPLES
from features import minute_features, FEATURE_NAMES


def load_record(name):
    """Return ECG signal and per-minute labels (1 = apnea, 0 = normal)."""
    path = str(RAW_DIR / name)
    rec = wfdb.rdrecord(path)
    ann = wfdb.rdann(path, "apn")
    ecg = rec.p_signal[:, 0]
    labels = np.array([1 if s == "A" else 0 for s in ann.symbol])
    # Only keep minutes that are fully covered by the signal
    n_minutes = min(len(labels), len(ecg) // MIN_SAMPLES)
    return ecg, labels[:n_minutes], n_minutes


def available_records():
    return [r for r in LEARNING_RECORDS if (RAW_DIR / f"{r}.hea").exists()
            and (RAW_DIR / f"{r}.apn").exists()]


def main():
    records = available_records()
    if not records:
        raise SystemExit(f"No records found in {RAW_DIR}. Run: python src/download_data.py")
    rows = []
    for name in records:
        t = time.time()
        ecg, labels, n = load_record(name)
        feats, peaks, _ = minute_features(ecg, n)
        df = pd.DataFrame(feats, columns=FEATURE_NAMES)
        df.insert(0, "label", labels)
        df.insert(0, "minute", np.arange(n))
        df.insert(0, "record", name)
        rows.append(df)
        print(f"{name}: {n} min, {labels.mean():.0%} apnea, {len(peaks)} beats "
              f"({time.time() - t:.1f}s)")
    out = pd.concat(rows, ignore_index=True)
    FEATURES_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(FEATURES_CSV, index=False)
    print(f"Saved {len(out)} minutes from {len(records)} records -> {FEATURES_CSV}")


if __name__ == "__main__":
    main()
