"""Write small SYNTHETIC records in WFDB format so the pipeline can be smoke-tested
without downloading PhysioNet. NOT used for any reported results.

Usage:  python tests/make_synthetic.py   (writes into data/raw/, same names as real records)
"""
import sys
from pathlib import Path
import numpy as np
import wfdb

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from config import RAW_DIR, FS

rng = np.random.default_rng(0)


def synth(name, hours=2.0):
    n_min = int(hours * 60)
    labels = np.zeros(n_min, int)
    m = 10
    while m < n_min - 5:  # random apnea episodes
        if rng.random() < 0.08:
            L = rng.integers(3, 15); labels[m:m + L] = 1; m += L + rng.integers(3, 10)
        m += 1
    t, beats, amps = 0.0, [], []
    T = n_min * 60
    while t < T:
        minute = int(t // 60)
        if labels[minute]:   # cyclic HR variation (~40 s cycle), weak breathing modulation
            rr = 0.95 + 0.15 * np.sin(2 * np.pi * t / 40)
            a = 1.0 + 0.03 * np.sin(2 * np.pi * 0.25 * t)
        else:                # normal: respiratory sinus arrhythmia at 0.25 Hz
            rr = 0.85 + 0.04 * np.sin(2 * np.pi * 0.25 * t)
            a = 1.0 + 0.15 * np.sin(2 * np.pi * 0.25 * t)
        rr += rng.normal(0, 0.015)
        beats.append(t); amps.append(a); t += rr
    n = int(T * FS)
    tt = np.arange(n) / FS
    ecg = rng.normal(0, 0.03, n)
    for b, a in zip(beats, amps):
        i0, i1 = int((b - 0.1) * FS), int((b + 0.4) * FS)
        if i0 < 0 or i1 >= n:
            continue
        x = tt[i0:i1] - b
        ecg[i0:i1] += a * np.exp(-(x / 0.012) ** 2) + 0.25 * np.exp(-((x - 0.25) / 0.04) ** 2)
    ecg += 0.1 * np.sin(2 * np.pi * 0.2 * tt)  # baseline wander
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    wfdb.wrsamp(name, fs=FS, units=["mV"], sig_name=["ECG"], p_signal=ecg[:, None],
                fmt=["16"], write_dir=str(RAW_DIR))
    wfdb.wrann(name, "apn", sample=np.arange(n_min) * 60 * FS,
               symbol=["A" if l else "N" for l in labels], write_dir=str(RAW_DIR))


if __name__ == "__main__":
    for r in ["a01", "a02", "a03", "a04", "a05", "a06", "a07", "b01", "b02", "c01", "c02", "c03"]:
        synth(r)
    print("Synthetic records written to", RAW_DIR)
