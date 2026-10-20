"""Train and evaluate the BreatheSync models with a subject-wise split.

  1. Detector   : is THIS minute apnea?          (core challenge task, timeline)
  2. Forecaster : will the NEXT minute be apnea?  (our innovation: early warning)
  3. Single-minute baseline: detector without time context, to show the value
     of continuous time-series modelling.

Usage:  python src/train.py
Output: models/*.joblib, results/metrics.json, results/per_record.csv,
        results/predictions.csv, results/*.png
"""
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lightgbm import LGBMClassifier
from sklearn.calibration import calibration_curve
from sklearn.model_selection import GroupKFold

from config import (FEATURES_CSV, MODELS_DIR, RESULTS_DIR, TEST_RECORDS,
                    DETECT_OFFSETS, FORECAST_OFFSETS, UNCERTAIN_BAND, RANDOM_STATE)
from features import FEATURE_NAMES
from metrics import (minute_metrics, event_metrics, forecast_metrics,
                     fill_short_gaps, to_events)


def add_context(df, offsets):
    """Stack features from neighbouring minutes (within the same recording)."""
    cols = []
    g = df.groupby("record", sort=False)
    parts = []
    for k in offsets:
        shifted = g[FEATURE_NAMES].shift(-k)
        shifted.columns = [f"{c}_t{k:+d}" for c in FEATURE_NAMES]
        parts.append(shifted)
        cols += list(shifted.columns)
    return pd.concat(parts, axis=1), cols


def make_model():
    return LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=31,
                          min_child_samples=30, subsample=0.8, subsample_freq=1,
                          colsample_bytree=0.8, random_state=RANDOM_STATE, verbose=-1)


def best_threshold(y, p):
    """Pick the decision threshold that maximises F1 on out-of-fold TRAINING predictions."""
    grid = np.linspace(0.2, 0.8, 61)
    f1s = []
    for t in grid:
        yhat = p >= t
        tp = (yhat & (y == 1)).sum(); fp = (yhat & (y == 0)).sum(); fn = (~yhat & (y == 1)).sum()
        f1s.append(2 * tp / (2 * tp + fp + fn) if tp else 0)
    return float(grid[int(np.argmax(f1s))])


def fit_task(train, X_cols, y_col):
    """Out-of-fold predictions (grouped by patient) -> threshold; then fit on all train."""
    oof = np.zeros(len(train))
    gkf = GroupKFold(n_splits=5)
    for tr, va in gkf.split(train, train[y_col], groups=train["record"]):
        m = make_model().fit(train.iloc[tr][X_cols], train.iloc[tr][y_col])
        oof[va] = m.predict_proba(train.iloc[va][X_cols])[:, 1]
    thr = best_threshold(train[y_col].values, oof)
    model = make_model().fit(train[X_cols], train[y_col])
    return model, thr


def main():
    MODELS_DIR.mkdir(exist_ok=True); RESULTS_DIR.mkdir(exist_ok=True)
    df = pd.read_csv(FEATURES_CSV).sort_values(["record", "minute"]).reset_index(drop=True)

    det_X, det_cols = add_context(df, DETECT_OFFSETS)
    fc_X, fc_cols = add_context(df, FORECAST_OFFSETS)
    single_cols = [f"{c}_t+0" for c in FEATURE_NAMES]
    df = pd.concat([df, det_X, fc_X.drop(columns=[c for c in fc_cols if c in det_X.columns])], axis=1).copy()
    df["label_next"] = df.groupby("record")["label"].shift(-1)

    test_recs = [r for r in TEST_RECORDS if r in set(df.record)]
    if not test_recs:  # fallback for partial downloads
        test_recs = sorted(df.record.unique())[-max(1, df.record.nunique() // 5):]
    is_test = df.record.isin(test_recs)
    train, test = df[~is_test].copy(), df[is_test].copy()
    print(f"Train: {train.record.nunique()} patients, {len(train)} min | "
          f"Test: {test.record.nunique()} patients ({', '.join(test_recs)}), {len(test)} min")

    # ---------------- 1) Detector (with time context) and single-minute baseline
    det_model, det_thr = fit_task(train, det_cols, "label")
    base_model, base_thr = fit_task(train, single_cols, "label")
    test["p_detect"] = det_model.predict_proba(test[det_cols])[:, 1]
    test["p_single"] = base_model.predict_proba(test[single_cols])[:, 1]

    # ---------------- 2) Forecaster (past only)
    tr_fc = train.dropna(subset=["label_next"])
    fc_model, fc_thr = fit_task(tr_fc, fc_cols, "label_next")
    test["p_forecast_next"] = fc_model.predict_proba(test[fc_cols])[:, 1]

    # ---------------- Evaluate per record (events never cross recordings)
    test["pred_detect"] = 0
    rows = []
    for rec, g in test.groupby("record"):
        y = g.label.values
        pred = fill_short_gaps((g.p_detect.values >= det_thr).astype(int))
        test.loc[g.index, "pred_detect"] = pred
        r = {"record": rec, "minutes": len(g), "apnea_frac": y.mean()}
        r.update({f"min_{k}": v for k, v in minute_metrics(y, g.p_detect.values, det_thr).items()})
        r.update(event_metrics(y, pred))
        rows.append(r)
    per_rec = pd.DataFrame(rows)
    per_rec.to_csv(RESULTS_DIR / "per_record.csv", index=False)

    y = test.label.values
    pooled_events = {k: [] for k in ["event_precision", "event_recall", "event_f1",
                                     "mean_iou", "onset_error_min", "offset_error_min"]}
    for k in pooled_events:
        pooled_events[k] = float(np.nanmean(per_rec[k]))

    fcm = test.dropna(subset=["label_next"])
    metrics = {
        "test_records": test_recs,
        "detector": {"threshold": det_thr, **minute_metrics(y, test.p_detect.values, det_thr),
                     **{f"{k}_mean_over_patients": v for k, v in pooled_events.items()}},
        "single_minute_baseline": {"threshold": base_thr,
                                   **minute_metrics(y, test.p_single.values, base_thr)},
        "robustness": {
            "per_patient_minute_f1_mean": float(per_rec.min_f1.mean()),
            "per_patient_minute_f1_std": float(per_rec.min_f1.std()),
            "per_patient_minute_f1_min": float(per_rec.min_f1.min()),
            "per_patient_minute_f1_max": float(per_rec.min_f1.max()),
        },
        "forecaster": {"threshold": fc_thr,
                       **forecast_metrics(fcm.label.values, fcm.label_next.values.astype(int),
                                          fcm.p_forecast_next.values, fc_thr, len(fcm))},
        "uncertain_fraction": float(((test.p_detect > UNCERTAIN_BAND[0]) &
                                     (test.p_detect < UNCERTAIN_BAND[1])).mean()),
    }
    with open(RESULTS_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=float)

    test[["record", "minute", "label", "p_detect", "pred_detect", "p_single",
          "p_forecast_next", "mean_rr"]].to_csv(RESULTS_DIR / "predictions.csv", index=False)
    joblib.dump({"model": det_model, "cols": det_cols, "thr": det_thr}, MODELS_DIR / "detector.joblib")
    joblib.dump({"model": fc_model, "cols": fc_cols, "thr": fc_thr}, MODELS_DIR / "forecaster.joblib")

    make_figures(test, per_rec, det_model, det_cols)
    print(json.dumps(metrics, indent=2, default=float))


def make_figures(test, per_rec, det_model, det_cols):
    # Calibration (reliability curve)
    fig, ax = plt.subplots(figsize=(5, 5))
    for col, name in [("p_detect", "Detector"), ("p_forecast_next", "Forecaster")]:
        d = test.dropna(subset=[col])
        target = d.label if col == "p_detect" else d.groupby("record").label.shift(-1)
        m = target.notna()
        pt, pp = calibration_curve(target[m].astype(int), d[col][m], n_bins=10)
        ax.plot(pp, pt, marker="o", label=name)
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")
    ax.set(xlabel="Predicted probability", ylabel="Observed apnea frequency",
           title="Reliability curve (test patients)")
    ax.legend(); fig.tight_layout(); fig.savefig(RESULTS_DIR / "calibration.png", dpi=150)

    # Robustness: per-patient F1
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.bar(per_rec.record, per_rec.min_f1, color="#2b6cb0", label="Minute F1")
    ax.plot(per_rec.record, per_rec.event_f1, "o", color="#ed8936", label="Event F1")
    ax.set(ylim=(0, 1.05), ylabel="F1", title="Per-patient performance (held-out patients)")
    ax.legend(); fig.tight_layout(); fig.savefig(RESULTS_DIR / "per_patient_f1.png", dpi=150)

    # Event-duration analysis
    true_d, pred_d = [], []
    for _, g in test.groupby("record"):
        true_d += [e - s for s, e in to_events(g.label.values)]
        pred_d += [e - s for s, e in to_events(g.pred_detect.values)]
    fig, ax = plt.subplots(figsize=(6, 3.5))
    bins = np.arange(1, max(true_d + pred_d + [2]) + 2, 2)
    ax.hist([true_d, pred_d], bins=bins, label=["True events", "Detected events"],
            color=["#90cdf4", "#2b6cb0"])
    ax.set(xlabel="Event duration (minutes)", ylabel="Count", title="Event-duration analysis")
    ax.legend(); fig.tight_layout(); fig.savefig(RESULTS_DIR / "event_durations.png", dpi=150)

    # Top features
    imp = pd.Series(det_model.feature_importances_, index=det_cols).nlargest(15)[::-1]
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.barh(imp.index, imp.values, color="#2b6cb0")
    ax.set(title="Top detector features", xlabel="Split importance")
    fig.tight_layout(); fig.savefig(RESULTS_DIR / "feature_importance.png", dpi=150)
    plt.close("all")


if __name__ == "__main__":
    main()
