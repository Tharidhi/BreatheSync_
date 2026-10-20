"""Evaluation metrics requested by Challenge 04:
event precision/recall/F1, temporal overlap, timing error, robustness and calibration,
plus forecasting metrics (warned onsets, false alarms per hour)."""
import numpy as np
from sklearn.metrics import roc_auc_score, brier_score_loss, f1_score


def fill_short_gaps(pred, max_gap=1):
    """Post-processing: merge apnea events separated by <= max_gap normal minutes."""
    pred = pred.copy()
    events = to_events(pred)
    for (s1, e1), (s2, e2) in zip(events[:-1], events[1:]):
        if s2 - e1 <= max_gap:
            pred[e1:s2] = 1
    return pred


def to_events(binary):
    """Contiguous runs of 1s -> list of (start, end) with end exclusive (minutes)."""
    b = np.concatenate([[0], np.asarray(binary, int), [0]])
    d = np.diff(b)
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def _iou(a, b):
    inter = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union else 0.0


def event_metrics(y_true, y_pred):
    """A predicted event is a hit if it overlaps a true event.
    For each true event we take its best-overlapping predicted event to measure
    temporal overlap (IoU) and onset/offset timing error (minutes)."""
    te, pe = to_events(y_true), to_events(y_pred)
    matched_pred = sum(any(_iou(p, t) > 0 for t in te) for p in pe)
    matched_true, ious, on_err, off_err = 0, [], [], []
    for t in te:
        overlaps = [(p, _iou(p, t)) for p in pe if _iou(p, t) > 0]
        if overlaps:
            matched_true += 1
            p, iou = max(overlaps, key=lambda x: x[1])
            ious.append(iou)
            on_err.append(abs(p[0] - t[0]))
            off_err.append(abs(p[1] - t[1]))
    prec = matched_pred / len(pe) if pe else np.nan
    rec = matched_true / len(te) if te else np.nan
    f1 = 2 * prec * rec / (prec + rec) if pe and te and (prec + rec) > 0 else np.nan
    return {
        "n_true_events": len(te), "n_pred_events": len(pe),
        "event_precision": prec, "event_recall": rec, "event_f1": f1,
        "mean_iou": float(np.mean(ious)) if ious else np.nan,
        "onset_error_min": float(np.mean(on_err)) if on_err else np.nan,
        "offset_error_min": float(np.mean(off_err)) if off_err else np.nan,
    }


def expected_calibration_error(y, p, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, bins) - 1, 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            ece += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(ece)


def minute_metrics(y, p, thr):
    yhat = (p >= thr).astype(int)
    tp = int(((yhat == 1) & (y == 1)).sum()); tn = int(((yhat == 0) & (y == 0)).sum())
    fp = int(((yhat == 1) & (y == 0)).sum()); fn = int(((yhat == 0) & (y == 1)).sum())
    return {
        "accuracy": (tp + tn) / len(y),
        "sensitivity": tp / (tp + fn) if tp + fn else np.nan,
        "specificity": tn / (tn + fp) if tn + fp else np.nan,
        "precision": tp / (tp + fp) if tp + fp else np.nan,
        "f1": f1_score(y, yhat, zero_division=0),
        "auroc": roc_auc_score(y, p) if len(np.unique(y)) > 1 else np.nan,
        "brier": brier_score_loss(y, p),
        "ece": expected_calibration_error(y, p),
    }


def forecast_metrics(cur, nxt, p_next, thr, minutes_total):
    """cur/nxt = true labels at minute t and t+1; p_next = forecast made at t for t+1.
    Onset = cur normal and next apnea: the case where an early warning matters."""
    alert = p_next >= thr
    onset = (cur == 0) & (nxt == 1)
    quiet = (cur == 0) & (nxt == 0)
    out = {
        "auroc_all": roc_auc_score(nxt, p_next) if len(np.unique(nxt)) > 1 else np.nan,
        "n_onsets": int(onset.sum()),
        "warned_onset_rate": float(alert[onset].mean()) if onset.any() else np.nan,
        "false_alarms_per_hour": float(alert[quiet].sum() / (minutes_total / 60)),
    }
    m = cur == 0  # among minutes that are currently normal, can we tell which tip into apnea?
    out["auroc_onset"] = roc_auc_score(nxt[m], p_next[m]) if len(np.unique(nxt[m])) > 1 else np.nan
    # Persistence baseline ("next minute = this minute") never warns an onset by definition
    out["persistence_accuracy_all"] = float((cur == nxt).mean())
    out["persistence_warned_onset_rate"] = 0.0
    return out
