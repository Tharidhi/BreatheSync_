"""BreatheSync demo: load an overnight recording -> timeline -> detected apnea periods
-> early-warning forecast.   Run:  streamlit run app.py"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))
from config import RAW_DIR, RESULTS_DIR, FS, MIN_SAMPLES, UNCERTAIN_BAND  # noqa: E402
from metrics import to_events, event_metrics  # noqa: E402

NAVY, BLUE, SKY, MIST, AMBER = "#1a365d", "#2b6cb0", "#63b3ed", "#bee3f8", "#dd6b20"

st.set_page_config(page_title="BreatheSync", page_icon="🌙", layout="wide")


@st.cache_data
def load_results():
    pred = pd.read_csv(RESULTS_DIR / "predictions.csv")
    with open(RESULTS_DIR / "metrics.json") as f:
        metrics = json.load(f)
    return pred, metrics


@st.cache_data
def load_ecg(record, start_min, n_min=2):
    import wfdb
    from features import detect_rpeaks
    path = RAW_DIR / record
    if not (path.with_suffix(".hea")).exists():
        return None
    sig_len = wfdb.rdheader(str(path)).sig_len
    b = min((start_min + n_min) * MIN_SAMPLES, sig_len)
    a = max(0, b - n_min * MIN_SAMPLES)
    sig = wfdb.rdrecord(str(path), sampfrom=a, sampto=b).p_signal[:, 0]
    peaks, _ = detect_rpeaks(sig)
    return sig, peaks, a


st.title("BreatheSync")
st.markdown("**Better breath, deeper sleep, brighter tomorrows.** Apnea detection and "
            "one-minute early warning from a single-lead ECG.")
st.caption("Research prototype only. Not a medical device and not intended for diagnosis.")

if not (RESULTS_DIR / "predictions.csv").exists():
    st.error("No results yet. Run `python src/build_dataset.py` then `python src/train.py`.")
    st.stop()

pred, metrics = load_results()
tab_night, tab_results, tab_how = st.tabs(["Overnight timeline", "Model results", "How it works"])

# =====================================================================  TIMELINE
with tab_night:
    c1, c2 = st.columns([1, 2])
    record = c1.selectbox("Held-out patient recording", sorted(pred.record.unique()))
    fc_thr_default = float(metrics["forecaster"]["threshold"])
    alert_thr = c2.slider("Vibration alert threshold (higher = fewer alerts, more missed warnings)",
                          0.05, 0.95, round(fc_thr_default, 2), 0.01)

    g = pred[pred.record == record].reset_index(drop=True)
    hours = g.minute / 60
    hr = 60 / g.mean_rr
    alerts = g.p_forecast_next >= alert_thr
    nxt = g.label.shift(-1)
    onset = (g.label == 0) & (nxt == 1)
    quiet = (g.label == 0) & (nxt == 0)
    ev = event_metrics(g.label.values, g.pred_detect.values)

    m = st.columns(6)
    m[0].metric("Recording", f"{len(g) / 60:.1f} h")
    m[1].metric("True apnea events", ev["n_true_events"])
    m[2].metric("Detected events", ev["n_pred_events"])
    m[3].metric("Event F1 (this night)", f"{ev['event_f1']:.2f}" if ev["event_f1"] == ev["event_f1"] else "–")
    m[4].metric("Onsets warned 1 min ahead", f"{int(alerts[onset].sum())}/{int(onset.sum())}")
    m[5].metric("False alarms / hour", f"{alerts[quiet].sum() / (len(g) / 60):.1f}")

    t_now = st.slider("Replay the night: drag to move through time (minutes)", 0, len(g) - 1, len(g) - 1)

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                        row_heights=[0.28, 0.36, 0.36],
                        subplot_titles=("Heart rate (bpm)", "Apnea detection: probability this minute",
                                        "Early warning: risk that the NEXT minute is apnea"))
    shown = g.minute <= t_now
    for s, e in to_events(g.label.values):           # true apnea periods (ground truth)
        if s <= t_now:
            fig.add_vrect(x0=s / 60, x1=min(e, t_now + 1) / 60, fillcolor=MIST, opacity=0.6,
                          line_width=0, layer="below")
    fig.add_trace(go.Scatter(x=hours[shown], y=hr[shown], line=dict(color=NAVY, width=1),
                             name="Heart rate"), 1, 1)
    fig.add_trace(go.Scatter(x=hours[shown], y=g.p_detect[shown], line=dict(color=BLUE, width=1.5),
                             name="Detection probability"), 2, 1)
    fig.add_trace(go.Scatter(x=hours[shown], y=g.pred_detect[shown] * 1.0, fill="tozeroy",
                             line=dict(color=BLUE, width=0), fillcolor="rgba(43,108,176,0.25)",
                             name="Detected apnea"), 2, 1)
    unc = shown & g.p_detect.between(*UNCERTAIN_BAND)
    fig.add_trace(go.Scatter(x=hours[unc], y=g.p_detect[unc], mode="markers",
                             marker=dict(color=AMBER, size=5), name="Uncertain minute"), 2, 1)
    fig.add_trace(go.Scatter(x=hours[shown], y=g.p_forecast_next[shown], line=dict(color=SKY, width=1.5),
                             name="Next-minute risk"), 3, 1)
    fig.add_hline(y=alert_thr, line=dict(color=AMBER, dash="dash", width=1), row=3, col=1)
    al = shown & alerts
    fig.add_trace(go.Scatter(x=hours[al], y=g.p_forecast_next[al], mode="markers",
                             marker=dict(symbol="triangle-up", color=AMBER, size=8),
                             name="Vibration alert"), 3, 1)
    fig.update_yaxes(range=[-0.02, 1.02], row=2, col=1)
    fig.update_yaxes(range=[-0.02, 1.02], row=3, col=1)
    fig.update_xaxes(title_text="Hours since recording start", range=[0, len(g) / 60], row=3, col=1)
    fig.update_layout(height=640, margin=dict(l=10, r=10, t=40, b=10),
                      legend=dict(orientation="h", y=-0.12), hovermode="x unified")
    st.plotly_chart(fig, width="stretch")
    st.caption("Light blue bands = true apnea minutes from the expert annotations.")

    row = g.iloc[t_now]
    status = "🔴 Apnea detected" if row.pred_detect else "🟢 Breathing normally"
    alert_txt = "📳 Gentle vibration alert sent" if alerts.iloc[t_now] else "No alert"
    truth = "apnea" if row.label else "normal"
    st.info(f"**Minute {int(row.minute)}** ({row.minute / 60:.1f} h) · {status} "
            f"(p = {row.p_detect:.2f}) · Next-minute risk {row.p_forecast_next:.2f} → {alert_txt} "
            f"· Expert label: {truth}")

    with st.expander("Inspect the raw ECG around this minute"):
        ecg = load_ecg(record, int(row.minute))
        if ecg is None:
            st.write("Raw signal files not found in data/raw.")
        else:
            sig, peaks, a = ecg
            t = a / FS + np.arange(len(sig)) / FS
            f2 = go.Figure(go.Scatter(x=t, y=sig, line=dict(color=NAVY, width=1), name="ECG"))
            f2.add_trace(go.Scatter(x=t[peaks], y=sig[peaks], mode="markers",
                                    marker=dict(color=AMBER, size=6), name="Detected R-peaks"))
            f2.update_layout(height=260, margin=dict(l=10, r=10, t=10, b=10),
                             xaxis_title="Seconds", yaxis_title="mV")
            st.plotly_chart(f2, width="stretch")

# =====================================================================  RESULTS
with tab_results:
    d, b, fc = metrics["detector"], metrics["single_minute_baseline"], metrics["forecaster"]
    st.subheader("Detection on held-out patients")
    st.write(f"Test patients never seen in training: {', '.join(metrics['test_records'])}")
    keys = ["accuracy", "sensitivity", "specificity", "precision", "f1", "auroc", "brier", "ece"]
    st.dataframe(pd.DataFrame({
        "Single-minute baseline": [b[k] for k in keys],
        "BreatheSync (±2 min context)": [d[k] for k in keys]}, index=keys).round(3),
        width="stretch")
    st.markdown("**Event-level** (mean over patients)")
    st.dataframe(pd.DataFrame({k.replace("_mean_over_patients", ""): [v] for k, v in d.items()
                               if k.endswith("_mean_over_patients")}).round(3),
                 width="stretch", hide_index=True)

    st.subheader("Early warning")
    st.dataframe(pd.DataFrame({
        "Metric": ["AUROC (all minutes)", "AUROC (currently-normal minutes only)",
                   "Onsets warned 1 min ahead", "False alarms per hour",
                   "Persistence baseline: onsets warned"],
        "Value": [fc["auroc_all"], fc["auroc_onset"], fc["warned_onset_rate"],
                  fc["false_alarms_per_hour"], fc["persistence_warned_onset_rate"]]}).round(3),
        width="stretch", hide_index=True)
    st.caption("A 'next minute = this minute' baseline already scores well on all minutes but can "
               "never warn before an onset, so onset-focused metrics are the honest test.")

    rb = metrics["robustness"]
    st.subheader("Robustness across patients")
    st.write(f"Minute F1 per patient: mean {rb['per_patient_minute_f1_mean']:.2f} "
             f"± {rb['per_patient_minute_f1_std']:.2f} (range {rb['per_patient_minute_f1_min']:.2f}"
             f"–{rb['per_patient_minute_f1_max']:.2f})")
    cols = st.columns(2)
    for i, name in enumerate(["per_patient_f1.png", "calibration.png",
                              "event_durations.png", "feature_importance.png"]):
        if (RESULTS_DIR / name).exists():
            cols[i % 2].image(str(RESULTS_DIR / name))

# =====================================================================  HOW IT WORKS
with tab_how:
    st.markdown("""
**1. Signal processing.** Band-pass filter the ECG, detect R-peaks (Pan-Tompkins style), clean the
RR intervals, and derive respiration from QRS-amplitude changes (ECG-derived respiration).

**2. Per-minute features.** Heart-rate variability (mean RR, SDNN, RMSSD, pNN50, HR range), spectral
power over the last 5 minutes (the very-low-frequency band captures the cyclic heart-rate swings that
apnea causes) and breathing-band power of the ECG-derived respiration signal.

**3. Detection.** Gradient-boosted trees classify each minute using features from the two minutes
before and after, because apnea is a pattern over time rather than a single snapshot.

**4. Early warning.** A second model sees only the past five minutes and estimates the risk that the
next minute will be apnea. Above the threshold, a gentle wearable vibration would be triggered.

**5. Evaluation.** Patient-wise split, minute and event metrics, temporal overlap, timing error,
per-patient robustness, calibration, and forecasting metrics focused on apnea onsets.

Data: PhysioNet Apnea-ECG Database (Penzel et al., 2000; Goldberger et al., 2000).
""")
