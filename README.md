# BreatheSync_
BreatheSync is an intelligent sleep-monitoring solution designed to predict potential sleep apnea events 30–60 seconds before they occur, shifting from reactive detection to proactive intervention.


Apnea detection and one-minute early warning from a single-lead ECG.

> ⚠️ **Research prototype only.** BreatheSync is not a medical device and makes no diagnostic claims.

## Team
**Team name:** BreatheSync
**Members:** TODO Name (Team Leader), TODO Name, TODO Name, TODO Name

## Selected problem statement
**Challenge 04: AI-Based Sleep Apnea Detection from Physiological Signals** (Biomedical Signal Processing)

## Problem description
TODO: paste the final problem paragraph from the pitch.

## Proposed solution
BreatheSync processes an overnight ECG recording and:
1. **Detects** apnea vs normal breathing for every minute of the night and shows the detected apnea periods on a timeline, as the challenge requires.
2. **Forecasts** whether the *next* minute will contain apnea, using only past signal. When the risk crosses a threshold, a gentle wearable vibration alert would be triggered. This moves from reacting after an event to warning before it.

## Key features
- Signal-processing pipeline: filtering, R-peak detection, RR cleaning, ECG-derived respiration
- Per-minute apnea detector that uses time context (±2 minutes)
- Next-minute early-warning model that uses past data only
- Interactive overnight timeline with a replay slider, true vs detected apnea periods and alert markers
- Raw ECG inspector with detected R-peaks
- Uncertainty flag for low-confidence minutes
- Quantitative evaluation: event P/R/F1, temporal overlap, timing error, per-patient robustness, calibration and event-duration analysis

## Technology stack
Python, NumPy, SciPy, pandas, scikit-learn, LightGBM, WFDB, Matplotlib, Plotly, Streamlit

## Methodology
**Data.** PhysioNet Apnea-ECG Database, the 35 labelled learning-set recordings (single-lead ECG at 100 Hz, roughly 7–10 h each). Experts labelled every minute as apnea or normal.

**Preprocessing.**
- Band-pass filter the ECG (5–15 Hz) and detect R-peaks with a Pan-Tompkins-style detector using adaptive thresholds.
- Reject implausible or ectopic RR intervals.
- Derive respiration from beat-to-beat QRS amplitude (ECG-derived respiration, EDR).
- Normalise per recording, using RR and amplitude relative to that recording's median.

**Features (per minute).**
- HRV: mean RR, SDNN, RMSSD, pNN50, HR min/max/range.
- QRS-amplitude statistics.
- Spectral power over the previous 5 minutes. The VLF band (0.01–0.04 Hz) captures the cyclic heart-rate variation typical of apnea; we also use LF, HF and the EDR breathing band.

**Models.** Gradient-boosted trees (LightGBM).
- *Detector:* features from minutes t−2 … t+2.
- *Forecaster:* features from minutes t−4 … t only, predicting the label at t+1.
- *Single-minute baseline:* detector features from minute t only. This shows the value of time-series context.

**Validation.**
- Patient-wise split: test patients `a02, a07, a12, a17, b02, c03, c08` are never seen in training.
- Decision thresholds are chosen on out-of-fold predictions from patient-grouped 5-fold cross-validation on the training patients only.
- Post-processing merges detected events separated by a single normal minute.

**Change from the screening proposal.** The Apnea-ECG labels are per minute, not exact onset times. Measuring a precise 30–60 s lead time is therefore not possible with this data, so we refined the forecasting target to *next-minute apnea*, giving up to about 60 s of warning at minute resolution. A "next minute = this minute" baseline scores highly on all minutes but can never warn before an onset. For that reason we report forecasting performance specifically on onsets: minutes that are normal now and become apnea next. We also dropped the CNN-LSTM and self-supervised pretraining to fit the one-day build and favour a working, explainable system (see Future scope).

## Architecture / workflow
```
ECG (100 Hz)
  → band-pass + R-peak detection → RR intervals + ECG-derived respiration
  → per-minute HRV / spectral / EDR features
  → context stacking ─┬─ Detector  (t−2…t+2) → apnea periods on timeline
                      └─ Forecaster (t−4…t)  → next-minute risk → vibration alert
  → evaluation + Streamlit timeline demo
```

## Installation
```bash
git clone <repo-url> && cd breathesync
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Usage
```bash
cd src
python download_data.py      # ~200 MB from PhysioNet (35 labelled records)
python build_dataset.py      # -> data/features.csv
python train.py              # -> models/, results/metrics.json, results/*.png
cd ..
streamlit run app.py         # interactive demo
```
To smoke-test without downloading, run `python tests/make_synthetic.py`, which writes small *synthetic* records. Never report results from synthetic data.

## Dataset information
- **PhysioNet Apnea-ECG Database v1.0.0.** T. Penzel, G.B. Moody, R.G. Mark, A.L. Goldberger, J.H. Peter. *The Apnea-ECG Database.* Computers in Cardiology 2000;27:255–258.
- Goldberger A., et al. *PhysioBank, PhysioToolkit, and PhysioNet.* Circulation 2000;101(23):e215–e220.
- Licence: Open Data Commons Attribution License v1.0. The raw data is not stored in this repository; it is downloaded by `src/download_data.py`.

## Models used
LightGBM gradient-boosted decision trees (400 trees, learning rate 0.03), trained from scratch on hand-crafted physiological features. No pre-trained models.

Held-out patients only (records a07, a12, a17, b02, c03, c08). Results from results/metrics.json.

| Metric | Single-minute baseline | BreatheSync detector |
|---|---|---|
| Accuracy | 0.736 | 0.846 |
| Sensitivity | 0.961 | 0.945 |
| Specificity | 0.609 | 0.790 |
| F1 | 0.725 | 0.816 |
| AUROC | 0.911 | 0.950 |
| Brier / ECE | 0.116 / 0.042 | 0.081 / 0.027 |

| Event-level (pooled over all 57 true events) | Value |
|---|---|
| Event precision / recall / F1 | 0.522 / 0.877 / 0.654 |
| Median temporal overlap (IoU) | 0.183 |
| Median onset / offset timing error (min) | 25.0 / 8.5 |
| Per-patient minute F1 (apnea patients, mean ± sd, range) | 0.772 ± 0.176, 0.586–0.971 |

| Early warning | Value |
|---|---|
| AUROC, currently-normal minutes | 0.827 |
| Onsets warned 1 min ahead | 0.737 (42 of 57) |
| False alarms per hour | 10.6 |
| Persistence baseline: onsets warned | 0 |
Figures: `results/per_patient_f1.png`, `results/calibration.png`, `results/event_durations.png`, `results/feature_importance.png`

## Screenshots / demo
TODO: add screenshots of the timeline tab and results tab, plus the demo video link.

## Limitations
- Minute-level labels limit timing precision to about ±1 minute; exact onset lead time can't be measured.
- Apnea-ECG has a small, mostly male adult cohort, so generalisation to other populations is untested.
- The ECG comes from clinical sensors, not consumer wearables.
- Early warning of onsets is harder than detection. We report false alarms and missed onsets honestly.
- The vibration alert is simulated in the demo.

## Future scope
- 1D-CNN + LSTM on raw RR/EDR sequences, and self-supervised pretraining.
- Multimodal signals (SpO₂, airflow) using the MIT-BIH Polysomnographic Database, whose 30-second annotations allow finer lead-time measurement.
- On-device inference with smartwatch/ring PPG and a real haptic alert.
- Clinical validation studies.
  
**Mobile app prototype** (Streamlit, phone layout): Home (night summary), Live Monitoring
  (replays a held-out night minute by minute with heart rate, HRV, signal quality and early-warning
  alerts), Detailed Analysis (night timeline vs expert labels, raw ECG with detected heartbeats, HRV
  trend), History, Insights (patient-friendly trends and plain-language reliability) and Settings
  (alert sensitivity with live trade-off, vibration toggle). Accessible navigation: 62px touch
  targets, labelled tabs, WCAG-checked contrast, keyboard focus states

## External resources & attributions
- PhysioNet Apnea-ECG Database (see citations above)
- Open-source libraries listed in `requirements.txt`
- R-peak detector inspired by Pan & Tompkins, *IEEE Trans. Biomed. Eng.* 1985
- **AI tool disclosure:** Claude (Anthropic) was used to assist with code scaffolding, debugging and documentation. All team members reviewed and understand the code.
  ## Record a02 could not be downloaded from PhysioNet during the event (server errors), so 34 of the 35 labelled recordings were used.
