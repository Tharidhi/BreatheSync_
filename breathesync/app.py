"""BreatheSync mobile app prototype (Streamlit).
Replays real held-out overnight ECG recordings through the trained models.
Run:  python -m streamlit run app.py"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))
from config import RAW_DIR, RESULTS_DIR, FEATURES_CSV, FS, MIN_SAMPLES  # noqa: E402
from metrics import to_events  # noqa: E402

INK, NAVY, PURPLE, PINK, CREAM = "#111530", "#2A3263", "#8869B7", "#F1C7D1", "#FFF3E5"
MUTED, GREEN = "rgba(255,243,229,0.80)", "#7FD1A8"

st.set_page_config(page_title="BreatheSync", page_icon="🌙", layout="centered")

# ------------------------------------------------------------------ styling
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700;800&display=swap');
html, body, [class*="css"], .stMarkdown, button, input {{
  font-family: 'Nunito', 'Segoe UI', system-ui, sans-serif !important; }}
header[data-testid="stHeader"], footer, #MainMenu {{ display: none !important; }}
.stApp {{ background: radial-gradient(circle at 50% 0%, #1c2150 0%, {INK} 60%); }}
[data-testid="stMainBlockContainer"] {{
  max-width: 430px; margin: 24px auto 24px; padding: 1.4rem 1.2rem 2rem;
  background: linear-gradient(180deg, #1a1f47 0%, {INK} 100%);
  border: 1px solid rgba(136,105,183,0.4); border-radius: 38px;
  box-shadow: 0 30px 80px rgba(0,0,0,0.45); }}
@media (max-width: 520px) {{
  [data-testid="stMainBlockContainer"] {{ margin: 0; border-radius: 0; border: 0; }} }}
.bs-brand {{ display:flex; align-items:center; gap:10px; font-weight:800; font-size:1.25rem; color:{CREAM}; }}
.bs-sub {{ color:{MUTED}; font-size:1rem; margin-top:-4px; }}
.bs-h1 {{ font-size:1.7rem; font-weight:800; color:{CREAM}; line-height:1.2; margin:0.6rem 0 0.2rem; }}
.bs-card {{ background:{NAVY}; border-radius:20px; padding:16px 18px; margin:10px 0;
  border:1px solid rgba(136,105,183,0.25); }}
.bs-card h4 {{ margin:0 0 6px; font-size:1rem; font-weight:700; color:{CREAM}; }}
.bs-muted {{ color:{MUTED}; font-size:0.95rem; }}
.bs-big {{ font-size:2rem; font-weight:800; color:{CREAM}; }}
.bs-row {{ display:flex; gap:10px; }}
.bs-row > div {{ flex:1; }}
.bs-ok {{ background:rgba(127,209,168,0.12); border:1px solid rgba(127,209,168,0.4); }}
.bs-alert {{ background:linear-gradient(160deg, #4a3557 0%, #3a2f52 100%); border:1px solid {PINK}; }}
.bs-alert h4 {{ color:{PINK}; font-size:1.25rem; }}
.bs-explain {{ background:{PINK}; color:{INK}; border-radius:16px; padding:12px 14px; font-size:0.98rem; line-height:1.45; }}
.bs-pill {{ display:inline-block; padding:6px 14px; border-radius:999px; font-size:0.95rem;
  background:rgba(136,105,183,0.25); color:{CREAM}; }}
.bs-vib {{ display:inline-block; margin-top:6px; padding:9px 18px; border-radius:999px; font-size:1.08rem;
  font-weight:700; background:rgba(241,199,209,0.18); border:1px solid {PINK}; color:{CREAM}; }}
.bs-chip-ok {{ color:{GREEN}; font-weight:700; }}
.bs-chip-ev {{ color:{PINK}; font-weight:700; }}
.bs-dot {{ display:inline-block; width:9px; height:9px; border-radius:50%; background:#ff6b8a;
  margin-right:6px; animation:pulse 1.4s infinite; }}
@keyframes pulse {{ 0%,100% {{opacity:1}} 50% {{opacity:0.3}} }}
@media (prefers-reduced-motion: reduce) {{ .bs-dot {{ animation:none; }} }}
.stButton > button {{ border-radius:999px !important; font-weight:700 !important; }}
.stButton > button[kind="primary"] {{ background:{PURPLE} !important; border:0 !important; color:{CREAM} !important; }}
/* Bottom navigation (BreatheSync nav spec): one deep-blue bar, pink inactive items,
   lavender active item with a rounded indicator bar. Lavender text is lightened (#B39DE0)
   to reach 5:1 contrast; #8869B7 is kept for the indicator. */
.st-key-nav_widget {{ position: sticky; bottom: 0; z-index: 50; padding: 12px 0 4px;
  background: linear-gradient(180deg, rgba(17,21,48,0) 0%, {INK} 35%); }}
.st-key-nav_widget [role="radiogroup"] {{ flex-wrap: nowrap !important; gap: 0 !important;
  background: {NAVY}; border-radius: 22px; padding: 4px 6px;
  box-shadow: 0 10px 30px rgba(0,0,0,0.35); border: 1px solid rgba(136,105,183,0.25); }}
.st-key-nav_widget button {{ position: relative; flex: 1 1 0 !important; min-width: 0 !important;
  min-height: 62px !important; padding: 6px 2px 10px !important; border: 0 !important;
  border-radius: 16px !important; background: transparent !important; color: {PINK} !important;
  box-shadow: none !important; }}
.st-key-nav_widget button span[data-has-shortcut] {{ display: flex !important; flex-direction: column !important;
  align-items: center !important; gap: 3px !important; }}
.st-key-nav_widget button p {{ font-size: 0.8rem !important; font-weight: 600 !important;
  line-height: 1.1 !important; color: inherit !important; }}
.st-key-nav_widget button span[data-testid="stIconMaterial"] {{ font-size: 1.6rem !important; color: inherit !important; }}
.st-key-nav_widget button[aria-checked="true"] {{ color: #B39DE0 !important; background: transparent !important; }}
.st-key-nav_widget button[aria-checked="true"] p {{ font-weight: 800 !important; }}
.st-key-nav_widget button[aria-checked="true"]::after {{ content: ""; position: absolute; bottom: 3px;
  left: 28%; right: 28%; height: 4px; border-radius: 4px; background: {PURPLE}; }}
.st-key-nav_widget button:hover {{ background: rgba(136,105,183,0.15) !important; }}
.st-key-nav_widget button:focus-visible {{ outline: 3px solid {CREAM} !important; outline-offset: -3px !important; }}
.bs-disclaimer {{ color:{MUTED}; font-size:0.85rem; text-align:center; margin-top:14px; }}
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------ data
@st.cache_data
def load():
    pred = pd.read_csv(RESULTS_DIR / "predictions.csv")
    with open(RESULTS_DIR / "metrics.json") as f:
        metrics = json.load(f)
    if FEATURES_CSV.exists():  # extra HRV detail when available locally
        feats = pd.read_csv(FEATURES_CSV, usecols=["record", "minute", "rmssd", "frac_valid"])
        pred = pred.merge(feats, on=["record", "minute"], how="left")
    return pred, metrics


@st.cache_data
def load_ecg(record, minute, seconds=10):
    import wfdb
    from features import detect_rpeaks
    path = RAW_DIR / record
    if not path.with_suffix(".hea").exists():
        return None
    sig_len = wfdb.rdheader(str(path)).sig_len
    a = min(minute * MIN_SAMPLES + 25 * FS, sig_len - seconds * FS)
    b = a + seconds * FS
    sig = wfdb.rdrecord(str(path), sampfrom=a, sampto=b).p_signal[:, 0]
    peaks, _ = detect_rpeaks(sig)
    return sig, peaks


def ring(pct, label, color=PINK, size=150):
    r, c = 52, 2 * np.pi * 52
    return f"""<div style="text-align:center;margin:8px 0">
    <svg width="{size}" height="{size}" viewBox="0 0 130 130" role="img" aria-label="{label} {pct:.0f}%">
      <circle cx="65" cy="65" r="{r}" fill="none" stroke="rgba(255,243,229,0.12)" stroke-width="10"/>
      <circle cx="65" cy="65" r="{r}" fill="none" stroke="{color}" stroke-width="10" stroke-linecap="round"
        stroke-dasharray="{c * pct / 100:.1f} {c:.1f}" transform="rotate(-90 65 65)"/>
      <text x="65" y="72" text-anchor="middle" fill="{CREAM}" font-size="26" font-weight="800">{pct:.0f}%</text>
    </svg><div class="bs-muted">{label}</div></div>"""


def dark_layout(fig, h):
    fig.update_layout(height=h, margin=dict(l=8, r=8, t=28, b=8), paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", font=dict(color=CREAM, family="Nunito, Segoe UI, sans-serif"),
                      showlegend=False, hovermode="x unified")
    fig.update_xaxes(gridcolor="rgba(255,243,229,0.07)", zeroline=False)
    fig.update_yaxes(gridcolor="rgba(255,243,229,0.07)", zeroline=False)
    return fig


if not (RESULTS_DIR / "predictions.csv").exists():
    st.error("No results yet. Run `python src/build_dataset.py` then `python src/train.py`.")
    st.stop()

pred, metrics = load()
records = sorted(pred.record.unique())
night_name = {r: f"Night {i + 1}" for i, r in enumerate(records)}
FC_THR = float(metrics["forecaster"]["threshold"])
SENS = {"Fewer alerts": 0.5, "Balanced": 0.3, "Maximum warning": FC_THR}

ss = st.session_state
if "screen" not in ss:
    # start on the night with the most apnea onsets: the most interesting demo
    on = pred.assign(nx=pred.groupby("record").label.shift(-1))
    on = on[(on.label == 0) & (on.nx == 1)].groupby("record").size()
    ss.update(screen="Home", record=on.idxmax() if len(on) else records[0], minute=0,
              playing=False, sensitivity="Balanced", vibration=True)

g = pred[pred.record == ss.record].reset_index(drop=True)
if ss.sensitivity not in SENS:
    ss.sensitivity = "Balanced"
thr = SENS[ss.sensitivity]
g["alert"] = g.p_forecast_next >= thr
g["hr"] = 60 / g.mean_rr
ss.minute = int(min(ss.minute, len(g) - 1))


def bound(widget, label, name, **kw):
    """Render a widget whose value lives in session_state[name] and survives screen changes."""
    wkey = f"w_{name}_{ss.screen}"
    ss[wkey] = ss[name]
    return widget(label, key=wkey, on_change=lambda: ss.update({name: ss[wkey]}), **kw)


def go_to(screen):
    ss.screen = screen
    ss.playing = False


def night_summary(d):
    det = to_events(d.pred_detect.values)
    return {"hours": len(d) / 60, "hr": np.nanmean(60 / d.mean_rr), "episodes": len(det),
            "apnea_min": int(d.pred_detect.sum())}


# ------------------------------------------------------------------ header
st.markdown(f"""<div class="bs-brand"><svg width="26" height="26" viewBox="0 0 24 24" aria-hidden="true">
<path d="M20 14.5A8.5 8.5 0 1 1 9.5 4a7 7 0 0 0 10.5 10.5z" fill="{PURPLE}"/></svg>BreatheSync</div>""",
            unsafe_allow_html=True)

# ================================================================== HOME
if ss.screen == "Home":
    s = night_summary(g)
    st.markdown(f'<div class="bs-h1">Good evening 🌙</div><div class="bs-sub">Your sleep matters.</div>',
                unsafe_allow_html=True)
    bound(st.selectbox, "Recorded night", "record", options=records,
          format_func=lambda r: f"{night_name[r]} ({r})")
    st.markdown(f"""<div class="bs-card"><h4>Night analysis</h4>
      <div class="bs-row"><div><div class="bs-muted">Avg. heart rate</div><div class="bs-big">{s['hr']:.0f}
      <span class="bs-muted">BPM</span></div></div><div><div class="bs-muted">Monitored</div>
      <div class="bs-big">{s['hours']:.1f}<span class="bs-muted"> h</span></div></div></div></div>""",
                unsafe_allow_html=True)
    if s["episodes"] == 0:
        st.markdown(f"""<div class="bs-card bs-ok"><h4><span class="bs-chip-ok">✓</span> No apnea patterns
        detected</h4><div class="bs-muted">Your breathing looked steady all night. Keep up the good
        sleep habits.</div></div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""<div class="bs-card bs-alert"><h4>⚠ {s['episodes']} possible apnea episodes</h4>
        <div class="bs-muted" style="color:{CREAM}">About {s['apnea_min']} minutes showed heart-rhythm
        patterns linked to disturbed breathing. If this happens often, consider talking to a doctor.</div>
        </div>""", unsafe_allow_html=True)
    fig = go.Figure(go.Scatter(x=g.minute / 60, y=g.p_detect, fill="tozeroy", line=dict(color=PURPLE, width=1.5),
                               fillcolor="rgba(136,105,183,0.35)", name="Apnea likelihood"))
    fig.update_yaxes(range=[0, 1], tickformat=".0%", title_text="Apnea likelihood")
    fig.update_xaxes(title_text="Hours into the night")
    st.plotly_chart(dark_layout(fig, 190), width="stretch", config={"displayModeBar": False})
    st.button("▶  Start monitoring", type="primary", width="stretch", on_click=go_to, args=("Live",))
    st.button("View detailed analysis", width="stretch", on_click=go_to, args=("Analysis",))

# ================================================================== LIVE
elif ss.screen == "Live":
    row = g.iloc[ss.minute]
    clock = f"{int(row.minute // 60)}h {int(row.minute % 60):02d}m"
    st.markdown(f'<div class="bs-h1">Live monitoring</div>'
                f'<span class="bs-pill"><span class="bs-dot"></span>Replaying {night_name[ss.record]} '
                f'· {clock}</span>', unsafe_allow_html=True)
    hrv = f"{row.rmssd * 1000:.0f} ms" if "rmssd" in g and pd.notna(row.get("rmssd")) else "–"
    q = row.get("frac_valid", np.nan)
    quality = "–" if pd.isna(q) else ("Good" if q >= 0.9 else "Fair" if q >= 0.7 else "Poor")
    st.markdown(f"""<div class="bs-card" style="text-align:center"><div class="bs-muted">Heart rate</div>
      <div class="bs-big" style="font-size:2.8rem">{row.hr:.0f} <span class="bs-muted">BPM</span></div>
      <div class="bs-row" style="margin-top:8px"><div><div class="bs-muted">HRV (RMSSD)</div><b>{hrv}</b></div>
      <div><div class="bs-muted">Signal quality</div><b>{quality}</b></div>
      <div><div class="bs-muted">Right now</div><b>{"Apnea pattern" if row.pred_detect else "Steady"}</b></div>
      </div></div>""", unsafe_allow_html=True)

    risk = row.p_forecast_next * 100
    if row.alert and row.pred_detect:
        st.markdown(f"""<div class="bs-card bs-alert" style="text-align:center"><h4>Apnea pattern ongoing</h4>
          <div style="color:{CREAM}">Heart-rhythm changes linked to disturbed breathing are happening now.</div>
          {ring(row.p_detect * 100, "Apnea likelihood")}</div>""", unsafe_allow_html=True)
    elif row.alert:
        vib = "📳 Gentle vibration sent" if ss.vibration else "Silent alert (vibration off)"
        st.markdown(f"""<div class="bs-card bs-alert" style="text-align:center"><h4>⚠ Possible apnea event</h4>
          <div style="color:{CREAM}">Predicted within the next minute</div>{ring(risk, "Apnea risk")}
          <div class="bs-vib">{vib}</div></div>
          <div class="bs-explain"><b>What this means</b><br>Your heart-rhythm pattern over the last few
          minutes looks like the build-up before an apnea episode. A gentle vibration can prompt a change
          of position before breathing is disrupted.</div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""<div class="bs-card bs-ok" style="text-align:center"><h4><span class="bs-chip-ok">✓</span>
          Breathing pattern looks normal</h4>{ring(risk, "Apnea risk, next minute", GREEN, 120)}</div>""",
                    unsafe_allow_html=True)

    w = g[(g.minute > ss.minute - 30) & (g.minute <= ss.minute)]
    fig = go.Figure(go.Scatter(x=w.minute, y=w.hr, line=dict(color=PINK, width=2), name="Heart rate"))
    a = w[w.alert]
    fig.add_trace(go.Scatter(x=a.minute, y=a.hr, mode="markers", marker=dict(color=PURPLE, size=9,
                  symbol="triangle-up"), name="Alert"))
    fig.update_layout(title=dict(text="Last 30 minutes", font=dict(size=13)))
    fig.update_xaxes(title_text="Minute of the night")
    st.plotly_chart(dark_layout(fig, 170), width="stretch", config={"displayModeBar": False})

    c1, c2, c3 = st.columns(3)
    if c1.button("⏸ Pause" if ss.playing else "▶ Play", type="primary", width="stretch"):
        ss.playing = not ss.playing
        st.rerun()
    if c2.button("⏭ Next alert", width="stretch"):
        # jump to the start of the next early warning (alert while breathing still looks normal)
        start = g.alert & ~g.alert.shift(1, fill_value=False) & (g.pred_detect == 0)
        nxt = g.index[(g.minute > ss.minute) & start]
        if not len(nxt):
            nxt = g.index[(g.minute > ss.minute) & g.alert]
        ss.minute = int(nxt[0]) if len(nxt) else ss.minute
        ss.playing = False
        st.rerun()
    if c3.button("⏹ Stop", width="stretch"):
        ss.playing, ss.minute = False, 0
        st.rerun()
    bound(st.slider, "Move through the night (minutes)", "minute", min_value=0, max_value=len(g) - 1)

# ================================================================== ANALYSIS
elif ss.screen == "Analysis":
    st.markdown(f'<div class="bs-h1">Detailed analysis</div><div class="bs-sub">{night_name[ss.record]} '
                f'({ss.record})</div>', unsafe_allow_html=True)
    view = st.segmented_control("View", ["Night timeline", "ECG", "HRV"], default="Night timeline",
                                label_visibility="collapsed")
    if view == "ECG":
        ecg = load_ecg(ss.record, ss.minute)
        if ecg is None:
            st.info("Raw ECG files are not available in this deployment.")
        else:
            sig, peaks = ecg
            t = np.arange(len(sig)) / FS
            fig = go.Figure(go.Scatter(x=t, y=sig, line=dict(color=PINK, width=1.4)))
            fig.add_trace(go.Scatter(x=t[peaks], y=sig[peaks], mode="markers",
                                     marker=dict(color=PURPLE, size=7), name="Detected heartbeat"))
            fig.update_layout(title=dict(text=f"ECG signal, 10 s at minute {ss.minute}", font=dict(size=13)))
            fig.update_xaxes(title_text="Seconds"); fig.update_yaxes(title_text="mV")
            st.plotly_chart(dark_layout(fig, 240), width="stretch", config={"displayModeBar": False})
            st.caption("Purple dots mark heartbeats (R-peaks) found by the signal-processing pipeline.")
        bound(st.slider, "Minute of the night", "minute", min_value=0, max_value=len(g) - 1)
    elif view == "HRV":
        y = g.rmssd * 1000 if "rmssd" in g else g.hr
        fig = go.Figure(go.Scatter(x=g.minute / 60, y=y, line=dict(color=PURPLE, width=1.5)))
        fig.update_layout(title=dict(text="HRV (RMSSD, ms)" if "rmssd" in g else "Heart rate (BPM)",
                                     font=dict(size=13)))
        fig.update_xaxes(title_text="Hours into the night")
        st.plotly_chart(dark_layout(fig, 240), width="stretch", config={"displayModeBar": False})
    else:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.12,
                            subplot_titles=("Apnea detected this minute", "Early-warning risk (next minute)"))
        for s0, e0 in to_events(g.label.values):
            fig.add_vrect(x0=s0 / 60, x1=e0 / 60, fillcolor="rgba(241,199,209,0.18)", line_width=0, layer="below")
        fig.add_trace(go.Scatter(x=g.minute / 60, y=g.p_detect, line=dict(color=CREAM, width=1),
                                 name="Detection probability"), 1, 1)
        fig.add_trace(go.Scatter(x=g.minute / 60, y=g.pred_detect, fill="tozeroy", line=dict(width=0),
                                 fillcolor="rgba(136,105,183,0.45)", name="Detected apnea"), 1, 1)
        fig.add_trace(go.Scatter(x=g.minute / 60, y=g.p_forecast_next, line=dict(color=PINK, width=1.2),
                                 name="Risk"), 2, 1)
        fig.add_hline(y=thr, line=dict(color=PURPLE, dash="dash", width=1), row=2, col=1)
        al = g[g.alert]
        fig.add_trace(go.Scatter(x=al.minute / 60, y=al.p_forecast_next, mode="markers",
                                 marker=dict(color=PURPLE, size=6, symbol="triangle-up"), name="Alert"), 2, 1)
        fig.update_yaxes(range=[0, 1.02])
        fig.update_xaxes(title_text="Hours into the night", row=2, col=1)
        fig.update_annotations(font_size=12)
        st.plotly_chart(dark_layout(fig, 380), width="stretch", config={"displayModeBar": False})
        st.caption("Purple area: apnea detected by BreatheSync. Pink bands: expert annotation, shown for "
                   "evaluation. Triangles: vibration alerts at the current sensitivity.")
    row = g.iloc[ss.minute]
    st.markdown(f"""<div class="bs-card"><h4>Key indicators at minute {ss.minute}</h4>
      <div class="bs-row"><div><div class="bs-muted">Heart rate</div><b>{row.hr:.0f} BPM</b></div>
      <div><div class="bs-muted">Apnea likelihood</div><b>{row.p_detect:.0%}</b></div>
      <div><div class="bs-muted">Next-minute risk</div><b>{row.p_forecast_next:.0%}</b></div></div></div>""",
                unsafe_allow_html=True)

# ================================================================== HISTORY
elif ss.screen == "History":
    st.markdown('<div class="bs-h1">History</div><div class="bs-sub">Recorded nights from held-out test '
                'patients</div>', unsafe_allow_html=True)

    def open_night(r):
        ss.record, ss.minute, ss.screen, ss.playing = r, 0, "Home", False

    for r in records:
        s = night_summary(pred[pred.record == r])
        chip = (f'<span class="bs-chip-ok">✓ Normal</span>' if s["episodes"] == 0 else
                f'<span class="bs-chip-ev">⚠ {s["episodes"]} possible events</span>')
        st.markdown(f"""<div class="bs-card"><div class="bs-row"><div><h4>{night_name[r]}</h4>
          <div class="bs-muted">{s['hours']:.1f} h · avg {s['hr']:.0f} BPM · record {r}</div></div>
          <div style="text-align:right;flex:0 0 auto">{chip}</div></div></div>""", unsafe_allow_html=True)
        st.button(f"Open {night_name[r]}", key=f"open_{r}", width="stretch", on_click=open_night, args=(r,))

# ================================================================== INSIGHTS
elif ss.screen == "Insights":
    st.markdown(f'<div class="bs-h1">Insights</div><div class="bs-sub">How {night_name[ss.record]} compares '
                f'with your other nights</div>', unsafe_allow_html=True)

    def night_stats(d):
        s = night_summary(d)
        s["steady"] = 1 - d.pred_detect.mean()
        s["hrv"] = d.rmssd.mean() * 1000 if "rmssd" in d else np.nan
        return s

    allstats = {r: night_stats(pred[pred.record == r]) for r in records}
    cur = allstats[ss.record]
    others = [allstats[r] for r in records if r != ss.record] or [cur]

    # Sleep quality: share of the night with steady breathing patterns
    quality = "Good" if cur["steady"] >= 0.9 else "Fair" if cur["steady"] >= 0.7 else "Needs attention"
    st.markdown(f"""<div class="bs-card"><h4>Breathing quality</h4><div class="bs-row" style="align-items:center">
      <div style="flex:0 0 130px">{ring(cur['steady'] * 100, quality, GREEN if quality == "Good" else PINK, 120)}</div>
      <div><div class="bs-big">{cur['hours']:.1f} h</div><div class="bs-muted">monitored, with steady breathing
      patterns for {cur['steady']:.0%} of the night</div></div></div></div>""", unsafe_allow_html=True)

    # Nightly overview
    per_night = [allstats[r]["apnea_min"] for r in records]
    fig = go.Figure(go.Bar(x=[night_name[r] for r in records], y=per_night,
                           marker_color=[PINK if r == ss.record else PURPLE for r in records]))
    fig.update_layout(title=dict(text="Minutes with apnea patterns, per night", font=dict(size=14)))
    st.plotly_chart(dark_layout(fig, 210), width="stretch", config={"displayModeBar": False})

    # Key trends vs the average of the other nights (lower apnea = better)
    def trend(name, now, ref, unit, lower_is_better):
        if pd.isna(now) or pd.isna(ref):
            return ""
        diff = now - ref
        if abs(diff) < 0.5:
            return (f'<div class="bs-row" style="padding:8px 0;border-bottom:1px solid rgba(255,243,229,0.08)">'
                    f'<div>{name}</div><div style="text-align:right"><b>{now:.0f} {unit}</b> '
                    f'<span style="color:{MUTED};font-weight:800">→ same</span></div></div>')
        good = (diff < 0) == lower_is_better or abs(diff) < 1e-9
        arrow = "↓" if diff < 0 else "↑" if diff > 0 else "→"
        color = GREEN if good else PINK
        return (f'<div class="bs-row" style="padding:8px 0;border-bottom:1px solid rgba(255,243,229,0.08)">'
                f'<div>{name}</div><div style="text-align:right"><b>{now:.0f} {unit}</b> '
                f'<span style="color:{color};font-weight:800">{arrow} {abs(diff):.0f}</span></div></div>')

    avg = lambda k: float(np.nanmean([o[k] for o in others]))
    rows = (trend("Apnea episodes", cur["episodes"], avg("episodes"), "", True)
            + trend("Minutes with apnea patterns", cur["apnea_min"], avg("apnea_min"), "min", True)
            + trend("Average heart rate", cur["hr"], avg("hr"), "BPM", True)
            + trend("Heart-rate variability", cur["hrv"], avg("hrv"), "ms", False))
    st.markdown(f"""<div class="bs-card"><h4>Key trends</h4><div class="bs-muted" style="margin-bottom:4px">
      Arrows compare this night with the average of your other nights.</div>{rows}</div>""",
                unsafe_allow_html=True)

    if cur["episodes"] > 0:
        st.markdown("""<div class="bs-explain"><b>What you can do</b><br>Sleeping on your side, avoiding alcohol
          close to bedtime and keeping a regular sleep schedule can reduce breathing interruptions. If apnea
          patterns show up on most nights, share these insights with a doctor.</div>""", unsafe_allow_html=True)

    fc, rb = metrics["forecaster"], metrics["robustness"]
    calm = rb.get("controls_specificity_mean")
    calm_txt = f" and stayed calm for about {calm:.0%} of the night for healthy sleepers" if calm else ""
    st.markdown(f"""<div class="bs-card"><h4>How reliable is BreatheSync?</h4><div class="bs-muted">In testing
      on people it had never seen, it gave a one-minute warning before about {fc['warned_onset_rate']:.0%} of
      apnea episodes{calm_txt}. It can still miss events or alert when nothing is wrong, so it does not replace
      a medical sleep test.</div></div>""", unsafe_allow_html=True)

# ================================================================== SETTINGS
elif ss.screen == "Settings":
    st.markdown('<div class="bs-h1">Settings</div>', unsafe_allow_html=True)
    st.markdown('<div class="bs-card"><h4>Alert sensitivity</h4><div class="bs-muted">More alerts catch more '
                'apnea onsets early, but wake you more often for nothing.</div></div>', unsafe_allow_html=True)
    bound(st.segmented_control, "Alert sensitivity", "sensitivity", options=list(SENS),
          label_visibility="collapsed")
    t = SENS[ss.sensitivity]
    a = pred.assign(nx=pred.groupby("record").label.shift(-1)).dropna(subset=["nx"])
    al = a.p_forecast_next >= t
    onset, quiet = (a.label == 0) & (a.nx == 1), (a.label == 0) & (a.nx == 0)
    st.markdown(f"""<div class="bs-card"><div class="bs-row">
      <div><div class="bs-muted">Onsets warned early</div><div class="bs-big">{al[onset].mean():.0%}</div></div>
      <div><div class="bs-muted">False alerts / hour</div><div class="bs-big">{al[quiet].sum() / (len(a) / 60):.1f}</div></div>
      </div><div class="bs-muted">Measured on the held-out test nights.</div></div>""", unsafe_allow_html=True)
    bound(st.toggle, "Vibration alerts", "vibration")
    st.markdown("""<div class="bs-card"><h4>About BreatheSync</h4><div class="bs-muted">Research prototype
      built for the AI Innovation Hackathon 2026. Models were trained on the PhysioNet Apnea-ECG Database and
      replay recorded nights from patients the models never saw. Not a medical device and not intended for
      diagnosis.</div></div>""", unsafe_allow_html=True)

# ------------------------------------------------------------------ bottom navigation
st.markdown('<div style="height:6px"></div>', unsafe_allow_html=True)
NAV = {"Home": ":material/home:", "History": ":material/history:", "Insights": ":material/bar_chart:",
       "Settings": ":material/settings:"}
if ss.screen == "Live":  # the Live tab only appears while monitoring (nav spec, special case)
    NAV["Live"] = ":material/vital_signs:"


def nav_changed():
    if ss.nav_widget:
        go_to(ss.nav_widget)


ss.nav_widget = ss.screen if ss.screen in NAV else None
st.segmented_control("Navigation", list(NAV), format_func=lambda s: f"{NAV[s]} {s}", key="nav_widget",
                     on_change=nav_changed, label_visibility="collapsed", width="stretch")
st.markdown('<div class="bs-disclaimer">Research prototype · not a medical device</div>', unsafe_allow_html=True)

# ------------------------------------------------------------------ playback loop (runs last)
if ss.screen == "Live" and ss.playing:
    time.sleep(0.7)
    if ss.minute < len(g) - 1:
        ss.minute += 1
    else:
        ss.playing = False
    st.rerun()
