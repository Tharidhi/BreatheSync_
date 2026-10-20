"""Signal processing: ECG -> R-peaks -> RR / HRV / ECG-derived respiration -> per-minute features.

Everything here is plain NumPy/SciPy so the team can explain every step to the judges.
"""
import numpy as np
from scipy import signal
from scipy.ndimage import median_filter

from config import FS, MIN_SAMPLES

_trapz = getattr(np, "trapezoid", None) or np.trapz  # works on NumPy 1.x and 2.x
INTERP_FS = 4.0      # resampling rate for the beat-to-beat series (Hz)
SPEC_WINDOW_S = 300  # 5-minute window (ending at the current minute) for spectral features


# ---------------------------------------------------------------- filtering & R-peaks
def bandpass(x, fs, lo, hi, order=3):
    b, a = signal.butter(order, [lo / (fs / 2), hi / (fs / 2)], btype="band")
    return signal.filtfilt(b, a, x)


def detect_rpeaks(ecg, fs=FS):
    """Pan-Tompkins-style detector: band-pass -> derivative^2 -> moving integration
    -> adaptive (per 10 s block) threshold -> refine to the true QRS peak."""
    ecg = np.nan_to_num(np.asarray(ecg, dtype=float))
    filt = bandpass(ecg, fs, 5, 15)
    energy = np.gradient(filt) ** 2
    win = int(0.12 * fs)
    mwi = np.convolve(energy, np.ones(win) / win, mode="same")

    # Adaptive threshold: 30% of the 98th percentile in each 10-second block
    block = 10 * fs
    n_blocks = int(np.ceil(len(mwi) / block))
    thr = np.empty_like(mwi)
    for b in range(n_blocks):
        seg = mwi[b * block:(b + 1) * block]
        thr[b * block:(b + 1) * block] = 0.3 * np.percentile(seg, 98)

    cand, _ = signal.find_peaks(mwi, height=thr, distance=int(0.3 * fs))

    # Refine each candidate to the largest |filtered ECG| value nearby (handles polarity)
    half = int(0.1 * fs)
    absf = np.abs(filt)
    peaks = np.array(
        [max(0, c - half) + np.argmax(absf[max(0, c - half):c + half]) for c in cand],
        dtype=int,
    )
    peaks = np.unique(peaks)
    # Enforce a 300 ms refractory period after refinement
    if len(peaks) > 1:
        keep = np.insert(np.diff(peaks) > 0.3 * fs, 0, True)
        peaks = peaks[keep]
    return peaks, filt


def clean_rr(peaks, fs=FS):
    """RR intervals (s) at each beat (assigned to the 2nd beat), with a validity mask
    that rejects physiologically implausible values and ectopic/missed beats."""
    t = peaks[1:] / fs
    rr = np.diff(peaks) / fs
    if len(rr) < 5:
        return t, rr, np.zeros(len(rr), bool)
    local_med = median_filter(rr, size=5, mode="nearest")
    valid = (rr > 0.3) & (rr < 2.0) & (np.abs(rr - local_med) < 0.3 * local_med)
    return t, rr, valid


# ---------------------------------------------------------------- spectral helpers
def _band_powers(times, values, t_end, bands):
    """Interpolate an irregular beat series onto a 4 Hz grid over the window
    [t_end - 5 min, t_end] and return power in each frequency band."""
    m = (times > t_end - SPEC_WINDOW_S) & (times <= t_end)
    if m.sum() < 60:  # need ~1 min of beats at least
        return [np.nan] * len(bands) + [np.nan]
    tt, vv = times[m], values[m]
    grid = np.arange(tt[0], tt[-1], 1 / INTERP_FS)
    if len(grid) < 128:
        return [np.nan] * len(bands) + [np.nan]
    y = signal.detrend(np.interp(grid, tt, vv))
    f, p = signal.welch(y, fs=INTERP_FS, nperseg=min(512, len(y)))
    out = [_trapz(p[(f >= lo) & (f < hi)], f[(f >= lo) & (f < hi)]) for lo, hi in bands]
    total = _trapz(p[(f >= 0.01) & (f < 0.5)], f[(f >= 0.01) & (f < 0.5)])
    return out + [total]


# ---------------------------------------------------------------- per-minute features
FEATURE_NAMES = [
    "n_beats", "frac_valid", "mean_rr", "rel_rr", "sd_rr", "rmssd", "pnn50",
    "hr_min", "hr_max", "hr_range", "amp_mean", "amp_sd",
    "rr_vlf", "rr_lf", "rr_hf", "rr_vlf_frac", "rr_lf_hf",
    "edr_vlf", "edr_resp", "edr_resp_frac",
]


def minute_features(ecg, n_minutes, fs=FS):
    """Return (features array [n_minutes x n_features], rpeaks, filtered ECG)."""
    peaks, filt = detect_rpeaks(ecg, fs)
    t_rr, rr, valid = clean_rr(peaks, fs)

    # ECG-derived respiration (EDR): breathing modulates QRS amplitude
    amp = np.abs(filt[peaks[1:]]) if len(peaks) > 1 else np.array([])
    rec_med_rr = np.median(rr[valid]) if valid.any() else np.nan
    rec_med_amp = np.median(amp[valid]) if valid.any() else np.nan

    tv, rrv, ampv = t_rr[valid], rr[valid], amp[valid] / rec_med_amp

    feats = np.full((n_minutes, len(FEATURE_NAMES)), np.nan)
    for m in range(n_minutes):
        t0, t1 = m * MIN_SAMPLES / fs, (m + 1) * MIN_SAMPLES / fs
        inm = (t_rr >= t0) & (t_rr < t1)
        vm = inm & valid
        n_beats = inm.sum()
        row = {"n_beats": n_beats, "frac_valid": vm.sum() / max(n_beats, 1)}
        if vm.sum() >= 10:
            r = rr[vm]
            hr = 60 / r
            d = np.diff(r)
            row.update(
                mean_rr=r.mean(), rel_rr=r.mean() / rec_med_rr, sd_rr=r.std(),
                rmssd=np.sqrt(np.mean(d ** 2)) if len(d) else np.nan,
                pnn50=np.mean(np.abs(d) > 0.05) if len(d) else np.nan,
                hr_min=hr.min(), hr_max=hr.max(), hr_range=hr.max() - hr.min(),
                amp_mean=(amp[vm] / rec_med_amp).mean(), amp_sd=(amp[vm] / rec_med_amp).std(),
            )
        # Spectral: VLF (0.01-0.04 Hz) captures the cyclic heart-rate variation of apnea
        vlf, lf, hf, tot = _band_powers(tv, rrv, t1, [(0.01, 0.04), (0.04, 0.15), (0.15, 0.4)])
        row.update(rr_vlf=vlf, rr_lf=lf, rr_hf=hf,
                   rr_vlf_frac=vlf / tot if tot and tot > 0 else np.nan,
                   rr_lf_hf=lf / hf if hf and hf > 0 else np.nan)
        evlf, eresp, etot = _band_powers(tv, ampv, t1, [(0.01, 0.04), (0.1, 0.5)])
        row.update(edr_vlf=evlf, edr_resp=eresp,
                   edr_resp_frac=eresp / etot if etot and etot > 0 else np.nan)
        feats[m] = [row.get(k, np.nan) for k in FEATURE_NAMES]
    return feats, peaks, filt
