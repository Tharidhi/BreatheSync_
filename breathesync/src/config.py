"""Shared paths and settings for BreatheSync."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
FEATURES_CSV = ROOT / "data" / "features.csv"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"

FS = 100                 # Apnea-ECG sampling rate (Hz)
MIN_SAMPLES = 60 * FS    # samples per one-minute label

# The 35 labelled "learning set" recordings of the PhysioNet Apnea-ECG Database
LEARNING_RECORDS = (
    [f"a{i:02d}" for i in range(1, 21)]
    + [f"b{i:02d}" for i in range(1, 6)]
    + [f"c{i:02d}" for i in range(1, 11)]
)

# Held-out test PATIENTS (never seen in training). Mix of a/b/c groups.
TEST_RECORDS = ["a02", "a07", "a12", "a17", "b02", "c03", "c08"]

# Context windows (in minutes) for the two tasks
DETECT_OFFSETS = [-2, -1, 0, 1, 2]      # detection may look slightly ahead (offline timeline)
FORECAST_OFFSETS = [-4, -3, -2, -1, 0]  # forecasting uses the PAST only -> no leakage

UNCERTAIN_BAND = (0.35, 0.65)  # probabilities inside this band are flagged "uncertain"
RANDOM_STATE = 42
