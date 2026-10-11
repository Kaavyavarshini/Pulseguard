"""
PulseGuard - Real Dataset Loader (PPG-DaLiA)
------------------------------------------------------------
Feeds REAL heart-rate data (not simulated) from the public PPG-DaLiA
dataset through the same PulseGuardEngine used in simulate_and_compare.py,
so you can show the panel a result validated on real physiological data,
not just synthetic data.

DOWNLOAD THE DATA FIRST (this cannot be done from this environment):
  1. Go to: https://archive.ics.uci.edu/dataset/495/ppg+dalia
  2. Click "Download" -- you'll get a zip containing one .pkl file
     per subject (S1.pkl, S2.pkl, ... S15.pkl), ~15 subjects total.
  3. Unzip it, and place ONE subject's .pkl file (e.g. S2.pkl) in the
     same folder as this script.
     (Pick any subject -- S2, S3, etc. S1 is fine too.)

WHAT'S IN THE FILE:
  Each .pkl is a Python dict with (among other things):
    data['label']  -> ground-truth heart rate (bpm), one value every
                       2 seconds, derived from the chest ECG sensor
                       (this is what we treat as "sensor readings")
  This is exactly what our engine needs -- a HR value over time -- so
  no PPG signal processing is required to validate the anomaly-
  detection logic itself.

Run with:
    pip install numpy --break-system-packages   (if not already installed)
    python3 load_real_dataset.py S2.pkl
"""

import pickle
import sys
from datetime import datetime, timedelta

from baseline_engine import PulseGuardEngine, Reading, Severity
from alert_manager import AlertManager, ConsoleNotificationChannel


def load_ppg_dalia_hr(pkl_path: str):
    """
    Loads ground-truth HR values from a PPG-DaLiA subject .pkl file.
    Returns a list of Reading objects, one every 2 seconds (the
    dataset's native label resolution).
    """
    with open(pkl_path, "rb") as f:
        # PPG-DaLiA pickles were saved under Python 2/3; latin1
        # encoding avoids unpickling errors across versions.
        data = pickle.load(f, encoding="latin1")

    hr_values = data["label"]  # ground-truth HR in bpm, one per 2s window

    readings = []
    t = datetime(2026, 8, 30, 8, 0, 0)
    for hr in hr_values:
        readings.append(Reading(timestamp=t, heart_rate=float(hr)))
        t += timedelta(seconds=2)

    return readings


def load_ppg_dalia_hr_csv(csv_path: str):
    """
    Loads the same HR series from a CSV export instead of the raw
    .pkl. Expects a header row and a 'heart_rate_bpm' column, with one
    row per 2-second window (the dataset's native label resolution) --
    i.e. exactly what load_ppg_dalia_hr() returns, just already
    extracted from the pickle.

    This exists so the validation can be reproduced without the
    multi-GB PPG-DaLiA download, and without needing the pickle to
    unpickle correctly across Python versions.
    """
    import csv as _csv

    readings = []
    t = datetime(2026, 8, 30, 8, 0, 0)
    with open(csv_path, newline="") as f:
        reader = _csv.DictReader(f)
        for row in reader:
            hr = row.get("heart_rate_bpm")
            if hr is None or hr == "":
                continue
            readings.append(Reading(timestamp=t, heart_rate=float(hr)))
            t += timedelta(seconds=2)
    return readings


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 load_real_dataset.py <path_to_subject_pkl>")
        print("Example: python3 load_real_dataset.py S2.pkl")
        sys.exit(1)

    pkl_path = sys.argv[1]
    print(f"Loading real HR data from: {pkl_path}")
    readings = load_ppg_dalia_hr(pkl_path)
    print(f"Loaded {len(readings)} real HR readings "
          f"({len(readings) * 2 / 60:.1f} minutes of recording)\n")

    engine = PulseGuardEngine(window_size=500, min_readings=50)
    alert_mgr = AlertManager(
        channels=[ConsoleNotificationChannel()],
        caregiver_contact="+91-XXXXXXXXXX (registered caregiver)",
    )

    events = []
    for r in readings:
        event = engine.process_reading(r)
        if event:
            events.append(event)
            alert_mgr.handle_event(event)

    by_severity = {}
    for e in events:
        by_severity[e.severity] = by_severity.get(e.severity, 0) + 1

    print("\n" + "=" * 70)
    print("RESULTS ON REAL PPG-DaLiA DATA")
    print("=" * 70)
    print(f"Total flagged events : {len(events)}")
    for sev in [Severity.MILD, Severity.MODERATE, Severity.EMERGENCY]:
        print(f"  {sev.value:10s}: {by_severity.get(sev, 0)}")
    print(f"Learned baseline (mean): {engine.baseline.mean:.1f} bpm")
    print(f"Learned baseline (SD)  : {engine.baseline.std_dev:.1f} bpm")
    print("=" * 70)
    print("\nNote: PPG-DaLiA subjects are healthy young/middle-aged adults")
    print("performing normal daily activities (walking, cycling, etc.),")
    print("not elderly cardiac patients -- so flagged events here mostly")
    print("reflect real activity-driven HR changes, not medical emergencies.")
    print("This still validates that the baseline-learning and severity-")
    print("tiering logic runs correctly on real, noisy physiological data")
    print("(not just clean synthetic data) -- worth stating exactly this")
    print("way to the panel so the claim is accurate, not overstated.")


if __name__ == "__main__":
    main()
