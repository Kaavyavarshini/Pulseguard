"""
PulseGuard - Real Dataset Loader (MIT-BIH, streamed - very small download)
----------------------------------------------------------------------------
Unlike PPG-DaLiA (which requires downloading a large multi-GB zip), this
script STREAMS just ONE recording directly from PhysioNet using the
`wfdb` Python library -- no manual download, no big zip file. A single
MIT-BIH record is only a few hundred KB to ~1-2 MB.

MIT-BIH Arrhythmia Database is the same dataset used by several papers
in the PulseGuard literature review, so using it here also lets you
say "validated against the same benchmark dataset cited in my own
literature review" to the panel.

SETUP (run once):
    pip install wfdb --break-system-packages
    (drop --break-system-packages if not needed on your machine)

RUN:
    python3 load_mitbih_stream.py
    (optionally: python3 load_mitbih_stream.py 101   -- to try a
     different record; valid IDs include 100-124, 200-234)

WHAT THIS DOES:
  1. Streams one ECG record + its beat annotations directly from
     PhysioNet's servers (small download, no manual file handling).
  2. Converts beat-to-beat R-peak timings into an instantaneous heart
     rate series (bpm) -- this is standard practice: HR = 60 / RR-interval.
  3. Feeds that real HR series through the exact same PulseGuardEngine
     used in simulate_and_compare.py.
"""

import sys
from datetime import datetime, timedelta

import wfdb

from baseline_engine import PulseGuardEngine, Reading, Severity
from alert_manager import AlertManager, ConsoleNotificationChannel
from evaluate import (
    compute_confusion,
    personalized_predictions,
    fixed_threshold_predictions,
)

# AAMI beat-class grouping: these symbols represent a "normal" beat
# morphology; everything else that IS a valid beat annotation (not a
# rhythm/quality marker) is treated as an abnormal beat for ground truth.
NORMAL_BEAT_SYMBOLS = {"N", "L", "R", "e", "j"}
ABNORMAL_BEAT_SYMBOLS = {"A", "a", "J", "S", "V", "E", "F", "/", "f", "Q"}
VALID_BEAT_SYMBOLS = NORMAL_BEAT_SYMBOLS | ABNORMAL_BEAT_SYMBOLS


def stream_mitbih_hr(record_id: str = "100"):
    """
    Streams one MIT-BIH record and its beat annotations from PhysioNet,
    and converts R-R intervals into an instantaneous heart-rate series.
    Requires an active internet connection (works on your machine even
    though it does not work inside this sandbox).

    Also returns ground_truth: for each HR reading, whether the beat
    ENDING that interval was annotated as abnormal (True) or normal
    (False). Because each HR value is derived from exactly one beat
    interval, this attaches ground truth exactly -- no time-window
    matching needed. Non-beat annotation markers (rhythm-change,
    signal-quality flags, etc.) are filtered out before computing
    intervals, so a skipped marker does not corrupt the RR interval
    on either side of it.
    """
    print(f"Streaming MIT-BIH record '{record_id}' from PhysioNet "
          f"(small download, ~1-2 MB)...")

    record = wfdb.rdrecord(record_id, pn_dir="mitdb")
    annotation = wfdb.rdann(record_id, "atr", pn_dir="mitdb")

    fs = record.fs  # sampling frequency, typically 360 Hz for MIT-BIH

    # Keep only genuine beat annotations (drop rhythm/quality markers
    # like '+', '~', etc.) so RR intervals are computed beat-to-beat.
    beats = [
        (sample, symbol)
        for sample, symbol in zip(annotation.sample, annotation.symbol)
        if symbol in VALID_BEAT_SYMBOLS
    ]

    readings = []
    ground_truth = []
    t = datetime(2026, 8, 30, 8, 0, 0)
    for i in range(1, len(beats)):
        prev_sample, _ = beats[i - 1]
        curr_sample, curr_symbol = beats[i]
        rr_seconds = (curr_sample - prev_sample) / fs
        if rr_seconds <= 0:
            continue
        hr_bpm = 60.0 / rr_seconds
        # sanity bound -- discard clearly invalid artifacts
        if 20 <= hr_bpm <= 250:
            readings.append(Reading(timestamp=t, heart_rate=round(hr_bpm, 1)))
            ground_truth.append(curr_symbol in ABNORMAL_BEAT_SYMBOLS)
            t += timedelta(seconds=rr_seconds)

    return readings, ground_truth


def main():
    record_id = sys.argv[1] if len(sys.argv) > 1 else "100"

    try:
        readings, ground_truth = stream_mitbih_hr(record_id)
    except Exception as exc:
        print(f"\nCould not stream from PhysioNet: {exc}")
        print("Check your internet connection and try again.")
        sys.exit(1)

    print(f"Loaded {len(readings)} real beat-to-beat HR readings "
          f"from record {record_id}\n")

    engine = PulseGuardEngine(window_size=200, min_readings=50)
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
    print(f"RESULTS ON REAL MIT-BIH DATA (record {record_id})")
    print("=" * 70)
    print(f"Total flagged events : {len(events)}")
    for sev in [Severity.MILD, Severity.MODERATE, Severity.EMERGENCY]:
        print(f"  {sev.value:10s}: {by_severity.get(sev, 0)}")
    print(f"Learned baseline (mean): {engine.baseline.mean:.1f} bpm")
    print(f"Learned baseline (SD)  : {engine.baseline.std_dev:.1f} bpm")
    print("=" * 70)
    print("\nNote: MIT-BIH subjects include real cardiac patients with")
    print("annotated arrhythmias, so flagged events here may correspond")
    print("to genuine clinically-relevant rhythm changes -- a stronger")
    print("validation than healthy-subject activity data, though this")
    print("is still ECG-derived HR, not the PPG signal your MAX30102")
    print("hardware will capture -- worth stating that distinction to")
    print("the panel rather than implying it's a 1:1 hardware match.")

    # --- Precision / Recall / F1 / FPR against real MIT-BIH annotations ---
    pg_predictions = personalized_predictions(readings)
    ft_predictions = fixed_threshold_predictions(readings)

    pg_metrics = compute_confusion(pg_predictions, ground_truth)
    ft_metrics = compute_confusion(ft_predictions, ground_truth)

    n_abnormal = sum(ground_truth)
    print("\n" + "=" * 70)
    print(f"EVALUATION METRICS (per-beat, vs. real MIT-BIH annotations)")
    print(f"Ground truth: {n_abnormal} / {len(ground_truth)} beats "
          f"annotated abnormal ({100 * n_abnormal / len(ground_truth):.1f}%)")
    print("=" * 70)
    print(pg_metrics.summary("PulseGuard (Personalized Baseline)"))
    print()
    print(ft_metrics.summary("Fixed-Threshold (60-100 bpm)"))
    print("=" * 70)
    print("\nImportant caveat: ground truth here is BEAT MORPHOLOGY")
    print("(is this an ectopic/abnormal-shaped beat?), while PulseGuard")
    print("detects HEART RATE deviation from personal baseline. These")
    print("are related but different signals -- a run of PVCs can occur")
    print("at a normal overall rate, and a rate deviation can happen")
    print("without any abnormal beat morphology. Low precision/recall")
    print("here does not necessarily mean the rate-based logic is wrong")
    print("-- it may be measuring a different kind of abnormality than")
    print("the ground truth captures. State this distinction plainly")
    print("if asked, rather than presenting the numbers as a direct")
    print("apples-to-apples clinical accuracy score.")


if __name__ == "__main__":
    main()
