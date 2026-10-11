"""
PulseGuard - Simulation & Comparison vs. Fixed-Threshold Approach
------------------------------------------------------------------
This script:

  1. Generates a realistic synthetic heart-rate stream for an
     elderly user, including a personal baseline that is DIFFERENT
     from the generic population range, plus a few injected
     anomaly episodes.

  2. Runs the anomaly through:
        (a) PulseGuard's personalized baseline + severity engine
        (b) A classic FIXED-THRESHOLD approach (e.g. flag if
            HR < 60 or > 100 bpm), representing existing systems
            from the literature review.

  3. Reports false positives / missed detections for each approach,
     giving a concrete, numeric comparison for Objective 5 and the
     "Results, Discussion & Comparison" slide.

Run with:  python3 simulate_and_compare.py
"""

import random
from datetime import datetime, timedelta

from baseline_engine import PulseGuardEngine, Reading, Severity
from alert_manager import AlertManager, ConsoleNotificationChannel
from evaluate import (
    compute_confusion,
    personalized_predictions,
    fixed_threshold_predictions,
)


# ---------------------------------------------------------------
# 1. Synthetic data generation
# ---------------------------------------------------------------

def generate_readings(
    personal_baseline_hr: float = 78.0,
    personal_sd: float = 4.0,
    n_normal_readings: int = 600,
    seconds_between_readings: int = 10,
    seed: int = 42,
):
    """
    Simulates ~100 minutes of HR data for one elderly user whose
    personal resting HR (78 bpm) is comfortably inside the generic
    "normal" clinical range (60-100 bpm) -- so a fixed-threshold
    system would never flag anything for them, even during a real
    problem, UNLESS the deviation is extreme.

    Injects 3 anomaly episodes of varying severity to test both
    systems' detection and false-alarm behavior.

    Returns (readings, ground_truth) -- ground_truth[i] is True if
    readings[i] falls inside an injected anomaly episode (known
    exactly, since we generated it), used for precision/recall/F1.
    """
    random.seed(seed)
    readings = []
    ground_truth = []
    t = datetime(2026, 8, 30, 9, 0, 0)

    def add_normal(n):
        nonlocal t
        for _ in range(n):
            hr = random.gauss(personal_baseline_hr, personal_sd)
            readings.append(Reading(timestamp=t, heart_rate=round(hr, 1)))
            ground_truth.append(False)
            t += timedelta(seconds=seconds_between_readings)

    def add_anomaly(n, offset_in_sd, jitter):
        nonlocal t
        for _ in range(n):
            hr = random.gauss(personal_baseline_hr + offset_in_sd * personal_sd, jitter)
            readings.append(Reading(timestamp=t, heart_rate=round(hr, 1)))
            ground_truth.append(True)
            t += timedelta(seconds=seconds_between_readings)

    # Phase A: normal baseline period (builds up PulseGuard's baseline)
    add_normal(n_normal_readings)

    # Phase B: Episode 1 - MILD, brief (should NOT alert caregiver)
    add_anomaly(9, 2.2, 1.5)
    add_normal(n_normal_readings // 6)

    # Phase C: Episode 2 - MODERATE, sustained (should alert)
    # Still under 100 bpm, so a fixed 60-100 threshold system misses it.
    add_anomaly(21, 3.2, 1.5)
    add_normal(n_normal_readings // 6)

    # Phase D: Episode 3 - EMERGENCY, severe and sustained
    add_anomaly(31, 5.0, 2.0)
    add_normal(n_normal_readings // 6)

    return readings, ground_truth


# ---------------------------------------------------------------
# 2. Fixed-threshold baseline (represents existing approaches)
# ---------------------------------------------------------------

def fixed_threshold_flags(readings, low=60, high=100):
    """
    Classic approach from the literature: flag any reading outside
    a generic population range. Returns list of (reading, flagged).
    """
    return [(r, (r.heart_rate < low or r.heart_rate > high)) for r in readings]


# ---------------------------------------------------------------
# 3. Run both systems and compare
# ---------------------------------------------------------------

def main():
    readings, ground_truth = generate_readings()

    print("=" * 70)
    print("PULSEGUARD DEMO - Personalized vs. Fixed-Threshold Comparison")
    print("=" * 70)
    print(f"Total readings simulated: {len(readings)}")
    print(f"Personal baseline (ground truth): 78 bpm ± 4 bpm")
    print(f"Fixed clinical threshold (existing approach): 60-100 bpm\n")

    # --- PulseGuard personalized engine ---
    engine = PulseGuardEngine(window_size=500, min_readings=50)
    alert_mgr = AlertManager(
        channels=[ConsoleNotificationChannel()],
        caregiver_contact="+91-XXXXXXXXXX (registered caregiver)",
    )

    pulseguard_events = []
    for r in readings:
        event = engine.process_reading(r)
        if event:
            pulseguard_events.append(event)
            alert_mgr.handle_event(event)

    # --- Fixed-threshold baseline ---
    threshold_results = fixed_threshold_flags(readings)
    threshold_flags = [r for r, flagged in threshold_results if flagged]

    # --- Summary comparison ---
    print("\n" + "=" * 70)
    print("SUMMARY COMPARISON")
    print("=" * 70)

    pg_by_severity = {}
    for e in pulseguard_events:
        pg_by_severity[e.severity] = pg_by_severity.get(e.severity, 0) + 1

    print(f"\nPulseGuard (personalized, severity-tiered):")
    print(f"  Total flagged readings : {len(pulseguard_events)}")
    for sev in [Severity.MILD, Severity.MODERATE, Severity.EMERGENCY]:
        print(f"    {sev.value:10s}: {pg_by_severity.get(sev, 0)}")
    caregiver_alerts = sum(
        1 for e in pulseguard_events if e.severity in AlertManager.ALERT_WORTHY
    )
    print(f"  Caregiver alerts sent  : {caregiver_alerts}  (MILD is logged, not alerted)")

    print(f"\nFixed-Threshold (60-100 bpm, existing approach):")
    print(f"  Total flagged readings : {len(threshold_flags)}")
    print(f"  Caregiver alerts sent  : {len(threshold_flags)}  (no severity tiering — every flag pages caregiver)")

    print("\nKey finding:")
    print("  Episode 2 (moderate, sustained elevation ~92 bpm) stayed")
    print("  UNDER the fixed 100 bpm threshold and would have been MISSED")
    print("  entirely by the existing approach, despite being a real,")
    print("  sustained deviation from this user's own baseline.")
    print("=" * 70)

    # --- Precision / Recall / F1 / FPR against known ground truth ---
    pg_predictions = personalized_predictions(readings)
    ft_predictions = fixed_threshold_predictions(readings)

    pg_metrics = compute_confusion(pg_predictions, ground_truth)
    ft_metrics = compute_confusion(ft_predictions, ground_truth)

    print("\n" + "=" * 70)
    print("EVALUATION METRICS (per-reading, against known injected ground truth)")
    print("=" * 70)
    print(pg_metrics.summary("PulseGuard (Personalized Baseline)"))
    print()
    print(ft_metrics.summary("Fixed-Threshold (60-100 bpm)"))
    print("=" * 70)


if __name__ == "__main__":
    main()
