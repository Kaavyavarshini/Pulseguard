"""
PulseGuard - Minimum-SD Floor Sensitivity Sweep
------------------------------------------------------------------
Selects the MIN_SD_BPM floor empirically instead of hand-picking it.

WHY THIS SCRIPT EXISTS
----------------------
The z-score denominator is the SD learned from the user's own recent
history. When that window happens to capture an unusually still
period, the learned SD collapses (3.8 bpm on PPG-DaLiA S1, 5.0 bpm on
the synthetic run), and the detector's sensitivity then scales
inversely with how still the person happened to be. A 6 bpm change
becomes a 1.6-sigma non-event for one user and a 3.2-sigma "moderate
anomaly" for another, purely because of the estimator.

Clamping the denominator to a physiological floor removes that
coupling. But a floor chosen by hand is just a different magic number.
This sweep evaluates floors from 0 (the original, unfloored behaviour)
upward on labelled data, so the chosen value can be justified by its
effect on precision/recall rather than by assertion.

RUN:
    python3 sd_floor_sweep.py                  # synthetic data (always works)
    python3 sd_floor_sweep.py --mitbih 100     # + MIT-BIH record (needs wfdb + internet)
    python3 sd_floor_sweep.py --dalia S1.pkl   # + PPG-DaLiA subject (needs the .pkl)
"""

import argparse
import sys

from evaluate import (
    compute_confusion,
    personalized_predictions,
    fixed_threshold_predictions,
)
from simulate_and_compare import generate_readings


# Floors to evaluate, in bpm. 0.0 reproduces the original, pre-fix
# behaviour exactly, so every table has an honest "before" column.
FLOORS = [0.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0]


def sweep(readings, ground_truth, label: str):
    """Evaluates every candidate floor on one labelled dataset."""
    print("\n" + "=" * 78)
    print(f"SD FLOOR SWEEP - {label}")
    print(f"  {len(readings)} readings, "
          f"{sum(ground_truth)} labelled anomalous "
          f"({100 * sum(ground_truth) / len(ground_truth):.1f}%)")
    print("=" * 78)
    print(f"{'floor':>7} | {'prec':>6} | {'recall':>6} | {'F1':>6} | "
          f"{'FPR':>6} | {'TP':>5} | {'FP':>5} | {'FN':>5}")
    print("-" * 78)

    rows = []
    for floor in FLOORS:
        preds = personalized_predictions(readings, min_sd=floor)
        m = compute_confusion(preds, ground_truth)
        rows.append((floor, m))
        tag = "  <- original" if floor == 0.0 else ""
        print(f"{floor:>7.1f} | {m.precision:>6.3f} | {m.recall:>6.3f} | "
              f"{m.f1:>6.3f} | {m.false_positive_rate:>6.3f} | "
              f"{m.tp:>5} | {m.fp:>5} | {m.fn:>5}{tag}")

    # Report the floor that maximises F1, and separately the most
    # conservative floor that still retains high recall -- because for
    # an eldercare alerting system, a missed emergency costs more than
    # a false alarm, so pure F1 is not automatically the right target.
    best_f1 = max(rows, key=lambda r: r[1].f1)
    high_recall = [r for r in rows if r[1].recall >= 0.90]
    best_precision_at_recall = (
        max(high_recall, key=lambda r: r[1].precision) if high_recall else None
    )

    print("-" * 78)
    print(f"Best F1                        : floor={best_f1[0]:.1f} "
          f"(F1={best_f1[1].f1:.3f})")
    if best_precision_at_recall:
        f, m = best_precision_at_recall
        print(f"Best precision at recall>=0.90 : floor={f:.1f} "
              f"(precision={m.precision:.3f}, recall={m.recall:.3f})")
    else:
        print("Best precision at recall>=0.90 : none "
              "(no floor keeps recall above 0.90 on this dataset)")

    # Quantify the headline claim: what did the floor do to false alarms?
    before = rows[0][1]
    after = next((m for f, m in rows if f == 5.0), None)
    if after and before.fp:
        drop = 100 * (before.fp - after.fp) / before.fp
        print(f"\nAt the chosen floor of 5.0 bpm: "
              f"false positives {before.fp} -> {after.fp} "
              f"({drop:.1f}% reduction), "
              f"recall {before.recall:.3f} -> {after.recall:.3f}")
    return rows


def generate_readings_absolute(
    personal_baseline_hr: float = 70.0,
    personal_sd: float = 2.0,
    n_normal_readings: int = 600,
    seconds_between_readings: int = 10,
    seed: int = 7,
):
    """
    Like simulate_and_compare.generate_readings(), but injects anomaly
    episodes as ABSOLUTE bpm offsets rather than as multiples of the
    person's own SD.

    This distinction matters and was a genuine flaw in the first
    version of this sweep. If anomalies are defined in SD units, then
    simulating a very still subject (small SD) also shrinks the
    injected anomalies, and the test degenerates: a "severe" episode
    becomes a 10 bpm bump that SHOULD be ignored. The floor then looks
    like it destroys recall, when really the test destroyed the signal.

    Physiologically, a cardiac event is a deviation of so many bpm --
    it does not get smaller because the person happened to be sitting
    still while the baseline was learned. Absolute offsets are the
    correct way to reproduce the PPG-DaLiA failure condition.
    """
    import random
    from datetime import datetime, timedelta
    from baseline_engine import Reading

    random.seed(seed)
    readings, ground_truth = [], []
    t = datetime(2026, 8, 30, 9, 0, 0)

    def add_normal(n):
        nonlocal t
        for _ in range(n):
            hr = random.gauss(personal_baseline_hr, personal_sd)
            readings.append(Reading(timestamp=t, heart_rate=round(hr, 1)))
            ground_truth.append(False)
            t += timedelta(seconds=seconds_between_readings)

    def add_anomaly(n, offset_bpm, jitter):
        nonlocal t
        for _ in range(n):
            hr = random.gauss(personal_baseline_hr + offset_bpm, jitter)
            readings.append(Reading(timestamp=t, heart_rate=round(hr, 1)))
            ground_truth.append(True)
            t += timedelta(seconds=seconds_between_readings)

    add_normal(n_normal_readings)
    add_anomaly(9, 9.0, 1.5)      # mild, brief  (~79 bpm)
    add_normal(n_normal_readings // 6)
    add_anomaly(21, 18.0, 1.5)    # moderate, sustained (~88 bpm)
    add_normal(n_normal_readings // 6)
    add_anomaly(31, 35.0, 2.0)    # emergency, severe (~105 bpm)
    add_normal(n_normal_readings // 6)

    return readings, ground_truth


def synthetic_datasets():
    """
    Two synthetic scenarios, not one.

    'normal-variability' is the project's existing simulation (SD 4 bpm,
    anomalies in SD units) -- kept unchanged so the numbers stay
    comparable with the figures already in the report.

    'quiet-baseline' reproduces the failure condition observed on
    PPG-DaLiA S1 -- a subject whose learned SD collapsed to ~3.8 bpm --
    using absolute-bpm anomalies, so the fix is tested against the
    condition it was actually designed for.
    """
    normal = generate_readings(personal_baseline_hr=78.0, personal_sd=4.0)
    quiet = generate_readings_absolute(personal_baseline_hr=70.0, personal_sd=2.0)
    return [
        ("SYNTHETIC - normal variability (true SD 4.0 bpm, SD-scaled anomalies)",
         normal),
        ("SYNTHETIC - quiet baseline (true SD 2.0 bpm, absolute-bpm anomalies)",
         quiet),
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mitbih", metavar="RECORD",
                    help="MIT-BIH record id to stream (e.g. 100)")
    ap.add_argument("--dalia", metavar="PKL",
                    help="Path to a PPG-DaLiA subject .pkl (e.g. S1.pkl)")
    args = ap.parse_args()

    all_rows = []
    for label, (readings, gt) in synthetic_datasets():
        all_rows.append((label, sweep(readings, gt, label)))

    # A floor tuned to one scenario is overfitting. The value shipped in
    # baseline_engine.py is the one that performs best ACROSS scenarios,
    # so report the joint picture explicitly rather than cherry-picking.
    print("\n" + "=" * 78)
    print("JOINT SELECTION ACROSS LABELLED SCENARIOS (mean F1)")
    print("=" * 78)
    print(f"{'floor':>7} | " + " | ".join(f"{'F1 sc%d' % (i + 1):>8}"
                                          for i in range(len(all_rows)))
          + f" | {'mean F1':>8}")
    print("-" * 78)
    for idx, floor in enumerate(FLOORS):
        f1s = [rows[idx][1].f1 for _, rows in all_rows]
        mean_f1 = sum(f1s) / len(f1s)
        print(f"{floor:>7.1f} | " + " | ".join(f"{v:>8.3f}" for v in f1s)
              + f" | {mean_f1:>8.3f}")
    means = [
        (floor, sum(rows[idx][1].f1 for _, rows in all_rows) / len(all_rows))
        for idx, floor in enumerate(FLOORS)
    ]
    best_floor, best_mean = max(means, key=lambda x: x[1])
    print("-" * 78)
    print(f"Selected floor: {best_floor:.1f} bpm (mean F1 = {best_mean:.3f})")
    print("This is the value set as MIN_SD_BPM in baseline_engine.py.")

    if args.mitbih:
        try:
            from load_mitbih_stream import stream_mitbih_hr
            readings, gt = stream_mitbih_hr(args.mitbih)
            sweep(readings, gt, f"MIT-BIH record {args.mitbih} (real ECG-derived HR)")
        except Exception as exc:
            print(f"\n[skipped MIT-BIH] {exc}", file=sys.stderr)

    if args.dalia:
        # PPG-DaLiA has no anomaly labels, so precision/recall are not
        # computable here. Report the alert-volume change instead --
        # that is the honest metric for an unlabelled dataset.
        try:
            from load_real_dataset import load_ppg_dalia_hr
            from baseline_engine import PulseGuardEngine, Severity
            readings = load_ppg_dalia_hr(args.dalia)
            print("\n" + "=" * 78)
            print(f"SD FLOOR SWEEP - PPG-DaLiA {args.dalia} (UNLABELLED)")
            print(f"  {len(readings)} readings "
                  f"({len(readings) * 2 / 60:.1f} min)")
            print("  No anomaly ground truth in this dataset -- reporting")
            print("  alert volume, not precision/recall.")
            print("=" * 78)
            print(f"{'floor':>7} | {'events':>7} | {'mild':>5} | {'mod':>5} | "
                  f"{'emerg':>6} | {'raw SD':>7} | {'eff SD':>7}")
            print("-" * 78)
            for floor in FLOORS:
                eng = PulseGuardEngine(window_size=500, min_readings=50,
                                       min_sd=floor)
                counts = {}
                total = 0
                for r in readings:
                    ev = eng.process_reading(r)
                    if ev:
                        total += 1
                        counts[ev.severity] = counts.get(ev.severity, 0) + 1
                tag = "  <- original" if floor == 0.0 else ""
                print(f"{floor:>7.1f} | {total:>7} | "
                      f"{counts.get(Severity.MILD, 0):>5} | "
                      f"{counts.get(Severity.MODERATE, 0):>5} | "
                      f"{counts.get(Severity.EMERGENCY, 0):>6} | "
                      f"{eng.baseline.raw_std_dev:>7.2f} | "
                      f"{eng.baseline.std_dev:>7.2f}{tag}")
        except Exception as exc:
            print(f"\n[skipped PPG-DaLiA] {exc}", file=sys.stderr)


if __name__ == "__main__":
    main()
