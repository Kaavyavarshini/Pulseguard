"""
PulseGuard - Full Re-Validation (before vs. after the SD floor)
------------------------------------------------------------------
Regenerates every number needed for the results chapter in one run,
reporting each metric BEFORE the minimum-SD floor (min_sd=0.0, the
original behaviour) and AFTER it (min_sd=MIN_SD_BPM), so the report
can show the fix's effect directly rather than asking the reader to
compare against an earlier draft.

RUN (synthetic only -- always works, no downloads):
    python3 revalidate.py

RUN (everything -- needs internet and the PPG-DaLiA file):
    pip install wfdb --break-system-packages
    python3 revalidate.py --mitbih 100 --dalia S1.pkl

Output is written to stdout and to revalidation_results.txt.
"""

import argparse
import sys
from datetime import datetime

from baseline_engine import (
    PulseGuardEngine,
    Severity,
    MIN_SD_BPM,
    MAX_FREEZE_SECONDS,
)
from evaluate import (
    compute_confusion,
    personalized_predictions,
    fixed_threshold_predictions,
)
from simulate_and_compare import generate_readings
from sd_floor_sweep import generate_readings_absolute


class Tee:
    """Writes to stdout and a file simultaneously."""

    def __init__(self, path):
        self.file = open(path, "w")

    def write(self, s):
        sys.__stdout__.write(s)
        self.file.write(s)

    def flush(self):
        sys.__stdout__.flush()
        self.file.flush()


def labelled_table(readings, ground_truth, label):
    """
    Prints a before/after/fixed-threshold comparison for a dataset
    that has per-reading ground truth.
    """
    before = compute_confusion(
        personalized_predictions(readings, min_sd=0.0), ground_truth)
    after = compute_confusion(
        personalized_predictions(readings, min_sd=MIN_SD_BPM), ground_truth)
    fixed = compute_confusion(
        fixed_threshold_predictions(readings), ground_truth)

    print("\n" + "=" * 78)
    print(label)
    print(f"  {len(readings)} readings, {sum(ground_truth)} labelled anomalous")
    print("=" * 78)
    print(f"{'Metric':<22} | {'PulseGuard':>12} | {'PulseGuard':>12} | "
          f"{'Fixed':>12}")
    print(f"{'':<22} | {'(before fix)':>12} | {'(after fix)':>12} | "
          f"{'60-100 bpm':>12}")
    print("-" * 78)
    rows = [
        ("Accuracy", "accuracy"),
        ("Precision", "precision"),
        ("Recall (Sensitivity)", "recall"),
        ("F1 Score", "f1"),
        ("False Positive Rate", "false_positive_rate"),
    ]
    for name, attr in rows:
        print(f"{name:<22} | {getattr(before, attr):>12.3f} | "
              f"{getattr(after, attr):>12.3f} | {getattr(fixed, attr):>12.3f}")
    print(f"{'TP / FP / FN / TN':<22} | "
          f"{f'{before.tp}/{before.fp}/{before.fn}/{before.tn}':>12} | "
          f"{f'{after.tp}/{after.fp}/{after.fn}/{after.tn}':>12} | "
          f"{f'{fixed.tp}/{fixed.fp}/{fixed.fn}/{fixed.tn}':>12}")
    print("-" * 78)
    if before.fp:
        drop = 100 * (before.fp - after.fp) / before.fp
        print(f"Effect of the SD floor: false positives {before.fp} -> "
              f"{after.fp} ({drop:.1f}% reduction), "
              f"precision {before.precision:.3f} -> {after.precision:.3f}, "
              f"recall {before.recall:.3f} -> {after.recall:.3f}")
    return before, after, fixed


def event_table(readings, label, window_size=500):
    """
    For datasets with no anomaly labels (PPG-DaLiA): reports alert
    volume and the learned baseline for the original engine and for
    each fix applied in turn.

    Reports BOTH episode count and per-reading flag rate, because
    episode-level counts alone are misleading: episode gating collapses
    a 20-minute continuous deviation into a single "event", which is
    exactly what concealed the stale-baseline problem in the first
    place. The per-reading rate is the honest measure of how much of
    the recording the detector considers abnormal.
    """
    import math
    from baseline_engine import MAX_FREEZE_SECONDS
    from evaluate import personalized_predictions

    NO_FREEZE_LIMIT = math.inf

    variants = [
        ("original", 0.0, NO_FREEZE_LIMIT),
        ("+ SD floor", MIN_SD_BPM, NO_FREEZE_LIMIT),
        ("+ bounded freeze", MIN_SD_BPM, MAX_FREEZE_SECONDS),
    ]

    print("\n" + "=" * 78)
    print(label)
    print(f"  {len(readings)} readings")
    print("  No anomaly ground truth -- reporting alert volume, not accuracy.")
    print("=" * 78)
    print(f"{'':<18} | {'episodes':>8} | {'mod':>4} | {'emrg':>4} | "
          f"{'flagged readings':>17} | {'raw SD':>7}")
    print("-" * 78)

    first_pct = None
    for tag, floor, freeze in variants:
        eng = PulseGuardEngine(window_size=window_size, min_readings=50,
                               min_sd=floor, max_freeze_seconds=freeze)
        counts, total = {}, 0
        for r in readings:
            ev = eng.process_reading(r)
            if ev:
                total += 1
                counts[ev.severity] = counts.get(ev.severity, 0) + 1

        flags = personalized_predictions(readings, min_sd=floor,
                                         max_freeze_seconds=freeze)
        n_flag = sum(flags)
        pct = 100 * n_flag / len(flags)
        if first_pct is None:
            first_pct = pct

        print(f"{tag:<18} | {total:>8} | "
              f"{counts.get(Severity.MODERATE, 0):>4} | "
              f"{counts.get(Severity.EMERGENCY, 0):>4} | "
              f"{n_flag:>6} ({pct:>5.1f}%) | "
              f"{eng.baseline.raw_std_dev:>7.2f}")

    fixed_flags = sum(
        1 for r in readings if r.heart_rate < 60 or r.heart_rate > 100)
    print(f"{'fixed 60-100 bpm':<18} | {'-':>8} | {'-':>4} | {'-':>4} | "
          f"{fixed_flags:>6} ({100 * fixed_flags / len(readings):>5.1f}%) | "
          f"{'-':>7}")
    print("-" * 78)
    print(f"Per-reading flag rate: {first_pct:.1f}% -> {pct:.1f}% "
          f"across the two fixes.")
    print("Note: this subject is a healthy adult performing daily activities")
    print("(walking, cycling), so most remaining flags are activity-driven HR")
    print("change, not pathology. This dataset bounds ALERT VOLUME; it cannot")
    print("measure detection accuracy, because it carries no anomaly labels.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mitbih", metavar="RECORD")
    ap.add_argument("--dalia", metavar="PKL")
    ap.add_argument("--dalia-csv", metavar="CSV",
                    help="PPG-DaLiA HR export as CSV (no .pkl needed)")
    args = ap.parse_args()

    sys.stdout = Tee("revalidation_results.txt")

    print("=" * 78)
    print("PULSEGUARD RE-VALIDATION")
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"MIN_SD_BPM = {MIN_SD_BPM}")
    print("=" * 78)

    r, gt = generate_readings()
    labelled_table(r, gt, "TABLE 2 - Synthetic data (SD-scaled anomalies)")

    r, gt = generate_readings_absolute()
    labelled_table(r, gt,
                   "TABLE 2b - Synthetic quiet baseline (absolute-bpm anomalies)")

    if args.mitbih:
        try:
            from load_mitbih_stream import stream_mitbih_hr
            r, gt = stream_mitbih_hr(args.mitbih)
            labelled_table(
                r, gt, f"TABLE 3 - MIT-BIH record {args.mitbih} (real ECG-derived HR)")
        except Exception as exc:
            print(f"\n[SKIPPED MIT-BIH] {exc}")
            print("  Install wfdb and ensure internet access, then re-run.")
    else:
        print("\n[MIT-BIH not run] pass --mitbih 100 to include Table 3.")

    if args.dalia or args.dalia_csv:
        try:
            if args.dalia_csv:
                from load_real_dataset import load_ppg_dalia_hr_csv
                src = args.dalia_csv
                r = load_ppg_dalia_hr_csv(src)
            else:
                from load_real_dataset import load_ppg_dalia_hr
                src = args.dalia
                r = load_ppg_dalia_hr(src)
            event_table(r, f"TABLE 4 - PPG-DaLiA {src} (real PPG-derived HR)")
        except Exception as exc:
            print(f"\n[SKIPPED PPG-DaLiA] {exc}")
            print("  Place the subject file beside this script, then re-run.")
    else:
        print("\n[PPG-DaLiA not run] pass --dalia S1.pkl to include Table 4.")

    print("\n" + "=" * 78)
    print("Saved to revalidation_results.txt")
    print("=" * 78)


if __name__ == "__main__":
    main()
