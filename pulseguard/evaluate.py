"""
PulseGuard - Evaluation Metrics (Precision / Recall / F1 / FPR)
------------------------------------------------------------------
Computes standard binary-classification metrics for anomaly
detection: "flagged by the system" = predicted positive,
"known/annotated ground-truth anomaly" = actual positive.

Ground truth is obtained two different ways depending on dataset:

  - Simulated data (simulate_and_compare.py): exact, because we
    injected the anomaly episodes ourselves and know precisely
    which readings are anomalous.

  - MIT-BIH data (load_mitbih_stream.py): each heart-rate value is
    derived from ONE beat-to-beat interval, and that beat's own
    annotation symbol (from PhysioNet's .atr file) tells us directly
    whether it was a normal or abnormal beat. Ground truth therefore
    attaches exactly to each HR reading -- no fuzzy time-window
    matching is needed, unlike event-level comparisons.
"""

from dataclasses import dataclass

from baseline_engine import (
    PersonalBaselineEngine,
    SeverityClassifier,
    MIN_SD_BPM,
    MAX_FREEZE_SECONDS,
)


@dataclass
class ConfusionMetrics:
    tp: int
    fp: int
    fn: int
    tn: int

    @property
    def precision(self) -> float:
        denom = self.tp + self.fp
        return self.tp / denom if denom else 0.0

    @property
    def recall(self) -> float:
        denom = self.tp + self.fn
        return self.tp / denom if denom else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    @property
    def false_positive_rate(self) -> float:
        denom = self.fp + self.tn
        return self.fp / denom if denom else 0.0

    @property
    def accuracy(self) -> float:
        total = self.tp + self.fp + self.fn + self.tn
        return (self.tp + self.tn) / total if total else 0.0

    def summary(self, label: str) -> str:
        return (
            f"{label}\n"
            f"  Accuracy            : {self.accuracy:.3f}\n"
            f"  Precision           : {self.precision:.3f}\n"
            f"  Recall (Sensitivity): {self.recall:.3f}\n"
            f"  F1 score            : {self.f1:.3f}\n"
            f"  False Positive Rate : {self.false_positive_rate:.3f}\n"
            f"  (TP={self.tp}, FP={self.fp}, FN={self.fn}, TN={self.tn})"
        )


def compute_confusion(predicted: list, actual: list) -> ConfusionMetrics:
    assert len(predicted) == len(actual), "predicted/actual length mismatch"
    tp = sum(1 for p, a in zip(predicted, actual) if p and a)
    fp = sum(1 for p, a in zip(predicted, actual) if p and not a)
    fn = sum(1 for p, a in zip(predicted, actual) if not p and a)
    tn = sum(1 for p, a in zip(predicted, actual) if not p and not a)
    return ConfusionMetrics(tp=tp, fp=fp, fn=fn, tn=tn)


def personalized_predictions(
    readings,
    min_sd: float = MIN_SD_BPM,
    max_freeze_seconds: float = MAX_FREEZE_SECONDS,
) -> list:
    """
    Per-reading anomaly flag using PulseGuard's personal baseline,
    WITHOUT the alert-dispatch escalation gating (that gating exists
    only to avoid re-alerting on every single reading of a sustained
    episode -- for evaluation we want a flag on every reading that
    deviates, so precision/recall reflect the underlying detector).
    """
    baseline = PersonalBaselineEngine(window_size=500, min_readings=50, min_sd=min_sd)
    predictions = []
    deviation_start = None
    for r in readings:
        if not baseline.is_ready:
            baseline.update(r.heart_rate)
            predictions.append(False)  # cold-start: not yet able to judge
            continue

        z = baseline.z_score(r.heart_rate)
        is_anomaly = abs(z) >= SeverityClassifier.MILD_Z
        predictions.append(is_anomaly)

        # Only let the baseline learn from non-anomalous readings, so a
        # sustained anomaly doesn't slowly redefine itself as "normal"
        # (same principle used in PulseGuardEngine) -- but bound that
        # freeze, so a long benign episode (exercise) does not leave the
        # baseline stale for the rest of the recording.
        if not is_anomaly:
            deviation_start = None
            baseline.update(r.heart_rate)
        else:
            if deviation_start is None:
                deviation_start = r.timestamp
            held = (r.timestamp - deviation_start).total_seconds()
            if held > max_freeze_seconds:
                baseline.update(r.heart_rate)

    return predictions


def fixed_threshold_predictions(readings, low: float = 60, high: float = 100) -> list:
    return [r.heart_rate < low or r.heart_rate > high for r in readings]
