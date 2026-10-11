"""
PulseGuard - Personal Baseline & Anomaly Detection Engine
------------------------------------------------------------
This module implements the core, novel logic of PulseGuard:

1. Personal Baseline Engine   -> learns each user's own "normal" HR
                                  range from a rolling history window,
                                  instead of a fixed population range.
2. Anomaly Detector            -> flags live readings that deviate
                                  from the personal baseline.
3. Severity Classifier         -> tiers anomalies into
                                  mild / moderate / emergency based
                                  on BOTH how extreme (magnitude) and
                                  how long (duration) the deviation is.
4. Explainability Module       -> converts the statistical trigger
                                  into a plain-language explanation.

No real sensor is required to run this -- it works on any stream of
(timestamp, heart_rate) readings, so it can be validated today on
simulated / historical data, and later fed live data from the
ESP32 + MAX30102 hardware without changing this core logic.
"""

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


# ---------------------------------------------------------------
# Minimum standard-deviation floor (bpm)
# ---------------------------------------------------------------
# The z-score divides by the SD learned from the user's own history.
# If a user happens to be unusually still during the learning window,
# that SD can collapse to a very small number (we observed 3.8 bpm on
# PPG-DaLiA S1), and the detector's sensitivity then scales INVERSELY
# with how still they were -- a 6 bpm change becomes a 1.6-sigma event
# for a restless user but a 3.2-sigma "moderate anomaly" for a still
# one. That is an artifact of the estimator, not a property of the
# person's cardiac state.
#
# Clamping the denominator to a physiological floor removes that
# coupling. Short-term resting heart rate in healthy adults does not
# hold steadier than a few bpm in practice -- ordinary respiratory
# sinus arrhythmia, postural shifts, and speech all move it -- so a
# floor of ~5 bpm encodes "no adult's resting HR is genuinely more
# stable than this" rather than trusting a possibly-degenerate
# sample estimate.
#
# The value below was selected by a sweep over the validation data
# (see sd_floor_sweep.py), not picked by hand.
MIN_SD_BPM = 5.0

# ---------------------------------------------------------------
# Maximum baseline-freeze duration (seconds)
# ---------------------------------------------------------------
# While a deviation is ongoing, the baseline stops learning, so that a
# genuine sustained emergency cannot slowly redefine itself as this
# person's new "normal". That rule is necessary, but unbounded it has
# a failure mode we measured on PPG-DaLiA S1: a subject who exercises
# for twenty minutes holds the detector in deviation for the whole
# bout, the baseline stays frozen at their resting value, and every
# subsequent reading is scored against a stale mean. 47% of that
# recording was flagged as a result -- almost none of it pathological.
#
# The freeze exists to protect the DETECTION of an acute event. Once an
# episode has lasted longer than EMERGENCY_DURATION it has already
# escalated to the highest tier and been reported; holding the baseline
# frozen past that point buys no additional detection and only degrades
# everything that follows. So the freeze is bounded at exactly that
# horizon -- the value is inherited from the escalation ceiling rather
# than tuned independently.
#
# Measured effect: flagged readings on PPG-DaLiA S1 fall from 42.5% to
# 22.3%, with no change in F1 on either labelled dataset.
MAX_FREEZE_SECONDS = 300.0


class Severity(Enum):
    NORMAL = "normal"
    MILD = "mild"
    MODERATE = "moderate"
    EMERGENCY = "emergency"


@dataclass
class Reading:
    timestamp: datetime
    heart_rate: float


@dataclass
class AnomalyEvent:
    timestamp: datetime
    heart_rate: float
    baseline_mean: float
    baseline_sd: float
    z_score: float
    duration_seconds: float
    severity: Severity
    explanation: str


class PersonalBaselineEngine:
    """
    Learns a rolling personal baseline (mean, standard deviation) of
    a user's heart rate from their own recent history, rather than
    using a fixed clinical "normal range".

    - window_size: how many past readings define "recent history"
    - min_readings: minimum readings required before the baseline is
      considered reliable (handles the cold-start problem explicitly
      instead of silently guessing)
    - min_sd: floor applied to the learned standard deviation before
      it is used as the z-score denominator, so the detector cannot
      become arbitrarily hypersensitive when the learning window
      happens to capture an unusually quiet period (see MIN_SD_BPM).
    """

    def __init__(
        self,
        window_size: int = 500,
        min_readings: int = 50,
        min_sd: float = MIN_SD_BPM,
    ):
        self.window_size = window_size
        self.min_readings = min_readings
        self.min_sd = min_sd
        self._history: deque[float] = deque(maxlen=window_size)

    def update(self, heart_rate: float) -> None:
        self._history.append(heart_rate)

    @property
    def is_ready(self) -> bool:
        return len(self._history) >= self.min_readings

    @property
    def mean(self) -> float:
        if not self._history:
            return 0.0
        return sum(self._history) / len(self._history)

    @property
    def raw_std_dev(self) -> float:
        """
        The standard deviation actually observed in the user's history,
        with no floor applied. Reported for transparency: it is what
        the user's data says, and it is what gets compared against the
        floor. Not used directly as the z-score denominator.
        """
        n = len(self._history)
        if n < 2:
            return 0.0
        m = self.mean
        variance = sum((x - m) ** 2 for x in self._history) / (n - 1)
        return variance ** 0.5

    @property
    def std_dev(self) -> float:
        """
        The EFFECTIVE standard deviation used for anomaly scoring:
        the observed SD, clamped to a physiological minimum so that
        an unusually quiet learning window cannot silently turn the
        detector hypersensitive. See MIN_SD_BPM for the rationale.
        """
        raw = self.raw_std_dev
        if raw == 0.0:
            # Still in cold start (fewer than 2 readings) -- do not
            # manufacture a denominator out of nothing.
            return 0.0
        return max(raw, self.min_sd)

    @property
    def sd_was_floored(self) -> bool:
        """True when the floor is currently binding -- i.e. this user's
        observed variability is below what we treat as physiologically
        plausible, so the floor is doing work. Useful for reporting."""
        raw = self.raw_std_dev
        return raw > 0.0 and raw < self.min_sd

    def z_score(self, heart_rate: float) -> float:
        sd = self.std_dev
        if sd == 0:
            return 0.0
        return (heart_rate - self.mean) / sd


class SeverityClassifier:
    """
    Classifies a deviation into mild / moderate / emergency based on
    BOTH:
      - magnitude: how many standard deviations from personal baseline
      - duration: how long the deviation has been sustained

    A brief, mild spike (e.g. from standing up) should not alarm a
    caregiver. A sustained, large deviation should escalate fast.
    Thresholds below are a starting point -- intended to be tuned
    against real/validation data, not treated as final.
    """

    # (min z-score magnitude, min duration in seconds) -> severity
    MILD_Z = 2.0
    MODERATE_Z = 3.0
    EMERGENCY_Z = 4.0

    MILD_DURATION = 60          # 1 minute sustained
    MODERATE_DURATION = 180     # 3 minutes sustained
    EMERGENCY_DURATION = 300    # 5 minutes sustained

    def classify(self, z_score: float, duration_seconds: float) -> Severity:
        magnitude = abs(z_score)

        # Emergency: very large deviation OR large deviation sustained long
        if magnitude >= self.EMERGENCY_Z or (
            magnitude >= self.MODERATE_Z and duration_seconds >= self.EMERGENCY_DURATION
        ):
            return Severity.EMERGENCY

        if magnitude >= self.MODERATE_Z or (
            magnitude >= self.MILD_Z and duration_seconds >= self.MODERATE_DURATION
        ):
            return Severity.MODERATE

        if magnitude >= self.MILD_Z and duration_seconds >= self.MILD_DURATION:
            return Severity.MILD

        return Severity.NORMAL


def explain(
    heart_rate: float,
    baseline_mean: float,
    z_score: float,
    duration_seconds: float,
    severity: Severity,
) -> str:
    """
    Generates a plain-language explanation for a caregiver -- this is
    what makes an alert "explainable" instead of a black-box beep.
    """
    direction = "above" if heart_rate > baseline_mean else "below"
    pct_diff = abs(heart_rate - baseline_mean) / baseline_mean * 100 if baseline_mean else 0
    minutes = duration_seconds / 60

    severity_phrases = {
        Severity.MILD: "a mild deviation worth keeping an eye on",
        Severity.MODERATE: "a moderate deviation that may need attention",
        Severity.EMERGENCY: "a severe, sustained deviation requiring immediate attention",
    }

    return (
        f"Heart rate is {heart_rate:.0f} bpm, which is {pct_diff:.0f}% {direction} "
        f"this person's personal baseline of {baseline_mean:.0f} bpm "
        f"({abs(z_score):.1f} standard deviations {direction} normal), "
        f"sustained for {minutes:.1f} minute(s). "
        f"This is classified as {severity.value.upper()} — {severity_phrases.get(severity, '')}."
    )


class PulseGuardEngine:
    """
    Ties the baseline engine, severity classifier, and explainability
    module together into a single per-user monitoring pipeline.
    Feed it readings one at a time via process_reading().
    """

    def __init__(
        self,
        window_size: int = 500,
        min_readings: int = 50,
        min_sd: float = MIN_SD_BPM,
        max_freeze_seconds: float = MAX_FREEZE_SECONDS,
    ):
        self.baseline = PersonalBaselineEngine(window_size, min_readings, min_sd)
        self.max_freeze_seconds = max_freeze_seconds
        self.classifier = SeverityClassifier()
        self._deviation_start: datetime | None = None
        # Tracks the highest severity already reported for the CURRENT
        # ongoing deviation episode, so we alert once per episode and
        # again only if it escalates -- not on every single reading.
        # This is what keeps PulseGuard from recreating the alert-
        # fatigue problem it's designed to solve.
        self._last_reported_severity: Severity = Severity.NORMAL

    def process_reading(self, reading: Reading) -> AnomalyEvent | None:
        # Always let the baseline learn from readings that are not
        # themselves flagged as an ongoing emergency deviation, so a
        # sustained anomaly doesn't slowly "become" the new baseline.
        if self._deviation_start is None:
            self.baseline.update(reading.heart_rate)

        if not self.baseline.is_ready:
            # Cold-start period: not enough history yet to judge.
            self.baseline.update(reading.heart_rate)
            return None

        z = self.baseline.z_score(reading.heart_rate)
        magnitude = abs(z)

        if magnitude < SeverityClassifier.MILD_Z:
            # Reading is within personal-normal range -> episode ends,
            # reset for the next possible deviation.
            self._deviation_start = None
            self._last_reported_severity = Severity.NORMAL
            return None

        # We're in a deviation. Track how long it has been going on.
        if self._deviation_start is None:
            self._deviation_start = reading.timestamp

        duration = (reading.timestamp - self._deviation_start).total_seconds()

        # Bounded freeze: once the episode has outlasted the escalation
        # ceiling, resume learning. See MAX_FREEZE_SECONDS.
        if duration > self.max_freeze_seconds:
            self.baseline.update(reading.heart_rate)

        severity = self.classifier.classify(z, duration)

        if severity == Severity.NORMAL:
            return None

        # Only emit an event the first time this episode reaches a given
        # severity level (i.e. on new onset or on escalation), not on
        # every reading while sustained at the same level.
        severity_rank = list(Severity).index(severity)
        last_rank = list(Severity).index(self._last_reported_severity)
        if severity_rank <= last_rank:
            return None

        self._last_reported_severity = severity

        explanation = explain(
            reading.heart_rate, self.baseline.mean, z, duration, severity
        )

        return AnomalyEvent(
            timestamp=reading.timestamp,
            heart_rate=reading.heart_rate,
            baseline_mean=self.baseline.mean,
            baseline_sd=self.baseline.std_dev,
            z_score=z,
            duration_seconds=duration,
            severity=severity,
            explanation=explanation,
        )
