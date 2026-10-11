"""
PulseGuard - SpO2 (Peripheral Oxygen Saturation) Monitor
------------------------------------------------------------------
Adds a second monitored vital alongside the personalized heart-rate
engine, using the SAME severity tiers, the same episode-gating, and
the same plain-language explainability contract -- so a caregiver
receives one consistent kind of alert regardless of which vital
triggered it.

WHY THIS IS NOT PERSONALIZED (the important design decision)
-------------------------------------------------------------
The whole argument of this project is that heart rate should be judged
against the individual, not the population. It would be tempting to
apply that same reasoning to SpO2. That would be wrong, and it is
worth being explicit about why, because the inconsistency is only
apparent.

Heart rate is personalized because a resting HR of 55 and a resting HR
of 85 are both entirely healthy -- the population range is wide, and a
person's own position within it carries no clinical meaning on its own.
The information is in the DEVIATION, not the absolute value.

Oxygen saturation does not behave that way. It is a near-saturating
measure: healthy adults sit at 95-100% almost regardless of age or
fitness, the usable range is narrow, and the absolute value itself is
what carries clinical meaning. Below roughly 90% indicates hypoxemia
for anybody. Learning a personal SpO2 baseline would actively cause
harm in the one case that matters most: a user whose saturation drifts
down over hours or days would have that decline absorbed into their
"personal normal", and the system would fall silent exactly as the
patient deteriorated. A rolling baseline normalises slow trends. For
HR that is a feature; for SpO2 it is a failure mode.

So SpO2 uses fixed clinical thresholds. This is a deliberate
asymmetry, not an unfinished feature -- personalizing here would be
less defensible, not more.

DURATION GATING
---------------
Transient SpO2 dips are extremely common artifacts: a finger shifting
on the sensor, cold peripheries, motion, or a held breath will all
produce brief drops. Acting on a single low sample would recreate the
alert-fatigue problem this project exists to solve, so a drop must be
SUSTAINED before it escalates -- the same magnitude-and-duration
principle used for heart rate.
"""

from dataclasses import dataclass
from datetime import datetime

from baseline_engine import Severity


# ---------------------------------------------------------------
# Sensor validity bounds
# ---------------------------------------------------------------
# The MAX30102 reports physiologically impossible values when the
# finger is off the sensor, poorly seated, or the signal is swamped by
# ambient light. Those readings must be DISCARDED, never treated as a
# desaturation event -- a sensor falling off a sleeping user must not
# page a caregiver at 3am. Anything below this bound in a conscious,
# ambulatory home user is a sensor fault, not a survivable reading.
MIN_PLAUSIBLE_SPO2 = 70.0
MAX_PLAUSIBLE_SPO2 = 100.0


@dataclass
class SpO2Reading:
    timestamp: datetime
    spo2: float


@dataclass
class SpO2Event:
    timestamp: datetime
    spo2: float
    threshold_crossed: float
    duration_seconds: float
    severity: Severity
    explanation: str


@dataclass
class SpO2Thresholds:
    """
    Fixed clinical cut-offs, in percent saturation.

    Defaults follow standard adult guidance: 95-100% is normal, the
    low-90s warrants attention, and below 90% is clinically significant
    hypoxemia.
    """
    mild_below: float = 95.0
    moderate_below: float = 92.0
    emergency_below: float = 88.0

    # Seconds a reading must stay below a cut-off before it escalates
    # to that tier. Emergencies are gated more briefly than mild
    # deviations: the more dangerous the reading, the less willing we
    # are to wait for confirmation.
    mild_duration: float = 60.0
    moderate_duration: float = 30.0
    emergency_duration: float = 15.0

    @classmethod
    def copd_profile(cls) -> "SpO2Thresholds":
        """
        Alternative preset for a user with known chronic hypoxemia
        (e.g. COPD), where a target saturation of 88-92% is clinically
        INTENTIONAL rather than an emergency. Applying the standard
        thresholds to such a user would generate continuous false
        alarms for a condition their clinician is deliberately
        maintaining.

        This is the correct way to handle individual variation in
        SpO2: a clinician-selected profile, not a self-learned
        baseline. The system is told the person's clinical context; it
        does not infer it from the data and risk normalising a decline.
        """
        return cls(
            mild_below=90.0,
            moderate_below=88.0,
            emergency_below=85.0,
        )


def explain_spo2(
    spo2: float,
    threshold_crossed: float,
    duration_seconds: float,
    severity: Severity,
) -> str:
    """
    Plain-language explanation, matching the contract used by the
    heart-rate explainability module so caregiver messages read
    consistently across both vitals.
    """
    minutes = duration_seconds / 60

    severity_phrases = {
        Severity.MILD: (
            "a mild reduction worth keeping an eye on, but not "
            "immediately dangerous"
        ),
        Severity.MODERATE: (
            "a moderate reduction that may need attention -- check on "
            "the person and consider contacting their doctor"
        ),
        Severity.EMERGENCY: (
            "a severe, sustained drop in oxygen requiring immediate "
            "attention"
        ),
    }

    return (
        f"Blood oxygen saturation is {spo2:.0f}%, which is below the "
        f"{threshold_crossed:.0f}% clinical threshold, sustained for "
        f"{minutes:.1f} minute(s). Normal is 95% or above for most "
        f"adults. This is classified as {severity.value.upper()} — "
        f"{severity_phrases.get(severity, '')}."
    )


class SpO2Monitor:
    """
    Fixed-threshold SpO2 monitor with duration gating and per-episode
    alert suppression.

    Mirrors the interface of PulseGuardEngine: feed it readings one at
    a time via process_reading(), which returns an SpO2Event only on
    the onset of a tier or on escalation -- never on every reading of
    a sustained episode.
    """

    def __init__(self, thresholds: SpO2Thresholds | None = None):
        self.thresholds = thresholds or SpO2Thresholds()
        self._episode_start: datetime | None = None
        self._last_reported_severity: Severity = Severity.NORMAL
        self.invalid_reading_count = 0

    def _tier_for(self, spo2: float, duration: float) -> tuple[Severity, float]:
        """
        Returns the severity this reading qualifies for given how long
        it has been sustained, plus the threshold it crossed.
        Evaluated most-severe-first so a catastrophic reading is not
        held back by a longer, less-urgent tier's duration gate.
        """
        t = self.thresholds
        if spo2 < t.emergency_below and duration >= t.emergency_duration:
            return Severity.EMERGENCY, t.emergency_below
        if spo2 < t.moderate_below and duration >= t.moderate_duration:
            return Severity.MODERATE, t.moderate_below
        if spo2 < t.mild_below and duration >= t.mild_duration:
            return Severity.MILD, t.mild_below
        return Severity.NORMAL, t.mild_below

    def process_reading(self, reading: SpO2Reading) -> SpO2Event | None:
        spo2 = reading.spo2

        # 1. Reject sensor faults before any clinical interpretation.
        if not (MIN_PLAUSIBLE_SPO2 <= spo2 <= MAX_PLAUSIBLE_SPO2):
            self.invalid_reading_count += 1
            return None

        # 2. Readings at or above the mild cut-off end any episode.
        if spo2 >= self.thresholds.mild_below:
            self._episode_start = None
            self._last_reported_severity = Severity.NORMAL
            return None

        # 3. We are below normal. Track how long this has been true.
        if self._episode_start is None:
            self._episode_start = reading.timestamp

        duration = (reading.timestamp - self._episode_start).total_seconds()
        severity, threshold = self._tier_for(spo2, duration)

        if severity == Severity.NORMAL:
            # Below normal, but not yet sustained long enough to act on.
            return None

        # 4. Emit only on onset or escalation, not on every reading.
        severity_rank = list(Severity).index(severity)
        last_rank = list(Severity).index(self._last_reported_severity)
        if severity_rank <= last_rank:
            return None

        self._last_reported_severity = severity

        return SpO2Event(
            timestamp=reading.timestamp,
            spo2=spo2,
            threshold_crossed=threshold,
            duration_seconds=duration,
            severity=severity,
            explanation=explain_spo2(spo2, threshold, duration, severity),
        )


class VitalsMonitor:
    """
    Runs the personalized HR engine and the fixed-threshold SpO2
    monitor side by side over a synchronised stream, returning any
    events from either.

    Keeping them as separate detectors -- rather than fusing them into
    one score -- preserves explainability: a caregiver is told which
    vital triggered the alert and why, instead of receiving a combined
    number that cannot be acted on.
    """

    def __init__(self, hr_engine, spo2_monitor: SpO2Monitor | None = None):
        self.hr_engine = hr_engine
        self.spo2_monitor = spo2_monitor or SpO2Monitor()

    def process(
        self,
        timestamp: datetime,
        heart_rate: float | None = None,
        spo2: float | None = None,
    ) -> list:
        """Returns a list of events (possibly empty) from both vitals."""
        from baseline_engine import Reading

        events = []
        if heart_rate is not None:
            ev = self.hr_engine.process_reading(
                Reading(timestamp=timestamp, heart_rate=heart_rate)
            )
            if ev:
                events.append(ev)
        if spo2 is not None:
            ev = self.spo2_monitor.process_reading(
                SpO2Reading(timestamp=timestamp, spo2=spo2)
            )
            if ev:
                events.append(ev)
        return events
