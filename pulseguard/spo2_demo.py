"""
PulseGuard - SpO2 Monitor Scenario Tests
------------------------------------------------------------------
Exercises the SpO2 monitor against five scenarios, chosen to cover
the cases that actually matter for a home eldercare device.

Note on validation honesty: unlike the heart-rate engine, these are
SYNTHETIC scenarios, because no public SpO2 dataset with anomaly
labels was available for this project (PPG-DaLiA and MIT-BIH both
carry heart rate but not saturation; MIMIC has SpO2 but requires
credentialed access). These scenarios therefore verify that the
DECISION LOGIC behaves correctly -- they are not a claim of clinical
accuracy on real patients. State it that way to a panel; the SpO2
component's real validation is the hardware phase.

Three of the five scenarios are negative cases (should NOT alert).
That ratio is deliberate: for an alerting system, proving it stays
quiet when it should is as important as proving it fires when it
should, and it is the half that usually goes untested.

RUN:
    python3 spo2_demo.py
"""

from datetime import datetime, timedelta

from alert_manager import AlertManager, ConsoleNotificationChannel
from baseline_engine import Severity
from spo2_monitor import SpO2Monitor, SpO2Reading, SpO2Thresholds


START = datetime(2026, 8, 30, 10, 0, 0)
INTERVAL = 5  # seconds between samples


def build(values):
    """Turns a list of SpO2 values into a timestamped reading stream."""
    return [
        SpO2Reading(timestamp=START + timedelta(seconds=i * INTERVAL), spo2=v)
        for i, v in enumerate(values)
    ]


def run(name, values, monitor=None, expect=None, why=""):
    monitor = monitor or SpO2Monitor()
    events = []
    for r in build(values):
        ev = monitor.process_reading(r)
        if ev:
            events.append(ev)

    peak = max(
        (e.severity for e in events),
        key=lambda s: list(Severity).index(s),
        default=Severity.NORMAL,
    )
    ok = "PASS" if peak == expect else "FAIL"

    print("\n" + "=" * 74)
    print(f"[{ok}] {name}")
    print(f"       {why}")
    print(f"       expected: {expect.value:<10} got: {peak.value}")
    if monitor.invalid_reading_count:
        print(f"       sensor-fault readings discarded: "
              f"{monitor.invalid_reading_count}")
    print("=" * 74)
    for e in events:
        print(f"  -> {e.severity.value.upper():10s} {e.explanation}")
    if not events:
        print("  -> no events (monitor stayed silent)")
    return ok == "PASS"


def main():
    results = []

    # 1. Healthy baseline -- must stay completely silent.
    results.append(run(
        "Healthy steady saturation",
        [98, 97, 98, 99, 98, 97, 98, 98, 97, 98] * 3,
        expect=Severity.NORMAL,
        why="Normal range throughout; any alert here is a false alarm.",
    ))

    # 2. Transient artifact dip -- the single most common false-alarm
    #    source in real devices. Finger shifts, value craters for a few
    #    seconds, recovers. Duration gating must absorb this.
    results.append(run(
        "Transient motion artifact (10s dip to 86%)",
        [98, 97, 98, 86, 85, 97, 98, 98, 97, 98] * 3,
        expect=Severity.NORMAL,
        why="Brief dip below even the emergency cut-off, but only 10s "
            "long -- duration gating must suppress it.",
    ))

    # 3. Sensor falls off -- implausible values must be discarded as a
    #    fault, never escalated as a catastrophic desaturation.
    results.append(run(
        "Sensor detached (implausible readings)",
        [98, 97, 0, 0, 12, 0, 0, 45, 0, 0, 98, 97, 98],
        expect=Severity.NORMAL,
        why="A sensor falling off a sleeping user must not page a "
            "caregiver at 3am.",
    ))

    # 4. Genuine sustained moderate desaturation -- must escalate.
    results.append(run(
        "Sustained desaturation to 91%",
        [98, 97, 98] + [91] * 20 + [97, 98],
        expect=Severity.MODERATE,
        why="Below the 92% cut-off and sustained well past the 30s "
            "gate -- this is a real event and must alert.",
    ))

    # 5. Severe sustained hypoxemia -- must reach EMERGENCY, and must
    #    do so quickly (shorter duration gate for the worst tier).
    results.append(run(
        "Severe sustained hypoxemia to 84%",
        [98, 97, 98] + [84] * 20 + [90, 95],
        expect=Severity.EMERGENCY,
        why="Below 88% sustained past the 15s emergency gate.",
    ))

    # 6. COPD profile -- a user whose clinician TARGETS 88-92%. The
    #    standard thresholds would alarm continuously; the clinical
    #    profile must not.
    copd_values = [90, 89, 91, 90, 89, 90, 91, 90] * 4
    print("\n\n" + "#" * 74)
    print("# COPD PROFILE COMPARISON")
    print("# Same readings (89-91%, a clinically intended range for this")
    print("# patient) judged by both threshold sets.")
    print("#" * 74)
    results.append(run(
        "COPD patient at 89-91% -- STANDARD thresholds",
        copd_values,
        expect=Severity.MODERATE,
        why="Standard cut-offs treat this as a moderate event, which "
            "for this patient is a continuous false alarm.",
    ))
    results.append(run(
        "COPD patient at 89-91% -- COPD profile",
        copd_values,
        monitor=SpO2Monitor(SpO2Thresholds.copd_profile()),
        expect=Severity.NORMAL,
        why="Clinician-set profile correctly treats this as this "
            "patient's intended target range.",
    ))

    # Demonstrate the alert-dispatch path end to end for one real event.
    print("\n\n" + "#" * 74)
    print("# CAREGIVER DISPATCH (same AlertManager as the HR engine,")
    print("# unmodified -- SpO2Event satisfies the same contract)")
    print("#" * 74)
    mgr = AlertManager(
        channels=[ConsoleNotificationChannel()],
        caregiver_contact="+91-XXXXXXXXXX (registered caregiver)",
    )
    mon = SpO2Monitor()
    for r in build([98, 97, 98] + [84] * 20):
        ev = mon.process_reading(r)
        if ev:
            mgr.handle_event(ev)

    passed = sum(results)
    print("\n" + "=" * 74)
    print(f"SCENARIO RESULTS: {passed}/{len(results)} passed")
    print("=" * 74)
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
