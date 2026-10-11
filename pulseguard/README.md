# PulseGuard

**A Personalized and Explainable Anomaly Detection System for Elderly Cardiac Monitoring Using Wearable Devices**

PulseGuard learns each person's own heart-rate baseline instead of applying a fixed 60-100 bpm range. It classifies deviations into severity tiers, explains every alert in plain language, and sends moderate and emergency alerts to a caregiver by SMS or email. It runs on datasets and on live ESP32 + MAX30102 hardware.

---

## Table of Contents

- [Why PulseGuard](#why-pulseguard)
- [Features](#features)
- [How It Works](#how-it-works)
- [Architecture](#architecture)
- [Hardware](#hardware)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Validation and Results](#validation-and-results)
- [Limitations](#limitations)
- [Future Work](#future-work)
- [Author](#author)

---

## Why PulseGuard

Most wearable monitors for elderly users flag a heart rate below 60 or above 100 bpm. That ignores age, fitness, medication and baseline health, and it fails in two ways:

- **False alarms** tire caregivers until they ignore alerts.
- **Missed emergencies** happen when a dangerous rate sits inside someone else's "normal" range.

PulseGuard compares each reading with that person's own history, so what counts as unusual is personal.

---

## Features

- **Personal baseline.** Rolling mean and standard deviation of the user's recent heart rate (default window 500 readings, 50 minimum before judging).
- **Severity tiers.** Mild, moderate and emergency, based on both how far and how long a reading deviates.
- **Explainable alerts.** Each alert says what the reading was, how far it is from the personal baseline, for how long, and the tier.
- **Alert-fatigue control.** Mild events are logged only. One alert per episode, repeated only on escalation.
- **Caregiver notification.** SMS through Twilio or email through SMTP, with a console channel for testing.
- **SpO2 monitor.** Fixed clinical thresholds, with a separate COPD profile.
- **Live mode.** The same engine runs on a live ESP32 serial stream.
- **Evaluation tooling.** Precision, recall, F1 and false-positive rate against a fixed-threshold baseline.

---

## How It Works

**Baseline.** From the rolling window of readings *x₁ … xₙ*:

```
μ      = mean(x)
σ_raw  = sample standard deviation of x
σ      = max(σ_raw, σ_min)        with σ_min = 5.0 bpm
z      = (x_live − μ) / σ
```

**Severity.** With `|z|` the deviation size and `d` the seconds it has lasted:

| Tier | Condition |
|------|-----------|
| Emergency | `|z| ≥ 4.0`, or (`|z| ≥ 3.0` and `d ≥ 300 s`) |
| Moderate | `|z| ≥ 3.0`, or (`|z| ≥ 2.0` and `d ≥ 180 s`) |
| Mild | `|z| ≥ 2.0` and `d ≥ 60 s` |
| Normal | otherwise |

**Two corrections found through validation:**

1. **SD floor (5 bpm).** A quiet learning window can shrink σ so much that ordinary movement looks like an anomaly. The floor was chosen by sweeping 0-10 bpm across labelled synthetic scenarios and keeping the best F1.
2. **Bounded baseline freeze (300 s).** The baseline freezes during a deviation so a real emergency cannot become the new normal. Left unbounded, a 20-minute bike ride left the baseline stale for the rest of the recording. Once an episode passes the emergency duration (already reported at the top tier), learning resumes.

**Example alert:**

```
Heart rate is 96 bpm, which is 24% above this person's personal baseline of 78 bpm
(3.6 standard deviations above normal), sustained for 2.5 minute(s).
This is classified as EMERGENCY — a severe, sustained deviation requiring immediate attention.
```

**SpO2** is not personalized. A rolling baseline would absorb a slow oxygen decline into "normal", which is the case where an alert matters most. SpO2 is checked against fixed cutoffs of 95%, 92% and 88% (mild, moderate, emergency), must stay low for a minimum duration before escalating, and implausible values are discarded as sensor faults.

---

## Architecture

```
MAX30102 ──I²C──▶ ESP32 ──serial──▶ live_serial_stream.py
 (HR + SpO2)    (filter, timestamp)          │
                                             ▼
 datasets (synthetic, MIT-BIH, PPG-DaLiA) ─▶ PulseGuardEngine
                                             │  baseline → z-score → severity → explanation
                                             ▼
                                        AlertManager ──▶ Twilio SMS / SMTP email / console
```

The detection logic is hardware-independent. The same `PulseGuardEngine` runs on synthetic data, public datasets and the live sensor.

---

## Hardware

| Component | Purpose |
|-----------|---------|
| ESP32-WROOM-32 | Sensor interfacing, data acquisition, serial communication |
| MH-ET LIVE MAX30102 | Optical heart-rate and SpO2 sensor |
| USB cable | Programming and live data transfer |
| Jumper wires, power supply | Connections and power |

The sensor connects over I²C (3.3 V, GND, SDA, SCL). GPIO pins depend on your firmware configuration. The firmware is C++ on the Arduino framework.

Live mode discards readings outside 20-250 bpm (for example, finger not yet seated) and timestamps by wall-clock time so an ESP32 reset doesn't corrupt the log.

---

## Repository Structure

| File | Role |
|------|------|
| `baseline_engine.py` | Baseline, severity classifier, explainability, `PulseGuardEngine` |
| `alert_manager.py` | Console, Twilio SMS and SMTP email channels, alert rules |
| `evaluate.py` | Precision, recall, F1, FPR and fixed-threshold comparison |
| `simulate_and_compare.py` | Synthetic data with injected anomalies |
| `load_mitbih_stream.py` | Streams MIT-BIH records from PhysioNet |
| `load_real_dataset.py` | Loads PPG-DaLiA subject recordings |
| `spo2_monitor.py` | SpO2 monitor with fixed clinical thresholds |
| `sd_floor_sweep.py` | Sweeps candidate SD floors |
| `revalidate.py` | Re-validates the fixes across datasets |
| `live_serial_stream.py` | Feeds live ESP32 serial output into the engine |

---

## Getting Started

**Requirements:** Python 3.9+, `numpy`, `wfdb` (MIT-BIH), `twilio` (SMS). The ESP32 and sensor are optional.

```bash
git clone https://github.com/Kaavyavarshini/<repo-name>.git
cd <repo-name>
pip install numpy wfdb twilio
```

**SMS credentials.** Live credentials are read from environment variables, never hard-coded:

```bash
export TWILIO_ACCOUNT_SID="..."
export TWILIO_AUTH_TOKEN="..."
export TWILIO_FROM_NUMBER="+1XXXXXXXXXX"
```

If any are missing, the run falls back to console-only alerts and lists what is missing. Check `live_serial_stream.py` for how the caregiver number is supplied. Never commit credentials.

---

## Usage

```bash
# Synthetic data: PulseGuard vs fixed 60-100 bpm
python simulate_and_compare.py

# Real ECG-derived data (MIT-BIH, streamed from PhysioNet)
python load_mitbih_stream.py

# Real PPG data (PPG-DaLiA subject pickle)
python load_real_dataset.py <path_to_subject_pkl>

# SpO2 scenario tests
python spo2_monitor.py

# Sweep the SD floor, then re-validate the fixes
python sd_floor_sweep.py --mitbih 100 --dalia <subject.pkl>
python revalidate.py --mitbih 100 --dalia <subject.pkl>

# Live hardware (add --sms for real SMS alerts)
python live_serial_stream.py
python live_serial_stream.py --sms
```

Flags beyond those shown (for example the serial port) depend on your scripts. Adjust as needed.

---

## Validation and Results

Validation ran in stages, each covering a weakness of the one before. Precision, recall, F1 and false-positive rate are reported separately, since a missed emergency and a false alarm cost different amounts in elder care.

### Synthetic data (961 readings)

| Metric | Before fixes | After fixes | Fixed threshold |
|--------|-------------:|------------:|----------------:|
| Accuracy | 0.869 | 0.984 | 0.942 |
| Precision | 0.324 | 0.883 | 1.000 |
| Recall | 0.984 | 0.869 | 0.082 |
| F1 | 0.488 | 0.876 | 0.152 |
| False-positive rate | 0.139 | 0.008 | 0.000 |

### MIT-BIH Record 100, real ECG-derived data (2,272 readings)

| Metric | Before fixes | After fixes | Fixed threshold |
|--------|-------------:|------------:|----------------:|
| Accuracy | 0.845 | 0.971 | 0.989 |
| Precision | 0.086 | 0.340 | 0.692 |
| Recall | 0.971 | 0.971 | 0.529 |
| F1 | 0.158 | 0.504 | 0.600 |
| False-positive rate | 0.157 | 0.029 | 0.004 |

PulseGuard caught 33 of 34 abnormal beats, versus 18 of 34 for the fixed threshold. Part of the precision gap is a ground-truth mismatch: MIT-BIH labels beat morphology, while PulseGuard detects deviation from a personal baseline.

The fixes cut false positives by 82-100% across datasets, with a recall cost of at most about 12 points on synthetic data and none on MIT-BIH.

### PPG-DaLiA Subject S1, real wrist PPG (153 min, 4,603 readings)

This subject's resting heart rate is often 49-55 bpm, which is normal but below the fixed 60 bpm cutoff.

| Variant | Episodes (mod / emerg) | Flagged readings |
|---------|-----------------------:|-----------------:|
| Original | 51 (30 / 21) | 2,155 (46.8%) |
| + SD floor | 44 (26 / 18) | 1,957 (42.5%) |
| + bounded freeze | 23 (15 / 7) | 1,028 (22.3%) |
| Fixed threshold | n/a | 829 (18.0%) |

The fixed threshold's flags are mostly for resting below 60 bpm, not for anything wrong. PulseGuard learned this subject's baseline and didn't flag those readings. PPG-DaLiA has no anomaly labels, so no precision or recall is reported for it.

### SpO2 monitor

All 7 scenario tests passed, including the 3 that should stay silent (steady reading, brief motion dip, detached sensor) and the same 89-91% reading judged under standard thresholds (moderate) and the COPD profile (normal). This validates the decision logic, not clinical accuracy.

### Live hardware (ESP32 + MAX30102)

- About 4 minutes, 230 readings
- Learned baseline: **84.7 ± 5.9 bpm**
- Firmware flagged 13 readings as anomalous; PulseGuard flagged one as significant: **107 bpm, 3.8σ above baseline, MODERATE**
- That event triggered a **real SMS** to a registered caregiver number through Twilio

---

## Limitations

- Severity thresholds are reasonable starting values, not yet statistically validated on a large, diverse, elderly patient set.
- Dataset PPG is cleaner than a live fingertip sensor, so live performance may differ. The hardware test was a single short session.
- The 300 s freeze bound was inherited from the emergency-duration threshold and tested on one real recording, not swept across subjects.
- Live SpO2 is not yet routed through the SpO2 monitor, so it is validated on scenario logic only.
- PulseGuard is a research prototype, **not a medical device**.

---

## Future Work

- Tune and validate thresholds on larger, elderly-specific recordings
- Sweep the freeze bound across multiple subjects
- Route live SpO2 through the SpO2 monitor and run longer live sessions
- Test a lightweight adaptive ML baseline, if it keeps explainability
- Run a caregiver study on whether the explanations are clear and useful

---

## Author

**Kaavya Varshini P**, M.Tech (Integrated) CSE (Business Analytics), VIT Chennai

[GitHub](https://github.com/Kaavyavarshini) · [LinkedIn](https://linkedin.com/in/kaavya-varshini-4b347025b)