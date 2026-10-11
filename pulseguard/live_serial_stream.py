"""
PulseGuard - Live Hardware Stream (ESP32 + MAX30102 over Serial)
------------------------------------------------------------------
Reads the ESP32's normal-mode serial output (DEBUG_RAW_SIGNAL = false
in the firmware) line by line and feeds each HR reading into the same
PulseGuardEngine + AlertManager pipeline already validated on
simulated data and MIT-BIH (see simulate_and_compare.py and
load_mitbih_stream.py) -- this is the "no real sensor required, but
works unchanged once you plug one in" promise from baseline_engine.py
actually being kept.

EXPECTED FIRMWARE OUTPUT FORMAT (normal mode, DEBUG_RAW_SIGNAL = false):
    Time(s)    Heart Rate    SpO2    Status
    ----------------------------------------
    190        84            100     NORMAL
    191        97            100     NORMAL
    ...
Each data row is 4 whitespace-separated fields: elapsed seconds,
integer bpm, SpO2 percent, and the firmware's own fixed-threshold
status (which we ignore -- PulseGuard's own personalized classifier
replaces it).

SETUP (run once):
    pip install pyserial --break-system-packages
    (drop --break-system-packages if not needed on your machine)

RUN:
    python3 live_serial_stream.py COM8
    python3 live_serial_stream.py COM8 --baud 115200
    python3 live_serial_stream.py /dev/ttyUSB0     (Linux/Mac)

Ctrl+C to stop -- prints a summary in the same style as
load_mitbih_stream.py before exiting.
"""

import argparse
import re
import sys
from datetime import datetime, timedelta

import serial

from baseline_engine import PulseGuardEngine, Reading, Severity
from alert_manager import AlertManager, ConsoleNotificationChannel

# Matches a normal-mode data row, e.g. "190        84            100     NORMAL"
# Tolerant of the exact column widths since snprintf's "%-Ns" padding
# can shift slightly between firmware revisions.
ROW_PATTERN = re.compile(
    r"^\s*(\d+)\s+(\d+)\s+(\d+)\s+(NORMAL|ANOMALY)\s*$"
)


def parse_line(line: str):
    """
    Returns (elapsed_seconds, heart_rate, spo2, firmware_status) if the
    line is a valid data row, otherwise None (headers, dashes, blank
    lines, and boot/debug prints are all silently skipped -- this is
    the same "just skip what doesn't parse" approach as filtering
    non-beat markers in load_mitbih_stream.py).
    """
    match = ROW_PATTERN.match(line)
    if not match:
        return None
    elapsed_s, hr, spo2, status = match.groups()
    return int(elapsed_s), float(hr), float(spo2), status


def stream_from_serial(port: str, baud: int, engine: PulseGuardEngine,
                        alert_mgr: AlertManager):
    """
    Opens the serial port and processes rows as they arrive, forever,
    until interrupted. Wall-clock time is used for each Reading's
    timestamp rather than the firmware's elapsed-seconds column, since
    that stays consistent even if the ESP32 resets mid-run.
    """
    events = []

    print(f"Opening {port} at {baud} baud... (Ctrl+C to stop)\n")
    with serial.Serial(port, baud, timeout=2) as ser:
        # Let the ESP32 finish its boot/reset sequence after the port
        # opens (this can happen automatically on some boards) before
        # we start trying to parse lines.
        ser.reset_input_buffer()

        while True:
            try:
                raw = ser.readline()
            except KeyboardInterrupt:
                break
            if not raw:
                continue  # read timeout, just keep listening

            line = raw.decode(errors="replace").rstrip()
            parsed = parse_line(line)
            if parsed is None:
                # Not a data row (header, dashes, boot message, or a
                # DEBUG_RAW_SIGNAL row if debug mode is still on) --
                # echo it so you can see what's coming through.
                if line.strip():
                    print(f"  (skipped) {line}")
                continue

            elapsed_s, hr, spo2, fw_status = parsed
            reading = Reading(timestamp=datetime.now(), heart_rate=hr)

            print(f"[{elapsed_s:>4}s] HR={hr:.0f} SpO2={spo2:.0f}% "
                  f"firmware={fw_status}", end="")

            if not engine.baseline.is_ready:
                remaining = engine.baseline.min_readings - len(engine.baseline._history)
                print(f"  (learning baseline, {max(remaining, 0)} readings to go)")
                continue

            event = engine.process_reading(reading)
            if event:
                events.append(event)
                print(f"  -> {event.severity.value.upper()}")
                alert_mgr.handle_event(event)
            else:
                print("  -> normal (personal baseline)")

    return events


def print_summary(events):
    by_severity = {}
    for e in events:
        by_severity[e.severity] = by_severity.get(e.severity, 0) + 1

    print("\n" + "=" * 70)
    print("LIVE HARDWARE SESSION SUMMARY")
    print("=" * 70)
    print(f"Total flagged events : {len(events)}")
    for sev in [Severity.MILD, Severity.MODERATE, Severity.EMERGENCY]:
        print(f"  {sev.value:10s}: {by_severity.get(sev, 0)}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Stream ESP32 PulseGuard serial output into the PulseGuardEngine pipeline."
    )
    parser.add_argument("port", help="Serial port, e.g. COM8 or /dev/ttyUSB0")
    parser.add_argument("--baud", type=int, default=115200,
                         help="Baud rate (must match Serial.begin() in the firmware; default 115200)")
    parser.add_argument("--window", type=int, default=200,
                         help="Baseline rolling window size (default 200, matches load_mitbih_stream.py)")
    parser.add_argument("--min-readings", type=int, default=50,
                         help="Readings required before the personal baseline is trusted (default 50)")
    parser.add_argument("--caregiver", default="+91-XXXXXXXXXX (registered caregiver)",
                         help="Caregiver contact string passed to AlertManager")
    args = parser.parse_args()

    engine = PulseGuardEngine(window_size=args.window, min_readings=args.min_readings)
    alert_mgr = AlertManager(
        channels=[ConsoleNotificationChannel()],
        caregiver_contact=args.caregiver,
    )

    events = []
    try:
        events = stream_from_serial(args.port, args.baud, engine, alert_mgr)
    except serial.SerialException as exc:
        print(f"\nCould not open {args.port}: {exc}")
        print("Check the port name in Arduino IDE (Tools > Port) and make sure")
        print("the Serial Monitor there is closed -- only one program can hold")
        print("the port at a time.")
        sys.exit(1)
    except KeyboardInterrupt:
        pass
    finally:
        print_summary(events)
        print(f"Learned baseline (mean): {engine.baseline.mean:.1f} bpm")
        print(f"Learned baseline (SD)  : {engine.baseline.std_dev:.1f} bpm")


if __name__ == "__main__":
    main()
