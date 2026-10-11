"""
PulseGuard - Alert & Notification Layer
------------------------------------------------------------
Sends caregiver notifications when a MODERATE or EMERGENCY anomaly
is detected. This module is written provider-agnostic: it defines a
clean interface (NotificationChannel) with a console implementation
you can run right now with zero credentials, plus ready-to-fill-in
Twilio (SMS) and SMTP (email) implementations for when real hardware
and API keys are set up post-review.
"""

from abc import ABC, abstractmethod
from baseline_engine import AnomalyEvent, Severity


class NotificationChannel(ABC):
    @abstractmethod
    def send(self, caregiver_contact: str, message: str) -> bool:
        """Return True if the notification was sent successfully."""
        raise NotImplementedError


class ConsoleNotificationChannel(NotificationChannel):
    """
    Zero-setup channel for demoing and testing the alert pipeline
    without any SMS/email credentials. Prints exactly what a real
    caregiver message would contain.
    """

    def send(self, caregiver_contact: str, message: str) -> bool:
        print(f"\n[ALERT SENT to {caregiver_contact}]")
        print(f"  {message}\n")
        return True


class TwilioSMSChannel(NotificationChannel):
    """
    Real SMS channel using Twilio. Requires:
        pip install twilio
    and environment variables / config for:
        TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_FROM_NUMBER
    Left unimplemented on purpose until credentials exist -- swap in
    for ConsoleNotificationChannel once hardware/API setup is done.
    """

    def __init__(self, account_sid: str, auth_token: str, from_number: str):
        self.account_sid = account_sid
        self.auth_token = auth_token
        self.from_number = from_number

    def send(self, caregiver_contact: str, message: str) -> bool:
        try:
            from twilio.rest import Client  # imported lazily
            client = Client(self.account_sid, self.auth_token)
            client.messages.create(
                body=message, from_=self.from_number, to=caregiver_contact
            )
            return True
        except Exception as exc:  # pragma: no cover
            print(f"[SMS send failed] {exc}")
            return False


class SMTPEmailChannel(NotificationChannel):
    """
    Real email channel using standard SMTP (e.g. Gmail SMTP,
    SendGrid, etc). Fill in smtp_host/port/credentials when ready.
    """

    def __init__(self, smtp_host: str, smtp_port: int, username: str, password: str):
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
        self.username = username
        self.password = password

    def send(self, caregiver_contact: str, message: str) -> bool:
        import smtplib
        from email.mime.text import MIMEText

        try:
            msg = MIMEText(message)
            msg["Subject"] = "PulseGuard Alert"
            msg["From"] = self.username
            msg["To"] = caregiver_contact

            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                server.starttls()
                server.login(self.username, self.password)
                server.sendmail(self.username, [caregiver_contact], msg.as_string())
            return True
        except Exception as exc:  # pragma: no cover
            print(f"[Email send failed] {exc}")
            return False


class AlertManager:
    """
    Decides WHETHER to alert (only MODERATE/EMERGENCY -- MILD is
    logged but does not page a caregiver, to avoid alert fatigue)
    and dispatches through one or more channels.
    """

    ALERT_WORTHY = {Severity.MODERATE, Severity.EMERGENCY}

    def __init__(self, channels: list[NotificationChannel], caregiver_contact: str):
        self.channels = channels
        self.caregiver_contact = caregiver_contact
        self.log: list[AnomalyEvent] = []

    def handle_event(self, event: AnomalyEvent) -> None:
        self.log.append(event)

        if event.severity not in self.ALERT_WORTHY:
            print(f"[Logged, not alerted - MILD] {event.explanation}")
            return

        message = (
            f"PulseGuard {event.severity.value.upper()} alert "
            f"at {event.timestamp.strftime('%H:%M:%S')}: {event.explanation}"
        )
        for channel in self.channels:
            channel.send(self.caregiver_contact, message)
