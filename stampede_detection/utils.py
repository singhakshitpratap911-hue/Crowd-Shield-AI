"""
utils.py
--------
Shared utilities: Twilio alert log, severity helpers, formatting.
All heavy detection logic lives in detectors.py + models/*.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import List

import pandas as pd


# ---------------------------------------------------------------------------
# Alert log
# ---------------------------------------------------------------------------
@dataclass
class TwilioAlert:
    timestamp: str
    module: str
    severity: str
    message: str
    recipient: str = ""
    sid: str = ""
    mode: str = "mock"          # "real" | "mock" | "error"


def append_alert(log: List[TwilioAlert], module: str, severity: str,
                 message: str, recipient: str = "",
                 sid: str = "", mode: str = "mock") -> TwilioAlert:
    alert = TwilioAlert(
        timestamp=datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
        module=module, severity=severity, message=message,
        recipient=recipient, sid=sid, mode=mode,
    )
    log.append(alert)
    if len(log) > 500:
        del log[0: len(log) - 500]
    return alert


def alerts_to_dataframe(log: List[TwilioAlert]) -> pd.DataFrame:
    if not log:
        return pd.DataFrame(columns=[
            "timestamp", "module", "severity", "message",
            "recipient", "sid", "mode",
        ])
    return pd.DataFrame([a.__dict__ for a in log])


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------
def severity_color(sev: str) -> str:
    return {
        "INFO":     "#0284c7",
        "WARNING":  "#d97706",
        "CRITICAL": "#dc2626",
    }.get(sev.upper(), "#6b7280")


def format_metric_delta(value: float, threshold: float) -> str:
    diff = value - threshold
    sign = "+" if diff >= 0 else ""
    return f"{sign}{diff:.1f} vs threshold"
