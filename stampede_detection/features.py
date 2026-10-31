"""
features.py
-----------
Non-ML analytics that sit on top of the real detectors:

    * Fusion Risk Engine       (combines real module readings)
    * Predictive Forecaster    (EMA + linear trend)
    * Weather / Event Context  (multiplier)
    * Evacuation Route Planner
    * Venue zones + status
    * Historical KPIs
    * Config coercion
    * Incident report renderer
    * Lightweight auth gate
    * SVG gauge helper
"""

from __future__ import annotations

import io
import math
import base64
import hashlib
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Thresholds + fusion weights
# ---------------------------------------------------------------------------
DEFAULT_THRESHOLDS: Dict[str, float] = {
    "VIDEO_COUNT_THRESHOLD":       150.0,
    "MOBILE_CLUSTER_THRESHOLD":    120.0,
    "SATELLITE_DENSITY_THRESHOLD": 4.0,
    "CCTV_VELOCITY_THRESHOLD":     0.15,
    "AUDIO_ANOMALY_THRESHOLD":     0.70,
    "FUSION_RISK_CRITICAL":        75.0,
    "FUSION_RISK_WARNING":         45.0,
}

FUSION_WEIGHTS: Dict[str, float] = {
    "video":     0.25,
    "mobile":    0.20,
    "satellite": 0.20,
    "cctv":      0.20,
    "audio":     0.15,
}

RISK_PALETTE = {
    "safe":     "#16a34a",
    "elevated": "#f59e0b",
    "critical": "#dc2626",
    "unknown":  "#6b7280",
}


# ---------------------------------------------------------------------------
# Fusion Risk Engine
# ---------------------------------------------------------------------------
@dataclass
class ModuleReading:
    module: str
    value: float
    normalized: float
    threshold: float
    contribution: float = 0.0


def normalize_reading(module: str, value: float, threshold: float) -> float:
    if threshold <= 0:
        return 0.0
    if module == "cctv":
        v = max(0.0, min(1.0, 1.0 - value / (threshold * 2)))
        return v
    return max(0.0, min(1.0, value / (threshold * 1.5)))


def compute_fusion_risk(
    readings: Dict[str, float],
    thresholds: Dict[str, float],
    context_multiplier: float = 1.0,
) -> Tuple[float, List[ModuleReading]]:
    m2t = {
        "video":     thresholds["VIDEO_COUNT_THRESHOLD"],
        "mobile":    thresholds["MOBILE_CLUSTER_THRESHOLD"],
        "satellite": thresholds["SATELLITE_DENSITY_THRESHOLD"],
        "cctv":      thresholds["CCTV_VELOCITY_THRESHOLD"],
        "audio":     thresholds["AUDIO_ANOMALY_THRESHOLD"],
    }
    breakdown: List[ModuleReading] = []
    total = 0.0
    for m, w in FUSION_WEIGHTS.items():
        raw = float(readings.get(m, 0.0))
        n = normalize_reading(m, raw, m2t[m])
        c = n * w
        total += c
        breakdown.append(ModuleReading(m, raw, n, m2t[m], c))
    score = min(100.0, max(0.0, total * 100.0 * context_multiplier))
    return score, breakdown


def risk_band(score: float, thresholds: Dict[str, float]) -> Tuple[str, str]:
    if score >= thresholds["FUSION_RISK_CRITICAL"]:
        return "CRITICAL", RISK_PALETTE["critical"]
    if score >= thresholds["FUSION_RISK_WARNING"]:
        return "ELEVATED", RISK_PALETTE["elevated"]
    return "SAFE", RISK_PALETTE["safe"]


# ---------------------------------------------------------------------------
# Predictive forecaster
# ---------------------------------------------------------------------------
def forecast_next_values(history: List[float], horizon: int = 15,
                         alpha: float = 0.35) -> List[float]:
    if len(history) < 4:
        return []
    h = np.asarray(history[-30:], dtype=float)
    ema = h[0]
    for v in h[1:]:
        ema = alpha * v + (1 - alpha) * ema
    x = np.arange(len(h))
    slope, intercept = np.polyfit(x, h, 1)
    fc = []
    for k in range(1, horizon + 1):
        blended = 0.6 * (ema + slope * k) + 0.4 * (intercept + slope * (len(h) + k))
        fc.append(max(0.0, float(blended)))
    return fc


def time_to_threshold(history: List[float], threshold: float,
                      horizon: int = 60) -> Optional[int]:
    for i, v in enumerate(forecast_next_values(history, horizon), 1):
        if v >= threshold:
            return i
    return None


# ---------------------------------------------------------------------------
# Weather / event context
# ---------------------------------------------------------------------------
@dataclass
class ContextInfo:
    condition: str
    temp_c: float
    humidity: float
    wind_kmh: float
    event_type: str
    expected_attendance: int
    multiplier: float


def build_context(condition: str = "Clear", temp_c: float = 24.0,
                  humidity: float = 55.0, wind_kmh: float = 8.0,
                  event_type: str = "Concert",
                  expected_attendance: int = 8000) -> ContextInfo:
    m = 1.0
    if condition.lower() in ("rain", "thunderstorm", "snow"):
        m *= 1.15
    if temp_c >= 32 or temp_c <= 2:
        m *= 1.10
    if humidity >= 80:
        m *= 1.05
    if wind_kmh >= 40:
        m *= 1.05
    if event_type.lower() in ("concert", "festival", "protest", "sports final"):
        m *= 1.10
    if expected_attendance >= 20000:
        m *= 1.10
    elif expected_attendance >= 8000:
        m *= 1.05
    m = min(m, 1.4)
    return ContextInfo(condition, temp_c, humidity, wind_kmh, event_type,
                       expected_attendance, round(m, 3))


# ---------------------------------------------------------------------------
# Evacuation planner
# ---------------------------------------------------------------------------
@dataclass
class Exit:
    name: str
    x: float
    y: float
    capacity_per_min: int = 800


DEFAULT_EXITS: List[Exit] = [
    Exit("Gate A (North)", 50, 95, 900),
    Exit("Gate B (East)",  95, 50, 700),
    Exit("Gate C (South)", 50, 5,  900),
    Exit("Gate D (West)",  5,  50, 600),
]


def recommend_exits(hotspots: List[Tuple[float, float, float]],
                    exits: Optional[List[Exit]] = None) -> List[Dict]:
    exits = exits or DEFAULT_EXITS
    ranked = []
    for ex in exits:
        danger = 0.0
        for (hx, hy, hi) in hotspots:
            d = math.hypot(ex.x - hx, ex.y - hy) + 1e-3
            danger += hi / d
        safety = ex.capacity_per_min / (1.0 + danger)
        ranked.append({
            "exit": ex.name, "x": ex.x, "y": ex.y,
            "capacity_per_min": ex.capacity_per_min,
            "danger": round(danger, 3),
            "safety_score": round(safety, 2),
        })
    ranked.sort(key=lambda r: r["safety_score"], reverse=True)
    for i, r in enumerate(ranked):
        r["rank"] = i + 1
        r["recommendation"] = (
            "PRIMARY" if i == 0 else
            "SECONDARY" if i == 1 else
            "AVOID" if r["danger"] > 15 else "STANDBY"
        )
    return ranked


# ---------------------------------------------------------------------------
# Analytics KPIs
# ---------------------------------------------------------------------------
def alert_kpis(df: pd.DataFrame) -> Dict[str, float]:
    if df is None or df.empty:
        return dict(total=0, critical=0, warning=0,
                    modules_involved=0, avg_per_hour=0.0, mttr_seconds=0.0)
    ts = pd.to_datetime(df["timestamp"].str.replace(" UTC", "", regex=False),
                        errors="coerce").dropna()
    dur_h = max(1.0, (ts.max() - ts.min()).total_seconds() / 3600.0) if len(ts) > 1 else 1.0
    crit_ts = ts[df["severity"].str.upper() == "CRITICAL"] if "severity" in df else ts
    mttr = float(crit_ts.sort_values().diff().dropna().dt.total_seconds().mean()) \
           if len(crit_ts) > 1 else 0.0
    return dict(
        total=int(len(df)),
        critical=int((df["severity"].str.upper() == "CRITICAL").sum()),
        warning=int((df["severity"].str.upper() == "WARNING").sum()),
        modules_involved=int(df["module"].nunique()),
        avg_per_hour=round(len(df) / dur_h, 2),
        mttr_seconds=round(mttr, 1),
    )


# ---------------------------------------------------------------------------
# Config coercion
# ---------------------------------------------------------------------------
def coerce_thresholds(user_input: Dict[str, float]) -> Dict[str, float]:
    out = dict(DEFAULT_THRESHOLDS)
    for k, v in (user_input or {}).items():
        try:
            out[k] = float(v)
        except Exception:
            pass
    return out


# ---------------------------------------------------------------------------
# Incident report (self-contained HTML)
# ---------------------------------------------------------------------------
INCIDENT_HTML = """<!doctype html>
<html><head><meta charset="utf-8"><title>Stampede Incident Report</title>
<style>
 body{{font-family:-apple-system,Segoe UI,Roboto,sans-serif;background:#f8fafc;color:#111827;margin:0;padding:32px;}}
 h1{{color:#111827;border-bottom:3px solid #dc2626;padding-bottom:10px;}}
 h2{{color:#0369a1;margin-top:28px;}}
 .card{{background:#ffffff;border:1px solid #e5e7eb;border-radius:12px;padding:18px;margin:10px 0;box-shadow:0 1px 3px rgba(0,0,0,0.06);}}
 .critical{{color:#dc2626;font-weight:800;}}
 .warning{{color:#d97706;font-weight:800;}}
 .safe{{color:#16a34a;font-weight:800;}}
 table{{width:100%;border-collapse:collapse;margin-top:8px;}}
 th,td{{border:1px solid #e5e7eb;padding:8px;text-align:left;font-size:13px;color:#111827;}}
 th{{background:#f1f5f9;color:#0f172a;}}
 .footer{{color:#6b7280;font-size:12px;margin-top:32px;}}
</style></head><body>
<h1>Stampede Incident Report</h1>
<div class="card">
  <b>Report ID:</b> {report_id}<br/>
  <b>Generated:</b> {generated_at}<br/>
  <b>Operator:</b> {operator}<br/>
  <b>Venue:</b> {venue}
</div>
<h2>Executive Summary</h2>
<div class="card">
  <p>Peak fusion risk score: <span class="{band_class}">{peak_risk:.1f} / 100 ({band})</span></p>
  <p>Total alerts dispatched: <b>{total_alerts}</b> ({critical} critical, {warning} warning)</p>
  <p>Modules involved: {modules_involved}</p>
  <p>Context: {ctx_event} · {ctx_condition} · {ctx_temp}°C · {ctx_attendance} attendees</p>
</div>
<h2>Alert Timeline</h2>
<div class="card">{alerts_table}</div>
<h2>Recommended Actions</h2>
<div class="card"><ul>{actions}</ul></div>
<div class="footer">Multi-Modal Stampede Detection Dashboard — v3 (real backends).</div>
</body></html>
"""


def render_incident_report(alerts_df: pd.DataFrame, peak_risk: float,
                           thresholds: Dict[str, float], context: ContextInfo,
                           operator: str = "on-duty",
                           venue: str = "Central Plaza") -> str:
    k = alert_kpis(alerts_df)
    band, _ = risk_band(peak_risk, thresholds)
    bc = {"CRITICAL": "critical", "ELEVATED": "warning", "SAFE": "safe"}[band]
    if alerts_df is None or alerts_df.empty:
        rows = "<i>No alerts recorded in this window.</i>"
    else:
        rows = "<table><tr><th>Time</th><th>Module</th><th>Severity</th><th>Message</th></tr>"
        for _, r in alerts_df.tail(50).iterrows():
            sc = {"CRITICAL": "critical", "WARNING": "warning"}.get(
                str(r.get("severity", "")).upper(), "safe")
            rows += (f"<tr><td>{r.get('timestamp','')}</td>"
                     f"<td>{r.get('module','')}</td>"
                     f"<td class='{sc}'>{r.get('severity','')}</td>"
                     f"<td>{r.get('message','')}</td></tr>")
        rows += "</table>"

    action_map = {
        "CRITICAL": [
            "Immediately open all primary exits and halt inbound flow.",
            "Deploy medical teams to the highest-density zone.",
            "Broadcast evacuation instructions on the PA system.",
            "Notify local emergency services and prepare a staging area.",
        ],
        "ELEVATED": [
            "Increase steward presence in identified hotspots.",
            "Slow inbound admissions; pre-position medical staff.",
            "Prepare — but do not yet open — secondary exits.",
        ],
        "SAFE": [
            "Continue routine monitoring at 1-minute cadence.",
            "Review threshold configuration for the next event tier.",
        ],
    }
    actions = "".join(f"<li>{a}</li>" for a in action_map[band])
    rid = hashlib.sha1(f"{datetime.utcnow().isoformat()}-{peak_risk}"
                       .encode()).hexdigest()[:10].upper()
    return INCIDENT_HTML.format(
        report_id=rid,
        generated_at=datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
        operator=operator, venue=venue,
        peak_risk=peak_risk, band=band, band_class=bc,
        total_alerts=k["total"], critical=k["critical"], warning=k["warning"],
        modules_involved=k["modules_involved"],
        ctx_event=context.event_type, ctx_condition=context.condition,
        ctx_temp=context.temp_c,
        ctx_attendance=f"{context.expected_attendance:,}",
        alerts_table=rows, actions=actions,
    )


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------
_VALID = {
    "operator": hashlib.sha256(b"stampede2026").hexdigest(),
    "admin":    hashlib.sha256(b"admin123").hexdigest(),
    "demo":     hashlib.sha256(b"demo").hexdigest(),
}


def verify_credentials(u: str, p: str) -> bool:
    if not u or not p:
        return False
    return _VALID.get(u.strip().lower()) == hashlib.sha256(p.encode()).hexdigest()


# ---------------------------------------------------------------------------
# SVG gauge (embedded so we do not need extra JS libs)
# ---------------------------------------------------------------------------
def gauge_svg_dataurl(score: float, color: str) -> str:
    score = max(0.0, min(100.0, float(score)))
    angle = math.radians(180 * (score / 100.0))
    r = 90
    cx, cy = 110, 110
    x = cx - r * math.cos(angle)
    y = cy - r * math.sin(angle)
    svg = f"""
<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 220 130' width='260'>
  <path d='M20 110 A90 90 0 0 1 200 110' fill='none' stroke='#e5e7eb' stroke-width='18'/>
  <path d='M20 110 A90 90 0 0 1 {x:.1f} {y:.1f}' fill='none'
        stroke='{color}' stroke-width='18' stroke-linecap='round'/>
  <text x='110' y='95' fill='#0f172a' font-size='32' font-family='Inter,Arial'
        font-weight='800' text-anchor='middle'>{score:.0f}</text>
  <text x='110' y='120' fill='#475569' font-size='12' font-family='Inter,Arial'
        font-weight='600' text-anchor='middle'>Fusion Risk / 100</text>
</svg>"""
    b64 = base64.b64encode(svg.encode()).decode()
    return f"data:image/svg+xml;base64,{b64}"
