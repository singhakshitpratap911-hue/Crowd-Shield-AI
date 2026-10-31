"""
app.py
------
Multi-Modal Stampede Detection & Crowd Monitoring Dashboard — v3

REAL AI backends (YOLOv8, CSRNet, P2PNet), REAL user data, REAL Twilio SMS.
Light-theme UI with WCAG-AA text contrast.
"""

from __future__ import annotations

import io
import os
import time
import tempfile
from datetime import datetime
from typing import List

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image

# Load .env for Twilio credentials
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

import detectors
import features
from features import (
    DEFAULT_THRESHOLDS, FUSION_WEIGHTS, RISK_PALETTE,
    compute_fusion_risk, risk_band,
    forecast_next_values, time_to_threshold,
    build_context, recommend_exits, coerce_thresholds,
    render_incident_report, verify_credentials, gauge_svg_dataurl,
    alert_kpis,
)
from utils import (
    TwilioAlert, append_alert, alerts_to_dataframe,
    severity_color, format_metric_delta,
)


# ---------------------------------------------------------------------------
# Page config + high-contrast light CSS
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Crowd Shield AI",
    page_icon="🛑",
    layout="wide",
    initial_sidebar_state="expanded",
)

CSS = """
<style>
:root {
  --bg:        #f8fafc;
  --panel:     #ffffff;
  --panel-2:   #f1f5f9;
  --text:      #0f172a;
  --text-soft: #334155;
  --muted:     #64748b;
  --border:    #cbd5e1;
  --primary:   #0369a1;
  --primary-2: #075985;
  --danger:    #dc2626;
  --warn:      #d97706;
  --ok:        #16a34a;
}

/* Base */
html, body, [class*="css"] {
  background: var(--bg) !important;
  color: var(--text) !important;
  font-family: -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
}

/* Headings — dark and readable */
h1, h2, h3, h4, h5, h6 {
  color: var(--text) !important;
  font-weight: 700 !important;
  letter-spacing: -0.01em;
}

/* All markdown paragraphs — force dark text */
.stMarkdown, .stMarkdown p, .stMarkdown li, .stMarkdown span,
.stCaption, .stText, label, .stRadio label, .stSelectbox label,
.stSlider label, .stCheckbox label, .stNumberInput label,
.stTextInput label, .stFileUploader label, .stDateInput label {
  color: var(--text) !important;
}
p, li { color: var(--text-soft) !important; font-size: 15px; }
small, .stCaption p { color: var(--muted) !important; }

/* Sidebar */
section[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #ffffff 0%, #f1f5f9 100%);
  border-right: 1px solid var(--border);
}
section[data-testid="stSidebar"] * { color: var(--text) !important; }

/* Metric cards */
div[data-testid="stMetric"] {
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 14px 18px;
  box-shadow: 0 2px 6px rgba(15,23,42,0.06);
}
div[data-testid="stMetricValue"] {
  color: var(--text) !important;
  font-weight: 800 !important;
  font-size: 30px !important;
}
div[data-testid="stMetricLabel"] {
  color: var(--muted) !important;
  text-transform: uppercase;
  letter-spacing: 1px;
  font-size: 12px !important;
  font-weight: 700 !important;
}
div[data-testid="stMetricDelta"] { color: var(--text-soft) !important; }

/* Alert banners */
.alert-critical {
  background: #fee2e2;
  color: #7f1d1d;
  padding: 14px 18px;
  border-left: 6px solid var(--danger);
  border-radius: 10px;
  font-weight: 700;
  font-size: 15px;
  margin: 10px 0;
  animation: flash 1.2s infinite alternate;
}
.alert-warning {
  background: #fef3c7;
  color: #78350f;
  padding: 12px 16px;
  border-left: 6px solid var(--warn);
  border-radius: 10px;
  font-weight: 600;
  margin: 10px 0;
}
.alert-ok {
  background: #dcfce7;
  color: #14532d;
  padding: 12px 16px;
  border-left: 6px solid var(--ok);
  border-radius: 10px;
  font-weight: 600;
  margin: 10px 0;
}
@keyframes flash {
  from { box-shadow: 0 0 0 rgba(220,38,38,0.0); }
  to   { box-shadow: 0 0 18px rgba(220,38,38,0.55); }
}

/* Log rows */
.log-row {
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 10px 12px;
  margin-bottom: 6px;
  font-family: "SF Mono", Menlo, Consolas, monospace;
  font-size: 13px;
  color: var(--text);
}
.log-time { color: var(--muted); }
.log-tag  {
  padding: 2px 8px;
  border-radius: 6px;
  font-size: 11px;
  font-weight: 800;
  color: white;
  margin-right: 8px;
}
.mode-badge {
  padding: 2px 6px;
  border-radius: 4px;
  font-size: 10px;
  font-weight: 700;
  margin-left: 6px;
}
.mode-real { background: #16a34a; color: white; }
.mode-mock { background: #64748b; color: white; }
.mode-error { background: #dc2626; color: white; }

/* Buttons — strong primary */
.stButton > button {
  background: linear-gradient(90deg, var(--primary), var(--primary-2));
  color: white !important;
  border: none;
  border-radius: 10px;
  padding: 10px 20px;
  font-weight: 700;
  font-size: 14px;
  box-shadow: 0 2px 6px rgba(3,105,161,0.25);
}
.stButton > button:hover { filter: brightness(1.08); }

/* Fusion hero card */
.fusion-card {
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 22px;
  box-shadow: 0 4px 14px rgba(15,23,42,0.08);
}
.fusion-band-label {
  font-size: 22px;
  font-weight: 800;
  margin-top: 6px;
}

/* Model badge */
.model-badge {
  display: inline-block;
  padding: 4px 10px;
  border-radius: 999px;
  background: #e0f2fe;
  color: #075985;
  font-weight: 700;
  font-size: 12px;
  margin-right: 6px;
}
.gpu-badge { background: #dcfce7; color: #14532d; }
.cpu-badge { background: #fef3c7; color: #78350f; }

/* Tables */
[data-testid="stDataFrame"] * { color: var(--text) !important; }

/* Hide streamlit chrome */
footer, #MainMenu { visibility: hidden; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
def _init():
    ss = st.session_state
    ss.setdefault("auth", False)
    ss.setdefault("operator", "")
    ss.setdefault("thresholds", dict(DEFAULT_THRESHOLDS))
    ss.setdefault("alert_log", [])
    ss.setdefault("video_history", [])
    ss.setdefault("last_readings", {
        "video": 0.0, "mobile": 0.0, "satellite": 0.0,
        "cctv": 1.0, "audio": 0.0,
    })
    ss.setdefault("peak_risk", 0.0)
    ss.setdefault("context", build_context())
    ss.setdefault("recipient", os.environ.get("EMERGENCY_RECIPIENT", ""))

_init()


# ---------------------------------------------------------------------------
# Login gate
# ---------------------------------------------------------------------------
def render_login():
    st.title("🛑 Crowd Shield AI")
    st.caption("Secure operator console. Sign in to load the AI dashboard.")
    with st.form("login"):
        c1, c2 = st.columns(2)
        u = c1.text_input("Username", value="demo")
        p = c2.text_input("Password", value="demo", type="password")
        submitted = st.form_submit_button("🔓 Sign in", type="primary")
    if submitted:
        if verify_credentials(u, p):
            st.session_state.auth = True
            st.session_state.operator = u.strip().lower()
            st.rerun()
        else:
            st.error("Invalid credentials.")
    with st.expander("Demo credentials"):
        st.code("demo / demo\noperator / stampede2026\nadmin / admin123")


if not st.session_state.auth:
    render_login()
    st.stop()


# ---------------------------------------------------------------------------
# Helper: banner + reading recorder
# ---------------------------------------------------------------------------
thr = st.session_state.thresholds


def banner(kind: str, text: str):
    cls = {"critical": "alert-critical", "warning": "alert-warning",
           "ok": "alert-ok"}[kind]
    icon = {"critical": "🚨", "warning": "⚠️", "ok": "✅"}[kind]
    st.markdown(f'<div class="{cls}">{icon} {text}</div>',
                unsafe_allow_html=True)


def record_reading(module: str, value: float):
    st.session_state.last_readings[module] = float(value)
    score, _ = compute_fusion_risk(
        st.session_state.last_readings, thr,
        st.session_state.context.multiplier)
    st.session_state.peak_risk = max(st.session_state.peak_risk, score)


def dispatch_alert(module: str, severity: str, message: str,
                   send_sms: bool = True):
    """Log an alert and — if SMS enabled AND severity=CRITICAL — send real Twilio SMS."""
    result = {"mode": "mock", "sid": ""}
    if send_sms and severity == "CRITICAL" and st.session_state.recipient:
        result = detectors.send_real_twilio_sms(
            message=f"[{module}] {message}",
            to_number=st.session_state.recipient,
        )
    append_alert(
        st.session_state.alert_log,
        module=module, severity=severity, message=message,
        recipient=st.session_state.recipient,
        sid=result.get("sid") or "",
        mode=result.get("mode", "mock"),
    )


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
st.sidebar.markdown(
    f"""
    <div style="text-align:center; padding: 8px 0 14px 0;">
      <h2 style="margin:0;">🛑 Crowd Shield AI</h2>
      <p style="color:#64748b; margin:2px 0; font-size:12px;">
        v3 · Real AI Backends
      </p>
      <p style="color:#0369a1; margin:4px 0; font-size:12px; font-weight:700;">
        👤 {st.session_state.operator}
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# GPU indicator
try:
    import torch
    gpu_ok = torch.cuda.is_available()
    dev_name = torch.cuda.get_device_name(0) if gpu_ok else "CPU"
    st.sidebar.markdown(
        f'<div style="text-align:center;margin-bottom:10px;">'
        f'<span class="model-badge {"gpu-badge" if gpu_ok else "cpu-badge"}">'
        f'{"⚡ GPU" if gpu_ok else "🖥️ CPU"}: {dev_name}</span></div>',
        unsafe_allow_html=True,
    )
except Exception:
    pass

module_choice = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Overview",
        "🔮 Fusion Risk",
        "🗺️ Live Map",
        "🎥 Video Detection",
        "📹 CCTV Optical Flow",
        "🛰️ Satellite / Aerial",
        "📡 Mobile Network",
        "🔊 Audio Anomaly",
        "📈 Predictive Forecast",
        "🚪 Evacuation Planner",
        "📊 Analytics",
        "⚙️ Config",
        "📨 Alert Log",
        "📄 Incident Report",
    ],
    index=0,
)

st.sidebar.markdown("---")
if st.sidebar.button("🧹 Clear alerts + readings"):
    st.session_state.alert_log.clear()
    st.session_state.video_history.clear()
    st.session_state.last_readings = {"video": 0, "mobile": 0,
                                       "satellite": 0, "cctv": 1, "audio": 0}
    st.session_state.peak_risk = 0.0
    st.rerun()
if st.sidebar.button("🚪 Sign out"):
    st.session_state.auth = False
    st.rerun()


# ===========================================================================
# 🏠 OVERVIEW
# ===========================================================================
def page_overview():
    st.title("🛑 Multi-Modal Stampede Detection")
    st.markdown(
        "Real-time crowd monitoring powered by **three AI backends** — "
        "YOLOv8 (Ultralytics), CSRNet (VGG-16 dilated), and P2PNet (point-based). "
        "All data is processed live from your uploads; no simulations."
    )

    ctx = st.session_state.context
    score, breakdown = compute_fusion_risk(
        st.session_state.last_readings, thr, ctx.multiplier)
    band, color = risk_band(score, thr)

    c1, c2 = st.columns([1, 2])
    with c1:
        st.markdown('<div class="fusion-card">', unsafe_allow_html=True)
        st.markdown(f"<img src='{gauge_svg_dataurl(score, color)}'/>",
                    unsafe_allow_html=True)
        st.markdown(
            f"<div class='fusion-band-label' style='color:{color};'>{band}</div>"
            f"<div style='color:#64748b;font-size:13px;'>"
            f"Context ×{ctx.multiplier} — {ctx.event_type}</div>",
            unsafe_allow_html=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("🎥 Video",     f"{st.session_state.last_readings['video']:.0f}")
        m2.metric("📡 Mobile",    f"{st.session_state.last_readings['mobile']:.0f}")
        m3.metric("🛰️ Satellite", f"{st.session_state.last_readings['satellite']:.2f}")
        m4.metric("📹 CCTV",      f"{st.session_state.last_readings['cctv']:.2f}")
        m5.metric("🔊 Audio",     f"{st.session_state.last_readings['audio']:.2f}")

        df = pd.DataFrame([{
            "module": b.module,
            "contribution %": round(b.contribution * 100, 1),
            "normalized": round(b.normalized, 3),
        } for b in breakdown])
        fig = px.bar(df, x="module", y="contribution %",
                     template="plotly_white", color="normalized",
                     color_continuous_scale="Reds",
                     title="Per-module fusion contribution")
        fig.update_layout(height=280, margin=dict(l=10, r=10, t=40, b=10),
                           font=dict(color="#0f172a"))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("🧠 How it works")
    st.markdown(
        """
        1. **🎥 Video / 🛰️ Satellite** — Upload MP4 or aerial JPG.
           Choose one of the **three real backends** to detect and count people:
           - **YOLOv8** — bounding-box detector (ultralytics)
           - **CSRNet** — dilated-conv density map (real VGG-16 backbone)
           - **P2PNet** — point-based head localizer (real VGG-16 backbone)
        2. **📹 CCTV Optical Flow** — Real Farneback optical flow on your uploaded footage.
        3. **🔊 Audio Anomaly** — Real FFT + spectral analysis on your WAV/MP3.
        4. **📡 Mobile Network** — Real sklearn DBSCAN on your CSV of pings.
        5. **🔮 Fusion Engine** — Combines all real readings → 0–100 risk score.
        6. **📨 Twilio SMS** — Real dispatch via the Twilio REST API (`.env` credentials).
        """
    )

    st.subheader("Recent alerts")
    df_a = alerts_to_dataframe(st.session_state.alert_log)
    if df_a.empty:
        st.info("No alerts yet. Try any detection module in the sidebar.")
    else:
        st.dataframe(df_a.tail(8), use_container_width=True)


# ===========================================================================
# 🔮 FUSION RISK
# ===========================================================================
def page_fusion():
    st.title("🔮 Fusion Risk Engine")
    st.caption("Combines all real module readings into a single 0-100 score, "
                "modulated by weather/event context.")

    readings = st.session_state.last_readings
    ctx = st.session_state.context
    score, breakdown = compute_fusion_risk(readings, thr, ctx.multiplier)
    band, color = risk_band(score, thr)

    c1, c2 = st.columns([1, 2])
    with c1:
        st.markdown(f"<img src='{gauge_svg_dataurl(score, color)}'/>",
                    unsafe_allow_html=True)
        st.markdown(
            f"<h2 style='color:{color};margin:6px 0'>{band}</h2>"
            f"<p style='color:#334155;font-size:13px;'>"
            f"Weights: {FUSION_WEIGHTS}<br/>"
            f"Context multiplier: <b>×{ctx.multiplier}</b> ({ctx.event_type})</p>",
            unsafe_allow_html=True,
        )
    with c2:
        df = pd.DataFrame([{
            "module": b.module, "raw": round(b.value, 3),
            "threshold": b.threshold, "normalized": round(b.normalized, 3),
            "contribution %": round(b.contribution * 100, 2),
        } for b in breakdown])
        st.dataframe(df, use_container_width=True)

        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=[b.normalized for b in breakdown] + [breakdown[0].normalized],
            theta=[b.module for b in breakdown] + [breakdown[0].module],
            fill="toself", line_color=color,
        ))
        fig.update_layout(
            template="plotly_white",
            polar=dict(radialaxis=dict(range=[0, 1])),
            height=340, font=dict(color="#0f172a"),
            title="Module danger radar (0-1)",
        )
        st.plotly_chart(fig, use_container_width=True)

    if band == "CRITICAL":
        banner("critical",
               f"FUSION RISK CRITICAL — score {score:.1f}/100. Consider evacuation.")
        # Rate-limit: only one CRITICAL fusion alert per session run
        recent = [a for a in st.session_state.alert_log if a.module == "Fusion"]
        if not recent or recent[-1].severity != "CRITICAL":
            dispatch_alert("Fusion", "CRITICAL",
                           f"Fusion risk {score:.1f}/100 (ctx ×{ctx.multiplier}).")
    elif band == "ELEVATED":
        banner("warning", f"Elevated fusion risk — {score:.1f}/100.")
    else:
        banner("ok", f"Safe — fusion risk {score:.1f}/100.")


# ===========================================================================
# 🎥 VIDEO DETECTION  (REAL YOLOv8 / CSRNet / P2PNet)
# ===========================================================================
def page_video():
    st.title("🎥 Video Detection")
    st.caption("Real AI inference on your uploaded video. Choose a backend, "
                "upload MP4/AVI, hit Start.")

    c1, c2 = st.columns([2, 1])
    with c2:
        backend = st.selectbox("AI backend",
                                 ["YOLOv8", "CSRNet", "P2PNet"], index=0,
                                 help="All three are real, live models.")
        conf = st.slider("YOLO confidence (used by YOLOv8)", 0.05, 0.9, 0.25, 0.05)
        stride = st.slider("Process every Nth frame", 1, 30, 5, 1,
                            help="Skip frames to keep CPU/GPU feasible.")
        max_frames_ui = st.slider("Max frames to process", 20, 500, 120, 10)

    uploaded = c1.file_uploader("Upload MP4 / AVI / MOV",
                                 type=["mp4", "avi", "mov", "mkv"])
    start = c1.button("▶️ Start real-time analysis", type="primary")

    if not start:
        banner("ok", "Ready. Upload a video and click Start.")
        return
    if uploaded is None:
        st.warning("Please upload a video file first.")
        return

    tmp = tempfile.NamedTemporaryFile(
        delete=False, suffix="." + uploaded.name.split(".")[-1])
    tmp.write(uploaded.read()); tmp.close()

    cap = cv2.VideoCapture(tmp.name)
    if not cap.isOpened():
        st.error("Could not decode video. Try re-encoding to H.264 MP4.")
        return

    frame_ph = st.empty()
    metric_ph = st.empty()
    chart_ph = st.empty()
    banner_ph = st.empty()

    counts, times, latencies = [], [], []
    triggered = False
    v_thr = int(thr["VIDEO_COUNT_THRESHOLD"])
    st.session_state.video_history = []

    idx = 0
    processed = 0
    with st.spinner(f"Loading {backend} model (first run may download weights)…"):
        # Warm-up the backend once so the spinner covers the load
        try:
            _dummy = Image.new("RGB", (320, 320))
            detectors.count_people(_dummy, backend=backend)
        except Exception as e:
            st.error(f"Failed to initialize {backend}: {e}")
            cap.release(); return

    while processed < max_frames_ui:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % stride != 0:
            idx += 1; continue

        pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        try:
            if backend == "YOLOv8":
                # Use the YOLO conf slider
                yolo = detectors.get_yolo()
                n, boxes, scores = yolo.predict(pil, conf=conf)
                annotated = detectors._draw_boxes(pil, boxes, scores)
                res = detectors.CountResult(
                    count=n, backend="YOLOv8",
                    annotated_image=annotated,
                    boxes=boxes,
                    latency_ms=0.0, device=yolo.info["device"],
                )
            else:
                res = detectors.count_people(pil, backend=backend)
        except Exception as e:
            st.error(f"Inference error: {e}")
            break

        frame_ph.image(res.annotated_image, use_container_width=True,
                        caption=f"{backend} · frame {idx}")

        counts.append(res.count); times.append(idx); latencies.append(res.latency_ms)
        record_reading("video", res.count)

        with metric_ph.container():
            k1, k2, k3, k4, k5 = st.columns(5)
            k1.metric("People", res.count,
                        format_metric_delta(res.count, v_thr))
            k2.metric("Backend", res.backend)
            k3.metric("Device", res.device.upper())
            k4.metric("Latency", f"{res.latency_ms:.0f} ms" if res.latency_ms else "—")
            score, _ = compute_fusion_risk(
                st.session_state.last_readings, thr,
                st.session_state.context.multiplier)
            k5.metric("🔮 Fusion", f"{score:.1f}/100")

        # Trend + forecast
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=times, y=counts, mode="lines+markers",
                                   line=dict(color="#0369a1", width=3),
                                   name="Count"))
        fig.add_hline(y=v_thr, line_dash="dash", line_color="#dc2626",
                        annotation_text=f"Threshold ({v_thr})")
        fc = forecast_next_values(counts, horizon=15)
        if fc:
            fig.add_trace(go.Scatter(
                x=list(range(len(counts), len(counts) + len(fc))),
                y=fc, mode="lines", name="Forecast",
                line=dict(color="#7c3aed", width=2, dash="dot"),
            ))
        fig.update_layout(template="plotly_white", height=320,
                            title=f"{backend} — people count over time",
                            xaxis_title="Frame index", yaxis_title="People",
                            font=dict(color="#0f172a"))
        chart_ph.plotly_chart(fig, use_container_width=True)

        if res.count >= v_thr:
            banner_ph.markdown(
                f'<div class="alert-critical">🚨 CROWD LIMIT EXCEEDED — '
                f'{res.count} people ({backend}, threshold {v_thr}).</div>',
                unsafe_allow_html=True)
            if not triggered:
                dispatch_alert("Video", "CRITICAL",
                                f"[{backend}] {res.count} people (thr {v_thr}).")
                triggered = True
        else:
            banner_ph.markdown(
                f'<div class="alert-ok">✅ Normal — {res.count} people.</div>',
                unsafe_allow_html=True)

        processed += 1
        idx += 1

    cap.release()
    st.session_state.video_history = counts
    st.success(f"Processed {processed} frames with {backend}. "
                 f"Avg latency: {np.mean(latencies) if latencies else 0:.0f} ms.")


# ===========================================================================
# 📹 CCTV OPTICAL FLOW  (REAL Farneback)
# ===========================================================================
def page_cctv():
    st.title("📹 CCTV — Real Optical-Flow Velocity")
    st.caption("Computes real Farneback dense optical flow between consecutive "
                "frames of your uploaded video. Low velocity == possible crush.")

    c1, c2 = st.columns([2, 1])
    with c2:
        stride = st.slider("Frame stride", 1, 15, 3, 1)
        window = st.slider("Moving-average window", 1, 20, 5, 1)
        max_frames_ui = st.slider("Max frames", 20, 400, 100, 10)

    uploaded = c1.file_uploader("Upload CCTV footage (MP4/AVI/MOV)",
                                 type=["mp4", "avi", "mov", "mkv"])
    start = c1.button("▶️ Analyze motion", type="primary")

    if not start:
        banner("ok", "Ready. Upload footage and click Analyze motion.")
        return
    if uploaded is None:
        st.warning("Please upload a video first."); return

    tmp = tempfile.NamedTemporaryFile(
        delete=False, suffix="." + uploaded.name.split(".")[-1])
    tmp.write(uploaded.read()); tmp.close()

    cap = cv2.VideoCapture(tmp.name)
    if not cap.isOpened():
        st.error("Could not decode."); return

    frame_ph = st.empty()
    metric_ph = st.empty()
    chart_ph = st.empty()
    banner_ph = st.empty()

    prev_gray = None
    raw, smooth = [], []
    triggered = False
    c_thr = float(thr["CCTV_VELOCITY_THRESHOLD"])

    idx = 0; processed = 0
    while processed < max_frames_ui:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % stride != 0:
            idx += 1; continue

        frame_small = cv2.resize(frame, (640, 360))
        gray = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)
        if prev_gray is None:
            prev_gray = gray; idx += 1; continue

        v, flow = detectors.optical_flow_velocity(prev_gray, gray)
        raw.append(v)
        s = float(np.mean(raw[-window:])) if raw else v
        smooth.append(s)
        record_reading("cctv", s)

        # Overlay flow visualization
        rgb_frame = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)
        flow_rgb = detectors.flow_to_rgb(flow)
        blended = cv2.addWeighted(rgb_frame, 0.6, flow_rgb, 0.4, 0)
        frame_ph.image(blended, use_container_width=True,
                        caption=f"Optical flow overlay · frame {idx}")

        with metric_ph.container():
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Instant velocity", f"{v:.2f}")
            k2.metric(f"Smoothed (MA-{window})", f"{s:.2f}",
                        format_metric_delta(s, c_thr))
            k3.metric("Frame", f"{idx}")
            score, _ = compute_fusion_risk(
                st.session_state.last_readings, thr,
                st.session_state.context.multiplier)
            k4.metric("🔮 Fusion", f"{score:.1f}/100")

        fig = go.Figure()
        fig.add_trace(go.Scatter(y=raw, name="Raw",
                                   line=dict(color="#94a3b8", width=1.5)))
        fig.add_trace(go.Scatter(y=smooth, name=f"MA-{window}",
                                   line=dict(color="#0369a1", width=3)))
        fig.add_hline(y=c_thr, line_dash="dash", line_color="#dc2626",
                        annotation_text=f"Crush threshold ({c_thr})")
        fig.update_layout(template="plotly_white", height=280,
                            title="Real crowd velocity (Farneback optical flow)",
                            font=dict(color="#0f172a"))
        chart_ph.plotly_chart(fig, use_container_width=True)

        if s <= c_thr:
            banner_ph.markdown(
                f'<div class="alert-critical">🚨 CCTV CRUSH ALERT — '
                f'velocity {s:.2f} (threshold {c_thr}).</div>',
                unsafe_allow_html=True)
            if not triggered:
                dispatch_alert("CCTV", "CRITICAL",
                                f"Optical-flow velocity collapsed to {s:.2f}.")
                triggered = True
        else:
            banner_ph.markdown(
                f'<div class="alert-ok">✅ Motion normal — {s:.2f}.</div>',
                unsafe_allow_html=True)

        prev_gray = gray
        processed += 1
        idx += 1

    cap.release()
    st.success(f"Analyzed {processed} frame pairs.")


# ===========================================================================
# 🛰️ SATELLITE / AERIAL  (REAL YOLOv8/CSRNet/P2PNet on user's image)
# ===========================================================================
def page_satellite():
    st.title("🛰️ Satellite / Aerial Density")
    st.caption("Upload an aerial or drone image. The selected AI backend "
                "runs REAL inference on it and estimates people per m².")

    c1, c2 = st.columns([2, 1])
    with c2:
        backend = st.selectbox("AI backend",
                                 ["YOLOv8", "CSRNet", "P2PNet"], index=1,
                                 help="CSRNet works best on aerial density.")
        area_m2 = st.number_input("Ground area (m²)", 100, 200000, 5000, 100,
                                    help="Actual physical footprint of the image.")
        overlay_opacity = st.slider("Density overlay opacity", 0.1, 1.0, 0.55, 0.05)

    uploaded = c1.file_uploader("Upload aerial / satellite image",
                                 type=["png", "jpg", "jpeg", "webp"])
    if uploaded is None:
        banner("ok", "Waiting for an aerial image upload.")
        return

    pil = Image.open(uploaded).convert("RGB")

    with st.spinner(f"Running {backend} on aerial image…"):
        try:
            res = detectors.count_people(pil, backend=backend)
        except Exception as e:
            st.error(f"{backend} error: {e}")
            return

    density_pm2 = res.count / max(1.0, area_m2)
    record_reading("satellite", density_pm2)
    s_thr = float(thr["SATELLITE_DENSITY_THRESHOLD"])

    c1.image(res.annotated_image, use_container_width=True,
              caption=f"{backend} → {res.count} people detected")

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Estimated people", res.count)
    k2.metric("Ground area", f"{area_m2:,} m²")
    k3.metric("Density", f"{density_pm2:.2f} p/m²",
                format_metric_delta(density_pm2, s_thr))
    k4.metric("Latency", f"{res.latency_ms:.0f} ms")

    if density_pm2 >= s_thr:
        banner("critical",
               f"CROWD DENSITY CRITICAL — {density_pm2:.2f} p/m² exceeds "
               f"{s_thr} p/m².")
        recent = [a for a in st.session_state.alert_log if a.module == "Satellite"]
        if not recent or recent[-1].severity != "CRITICAL":
            dispatch_alert("Satellite", "CRITICAL",
                            f"[{backend}] Aerial density {density_pm2:.2f} p/m².")
    elif density_pm2 >= s_thr * 0.75:
        banner("warning",
               f"Elevated density — {density_pm2:.2f} p/m² approaching limit.")
    else:
        banner("ok", f"Safe — {density_pm2:.2f} p/m².")


# ===========================================================================
# 📡 MOBILE NETWORK  (REAL DBSCAN on user CSV)
# ===========================================================================
def page_mobile():
    st.title("📡 Mobile Network — Real DBSCAN")
    st.caption("Upload a CSV of cellular pings with `x, y` columns "
                "(and optional `device_id`, `signal_dbm`). Runs live sklearn DBSCAN.")

    c1, c2 = st.columns([2, 1])
    with c2:
        eps = st.slider("DBSCAN ε (radius)", 0.5, 20.0, 3.5, 0.5)
        min_samples = st.slider("min_samples", 3, 50, 15, 1)
        x_col = st.text_input("X column name", value="x")
        y_col = st.text_input("Y column name", value="y")

    uploaded = c1.file_uploader("Upload mobile pings CSV", type=["csv"])
    st.caption("💡 A working sample is included at `sample_data/mobile_pings.csv`.")

    if uploaded is None:
        banner("ok", "Waiting for a CSV upload.")
        return

    try:
        df = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Could not read CSV: {e}"); return

    if x_col not in df.columns or y_col not in df.columns:
        st.error(f"CSV must contain '{x_col}' and '{y_col}' columns. "
                    f"Found: {list(df.columns)}")
        return

    result_df, n_clusters, largest = detectors.run_dbscan_on_csv(
        df, x_col=x_col, y_col=y_col, eps=eps, min_samples=min_samples,
    )
    record_reading("mobile", largest)

    m_thr = int(thr["MOBILE_CLUSTER_THRESHOLD"])
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Devices in CSV", len(df))
    k2.metric("Clusters found", n_clusters)
    k3.metric("Largest cluster", largest, format_metric_delta(largest, m_thr))
    score, _ = compute_fusion_risk(
        st.session_state.last_readings, thr,
        st.session_state.context.multiplier)
    k4.metric("🔮 Fusion", f"{score:.1f}/100")

    fig = px.scatter(result_df, x=x_col, y=y_col,
                       color=result_df["cluster"].astype(str),
                       template="plotly_white", height=520,
                       title=f"DBSCAN — {n_clusters} clusters "
                             f"(-1 = noise)")
    fig.update_layout(font=dict(color="#0f172a"))
    st.plotly_chart(fig, use_container_width=True)

    if largest >= m_thr:
        banner("critical",
               f"SIGNALING OVERLOAD — {largest} devices in one cluster.")
        recent = [a for a in st.session_state.alert_log if a.module == "Mobile"]
        if not recent or recent[-1].severity != "CRITICAL":
            dispatch_alert("Mobile", "CRITICAL",
                            f"DBSCAN found {largest} devices in one cluster.")
    else:
        banner("ok", f"Cellular density normal — largest cluster {largest}.")


# ===========================================================================
# 🔊 AUDIO ANOMALY  (REAL librosa/FFT on user WAV/MP3)
# ===========================================================================
def page_audio():
    st.title("🔊 Audio Anomaly — Real FFT / Spectral Analysis")
    st.caption("Upload a real WAV/MP3 clip from your event PA/stage mic. "
                "The 1.5–4 kHz 'panic band' energy ratio is measured live.")

    uploaded = st.file_uploader("Upload audio (WAV / MP3)",
                                 type=["wav", "mp3", "flac", "ogg", "m4a"])
    if uploaded is None:
        banner("ok", "Waiting for an audio upload.")
        return

    with st.spinner("Analyzing audio (real FFT + spectral features)…"):
        try:
            info = detectors.analyze_audio(uploaded.read())
        except Exception as e:
            st.error(f"Audio decode error: {e}")
            st.info("If MP3 fails, install librosa and audioread: "
                     "`pip install librosa audioread`")
            return

    score = info["anomaly_score"]
    record_reading("audio", score)
    a_thr = float(thr["AUDIO_ANOMALY_THRESHOLD"])

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Anomaly score", f"{score:.2f}",
                format_metric_delta(score, a_thr))
    k2.metric("Panic-band ratio", f"{info['panic_band_ratio']:.3f}")
    k3.metric("Spectral centroid", f"{info['spectral_centroid']:.0f} Hz")
    k4.metric("Zero-crossing rate", f"{info['zero_crossing_rate']:.4f}")

    y = info["waveform"]
    if len(y) > 0:
        t = np.linspace(0, info["duration_s"], len(y), endpoint=False)
        # Downsample for plot if huge
        if len(y) > 20000:
            step = len(y) // 20000
            t = t[::step]; y = y[::step]
        fig_w = go.Figure()
        fig_w.add_trace(go.Scatter(x=t, y=y, line=dict(color="#0369a1"),
                                     name="Waveform"))
        fig_w.update_layout(template="plotly_white", height=200,
                              title="Waveform",
                              xaxis_title="Time (s)", yaxis_title="Amplitude",
                              font=dict(color="#0f172a"))
        st.plotly_chart(fig_w, use_container_width=True)

    if len(info["freqs"]) > 0:
        # Downsample spectrum too
        f = info["freqs"]; m = info["mags"]
        if len(f) > 5000:
            step = len(f) // 5000
            f = f[::step]; m = m[::step]
        fig_s = go.Figure()
        fig_s.add_trace(go.Scatter(x=f, y=m, line=dict(color="#7c3aed"),
                                     name="Magnitude"))
        fig_s.add_vrect(x0=1500, x1=4000, fillcolor="#dc2626", opacity=0.15,
                          line_width=0, annotation_text="Panic band")
        fig_s.update_layout(template="plotly_white", height=280,
                              title="FFT spectrum (real, from uploaded audio)",
                              xaxis_title="Frequency (Hz)",
                              yaxis_title="Magnitude",
                              font=dict(color="#0f172a"))
        st.plotly_chart(fig_s, use_container_width=True)

    if score >= a_thr:
        banner("critical", f"PANIC AUDIO DETECTED — score {score:.2f}.")
        recent = [a for a in st.session_state.alert_log if a.module == "Audio"]
        if not recent or recent[-1].severity != "CRITICAL":
            dispatch_alert("Audio", "CRITICAL",
                            f"Audio anomaly score {score:.2f}.")
    else:
        banner("ok", f"Ambient audio nominal — score {score:.2f}.")


# ===========================================================================
# 📈 FORECAST
# ===========================================================================
def page_forecast():
    st.title("📈 Predictive Crowd Forecast")
    st.caption("EMA + linear-trend projection on the REAL count history from "
                "your last Video Detection run.")
    hist = st.session_state.video_history
    if len(hist) < 5:
        st.info("Run 🎥 Video Detection first to accumulate history.")
        return
    horizon = st.slider("Forecast horizon (frames)", 5, 60, 20, 1)
    v_thr = int(thr["VIDEO_COUNT_THRESHOLD"])
    fc = forecast_next_values(hist, horizon=horizon)
    ttb = time_to_threshold(hist, v_thr, horizon=horizon)

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=hist, name="Real history",
                               line=dict(color="#0369a1", width=3)))
    fig.add_trace(go.Scatter(
        x=list(range(len(hist), len(hist) + len(fc))),
        y=fc, name="Forecast",
        line=dict(color="#7c3aed", width=3, dash="dot"),
    ))
    fig.add_hline(y=v_thr, line_dash="dash", line_color="#dc2626",
                    annotation_text=f"Threshold ({v_thr})")
    fig.update_layout(template="plotly_white", height=400,
                        title="Crowd forecast", font=dict(color="#0f172a"))
    st.plotly_chart(fig, use_container_width=True)

    k1, k2, k3 = st.columns(3)
    k1.metric("Latest actual", int(hist[-1]))
    k2.metric("Projected peak", f"{max(fc):.0f}" if fc else "—")
    k3.metric("Frames until breach", ttb if ttb is not None else "No breach")

    if ttb is not None and ttb <= 5:
        banner("critical", f"IMMINENT BREACH — {ttb} frames.")
    elif ttb is not None:
        banner("warning", f"Projected breach in {ttb} frames.")
    else:
        banner("ok", "No breach in the forecast horizon.")


# ===========================================================================
# 🗺️ LIVE MAP
# ===========================================================================
def page_map():
    st.title("🗺️ Live Venue Map")
    st.caption("Zones' current density is driven by the latest real satellite "
                "reading (density) × zone weight.")

    base = st.session_state.last_readings["satellite"]
    zone_weights = {
        "Main Stage": 1.4, "VIP Pit": 1.25, "Food Court": 0.65,
        "Merch Area": 0.55, "North Entrance": 0.9, "South Entrance": 0.85,
    }
    zone_xy = {
        "Main Stage": (50, 60), "VIP Pit": (50, 45),
        "Food Court": (20, 30), "Merch Area": (80, 30),
        "North Entrance": (50, 90), "South Entrance": (50, 10),
    }
    crit = float(thr["SATELLITE_DENSITY_THRESHOLD"])
    rows = []
    for zn, w in zone_weights.items():
        d = base * w
        status = ("CRITICAL" if d >= crit else
                    "ELEVATED" if d >= crit * 0.75 else "SAFE")
        rows.append({"zone": zn, "x": zone_xy[zn][0], "y": zone_xy[zn][1],
                       "density": round(d, 2), "status": status})
    df = pd.DataFrame(rows)

    fig = px.scatter(df, x="x", y="y", size="density", color="status",
                       text="zone", size_max=48,
                       color_discrete_map={
                           "SAFE": RISK_PALETTE["safe"],
                           "ELEVATED": RISK_PALETTE["elevated"],
                           "CRITICAL": RISK_PALETTE["critical"],
                       },
                       template="plotly_white", height=560)
    fig.update_traces(textposition="top center")
    fig.update_layout(
        xaxis=dict(range=[0, 100], title="X"),
        yaxis=dict(range=[0, 100], title="Y"),
        title=f"Venue map — driven by real satellite reading "
              f"({base:.2f} p/m²)",
        font=dict(color="#0f172a"),
    )
    for ex in features.DEFAULT_EXITS:
        fig.add_trace(go.Scatter(
            x=[ex.x], y=[ex.y], mode="markers+text",
            marker=dict(symbol="square", size=18, color="#0369a1",
                          line=dict(color="white", width=2)),
            text=[f"🚪 {ex.name}"], textposition="bottom center",
            showlegend=False,
        ))
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(df, use_container_width=True)

    if (df["status"] == "CRITICAL").any():
        banner("critical",
               f"Critical zones: {', '.join(df.loc[df['status']=='CRITICAL','zone'])}.")
    elif (df["status"] == "ELEVATED").any():
        banner("warning",
               f"Elevated zones: {', '.join(df.loc[df['status']=='ELEVATED','zone'])}.")
    else:
        banner("ok", "All zones nominal.")


# ===========================================================================
# 🚪 EVACUATION PLANNER
# ===========================================================================
def page_evac():
    st.title("🚪 Evacuation Route Recommender")
    st.caption("Ranks the 4 gates by safety given current hotspots.")
    n_h = st.slider("Number of hotspots", 1, 5, 2)
    hotspots = []
    cols = st.columns(n_h)
    for i in range(n_h):
        with cols[i]:
            st.markdown(f"**Hotspot #{i+1}**")
            x = st.slider(f"x{i}", 0, 100, [50, 30, 70, 20, 80][i % 5], key=f"hx{i}")
            y = st.slider(f"y{i}", 0, 100, [55, 60, 40, 80, 20][i % 5], key=f"hy{i}")
            inten = st.slider(f"intensity{i}", 0.1, 10.0, 4.0, 0.1, key=f"hi{i}")
            hotspots.append((float(x), float(y), float(inten)))

    ranked = recommend_exits(hotspots)
    df_r = pd.DataFrame(ranked)

    fig = go.Figure()
    hx, hy, hi = zip(*hotspots)
    fig.add_trace(go.Scatter(
        x=hx, y=hy, mode="markers",
        marker=dict(size=[i*6 for i in hi], color="#dc2626", opacity=0.55),
        name="Hotspots",
    ))
    color_map = {"PRIMARY": "#16a34a", "SECONDARY": "#0369a1",
                   "STANDBY": "#d97706", "AVOID": "#dc2626"}
    fig.add_trace(go.Scatter(
        x=df_r["x"], y=df_r["y"], mode="markers+text",
        marker=dict(size=22, symbol="square",
                      color=[color_map[r] for r in df_r["recommendation"]],
                      line=dict(color="white", width=2)),
        text=df_r["exit"], textposition="top center", name="Exits",
    ))
    fig.update_layout(template="plotly_white", height=500,
                        xaxis=dict(range=[0, 100]),
                        yaxis=dict(range=[0, 100]),
                        font=dict(color="#0f172a"))
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(df_r, use_container_width=True)
    top = df_r.iloc[0]
    banner("ok", f"Primary: <b>{top['exit']}</b> — safety score "
                    f"{top['safety_score']}, capacity {top['capacity_per_min']}/min.")


# ===========================================================================
# 📊 ANALYTICS
# ===========================================================================
def page_analytics():
    st.title("📊 Historical Analytics")
    st.caption("KPIs and trends of REAL alerts dispatched in this session.")
    df = alerts_to_dataframe(st.session_state.alert_log)
    k = alert_kpis(df)
    c = st.columns(5)
    c[0].metric("Total", k["total"])
    c[1].metric("Critical", k["critical"])
    c[2].metric("Warnings", k["warning"])
    c[3].metric("Alerts / hr", k["avg_per_hour"])
    c[4].metric("MTTR (s)", k["mttr_seconds"])
    if df.empty:
        st.info("No alerts yet — run detection modules to populate.")
        return
    df["ts"] = pd.to_datetime(df["timestamp"].str.replace(" UTC", "", regex=False),
                                errors="coerce")
    df = df.dropna(subset=["ts"])
    df["minute"] = df["ts"].dt.floor("min")
    per_min = df.groupby(["minute", "severity"]).size().reset_index(name="count")
    fig = px.bar(per_min, x="minute", y="count", color="severity",
                   template="plotly_white",
                   color_discrete_map={"CRITICAL": "#dc2626",
                                         "WARNING": "#d97706",
                                         "INFO": "#0284c7"},
                   title="Alerts per minute")
    fig.update_layout(font=dict(color="#0f172a"))
    st.plotly_chart(fig, use_container_width=True)

    heat = df.pivot_table(index="module", columns="severity",
                            values="timestamp", aggfunc="count", fill_value=0)
    fig2 = px.imshow(heat, template="plotly_white", aspect="auto",
                       color_continuous_scale="Reds",
                       title="Module × severity heatmap")
    fig2.update_layout(font=dict(color="#0f172a"))
    st.plotly_chart(fig2, use_container_width=True)


# ===========================================================================
# ⚙️ CONFIG
# ===========================================================================
def page_config():
    st.title("⚙️ Configuration")
    st.caption("Tune thresholds, weather/event context, Twilio recipient.")
    with st.form("cfg"):
        st.subheader("Thresholds")
        c1, c2, c3 = st.columns(3)
        v  = c1.number_input("Video count", 10.0, 2000.0, float(thr["VIDEO_COUNT_THRESHOLD"]), 5.0)
        m  = c1.number_input("Mobile cluster", 10.0, 2000.0, float(thr["MOBILE_CLUSTER_THRESHOLD"]), 5.0)
        s  = c2.number_input("Satellite p/m²", 0.5, 20.0, float(thr["SATELLITE_DENSITY_THRESHOLD"]), 0.1)
        cc = c2.number_input("CCTV velocity", 0.01, 2.0, float(thr["CCTV_VELOCITY_THRESHOLD"]), 0.01)
        a  = c3.number_input("Audio anomaly", 0.05, 1.0, float(thr["AUDIO_ANOMALY_THRESHOLD"]), 0.05)
        fc = c3.number_input("Fusion CRITICAL", 10.0, 100.0, float(thr["FUSION_RISK_CRITICAL"]), 1.0)
        fw = c3.number_input("Fusion WARNING", 10.0, 100.0, float(thr["FUSION_RISK_WARNING"]), 1.0)

        st.subheader("Weather & Event")
        c4, c5, c6 = st.columns(3)
        cond = c4.selectbox("Weather",
                              ["Clear", "Rain", "Thunderstorm", "Snow", "Fog"])
        temp = c4.number_input("Temperature (°C)", -20.0, 50.0,
                                 float(st.session_state.context.temp_c), 0.5)
        hum  = c5.number_input("Humidity (%)", 0.0, 100.0,
                                 float(st.session_state.context.humidity), 1.0)
        wind = c5.number_input("Wind (km/h)", 0.0, 200.0,
                                 float(st.session_state.context.wind_kmh), 1.0)
        etype = c6.selectbox("Event",
                                ["Concert", "Festival", "Sports Final",
                                 "Protest", "Religious Gathering", "Fair"])
        att = c6.number_input("Expected attendance", 100, 500000,
                                int(st.session_state.context.expected_attendance), 500)

        st.subheader("Twilio SMS recipient")
        recipient = st.text_input("E.164 phone (e.g. +15551234567)",
                                     value=st.session_state.recipient)

        saved = st.form_submit_button("💾 Save", type="primary")

    if saved:
        st.session_state.thresholds = coerce_thresholds({
            "VIDEO_COUNT_THRESHOLD": v, "MOBILE_CLUSTER_THRESHOLD": m,
            "SATELLITE_DENSITY_THRESHOLD": s, "CCTV_VELOCITY_THRESHOLD": cc,
            "AUDIO_ANOMALY_THRESHOLD": a,
            "FUSION_RISK_CRITICAL": fc, "FUSION_RISK_WARNING": fw,
        })
        st.session_state.context = build_context(
            condition=cond, temp_c=temp, humidity=hum,
            wind_kmh=wind, event_type=etype,
            expected_attendance=int(att),
        )
        st.session_state.recipient = recipient.strip()
        st.success(f"Saved. Context multiplier now "
                    f"×{st.session_state.context.multiplier}.")

    st.subheader("Twilio status")
    has_creds = all(os.environ.get(k) for k in
                     ["TWILIO_ACCOUNT_SID", "TWILIO_AUTH_TOKEN", "TWILIO_FROM_NUMBER"])
    if has_creds and st.session_state.recipient:
        st.success("✅ Real Twilio SMS is configured. Critical alerts will be sent.")
    else:
        missing = []
        if not has_creds:
            missing.append("credentials in `.env`")
        if not st.session_state.recipient:
            missing.append("recipient number")
        st.warning(f"⚠️ Twilio in MOCK mode. Missing: {', '.join(missing)}.")


# ===========================================================================
# 📨 ALERT LOG
# ===========================================================================
def page_alerts():
    st.title("📨 Alert Log")
    log = st.session_state.alert_log
    df = alerts_to_dataframe(log)
    c = st.columns(4)
    c[0].metric("Total", len(log))
    c[1].metric("Critical",
                  int((df["severity"] == "CRITICAL").sum()) if not df.empty else 0)
    c[2].metric("Real SMS sent",
                  int((df["mode"] == "real").sum()) if not df.empty else 0)
    c[3].metric("Modules",
                  df["module"].nunique() if not df.empty else 0)
    st.markdown("---")
    if df.empty:
        st.info("No alerts yet.")
        return
    for a in reversed(log):
        color = severity_color(a.severity)
        mode_cls = {"real": "mode-real", "mock": "mode-mock",
                     "error": "mode-error"}[a.mode]
        sid_str = f" · sid: <code>{a.sid[:12]}…</code>" if a.sid else ""
        st.markdown(
            f'''<div class="log-row">
              <span class="log-tag" style="background:{color};">{a.severity}</span>
              <b>{a.module}</b>
              <span class="log-time"> · {a.timestamp}</span>
              <span class="mode-badge {mode_cls}">{a.mode.upper()}</span>
              <br/>📱 <i>{a.recipient or "no recipient"}</i> — {a.message}{sid_str}
            </div>''',
            unsafe_allow_html=True,
        )
    st.download_button(
        "⬇️ Download CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="stampede_alerts.csv", mime="text/csv",
    )


# ===========================================================================
# 📄 INCIDENT REPORT
# ===========================================================================
def page_report():
    st.title("📄 Incident Report Exporter")
    df = alerts_to_dataframe(st.session_state.alert_log)
    peak = float(st.session_state.peak_risk)
    band, _ = risk_band(peak, thr)
    c = st.columns(3)
    c[0].metric("Peak fusion risk", f"{peak:.1f}/100")
    c[1].metric("Peak band", band)
    c[2].metric("Alerts in report", len(df))
    venue = st.text_input("Venue name", "Central Plaza")
    html = render_incident_report(
        df, peak, thr, st.session_state.context,
        st.session_state.operator, venue,
    )
    st.subheader("Preview")
    st.components.v1.html(html, height=600, scrolling=True)
    st.download_button(
        "⬇️ Download report (HTML — print-to-PDF ready)",
        data=html.encode("utf-8"),
        file_name=f"incident_report_{datetime.utcnow():%Y%m%d_%H%M%S}.html",
        mime="text/html",
    )


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------
ROUTER = {
    "🏠": page_overview, "🔮": page_fusion, "🗺️": page_map,
    "🎥": page_video, "📹": page_cctv, "🛰️": page_satellite,
    "📡": page_mobile, "🔊": page_audio, "📈": page_forecast,
    "🚪": page_evac, "📊": page_analytics, "⚙️": page_config,
    "📨": page_alerts, "📄": page_report,
}
for icon, fn in ROUTER.items():
    if module_choice.startswith(icon):
        fn(); break
