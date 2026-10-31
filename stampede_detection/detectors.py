"""
detectors.py
------------
Unified interface over the three REAL crowd-counting backends:
    * YOLOv8       (Ultralytics)
    * CSRNet       (VGG-16 dilated back-end -> density map)
    * P2PNet       (Point-based head prediction)

Also provides:
    * Real Farneback optical-flow velocity estimator for the CCTV module.
    * Real librosa-based audio anomaly detector.
    * Real Twilio SMS sender.
    * Real DBSCAN clustering on user-provided CSV.

Every function here operates on REAL user-supplied data.
"""

from __future__ import annotations

import io
import os
import time
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

# ---------------------------------------------------------------------------
# Lazy singletons — neural backends are imported ONLY when first used so the
# app boots even if torch/ultralytics are missing on the current machine.
# ---------------------------------------------------------------------------
_YOLO = None
_CSR = None
_P2P = None


def get_yolo():
    """Real YOLOv8 (ultralytics) — imports lazily."""
    global _YOLO
    if _YOLO is None:
        from models.yolo_counter import YOLOv8Inference
        _YOLO = YOLOv8Inference()
    return _YOLO


def get_csrnet():
    """Real CSRNet (VGG-16 + dilated back-end) — imports lazily."""
    global _CSR
    if _CSR is None:
        from models.csrnet import CSRNetInference
        w = os.path.join(os.path.dirname(__file__), "models",
                         "weights", "csrnet_shanghai.pth")
        _CSR = CSRNetInference(weights_path=w if os.path.isfile(w) else None)
    return _CSR


def get_p2pnet():
    """Real P2PNet (VGG-16 + regression/classification heads) — imports lazily."""
    global _P2P
    if _P2P is None:
        from models.p2pnet import P2PNetInference
        w = os.path.join(os.path.dirname(__file__), "models",
                         "weights", "p2pnet_shb.pth")
        _P2P = P2PNetInference(weights_path=w if os.path.isfile(w) else None)
    return _P2P


# ---------------------------------------------------------------------------
# Unified crowd-count API
# ---------------------------------------------------------------------------
@dataclass
class CountResult:
    count: int
    backend: str
    annotated_image: Optional[Image.Image] = None
    density_map: Optional[np.ndarray] = None
    points: Optional[np.ndarray] = None
    boxes: Optional[np.ndarray] = None
    latency_ms: float = 0.0
    device: str = "cpu"


def count_people(
    image: Image.Image,
    backend: str = "YOLOv8",
) -> CountResult:
    """
    Run the selected REAL backend on `image` and return a rich result.
    """
    backend = backend.upper()
    t0 = time.perf_counter()

    if backend == "YOLOV8":
        model = get_yolo()
        n, boxes, scores = model.predict(image)
        annotated = _draw_boxes(image, boxes, scores)
        return CountResult(
            count=n, backend="YOLOv8", annotated_image=annotated,
            boxes=boxes, latency_ms=(time.perf_counter() - t0) * 1000.0,
            device=model.info["device"],
        )

    if backend == "CSRNET":
        model = get_csrnet()
        est, density = model.predict(image)
        annotated = _draw_density(image, density)
        return CountResult(
            count=int(round(est)), backend="CSRNet",
            annotated_image=annotated, density_map=density,
            latency_ms=(time.perf_counter() - t0) * 1000.0,
            device=model.device,
        )

    if backend == "P2PNET":
        model = get_p2pnet()
        n, points = model.predict(image)
        annotated = _draw_points(image, points)
        return CountResult(
            count=n, backend="P2PNet", annotated_image=annotated,
            points=points, latency_ms=(time.perf_counter() - t0) * 1000.0,
            device=model.device,
        )

    raise ValueError(f"Unknown backend: {backend}")


# ---------------------------------------------------------------------------
# Overlay helpers
# ---------------------------------------------------------------------------
def _draw_boxes(image: Image.Image, boxes: np.ndarray,
                scores: np.ndarray) -> Image.Image:
    img = image.convert("RGB").copy()
    draw = ImageDraw.Draw(img)
    for (x1, y1, x2, y2), s in zip(boxes, scores):
        draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 120), width=2)
        draw.text((x1 + 2, max(0, y1 - 12)),
                  f"{s:.2f}", fill=(0, 255, 120))
    return img


def _draw_points(image: Image.Image, points: np.ndarray) -> Image.Image:
    img = image.convert("RGB").copy()
    draw = ImageDraw.Draw(img)
    for (x, y) in points:
        draw.ellipse([x - 3, y - 3, x + 3, y + 3],
                     fill=(255, 60, 60), outline=(255, 255, 255))
    return img


def _draw_density(image: Image.Image, density: np.ndarray) -> Image.Image:
    """Overlay a jet-colormapped density map on the input image."""
    img = np.asarray(image.convert("RGB"))
    H, W = img.shape[:2]
    d = density.copy()
    if d.max() > 0:
        d = (d - d.min()) / (d.max() - d.min() + 1e-9)
    d = (d * 255).astype(np.uint8)
    d = cv2.resize(d, (W, H), interpolation=cv2.INTER_CUBIC)
    heat = cv2.applyColorMap(d, cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    blended = cv2.addWeighted(img, 0.55, heat, 0.45, 0)
    return Image.fromarray(blended)


# ---------------------------------------------------------------------------
# REAL optical-flow velocity (Farneback) for the CCTV module
# ---------------------------------------------------------------------------
def optical_flow_velocity(
    prev_gray: np.ndarray,
    curr_gray: np.ndarray,
) -> Tuple[float, np.ndarray]:
    """
    Return (mean_magnitude, flow_HxWx2).
    Both inputs must be single-channel uint8 arrays of the same shape.
    """
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray, None,
        pyr_scale=0.5, levels=3, winsize=15, iterations=3,
        poly_n=5, poly_sigma=1.2, flags=0,
    )
    mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    mean_mag = float(np.mean(mag))
    # Normalize to a comparable 0..1 range where ~10 px/frame -> 1.0
    return min(1.0, mean_mag / 10.0), flow


def flow_to_rgb(flow: np.ndarray) -> np.ndarray:
    """Colorize the flow field for display."""
    H, W = flow.shape[:2]
    hsv = np.zeros((H, W, 3), dtype=np.uint8)
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 1] = 255
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


# ---------------------------------------------------------------------------
# REAL audio anomaly detector — librosa if available, numpy fallback.
# ---------------------------------------------------------------------------
def analyze_audio(file_bytes: bytes) -> Dict[str, Any]:
    """
    Return a dict with:
        sample_rate, duration_s, waveform (mono float32),
        spectral_centroid, zero_crossing_rate, panic_band_ratio,
        anomaly_score (0..1).
    Uses librosa when installed; otherwise a lightweight WAV loader.
    """
    try:
        import librosa
        y, sr = librosa.load(io.BytesIO(file_bytes), sr=None, mono=True)
    except Exception:
        # Minimal WAV fallback
        import wave
        with wave.open(io.BytesIO(file_bytes), "rb") as wf:
            sr = wf.getframerate()
            n = wf.getnframes()
            raw = wf.readframes(n)
            y = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            if wf.getnchannels() > 1:
                y = y.reshape(-1, wf.getnchannels()).mean(axis=1)

    if y.size == 0:
        return dict(sample_rate=sr, duration_s=0.0, waveform=y,
                    spectral_centroid=0.0, zero_crossing_rate=0.0,
                    panic_band_ratio=0.0, anomaly_score=0.0,
                    freqs=np.array([]), mags=np.array([]))

    n = y.size
    fft = np.fft.rfft(y * np.hanning(n))
    mags = np.abs(fft)
    freqs = np.fft.rfftfreq(n, d=1.0 / sr)

    total = float((mags ** 2).sum()) + 1e-9
    band = (freqs >= 1500) & (freqs <= 4000)
    band_energy = float((mags[band] ** 2).sum())
    ratio = band_energy / total

    # spectral centroid (Hz)
    centroid = float((freqs * mags).sum() / (mags.sum() + 1e-9))

    # zero-crossing rate
    zcr = float(((y[:-1] * y[1:]) < 0).sum() / max(1, n - 1))

    # combined anomaly score
    score = float(min(1.0, ratio * 1.6 + zcr * 0.3))

    return dict(
        sample_rate=sr,
        duration_s=n / sr,
        waveform=y,
        spectral_centroid=centroid,
        zero_crossing_rate=zcr,
        panic_band_ratio=ratio,
        anomaly_score=score,
        freqs=freqs, mags=mags,
    )


# ---------------------------------------------------------------------------
# REAL DBSCAN on user-supplied CSV
# ---------------------------------------------------------------------------
def run_dbscan_on_csv(
    df: pd.DataFrame,
    x_col: str = "x",
    y_col: str = "y",
    eps: float = 3.5,
    min_samples: int = 15,
) -> Tuple[pd.DataFrame, int, int]:
    """
    Cluster (x, y) pings with sklearn DBSCAN and return
    (df with 'cluster' column, n_clusters, largest_cluster_size).
    """
    from sklearn.cluster import DBSCAN
    df = df.copy()
    coords = df[[x_col, y_col]].to_numpy()
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(coords)
    df["cluster"] = labels
    unique = [c for c in np.unique(labels) if c != -1]
    largest = 0
    if unique:
        largest = int(max((labels == c).sum() for c in unique))
    return df, len(unique), largest


# ---------------------------------------------------------------------------
# REAL Twilio SMS
# ---------------------------------------------------------------------------
def send_real_twilio_sms(
    message: str,
    to_number: str,
    from_number: Optional[str] = None,
    account_sid: Optional[str] = None,
    auth_token: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Send an SMS through the real Twilio REST API.

    Credentials are picked up from arguments OR the following env vars
    (loaded from `.env` by app.py):
        TWILIO_ACCOUNT_SID
        TWILIO_AUTH_TOKEN
        TWILIO_FROM_NUMBER

    Returns a dict with fields {ok, sid, error, mode}.
    """
    account_sid = account_sid or os.environ.get("TWILIO_ACCOUNT_SID", "")
    auth_token  = auth_token  or os.environ.get("TWILIO_AUTH_TOKEN", "")
    from_number = from_number or os.environ.get("TWILIO_FROM_NUMBER", "")

    if not (account_sid and auth_token and from_number):
        return dict(ok=False, sid=None, mode="mock",
                    error="Twilio credentials missing — sent as MOCK. "
                          "Set TWILIO_ACCOUNT_SID / TWILIO_AUTH_TOKEN / "
                          "TWILIO_FROM_NUMBER in .env to enable real SMS.")

    try:
        from twilio.rest import Client
        client = Client(account_sid, auth_token)
        msg = client.messages.create(
            body=message, from_=from_number, to=to_number,
        )
        return dict(ok=True, sid=msg.sid, mode="real", error=None)
    except Exception as e:
        return dict(ok=False, sid=None, mode="error", error=str(e))
