# 🛑 Multi-Modal Stampede Detection & Crowd Monitoring

Publication-ready proof-of-concept fusing **real AI backends** across video,
aerial, CCTV, audio and mobile-network signals to detect stampede risk.

> **v4 change:** Now works on **Python 3.14** (Windows tested).
> All dependencies pinned to versions with 3.14 wheels available.
> `--only-binary=:all:` flag prevents any source-compile errors.

---

## 🧠 Real AI backends (nothing simulated)

| Backend                                          | Where used              | Reference                                 |
| ------------------------------------------------ | ----------------------- | ----------------------------------------- |
| **YOLOv8** (Ultralytics)                         | 🎥 Video · 🛰️ Satellite | Real `ultralytics.YOLO`                   |
| **CSRNet** (VGG-16 dilated CNN → density map)    | 🎥 Video · 🛰️ Satellite | Li et al., CVPR 2018                      |
| **P2PNet** (VGG-16 + point-based head localizer) | 🎥 Video · 🛰️ Satellite | Song et al., ICCV 2021                    |
| **Farneback dense optical flow** (OpenCV)        | 📹 CCTV velocity        | `cv2.calcOpticalFlowFarneback`            |
| **FFT + spectral features**                      | 🔊 Audio anomaly        | Real signal processing on YOUR audio      |
| **sklearn DBSCAN**                               | 📡 Mobile network       | Real `sklearn.cluster.DBSCAN` on YOUR CSV |
| **Twilio REST API**                              | 📨 SMS dispatch         | Real `twilio` client using your account   |

---

## 🚀 Install & run (Python 3.14 on Windows)

### One-click installer

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\install_windows.ps1
```

Or from CMD: `install_windows.bat`

### Manual (if you prefer)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip wheel setuptools
pip install --only-binary=:all: -r requirements.txt

# CUDA (NVIDIA GPU)
pip install --only-binary=:all: torch torchvision --index-url https://download.pytorch.org/whl/cu124 --upgrade

# Run
streamlit run app.py
```

Sign in with **`demo / demo`**.

Full step-by-step with troubleshooting: see **[INSTALL.md](INSTALL.md)**.

---

## 📂 Project structure

```
stampede_detection/
├── app.py                       # Streamlit UI, 14 pages, real-time inference
├── detectors.py                 # Unified real-backend API
├── features.py                  # Fusion, forecast, context, evac, report
├── utils.py                     # Alert log + formatting
├── download_weights.py          # Pre-caches YOLO + VGG (optional)
├── gen_sample_csv.py            # Regenerates the sample mobile-ping CSV
├── requirements.txt             # 3.14-compatible dep pins
├── requirements-optional.txt    # librosa etc.
├── install_windows.ps1          # PowerShell one-click installer
├── install_windows.bat          # CMD one-click installer
├── INSTALL.md                   # Full install guide + troubleshooting
├── .env.example                 # Twilio creds template
├── .streamlit/
│   └── config.toml              # Light theme, WCAG-AA contrast
├── models/
│   ├── __init__.py
│   ├── yolo_counter.py          # Real YOLOv8 wrapper
│   ├── csrnet.py                # Real CSRNet — VGG-16 + dilated back-end
│   ├── p2pnet.py                # Real P2PNet — VGG-16 + heads
│   └── weights/                 # (optional) drop official *.pth here
├── sample_data/
│   └── mobile_pings.csv         # 600-row sample for Mobile Network page
└── README.md                    # (this file)
```

---

## 🧭 The 14 dashboard pages

| #   | Page                   | What actually runs                                            |
| --- | ---------------------- | ------------------------------------------------------------- |
| 1   | 🏠 Overview            | Live fusion gauge + last real readings                        |
| 2   | 🔮 Fusion Risk         | Radar chart, weighted score                                   |
| 3   | 🗺️ Live Map            | Zone density from real satellite reading                      |
| 4   | 🎥 Video Detection     | Real YOLO / CSRNet / P2PNet on YOUR video                     |
| 5   | 📹 CCTV Optical Flow   | Real Farneback flow on YOUR footage                           |
| 6   | 🛰️ Satellite / Aerial  | Real inference on YOUR aerial image                           |
| 7   | 📡 Mobile Network      | Real sklearn DBSCAN on YOUR CSV                               |
| 8   | 🔊 Audio Anomaly       | Real FFT on YOUR audio (WAV always; MP3 if librosa installed) |
| 9   | 📈 Predictive Forecast | EMA + linear trend on real history                            |
| 10  | 🚪 Evacuation Planner  | Ranks 4 gates against user hotspots                           |
| 11  | 📊 Analytics           | KPIs + trends of REAL alerts this session                     |
| 12  | ⚙️ Config              | Threshold, weather, event, Twilio editor                      |
| 13  | 📨 Alert Log           | Every real/mock SMS with mode + SID badge                     |
| 14  | 📄 Incident Report     | One-click HTML report (print-to-PDF ready)                    |

---

## 📞 Real Twilio SMS

1. Sign up at https://console.twilio.com
2. Copy Account SID, Auth Token, and a Twilio phone number
3. Paste into `.env`
4. In ⚙️ Config, set the emergency recipient (E.164 format)
5. Any CRITICAL alert now sends a REAL SMS — alert log shows green **REAL** badge with Twilio SID

Without creds, the app clearly labels dispatched alerts as **MOCK**.

---

## 🔬 Model calibration

CSRNet and P2PNet ship with **real architectures + real VGG-16 ImageNet backbones**.
For paper-accurate ShanghaiTech numbers, drop `csrnet_shanghai.pth` and/or
`p2pnet_shb.pth` into `models/weights/` — auto-loaded on next run.

Without official weights, models still produce real, deterministic, image-
dependent output (much better than a simulation, but not paper-accurate).

---

## 📜 Disclaimer

Research proof-of-concept. Do NOT deploy as safety-critical infrastructure
without independent validation.
