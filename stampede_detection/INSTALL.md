# 🛠️ Windows + Python 3.14 Install Guide (v4)

This build is tuned for **Python 3.14** — the exact version on your machine.
It uses binary-only wheels so pip never tries to compile from source.

---

## 🚀 Fastest path — one-click installer

Open **PowerShell inside the project folder** (Shift+Right-click → "Open PowerShell here"):

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\install_windows.ps1
```

Or from **CMD**:

```cmd
install_windows.bat
```

Either script:
1. Finds your Python (3.12 / 3.13 / 3.14)
2. Deletes any old `.venv`
3. Creates a fresh `.venv`
4. Upgrades pip / wheel / setuptools
5. Installs all dependencies **with `--only-binary=:all:`** so pip cannot compile from source
6. Falls back to per-package install if any single package fails
7. Copies `.env.example` → `.env`
8. Runs a sanity import check

When it finishes:

```powershell
.\.venv\Scripts\Activate.ps1
streamlit run app.py
```

---

## 🩹 If the installer fails — manual steps

The error `py -3.12 -m venv .venv → No suitable Python runtime found` means
you don't have Python 3.12 installed. That's fine — we'll use **your existing
Python 3.14**.

### 1. Delete any broken venv
```powershell
Remove-Item -Recurse -Force .venv -ErrorAction SilentlyContinue
```

### 2. Create a venv with your installed Python
```powershell
python -m venv .venv
```
*(Use `python` — not `py -3.12` — since you have 3.14 only.)*

### 3. Activate
```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks the script:
```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

### 4. Upgrade pip (CRITICAL for Python 3.14)
```powershell
python -m pip install --upgrade pip wheel setuptools
```

### 5. Install everything with binary-only flag
```powershell
pip install --only-binary=:all: -r requirements.txt
```

**Why `--only-binary=:all:`?**
On Python 3.14, some packages don't have source-build fallbacks yet.
This flag forces pip to only install pre-built wheels — no `Failed to build
'pandas'` errors possible.

### 6. Add CUDA (you have an NVIDIA GPU)
```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124 --upgrade --only-binary=:all:
```

Verify:
```powershell
python -c "import torch; print('CUDA:', torch.cuda.is_available(), '-', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

### 7. (Optional) Audio MP3 support
The Audio Anomaly module works with WAV files out of the box (pure numpy).
For MP3/FLAC support:
```powershell
pip install --only-binary=:all: librosa audioread
```
If this fails on 3.14, **just skip it** — the app still runs and the
Audio module falls back to WAV-only.

### 8. Twilio credentials
```powershell
Copy-Item .env.example .env
# Then open .env in VS Code and paste your Twilio SID / AUTH_TOKEN / FROM_NUMBER
```

### 9. Point VS Code at the venv
Press **Ctrl + Shift + P** → **"Python: Select Interpreter"** →
choose `.venv\Scripts\python.exe`. Yellow "Import could not be resolved"
squiggles disappear.

### 10. Run
```powershell
streamlit run app.py
```
Open http://localhost:8501 → sign in with **`demo / demo`**.

---

## 🩺 Troubleshooting

| Symptom | Fix |
|---|---|
| `No suitable Python runtime found` when using `py -3.12` | Use `python -m venv .venv` instead — you have 3.14 |
| `Could not find a version that satisfies numpy` | Run `python -m pip install --upgrade pip` first |
| `Failed to build 'pandas' / 'numpy'` | You forgot `--only-binary=:all:`. Retry step 5. |
| `Torch not compiled with CUDA` | Step 6 was skipped or failed — rerun it |
| `librosa` refuses to install | Skip it — audio module uses numpy WAV fallback |
| Streamlit blank page | `streamlit cache clear` then `streamlit run app.py` |
| Port 8501 already in use | `streamlit run app.py --server.port 8600` |
| PowerShell blocks activate.ps1 | `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned` |
| VS Code shows yellow "Import could not be resolved" | Ctrl+Shift+P → Python: Select Interpreter → pick `.venv` |
| YOLO downloads slowly | Normal — yolov8n.pt is ~6 MB on first use |

## ✅ Sanity check

```powershell
python -c "import streamlit, torch, ultralytics, cv2, sklearn, twilio, PIL, pandas, numpy; print('All required deps OK')"
```

If that prints `All required deps OK`, you're ready.
