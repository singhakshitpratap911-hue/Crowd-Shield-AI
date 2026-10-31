# =============================================================================
# install_windows.ps1  —  One-click installer for Windows + Python 3.14
# =============================================================================
# Usage (from an elevated PowerShell inside the project folder):
#   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
#   .\install_windows.ps1
# =============================================================================

Write-Host "== Stampede Detection v4 — Windows installer ==" -ForegroundColor Cyan
Write-Host ""

# --- 1. Verify Python is available ---
$pythonCmd = $null
foreach ($cmd in @("python", "py -3.14", "py")) {
    try {
        $ver = & cmd /c "$cmd --version 2>&1"
        if ($LASTEXITCODE -eq 0 -and $ver -match "Python 3\.1[234]") {
            $pythonCmd = $cmd
            Write-Host "[OK] Found: $ver ($cmd)" -ForegroundColor Green
            break
        }
    } catch { }
}
if (-not $pythonCmd) {
    Write-Host "[ERROR] Python 3.12+ not found. Install from python.org first." -ForegroundColor Red
    exit 1
}

# --- 2. Delete any broken .venv ---
if (Test-Path .venv) {
    Write-Host "[..] Removing old .venv ..." -ForegroundColor Yellow
    Remove-Item -Recurse -Force .venv
}

# --- 3. Create fresh venv ---
Write-Host "[..] Creating .venv with $pythonCmd ..." -ForegroundColor Yellow
& cmd /c "$pythonCmd -m venv .venv"
if (-not (Test-Path .venv\Scripts\python.exe)) {
    Write-Host "[ERROR] venv creation failed." -ForegroundColor Red
    exit 1
}

# --- 4. Activate ---
& .\.venv\Scripts\Activate.ps1

# --- 5. Upgrade pip + wheel ---
Write-Host "[..] Upgrading pip / wheel / setuptools ..." -ForegroundColor Yellow
python -m pip install --upgrade pip wheel setuptools

# --- 6. Install requirements with binary-only flag ---
#     This is the CRITICAL step — --only-binary=:all: prevents pip from
#     ever trying to compile numpy/pandas from source on Python 3.14.
Write-Host "[..] Installing base requirements (binary-only wheels) ..." -ForegroundColor Yellow
python -m pip install --only-binary=:all: -r requirements.txt

if ($LASTEXITCODE -ne 0) {
    Write-Host "[WARN] Some packages have no 3.14 wheel yet." -ForegroundColor Yellow
    Write-Host "        Falling back to per-package install (skip failures) ..." -ForegroundColor Yellow
    $pkgs = @(
        "numpy>=2.1", "pandas>=2.2.3", "streamlit>=1.38", "plotly>=5.24",
        "pillow>=10.4", "opencv-python-headless>=4.10",
        "torch>=2.5", "torchvision>=0.20", "ultralytics>=8.3",
        "soundfile>=0.12", "twilio>=9.0", "python-dotenv>=1.0",
        "scikit-learn>=1.5"
    )
    foreach ($p in $pkgs) {
        Write-Host "  - installing $p" -ForegroundColor Gray
        python -m pip install --only-binary=:all: $p
    }
}

# --- 7. Optional: librosa (audio MP3 support) ---
Write-Host "[..] Attempting optional librosa install (audio MP3 support) ..." -ForegroundColor Yellow
python -m pip install --only-binary=:all: librosa audioread 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "[INFO] librosa not available on this Python version." -ForegroundColor Cyan
    Write-Host "       Audio module will still work with WAV files (numpy fallback)." -ForegroundColor Cyan
}

# --- 8. Sanity check ---
Write-Host ""
Write-Host "[..] Running sanity check ..." -ForegroundColor Yellow
python -c @"
import importlib, sys
mods = ['streamlit', 'numpy', 'pandas', 'plotly', 'PIL', 'cv2',
        'torch', 'torchvision', 'ultralytics', 'sklearn', 'twilio', 'dotenv']
ok, miss = [], []
for m in mods:
    try:
        importlib.import_module(m); ok.append(m)
    except Exception as e:
        miss.append((m, str(e)[:60]))
print('OK :', ', '.join(ok))
if miss:
    print('MISSING:')
    for m, e in miss: print(f'  {m}: {e}')
    sys.exit(1)
print('All required deps present.')
"@

# --- 9. Copy .env.example ---
if (-not (Test-Path .env)) {
    Copy-Item .env.example .env
    Write-Host "[OK] Created .env — edit it to add your Twilio credentials." -ForegroundColor Green
}

Write-Host ""
Write-Host "==============================================" -ForegroundColor Green
Write-Host "  Install complete.  Run the app with:"        -ForegroundColor Green
Write-Host "    .\.venv\Scripts\Activate.ps1"              -ForegroundColor Cyan
Write-Host "    streamlit run app.py"                       -ForegroundColor Cyan
Write-Host "==============================================" -ForegroundColor Green
