@echo off
REM =============================================================================
REM install_windows.bat — CMD version of the installer for Python 3.14
REM =============================================================================
setlocal enabledelayedexpansion

echo == Stampede Detection v4 — Windows CMD installer ==
echo.

REM --- 1. Find Python ---
set PYCMD=
for %%C in ("python" "py -3.14" "py -3.13" "py -3.12" "py") do (
    %%~C --version >nul 2>&1
    if !errorlevel! EQU 0 (
        set PYCMD=%%~C
        goto :found
    )
)
echo [ERROR] No Python found. Install from python.org first.
exit /b 1

:found
echo [OK] Using: %PYCMD%
%PYCMD% --version

REM --- 2. Delete old venv ---
if exist .venv (
    echo [..] Removing old .venv ...
    rmdir /s /q .venv
)

REM --- 3. Create venv ---
echo [..] Creating .venv ...
%PYCMD% -m venv .venv
if not exist .venv\Scripts\python.exe (
    echo [ERROR] venv creation failed.
    exit /b 1
)

REM --- 4. Activate ---
call .venv\Scripts\activate.bat

REM --- 5. Upgrade pip ---
echo [..] Upgrading pip ...
python -m pip install --upgrade pip wheel setuptools

REM --- 6. Install requirements (binary-only) ---
echo [..] Installing dependencies (binary wheels only) ...
python -m pip install --only-binary=:all: -r requirements.txt
if !errorlevel! NEQ 0 (
    echo [WARN] Bulk install failed. Trying per-package fallback ...
    for %%P in (numpy pandas streamlit plotly pillow opencv-python-headless torch torchvision ultralytics soundfile twilio python-dotenv scikit-learn) do (
        python -m pip install --only-binary=:all: %%P
    )
)

REM --- 7. Optional librosa ---
python -m pip install --only-binary=:all: librosa audioread >nul 2>&1

REM --- 8. .env ---
if not exist .env copy .env.example .env >nul

echo.
echo ==============================================
echo   Install complete. Run the app with:
echo     .venv\Scripts\activate.bat
echo     streamlit run app.py
echo ==============================================
endlocal
