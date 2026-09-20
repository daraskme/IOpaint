@echo off
setlocal EnableDelayedExpansion
REM ============================================================
REM  IOPaint one-click launcher for Windows
REM  - First run: installs uv, creates a Python env, installs
REM    PyTorch (CUDA 12.8 if an NVIDIA GPU is present, else CPU)
REM    and the latest IOPaint release wheel from GitHub.
REM  - Later runs: starts IOPaint immediately.
REM
REM  Everything (Python, packages, uv cache, downloaded models)
REM  lives under %LOCALAPPDATA%\IOPaint by default. Override any of
REM  the settings below either as environment variables or in an
REM  IOPaint-OneClick.cfg file placed next to this .bat
REM  (KEY=VALUE per line, "#" starts a comment; a template is
REM  written on first run):
REM
REM    IOPAINT_HOME        install root        (default %LOCALAPPDATA%\IOPaint)
REM    IOPAINT_MODEL_DIR   model download dir  (default %IOPAINT_HOME%\models)
REM    IOPAINT_MODEL       model to start with (default lama)
REM    IOPAINT_PORT        HTTP port           (default 8080)
REM    IOPAINT_EXTRA_ARGS  extra "iopaint start" arguments
REM    UV_CACHE_DIR        uv download cache   (default %IOPAINT_HOME%\uv-cache)
REM    UV_PYTHON_INSTALL_DIR  uv-managed Python (default %IOPAINT_HOME%\python)
REM ============================================================

set "REPO=daraskme/IOpaint"
set "CFG=%~dp0IOPaint-OneClick.cfg"

if exist "%CFG%" (
    for /f "usebackq eol=# tokens=1,* delims==" %%a in ("%CFG%") do (
        if not "%%a"=="" set "%%a=%%b"
    )
)

if not defined IOPAINT_HOME set "IOPAINT_HOME=%LOCALAPPDATA%\IOPaint"
if not defined IOPAINT_MODEL set "IOPAINT_MODEL=lama"
if not defined IOPAINT_PORT set "IOPAINT_PORT=8080"
if not defined UV_CACHE_DIR set "UV_CACHE_DIR=%IOPAINT_HOME%\uv-cache"
if not defined UV_PYTHON_INSTALL_DIR set "UV_PYTHON_INSTALL_DIR=%IOPAINT_HOME%\python"
if not defined IOPAINT_MODEL_DIR (
    set "IOPAINT_MODEL_DIR=%IOPAINT_HOME%\models"
    REM Installs made by older launcher versions downloaded models to
    REM %USERPROFILE%\.cache; keep using them instead of re-downloading.
    if not exist "%IOPAINT_HOME%\models" if exist "%USERPROFILE%\.cache\torch\hub\checkpoints" (
        set "IOPAINT_MODEL_DIR=%USERPROFILE%\.cache"
    )
)
if "%IOPAINT_HOME:~-1%"=="\" set "IOPAINT_HOME=%IOPAINT_HOME:~0,-1%"
if "%IOPAINT_MODEL_DIR:~-1%"=="\" set "IOPAINT_MODEL_DIR=%IOPAINT_MODEL_DIR:~0,-1%"

set "APPDIR=%IOPAINT_HOME%"
set "VENV=%APPDIR%\env"
set "IOPAINT_EXE=%VENV%\Scripts\iopaint.exe"
set "HF_HUB_DISABLE_SYMLINKS_WARNING=1"

if not exist "%CFG%" call :write_cfg_template

if exist "%IOPAINT_EXE%" goto :run

echo === IOPaint first-time setup ===
echo Install location: %APPDIR%
echo Model directory:  %IOPAINT_MODEL_DIR%
echo uv cache:         %UV_CACHE_DIR%
echo (edit "%CFG%" to change these)
echo.

where uv >nul 2>nul
if not errorlevel 1 goto :have_uv
echo [1/4] Installing uv package manager...
powershell -NoProfile -ExecutionPolicy Bypass -Command "irm https://astral.sh/uv/install.ps1 | iex"
set "PATH=%USERPROFILE%\.local\bin;%PATH%"
where uv >nul 2>nul
if errorlevel 1 (
    echo ERROR: uv installation failed. Install it manually from https://docs.astral.sh/uv/
    goto :fail
)
:have_uv

echo [2/4] Creating Python environment...
if not exist "%APPDIR%" mkdir "%APPDIR%"
uv venv "%VENV%" --python 3.12
if errorlevel 1 goto :fail

where nvidia-smi >nul 2>nul
if errorlevel 1 goto :torch_cpu
echo [3/4] NVIDIA GPU detected - installing PyTorch with CUDA 12.8...
echo       NOTE: needs a driver supporting CUDA 12.8+ (Blackwell/Ada/Ampere are fine).
uv pip install --python "%VENV%" torch torchvision --torch-backend=cu128
if errorlevel 1 (
    echo ERROR: CUDA PyTorch install failed. Update your NVIDIA driver and retry.
    goto :fail
)
goto :torch_done
:torch_cpu
echo [3/4] No NVIDIA GPU detected - installing CPU PyTorch...
uv pip install --python "%VENV%" torch torchvision --torch-backend=cpu
if errorlevel 1 goto :fail
:torch_done

echo [4/4] Downloading the latest IOPaint release...
set "WHEEL_URL="
for /f "usebackq delims=" %%u in (`powershell -NoProfile -Command "$r = Invoke-RestMethod 'https://api.github.com/repos/%REPO%/releases?per_page=5'; $a = $r | ForEach-Object assets | Where-Object name -like '*.whl' | Select-Object -First 1; $a.browser_download_url"`) do set "WHEEL_URL=%%u"
if "!WHEEL_URL!"=="" (
    echo ERROR: could not find a release wheel for %REPO%.
    goto :fail
)
echo       %WHEEL_URL%
uv pip install --python "%VENV%" "!WHEEL_URL!"
if errorlevel 1 goto :fail

echo.
echo Setup finished successfully.
echo.

:run
if not exist "%IOPAINT_MODEL_DIR%" mkdir "%IOPAINT_MODEL_DIR%"
where nvidia-smi >nul 2>nul
if errorlevel 1 (set "DEVICE=cpu") else (set "DEVICE=cuda")
echo Starting IOPaint (model: %IOPAINT_MODEL%, device: !DEVICE!, models in: %IOPAINT_MODEL_DIR%)
echo The browser will open automatically. Close this window to stop IOPaint.
"%IOPAINT_EXE%" start --model "%IOPAINT_MODEL%" --device !DEVICE! --port %IOPAINT_PORT% --model-dir "%IOPAINT_MODEL_DIR%" --inbrowser %IOPAINT_EXTRA_ARGS%
goto :eof

:write_cfg_template
(
    echo # IOPaint-OneClick settings. Remove the leading "#" to enable a line.
    echo # Paths may contain spaces; do not wrap values in quotes.
    echo #IOPAINT_HOME=D:\IOPaint
    echo #IOPAINT_MODEL_DIR=D:\IOPaint\models
    echo #IOPAINT_MODEL=lama
    echo #IOPAINT_PORT=8080
    echo #IOPAINT_EXTRA_ARGS=--low-mem
    echo #UV_CACHE_DIR=D:\IOPaint\uv-cache
    echo #UV_PYTHON_INSTALL_DIR=D:\IOPaint\python
) > "%CFG%" 2>nul
exit /b 0

:fail
echo.
echo Setup failed. See the messages above for details.
pause
exit /b 1
