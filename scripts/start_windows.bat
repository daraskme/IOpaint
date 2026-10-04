@echo off
setlocal

cd /d "%~dp0\.."
if not exist ".iopaint-env\Scripts\activate.bat" (
    echo ERROR: .iopaint-env is missing. Run scripts\install_windows.bat first.
    exit /b 1
)

set "IOPAINT_DEVICE=cpu"
.iopaint-env\Scripts\python.exe -c "import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)" >nul 2>&1
if not errorlevel 1 set "IOPAINT_DEVICE=cuda"

call .iopaint-env\Scripts\activate.bat
iopaint start --model lama --device %IOPAINT_DEVICE% --port 8080 --inbrowser
