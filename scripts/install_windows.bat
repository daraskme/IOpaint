@echo off
setlocal

cd /d "%~dp0\.."

where uv >nul 2>&1
if errorlevel 1 (
    echo Installing uv...
    powershell -NoProfile -ExecutionPolicy Bypass -Command "irm https://astral.sh/uv/install.ps1 | iex"
    if errorlevel 1 (
        echo ERROR: uv installation failed.
        exit /b 1
    )
    set "PATH=%USERPROFILE%\.local\bin;%PATH%"
)

where uv >nul 2>&1
if errorlevel 1 (
    echo ERROR: uv was installed but is not on PATH. Open a new terminal and retry.
    exit /b 1
)

echo Creating .iopaint-env...
uv venv .iopaint-env --clear
if errorlevel 1 exit /b 1
set "IOPAINT_PYTHON=.iopaint-env\Scripts\python.exe"

where nvidia-smi >nul 2>&1
if errorlevel 1 goto install_cpu

echo NVIDIA GPU detected. Installing CUDA 12.8 PyTorch wheels...
uv pip install --python "%IOPAINT_PYTHON%" torch torchvision --torch-backend=cu128
if errorlevel 1 goto cuda_error
"%IOPAINT_PYTHON%" -c "import sys, torch; print('PyTorch CUDA:', torch.version.cuda); sys.exit(0 if torch.cuda.is_available() else 1)"
if errorlevel 1 goto cuda_error
goto install_iopaint

:install_cpu
echo No NVIDIA GPU detected. Installing CPU PyTorch wheels...
uv pip install --python "%IOPAINT_PYTHON%" torch torchvision --torch-backend=cpu
if errorlevel 1 exit /b 1
goto install_iopaint

:cuda_error
echo ERROR: CUDA PyTorch could not use this GPU.
echo Update the NVIDIA driver to one that supports CUDA 12.8 wheels, then retry.
echo IOPaint will not silently fall back to CPU on a detected NVIDIA system.
exit /b 1

:install_iopaint
echo Installing iopaint-ng...
uv pip install --python "%IOPAINT_PYTHON%" iopaint-ng
REM Before the first PyPI release, install a locally built wheel instead:
REM uv pip install --python "%IOPAINT_PYTHON%" --find-links dist iopaint-ng
if errorlevel 1 (
    echo ERROR: iopaint-ng installation failed.
    echo If it is not published yet, build a wheel and use the local-wheel command above.
    exit /b 1
)

echo Installation complete. Run scripts\start_windows.bat to launch IOPaint.
exit /b 0
