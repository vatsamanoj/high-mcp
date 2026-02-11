@echo off
REM Robust script to run Key Harvester using the local venv

set VENV_PYTHON=venv\Scripts\python.exe

if not exist "%VENV_PYTHON%" (
    echo [ERROR] Virtual Environment not found at %VENV_PYTHON%
    echo Please create it or check your path.
    exit /b 1
)

echo [INFO] Using Python: %VENV_PYTHON%
echo [INFO] Installing missing dependencies...
"%VENV_PYTHON%" -m pip install -r requirements.txt > nul 2>&1

echo [INFO] Starting Key Harvester...
"%VENV_PYTHON%" key_harvester/harvester.py
pause
