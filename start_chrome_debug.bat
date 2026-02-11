@echo off
echo =================================================================
echo       STARTING CHROME IN REMOTE DEBUGGING MODE
echo =================================================================
echo.
echo 1. Please ensure ALL other Google Chrome windows are CLOSED.
echo    (If they are not closed, this command will just open a new tab 
echo    in the existing non-debuggable instance and FAIL).
echo.
echo 2. This will launch Chrome using your DEFAULT profile.
echo.
pause

echo Killing any stray Chrome processes...
taskkill /F /IM chrome.exe /T >nul 2>&1
timeout /t 1 /nobreak >nul
taskkill /F /IM chrome.exe /T >nul 2>&1

echo Checking if Chrome is truly dead...
tasklist | find /i "chrome.exe" >nul
if %ERRORLEVEL% == 0 (
    echo [WARNING] Chrome processes are STILL running!
    echo Please manually close them via Task Manager or System Tray.
    echo Script will try to proceed, but it might fail...
    pause
)

echo Starting Chrome...

set "CHROME_PATH=C:\Program Files\Google\Chrome\Application\chrome.exe"
if not exist "%CHROME_PATH%" (
    set "CHROME_PATH=C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"
)
if not exist "%CHROME_PATH%" (
    set "CHROME_PATH=%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe"
)

if not exist "%CHROME_PATH%" (
    echo [ERROR] Could not find chrome.exe in standard locations.
    echo Please edit this script to set the correct path.
    pause
    exit /b
)

echo Found Chrome at: "%CHROME_PATH%"
echo Launching with remote debugging on port 9222...
start "" "%CHROME_PATH%" --remote-debugging-port=9222 --user-data-dir="%LOCALAPPDATA%\Google\Chrome\User Data"

echo.
echo Waiting for Chrome to initialize (max 10s)...

set "RETRIES=0"
:CHECK_LOOP
timeout /t 2 /nobreak >nul
python debug_chrome_connection.py >nul 2>&1
if %ERRORLEVEL% == 0 (
    echo.
    echo ✅ SUCCESS! Chrome is listening on port 9222.
    echo You can now run 'run_harvester.bat' (Option 2).
    echo.
    goto :END
)

set /a "RETRIES+=1"
if %RETRIES% LSS 5 (
    echo ... attempting to connect (%RETRIES%/5)
    goto :CHECK_LOOP
)

echo.
echo ❌ FAILURE: Chrome did not start debugging mode in time.
echo.
echo POSSIBLE CAUSES:
echo 1. Chrome didn't close fully (check System Tray icons!).
echo 2. A zombie chrome.exe process is stuck (check Task Manager).
echo 3. You need to run this script as Administrator.
echo.
echo TRUBLESHOOTING:
echo Try running: python debug_chrome_connection.py
echo to see the exact error.

:END
pause
