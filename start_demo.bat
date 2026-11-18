@echo off
rem Opens the demo in your browser. Close this window to stop it.
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Run setup.bat once first.
    pause
    exit /b 1
)
if not exist "artifacts\bottle\meta.json" ".venv\Scripts\python.exe" -m src.train
echo Starting the demo. It opens in your browser; close this window to stop it.
".venv\Scripts\python.exe" app.py %*
pause
