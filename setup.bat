@echo off
rem One-time setup: Python environment, packages (about 1 GB), MVTec AD images (about 500 MB), training.
cd /d "%~dp0"
echo.
echo === Visual Inspection: setup ===
if not exist ".venv\Scripts\python.exe" (
    echo [1/3] Creating the Python environment in .venv ...
    py -3 -m venv .venv 2>nul || python -m venv .venv
)
if not exist ".venv\Scripts\python.exe" (
    echo Could not create .venv. Install Python 3.12 or newer from python.org, then run setup.bat again.
    pause
    exit /b 1
)
echo [2/3] Installing packages - the first time takes several minutes ...
".venv\Scripts\python.exe" -m pip install --disable-pip-version-check -q -r requirements.txt
if errorlevel 1 (
    echo Installing packages failed. Check the internet connection, then run setup.bat again.
    pause
    exit /b 1
)
echo [3/3] Downloading the images and training one model per product - about 5 minutes ...
".venv\Scripts\python.exe" -m src.train
if errorlevel 1 (
    pause
    exit /b 1
)
echo.
echo Setup finished. Double-click start_demo.bat to open the demo.
pause
