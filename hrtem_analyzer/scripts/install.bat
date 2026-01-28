@echo off
REM HR-TEM Analyzer Installation Script (Windows)
REM ==============================================

echo ========================================
echo   HR-TEM Analyzer Installer
echo ========================================
echo.

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    echo Please install Python 3.9+ from https://python.org
    pause
    exit /b 1
)

REM Get Python version
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VER=%%i
echo Detected Python version: %PYTHON_VER%

REM Navigate to project directory
cd /d "%~dp0\.."

echo.
echo Installation options:
echo   1) Minimal (CLI only, no GUI)
echo   2) Standard (GUI included)
echo   3) Full (GUI + all optional features)
echo.
set /p option="Select option [2]: "
if "%option%"=="" set option=2

REM Create virtual environment
if not exist "venv" (
    echo.
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Upgrade pip
echo.
echo Upgrading pip...
python -m pip install --upgrade pip

REM Install based on option
if "%option%"=="1" (
    echo.
    echo Installing minimal version...
    pip install -r requirements-minimal.txt
) else if "%option%"=="3" (
    echo.
    echo Installing full version...
    pip install -r requirements.txt
    pip install PyWavelets scikit-image
) else (
    echo.
    echo Installing standard version with GUI...
    pip install -r requirements.txt
)

REM Install the package
echo.
echo Installing HR-TEM Analyzer...
pip install -e .

echo.
echo ========================================
echo   Installation Complete!
echo ========================================
echo.
echo To run the GUI:
echo   venv\Scripts\activate
echo   python scripts\run_gui.py
echo.
echo Or create a shortcut to run_gui.bat
echo.
pause
