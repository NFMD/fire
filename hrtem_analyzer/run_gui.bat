@echo off
REM HR-TEM Analyzer GUI Launcher (Windows)
REM ======================================
REM Double-click this file to run the GUI

cd /d "%~dp0"

REM Check if virtual environment exists
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
)

python scripts\run_gui.py

pause
