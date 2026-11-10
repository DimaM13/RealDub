@echo off
cd /d "%~dp0"
title RealDub Installer

echo ===================================================
echo               RealDub Installer
echo ===================================================

if exist ".venv\Scripts\python.exe" (
    start "" ".venv\Scripts\python.exe" installer.py
    exit /b 0
)

where python >nul 2>nul
if %errorlevel% equ 0 (
    python -m pip install -q PySide6
    start "" python installer.py
    exit /b 0
)

where py >nul 2>nul
if %errorlevel% equ 0 (
    py -m pip install -q PySide6
    start "" py installer.py
    exit /b 0
)

echo [Error] Python 3 not found in PATH!
echo Please download Python from: https://www.python.org/downloads/
echo Make sure to check "Add Python to PATH" during installation.
pause
