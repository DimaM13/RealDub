@echo off
chcp 65001 >nul
cd /d "%~dp0"
title Встановлення RealDub

echo ===================================================
echo     🌐 Запуск Майстра Встановлення RealDub
echo ===================================================

where python >nul 2>nul
if %errorlevel% neq 0 (
    echo.
    echo ❌ Python не знайдено в системі!
    echo Для роботи програми потрібен Python 3.10, 3.11 або 3.12.
    echo Будь ласка, завантажте його з офіційного сайту: https://www.python.org/downloads/
    echo (Обов'язково поставте галочку "Add Python to PATH" під час встановлення).
    echo.
    pause
    exit /b 1
)

if exist .venv\Scripts\python.exe (
    start "" .venv\Scripts\python.exe installer.py
) else (
    python -m pip install -q PySide6
    start "" python installer.py
)
