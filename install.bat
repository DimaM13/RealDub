@echo off
chcp 65001 >nul
cd /d "%~dp0"
if not exist .venv (
  echo Створюю віртуальне середовище...
  python -m venv .venv
)
call .venv\Scripts\activate.bat
python -m pip install -U pip
pip install -r requirements.txt
pip install nvidia-cublas-cu12 nvidia-cudnn-cu12==9.*
python download_models.py
pause
