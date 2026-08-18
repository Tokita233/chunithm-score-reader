@echo off
chcp 65001 >nul
cd /d "%~dp0"
where py >nul 2>nul
if errorlevel 1 (
  echo 未找到 Python。请先从 https://www.python.org/downloads/ 安装 Python 3.11 或 3.12。
  pause
  exit /b 1
)
if not exist ".venv\Scripts\python.exe" (
  echo 首次运行：正在创建环境并安装 OCR 组件...
  py -3 -m venv .venv
  call .venv\Scripts\activate.bat
  python -m pip install --upgrade pip
  pip install -r requirements.txt
) else (
  call .venv\Scripts\activate.bat
)
python app.py
pause
