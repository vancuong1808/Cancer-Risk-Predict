@echo off
REM Run Cancer Risk app with project venv (double-click this file)
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo [ERROR] .venv not found. Create it first:
  echo C:\Users\ADMIN\AppData\Local\Programs\Python\Python310\python.exe -m venv .venv
  echo .venv\Scripts\python.exe -m pip install -r requirements.txt
  pause
  exit /b 1
)
.venv\Scripts\python.exe -m streamlit run app/app.py
pause
