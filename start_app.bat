@echo off
rem Start the light Morse app in the browser (port 8503, uploads up to 1 GB).
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  echo Run setup.bat first.
  pause
  exit /b 1
)
.venv\Scripts\python.exe -m streamlit run app.py --server.port 8503 --server.maxUploadSize 1024
