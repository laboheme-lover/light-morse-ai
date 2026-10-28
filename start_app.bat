@echo off
rem Start the light Morse app in the browser (port 8503, uploads up to 1 GB).
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  echo Run setup.bat first.
  pause
  exit /b 1
)
rem headless: skip Streamlit's first-run e-mail question. Open the browser after the server has started.
start "" /b cmd /c "ping -n 7 127.0.0.1 >nul & start http://localhost:8503"
.venv\Scripts\python.exe -m streamlit run app.py --server.headless true --server.port 8503 --server.maxUploadSize 1024
