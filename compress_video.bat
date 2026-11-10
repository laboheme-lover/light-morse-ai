@echo off
rem Drag and drop video files onto this file. It writes a small copy (<name>_small.mp4) next to each video
rem for the web version (960x540, MPEG-4, same frame rate).
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  echo Run setup.bat first.
  pause
  exit /b 1
)
if "%~1"=="" (
  echo Drag and drop video files onto compress_video.bat.
  pause
  exit /b 1
)
.venv\Scripts\python.exe compress_video.py %*
pause
