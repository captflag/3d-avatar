@echo off
rem Start the avatar maker. Double-click this file, or run it from a terminal.
rem Leave the window open; closing it stops the server. Add --dev for the team's
rem tools (testing studio, review, gallery), --lan to open it to the local network.
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo No virtualenv yet. See README.md "Set up" first.
  pause
  exit /b 1
)
if not exist "assets\face_landmarker.task" (
  echo Model files missing. See README.md "Set up" first.
  pause
  exit /b 1
)

echo Starting the avatar maker. Open http://localhost:8777
echo It listens on this PC only. For players on the same network, run
echo   .venv\Scripts\python.exe app.py --lan
echo.
.venv\Scripts\python.exe app.py %*
pause
