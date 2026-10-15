@echo off
rem Start the avatar studio for the team. Double-click this file, or run it from
rem a terminal. Leave the window open; closing it stops the server.
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

echo Starting avatar studio. Open http://localhost:8777
echo Teammates on the same network use this PC's address, shown below.
echo.
.venv\Scripts\python.exe app.py
pause
