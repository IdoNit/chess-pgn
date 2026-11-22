@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo Loading Chess PGN images, this takes a few minutes...
docker load -i chesspgn-images.tar
if errorlevel 1 (
  echo Docker Desktop is not running. Start it and try again.
  pause
  exit /b 1
)
echo Done. Use "Start Chess PGN" from now on.
pause
