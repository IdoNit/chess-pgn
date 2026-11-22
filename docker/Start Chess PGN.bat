@echo off
chcp 65001 >nul
set "CHESSPGN_DIR=%USERPROFILE%\Documents\Chess PGN"
if not exist "%CHESSPGN_DIR%" mkdir "%CHESSPGN_DIR%"
cd /d "%~dp0"
docker compose up -d --no-build
if errorlevel 1 (
  echo Docker Desktop is not running. Start it, wait a minute, and try again.
  pause
  exit /b 1
)
for /l %%i in (1,1,60) do (
  curl -s -o nul http://localhost:8321/ && goto open
  timeout /t 2 >nul
)
:open
start "" http://localhost:8321/
