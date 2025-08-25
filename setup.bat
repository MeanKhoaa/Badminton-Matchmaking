@echo off
title Badminton Matchmaking - Setup & Run

REM --- Check Python ---
where python >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
  where python3 >nul 2>nul
  if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Python is not installed.
    echo Please install Python 3.10+ from https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during install.
    pause
    exit /b 1
  ) else (
    set "PY=python3"
  )
) else (
  set "PY=python"
)

echo Python found: %PY%
echo.

REM --- Move to script directory (repo root) ---
cd /d "%~dp0"

REM --- Create outputs/logs folders (if missing) ---
if not exist outputs mkdir outputs
if not exist logs mkdir logs

REM --- Run session UI ---
%PY% session_ui.py
if %ERRORLEVEL% NEQ 0 (
  echo.
  echo Something went wrong running session_ui.py
  pause
  exit /b 1
)

echo.
echo Done.
pause
