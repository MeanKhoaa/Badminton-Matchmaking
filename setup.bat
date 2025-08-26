@echo off
REM ================================================
REM Build both EXE files for Badminton Matchmaking
REM Requires: Python + pip + pyinstaller installed
REM ================================================

echo Cleaning old builds...
rmdir /s /q build dist
del /q *.spec

echo Installing PyInstaller if missing...
pip show pyinstaller >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    pip install pyinstaller
)

echo Building BadmintonSetup.exe (player recorder)...
pyinstaller --onefile --clean --noconfirm --name BadmintonSetup app.py

echo Building BadmintonSession.exe (session scheduler)...
pyinstaller --onefile --clean --noconfirm --name BadmintonSession session_ui.py

echo ================================================
echo Done! Check the dist\ folder for:
echo   BadmintonSetup.exe
echo   BadmintonSession.exe
echo ================================================

pause
