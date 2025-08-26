#!/usr/bin/env bash
# ================================================
# Build both executables for Badminton Matchmaking (Linux)
# Requires: Python 3 + pip + pyinstaller installed
# ================================================

set -e

echo "Cleaning old builds..."
rm -rf build dist *.spec

echo "Installing PyInstaller if missing..."
if ! pip3 show pyinstaller > /dev/null 2>&1; then
    pip3 install pyinstaller
fi

echo "Building BadmintonSetup-linux (player recorder)..."
pyinstaller --onefile --clean --noconfirm --name BadmintonSetup-linux app.py

echo "Building BadmintonSession-linux (session scheduler)..."
pyinstaller --onefile --clean --noconfirm --name BadmintonSession-linux session_ui.py

echo "================================"
echo "Done! Check the dist/ folder for:"
echo "  BadmintonSetup-linux"
echo "  BadmintonSession-linux"
echo "================================"
