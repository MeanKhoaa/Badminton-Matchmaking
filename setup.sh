#!/usr/bin/env bash
set -euo pipefail

# cd to repo root (this script's folder)
cd "$(dirname "$0")"

# pick python command
if command -v python3 >/dev/null 2>&1; then
  PY=python3
elif command -v python >/dev/null 2>&1; then
  PY=python
else
  cat <<'EOF'
Python is not installed.
Please install Python 3.10+ from https://www.python.org/downloads/
(On macOS you can also: brew install python)
EOF
  exit 1
fi
echo "Python found: $($PY --version)"

# create output dirs
mkdir -p outputs logs

# run
exec "$PY" session_ui.py
