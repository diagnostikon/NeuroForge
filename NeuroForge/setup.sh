#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

echo "========================================"
echo "  Rebirth AI — Setup"
echo "========================================"
echo ""

# ---------- Python version ----------
PYTHON=$(command -v python3 || command -v python || echo "")
if [ -z "$PYTHON" ]; then
    echo "ERROR: Python not found. Install Python 3.8+ first."
    exit 1
fi

PY_VER=$("$PYTHON" --version 2>&1 | grep -oP '\d+\.\d+')
PY_MAJ=$(echo "$PY_VER" | cut -d. -f1)
PY_MIN=$(echo "$PY_VER" | cut -d. -f2)

if [ "$PY_MAJ" -lt 3 ] || { [ "$PY_MAJ" -eq 3 ] && [ "$PY_MIN" -lt 8 ]; }; then
    echo "ERROR: Python 3.8+ required (found $PY_VER)"
    exit 1
fi
echo "  Python $PY_VER — OK"

# ---------- Virtual environment (optional) ----------
if [ -n "${VIRTUAL_ENV:-}" ]; then
    echo "  Virtual env active: $VIRTUAL_ENV"
else
    if [ -d .venv ]; then
        echo "  Activating existing .venv ..."
        source .venv/bin/activate
    else
        echo "  Creating virtual environment (.venv) ..."
        "$PYTHON" -m venv .venv
        source .venv/bin/activate
    fi
    PYTHON="python"
fi

# ---------- Dependencies ----------
echo ""
echo "  Installing numpy ..."
"$PYTHON" -m pip install --quiet --upgrade pip
"$PYTHON" -m pip install --quiet numpy

# ---------- Verify ----------
echo ""
echo "  Verifying import ..."
"$PYTHON" -c "import numpy; print(f'    numpy {numpy.__version__} — OK')"

# ---------- Run tests ----------
echo ""
echo "  Running tests ..."
"$PYTHON" tests.py

echo ""
echo "========================================"
echo "  Setup complete!"
echo "========================================"
echo ""
echo "  Start chatting:"
echo "    $PYTHON chat.py"
echo ""
echo "  Train from scratch:"
echo "    rm language_learner.pkl && $PYTHON chat.py"
echo ""
