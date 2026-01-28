#!/bin/bash
# HR-TEM Analyzer Installation Script (Linux/macOS)
# =================================================

set -e

echo "========================================"
echo "  HR-TEM Analyzer Installer"
echo "========================================"
echo ""

# Check Python version
PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "Detected Python version: $PYTHON_VERSION"

# Check if Python >= 3.9
if python3 -c "import sys; exit(0 if sys.version_info >= (3, 9) else 1)"; then
    echo "✓ Python version OK"
else
    echo "✗ Error: Python 3.9 or higher is required"
    echo "  Please install Python 3.9+ and try again"
    exit 1
fi

# Navigate to script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

echo ""
echo "Installation options:"
echo "  1) Minimal (CLI only, no GUI)"
echo "  2) Standard (GUI included)"
echo "  3) Full (GUI + all optional features)"
echo ""
read -p "Select option [2]: " option
option=${option:-2}

# Create virtual environment if not in one
if [ -z "$VIRTUAL_ENV" ]; then
    echo ""
    echo "Creating virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    echo "✓ Virtual environment created and activated"
fi

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Install based on option
case $option in
    1)
        echo ""
        echo "Installing minimal version..."
        pip install -r requirements-minimal.txt
        ;;
    2)
        echo ""
        echo "Installing standard version with GUI..."
        pip install -r requirements.txt
        ;;
    3)
        echo ""
        echo "Installing full version..."
        pip install -r requirements.txt
        pip install PyWavelets scikit-image
        ;;
    *)
        echo "Invalid option. Installing standard version..."
        pip install -r requirements.txt
        ;;
esac

# Install the package in development mode
echo ""
echo "Installing HR-TEM Analyzer..."
pip install -e .

echo ""
echo "========================================"
echo "  Installation Complete!"
echo "========================================"
echo ""
echo "To run the GUI:"
echo "  source venv/bin/activate"
echo "  python scripts/run_gui.py"
echo ""
echo "Or from command line:"
echo "  hrtem-analyze --help"
echo ""
