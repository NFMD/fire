#!/usr/bin/env python3
"""
HR-TEM Analyzer GUI Launcher

Launch the graphical user interface for HR-TEM image analysis.

Usage:
    python run_gui.py
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from gui import run_app


def main():
    """Main entry point"""
    run_app()


if __name__ == '__main__':
    main()
