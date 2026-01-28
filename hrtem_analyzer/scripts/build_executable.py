#!/usr/bin/env python3
"""
Build executable for HR-TEM Analyzer

Creates a standalone executable that can be distributed
to users without Python installed.

Requirements:
    pip install pyinstaller

Usage:
    python build_executable.py [--onefile]

Options:
    --onefile    Create a single executable file (slower startup)
    --clean      Clean build directories before building
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def clean_build_dirs(project_root: Path):
    """Clean build directories"""
    dirs_to_clean = ['build', 'dist', '__pycache__']

    for dir_name in dirs_to_clean:
        dir_path = project_root / dir_name
        if dir_path.exists():
            print(f"Removing {dir_path}...")
            shutil.rmtree(dir_path)

    # Clean .pyc files
    for pyc in project_root.rglob('*.pyc'):
        pyc.unlink()

    # Clean .pyo files
    for pyo in project_root.rglob('*.pyo'):
        pyo.unlink()


def build_executable(onefile: bool = False, clean: bool = False):
    """Build the executable"""
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)

    # Check if pyinstaller is installed
    try:
        import PyInstaller
        print(f"PyInstaller version: {PyInstaller.__version__}")
    except ImportError:
        print("PyInstaller not found. Installing...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'pyinstaller'])

    # Clean if requested
    if clean:
        print("\n=== Cleaning build directories ===")
        clean_build_dirs(project_root)

    print("\n=== Building HR-TEM Analyzer ===")
    print(f"Platform: {sys.platform}")
    print(f"Python: {sys.version}")

    # Build command
    cmd = [sys.executable, '-m', 'PyInstaller']

    if onefile:
        print("Mode: Single file executable")
        cmd.extend(['--onefile', '--windowed'])
        cmd.extend(['--name', 'HRTEM-Analyzer'])
        cmd.append('scripts/run_gui.py')
    else:
        print("Mode: Directory distribution")
        cmd.append('hrtem_analyzer.spec')

    # Add common options
    cmd.extend(['--noconfirm'])

    print(f"\nRunning: {' '.join(cmd)}")
    subprocess.check_call(cmd)

    # Report output location
    dist_dir = project_root / 'dist'
    print("\n=== Build Complete ===")

    if onefile:
        exe_name = 'HRTEM-Analyzer.exe' if sys.platform == 'win32' else 'HRTEM-Analyzer'
        exe_path = dist_dir / exe_name
        if exe_path.exists():
            size_mb = exe_path.stat().st_size / (1024 * 1024)
            print(f"Executable: {exe_path}")
            print(f"Size: {size_mb:.1f} MB")
    else:
        app_dir = dist_dir / 'HRTEM-Analyzer'
        if app_dir.exists():
            total_size = sum(f.stat().st_size for f in app_dir.rglob('*') if f.is_file())
            size_mb = total_size / (1024 * 1024)
            print(f"Application folder: {app_dir}")
            print(f"Total size: {size_mb:.1f} MB")

    print("\n=== Distribution Instructions ===")
    if onefile:
        print("1. Copy the executable to your target machine")
        print("2. Run it directly (no installation needed)")
    else:
        print("1. Copy the entire 'HRTEM-Analyzer' folder to your target machine")
        print("2. Run 'HRTEM-Analyzer.exe' inside the folder")

    if sys.platform == 'darwin':
        print("\nFor macOS: You can also distribute the .app bundle")


def main():
    parser = argparse.ArgumentParser(description='Build HR-TEM Analyzer executable')
    parser.add_argument('--onefile', action='store_true',
                        help='Create single file executable')
    parser.add_argument('--clean', action='store_true',
                        help='Clean build directories before building')

    args = parser.parse_args()
    build_executable(onefile=args.onefile, clean=args.clean)


if __name__ == '__main__':
    main()
