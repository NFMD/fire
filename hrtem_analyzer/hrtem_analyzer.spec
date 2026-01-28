# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for HR-TEM Analyzer

Build commands:
    Windows: pyinstaller hrtem_analyzer.spec
    macOS:   pyinstaller hrtem_analyzer.spec
    Linux:   pyinstaller hrtem_analyzer.spec

For one-folder distribution (faster startup):
    pyinstaller hrtem_analyzer.spec

For single-file distribution (slower startup but easier to distribute):
    pyinstaller --onefile hrtem_analyzer.spec
"""

import sys
from pathlib import Path

# Get the project root directory
project_root = Path(SPECPATH)

block_cipher = None

# Collect all source files
a = Analysis(
    [str(project_root / 'scripts' / 'run_gui.py')],
    pathex=[str(project_root), str(project_root / 'src')],
    binaries=[],
    datas=[
        # Include config files
        (str(project_root / 'config'), 'config'),
    ],
    hiddenimports=[
        # PyQt6 components
        'PyQt6.QtCore',
        'PyQt6.QtGui',
        'PyQt6.QtWidgets',
        'PyQt6.sip',
        # Scientific computing
        'numpy',
        'numpy.core._methods',
        'numpy.lib.format',
        'scipy',
        'scipy.ndimage',
        'scipy.signal',
        'scipy.optimize',
        'scipy.interpolate',
        'cv2',
        'PIL',
        'PIL.Image',
        'tifffile',
        # Optional wavelets
        'pywt',
        # Logging
        'loguru',
        # Core modules
        'src.core',
        'src.core.image_loader',
        'src.core.preprocessor',
        'src.core.baseline_detector',
        'src.core.thickness_measurer',
        'src.core.result_exporter',
        'src.core.advanced_analysis',
        'src.core.enhanced_measurer',
        'src.core.precision_measurement',
        'src.pipeline',
        'src.pipeline.inference_pipeline',
        'src.pipeline.batch_processor',
        'src.gui',
        'src.gui.main_window',
        'src.gui.widgets',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Exclude unnecessary modules to reduce size
        'torch',
        'torchvision',
        'tensorflow',
        'keras',
        'matplotlib',
        'pandas',
        'IPython',
        'jupyter',
        'notebook',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='HRTEM-Analyzer',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,  # Set to True for debugging
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    # icon='resources/icon.ico',  # Uncomment if you have an icon
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='HRTEM-Analyzer',
)

# For macOS app bundle
if sys.platform == 'darwin':
    app = BUNDLE(
        coll,
        name='HRTEM-Analyzer.app',
        icon=None,  # Set path to .icns file if available
        bundle_identifier='com.hrtem.analyzer',
        info_plist={
            'NSHighResolutionCapable': 'True',
            'CFBundleShortVersionString': '1.0.0',
        },
    )
