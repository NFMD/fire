# HR-TEM Image Analyzer

Automated Critical Dimension (CD) measurement system for High-Resolution Transmission Electron Microscopy (HR-TEM) images using multi-method analysis and deep learning.

## Features

- **Automatic Scale Extraction**: Reads scale information from TIFF metadata (FEI, JEOL, Hitachi, ImageJ)
- **Auto-Leveling**: Automatic image orientation correction using Hough transform
- **Multi-Method CD Measurement**: Uses multiple edge detection algorithms for high precision
- **Parallel Processing**: Process up to 20 images in parallel per working group
- **Memory Efficient**: Streaming processing with automatic memory management
- **Consensus Algorithms**: Combines measurements using trimmed mean, median, or weighted average
- **Result Visualization**: Exports annotated JPEG images with measurements and scale bars

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    HR-TEM Analyzer Pipeline                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Input TIFF ──► Scale Extraction ──► Auto-Leveling ──► Baseline     │
│                                                          Detection   │
│                                                              │       │
│                                                              ▼       │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │              Multi-Variant Processing                         │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐          │   │
│  │  │ Preprocess  │  │ Preprocess  │  │ Preprocess  │   ...    │   │
│  │  │ Original    │  │ CLAHE       │  │ Bilateral   │          │   │
│  │  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘          │   │
│  │         │                │                │                  │   │
│  │         ▼                ▼                ▼                  │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐          │   │
│  │  │ Rotation    │  │ Rotation    │  │ Rotation    │   ...    │   │
│  │  │ -2°,-1°,0°  │  │ -2°,-1°,0°  │  │ -2°,-1°,0°  │          │   │
│  │  │ +1°,+2°     │  │ +1°,+2°     │  │ +1°,+2°     │          │   │
│  │  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘          │   │
│  │         │                │                │                  │   │
│  │         └────────────────┼────────────────┘                  │   │
│  │                          ▼                                   │   │
│  │  ┌──────────────────────────────────────────────────────┐   │   │
│  │  │           Multi-Method Edge Detection                 │   │   │
│  │  │  Sobel │ Canny │ Laplacian │ Gradient │ Morphological │   │   │
│  │  └──────────────────────────────────────────────────────┘   │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                    │                                 │
│                                    ▼                                 │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                  Consensus & Aggregation                      │   │
│  │   - Outlier removal (3σ)                                     │   │
│  │   - Trimmed mean / Median / Weighted mean                    │   │
│  │   - Statistical confidence calculation                        │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                    │                                 │
│                                    ▼                                 │
│  Output: JPEG with annotations + JSON/CSV data                      │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

## Installation

### Option 1: Standalone Executable (No Python Required)

Download the pre-built executable for your platform:
- **Windows**: `HRTEM-Analyzer.exe`
- **macOS**: `HRTEM-Analyzer.app`
- **Linux**: `HRTEM-Analyzer`

Just double-click to run - no installation needed!

### Option 2: Quick Install Script

**Linux/macOS:**
```bash
cd hrtem_analyzer
chmod +x scripts/install.sh
./scripts/install.sh
```

**Windows:**
```cmd
cd hrtem_analyzer
scripts\install.bat
```

### Option 3: pip Install

```bash
cd hrtem_analyzer

# Standard installation (with GUI)
pip install -r requirements.txt
pip install -e .

# Or minimal (CLI only)
pip install -r requirements-minimal.txt
pip install -e .

# Or full installation (all features)
pip install -e ".[full]"
```

### Option 4: Install from PyPI (Coming Soon)

```bash
pip install hrtem-analyzer
pip install hrtem-analyzer[gui]    # with GUI
pip install hrtem-analyzer[full]   # all features
```

## Building Standalone Executable

To create an executable for distribution:

```bash
# Install PyInstaller
pip install pyinstaller

# Build directory distribution (recommended)
python scripts/build_executable.py

# Or single file (slower startup but easier to share)
python scripts/build_executable.py --onefile

# With clean build
python scripts/build_executable.py --clean
```

The executable will be in the `dist/` folder.

## Building Installer (.exe Setup)

For a professional installer that users can run to install the application:

### Windows Installer (.exe Setup)

```bash
# 1. Install Inno Setup from https://jrsoftware.org/isinfo.php
# 2. Build the installer
python scripts/build_installer.py
```

This creates `installer_output/HRTEM-Analyzer-Setup-1.0.0.exe`

### macOS Installer (.dmg)

```bash
python scripts/build_installer.py
```

Creates `installer_output/HRTEM-Analyzer-1.0.0-macOS.dmg`

### Linux Packages (.deb, .rpm)

```bash
# Install fpm first: gem install fpm
python scripts/build_installer.py
```

Creates `.deb` and `.rpm` packages in `installer_output/`

## Quick Start

### Graphical User Interface (GUI)

The easiest way to use HR-TEM Analyzer is through the GUI:

```bash
python scripts/run_gui.py
```

**GUI Features:**
- Drag & drop TIFF images
- Interactive baseline (0-point) selection with click
- Real-time image preview with zoom/pan
- Configure all measurement parameters
- Visual progress tracking
- Results table with statistics
- Export to JPEG/JSON/CSV

![GUI Screenshot](docs/gui_screenshot.png)

### Command Line Interface (CLI)

#### Single Image Analysis

```bash
python scripts/analyze.py single sample.tiff -o results/ -d 5 10 15 20
```

### Batch Processing

```bash
python scripts/analyze.py batch img1.tiff img2.tiff img3.tiff -o results/ -w 4
```

### Directory Processing

```bash
python scripts/analyze.py directory ./images/ -o results/ --pattern "*.tif*"
```

### High Precision Mode

```bash
python scripts/analyze.py directory ./images/ -o results/ --high-precision -w 8
```

## Python API

```python
from src.pipeline.inference_pipeline import create_pipeline

# Create pipeline
pipeline = create_pipeline(
    output_dir='./results',
    depths_nm=[5, 10, 15, 20, 25],
    max_workers=4,
    high_precision=True
)

# Process single image
result = pipeline.process_single('sample.tiff')

print(f"Measurements at depth 10nm: {result['measurements'][10]['thickness_nm']:.2f} nm")

# Process batch
results = pipeline.process_batch(['img1.tiff', 'img2.tiff', 'img3.tiff'])
```

## Configuration

### Pipeline Configuration

```python
from src.pipeline.inference_pipeline import PipelineConfig

config = PipelineConfig(
    # Measurement depths (nm from baseline)
    depths_nm=[5, 10, 15, 20, 25],

    # Preprocessing variants
    preprocessing_methods=['original', 'clahe', 'bilateral_filter', 'gaussian_blur'],

    # Rotation angles for multi-angle analysis
    rotation_angles=[-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0],

    # Edge detection methods
    edge_methods=['sobel', 'canny', 'gradient', 'morphological', 'scharr'],

    # Consensus settings
    consensus_method='trimmed_mean',  # 'mean', 'median', 'trimmed_mean', 'weighted_mean'
    trim_percentage=0.1,

    # Parallel processing
    max_workers=4,
    memory_limit_mb=4096,
)
```

## Output Files

### Visualization (JPEG)

Annotated image with:
- Baseline (0-point) marker in red
- Measurement lines at each depth in different colors
- Edge markers with thickness values
- Scale bar
- Information panel with statistics

### Data (JSON)

```json
{
  "source_path": "sample.tiff",
  "timestamp": "2024-01-15T10:30:00",
  "scale": {
    "scale_nm_per_pixel": 0.1234,
    "source": "fei_metadata"
  },
  "baseline": {
    "y_position": 256,
    "confidence": 0.95,
    "method": "gradient"
  },
  "measurements": {
    "10.0": {
      "thickness_nm": 42.35,
      "thickness_std": 0.82,
      "thickness_min": 40.5,
      "thickness_max": 44.2,
      "confidence": 0.91,
      "num_measurements": 35
    }
  }
}
```

### Summary (CSV)

| filename | scale_nm_per_pixel | baseline_y | thickness_10nm | std_10nm | ... |
|----------|-------------------|------------|----------------|----------|-----|
| img1.tiff | 0.1234 | 256 | 42.35 | 0.82 | ... |

## Multi-Method Precision

The analyzer achieves high precision by combining:

1. **5 Preprocessing Methods**: Original, CLAHE, Bilateral Filter, Gaussian Blur, Median Filter
2. **7 Rotation Angles**: -2°, -1°, -0.5°, 0°, +0.5°, +1°, +2°
3. **6 Edge Detection Methods**: Sobel, Canny, Laplacian, Gradient, Morphological, Scharr
4. **5-7 Measurement Lines per Depth**

Total: Up to **5 × 7 × 6 × 7 = 1,470 individual measurements** per depth, aggregated using robust statistics.

## Memory Management

- Working groups limited to 20 images
- Streaming variant generation (one at a time)
- Automatic garbage collection between batches
- Memory monitoring with configurable limits

## Advanced Features

### Gatan Digital Micrograph-Style Analysis
- **Line Profile Analysis**: FWHM, 10-90% threshold, derivative-based, sigmoid fitting
- **FFT Calibration**: Automatic scale verification using known lattice spacing
- **Background Subtraction**: Rolling ball, polynomial, top-hat, gaussian
- **Drift Correction**: Cross-correlation based drift compensation

### Precision Measurement Mode
- **Sub-pixel Edge Detection**: Gaussian, parabolic, centroid, spline interpolation
- **ESF/LSF Analysis**: Edge Spread Function and Line Spread Function analysis
- **Advanced Denoising**: Non-local means, bilateral, wavelet, anisotropic diffusion
- **Multi-scale Wavelet**: Robust edge detection across multiple scales
- **Monte Carlo Uncertainty**: Statistical uncertainty quantification
- **Atomic Column Fitting**: For crystalline materials

## Requirements

### Minimum Requirements (CLI)
- Python 3.9+
- numpy, opencv-python, Pillow, tifffile, scipy, loguru

### GUI Requirements
- PyQt6 >= 6.5.0

### Full Features
- PyWavelets (for wavelet analysis)
- scikit-image (for additional algorithms)

## Distribution to Others

### For Users Without Python

1. Build the executable:
   ```bash
   python scripts/build_executable.py
   ```

2. Share the `dist/HRTEM-Analyzer` folder or `HRTEM-Analyzer.exe`

3. Users can run it directly without installing anything

### For Users With Python

Share a ZIP with instructions to run:
```bash
pip install -r requirements.txt
python scripts/run_gui.py
```

## License

MIT License
