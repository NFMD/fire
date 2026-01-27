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

```bash
cd hrtem_analyzer
pip install -r requirements.txt
```

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

## Requirements

- Python 3.8+
- numpy, opencv-python, tifffile, scikit-image
- scipy (for statistics)
- loguru (for logging)

## License

MIT License
