"""
Core modules for HR-TEM analysis

Includes:
- Image loading (TIFF, DM3/DM4)
- Preprocessing and denoising
- Edge detection and measurement
- Scale bar detection (OCR)
- Advanced sub-pixel analysis
- Phase congruency edge detection
"""
from .image_loader import TIFFLoader, UniversalImageLoader, ScaleInfo
from .preprocessor import ImagePreprocessor
from .baseline_detector import BaselineDetector
from .thickness_measurer import ThicknessMeasurer, MeasurementResult
from .result_exporter import ResultExporter, NumpyEncoder, convert_numpy_types

# Advanced analysis (Gatan DM-style features)
from .advanced_analysis import (
    LineProfileAnalyzer,
    FFTAnalyzer,
    DriftCorrector,
    BackgroundSubtractor,
    IntensityCalibrator,
    StatisticalAnalyzer,
)
from .enhanced_measurer import EnhancedCDMeasurer, EnhancedMeasurementResult, HybridMeasurer
from .scale_bar_detector import ScaleBarDetector

# Precision measurement (sub-pixel, ESF/LSF, wavelet, Monte Carlo)
from .precision_measurement import (
    PrecisionCDMeasurer,
    PrecisionMeasurementResult,
    SubPixelMethod,
    SubPixelEdgeDetector,
    ESFLSFAnalyzer,
    AdvancedDenoiser as PrecisionDenoiser,
    MultiScaleWaveletAnalyzer,
    MonteCarloUncertainty,
    AtomicColumnFitter,
)

# State-of-the-art edge fitting
from .advanced_edge_fitting import (
    ESFLSFFitter,
    GaussianProcessEdgeFitter,
    WaveletEdgeFitter,
    MCMCEdgeFitter,
    EnsembleEdgeFitter,
    EdgeFitResult,
    fit_edge_profile,
)

# Advanced denoising and phase congruency
from .advanced_denoising import (
    AdvancedDenoiser,
    PhaseCongruencyEdgeDetector,
    denoise_tem_image,
    detect_edges_phase_congruency,
)

__all__ = [
    # Basic modules
    'TIFFLoader',
    'UniversalImageLoader',
    'ScaleInfo',
    'ImagePreprocessor',
    'BaselineDetector',
    'ThicknessMeasurer',
    'MeasurementResult',
    'ResultExporter',
    'NumpyEncoder',
    'convert_numpy_types',
    # Advanced analysis
    'LineProfileAnalyzer',
    'FFTAnalyzer',
    'DriftCorrector',
    'BackgroundSubtractor',
    'IntensityCalibrator',
    'StatisticalAnalyzer',
    'EnhancedCDMeasurer',
    'EnhancedMeasurementResult',
    'HybridMeasurer',
    'ScaleBarDetector',
    # Precision measurement
    'PrecisionCDMeasurer',
    'PrecisionMeasurementResult',
    'SubPixelMethod',
    'SubPixelEdgeDetector',
    'ESFLSFAnalyzer',
    'PrecisionDenoiser',
    'MultiScaleWaveletAnalyzer',
    'MonteCarloUncertainty',
    'AtomicColumnFitter',
    # State-of-the-art edge fitting
    'ESFLSFFitter',
    'GaussianProcessEdgeFitter',
    'WaveletEdgeFitter',
    'MCMCEdgeFitter',
    'EnsembleEdgeFitter',
    'EdgeFitResult',
    'fit_edge_profile',
    # Advanced denoising
    'AdvancedDenoiser',
    'PhaseCongruencyEdgeDetector',
    'denoise_tem_image',
    'detect_edges_phase_congruency',
]
