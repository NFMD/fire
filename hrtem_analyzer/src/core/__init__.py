"""
Core modules for HR-TEM analysis
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
    AdvancedDenoiser,
    MultiScaleWaveletAnalyzer,
    MonteCarloUncertainty,
    AtomicColumnFitter,
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
    'AdvancedDenoiser',
    'MultiScaleWaveletAnalyzer',
    'MonteCarloUncertainty',
    'AtomicColumnFitter',
]
