"""
Deep Learning Module for HR-TEM CD Measurement

Provides lightweight neural network models optimized for
CPU and integrated GPU (Intel UHD, AMD Vega, etc.)

Components:
- models: Neural network architectures
- dataset: Training data management
- trainer: Training pipeline
- inference: Prediction API
"""

# Check for PyTorch availability
try:
    import torch
    PYTORCH_AVAILABLE = True
    PYTORCH_VERSION = torch.__version__
except ImportError:
    PYTORCH_AVAILABLE = False
    PYTORCH_VERSION = None

from .models import (
    create_model,
    get_device,
    CDMeasurementNet,
    EdgeSegmentationNet,
    EnsembleModel,
    MODEL_INFO,
)

from .dataset import (
    TrainingDataManager,
    ImageAnnotation,
    CDAnnotation,
    DataAugmentor,
)

from .trainer import (
    Trainer,
    TrainingConfig,
    train_model,
)

from .inference import (
    DeepLearningInference,
    InferenceResult,
    CDPrediction,
    HybridMeasurer,
    load_inference_engine,
)

__all__ = [
    # Availability checks
    'PYTORCH_AVAILABLE',
    'PYTORCH_VERSION',
    # Models
    'create_model',
    'get_device',
    'CDMeasurementNet',
    'EdgeSegmentationNet',
    'EnsembleModel',
    'MODEL_INFO',
    # Dataset
    'TrainingDataManager',
    'ImageAnnotation',
    'CDAnnotation',
    'DataAugmentor',
    # Training
    'Trainer',
    'TrainingConfig',
    'train_model',
    # Inference
    'DeepLearningInference',
    'InferenceResult',
    'CDPrediction',
    'HybridMeasurer',
    'load_inference_engine',
]


def check_requirements():
    """Check if all requirements are met for deep learning"""
    issues = []

    if not PYTORCH_AVAILABLE:
        issues.append("PyTorch not installed. Install with: pip install torch")

    try:
        import numpy
    except ImportError:
        issues.append("NumPy not installed. Install with: pip install numpy")

    return {
        'ready': len(issues) == 0,
        'issues': issues,
        'pytorch_version': PYTORCH_VERSION,
        'device': str(get_device()) if PYTORCH_AVAILABLE else 'N/A',
    }
