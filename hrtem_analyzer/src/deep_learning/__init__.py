"""
Deep Learning Module for HR-TEM CD Measurement

Provides neural network models optimized for both
CPU and GPU, including state-of-the-art architectures.

Components:
- models: Standard neural network architectures (MobileNet, EfficientNet)
- advanced_models: State-of-the-art models (Attention U-Net, Swin Transformer, Deep Ensemble)
- dataset: Training data management
- trainer: Standard training pipeline
- advanced_trainer: Advanced training with mixed precision, multi-loss, uncertainty
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
    EfficientNetEncoder,
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

# Advanced modules (state-of-the-art)
try:
    from .advanced_models import (
        create_advanced_model,
        AttentionUNet,
        SwinUNet,
        DeepEnsemble,
        HybridCDNet,
        ADVANCED_MODEL_INFO,
    )
    from .advanced_trainer import (
        AdvancedTrainer,
        AdvancedTrainerConfig,
        EnsembleTrainer,
        CombinedLoss,
        DiceLoss,
        FocalLoss,
        BoundaryLoss,
    )
    ADVANCED_MODELS_AVAILABLE = True
except ImportError:
    ADVANCED_MODELS_AVAILABLE = False
    create_advanced_model = None
    AttentionUNet = None
    SwinUNet = None
    DeepEnsemble = None
    HybridCDNet = None
    ADVANCED_MODEL_INFO = {}
    AdvancedTrainer = None
    AdvancedTrainerConfig = None
    EnsembleTrainer = None

__all__ = [
    # Availability checks
    'PYTORCH_AVAILABLE',
    'PYTORCH_VERSION',
    'ADVANCED_MODELS_AVAILABLE',
    # Models
    'create_model',
    'get_device',
    'CDMeasurementNet',
    'EdgeSegmentationNet',
    'EfficientNetEncoder',
    'EnsembleModel',
    'MODEL_INFO',
    # Advanced Models
    'create_advanced_model',
    'AttentionUNet',
    'SwinUNet',
    'DeepEnsemble',
    'HybridCDNet',
    'ADVANCED_MODEL_INFO',
    # Dataset
    'TrainingDataManager',
    'ImageAnnotation',
    'CDAnnotation',
    'DataAugmentor',
    # Training
    'Trainer',
    'TrainingConfig',
    'train_model',
    # Advanced Training
    'AdvancedTrainer',
    'AdvancedTrainerConfig',
    'EnsembleTrainer',
    'CombinedLoss',
    'DiceLoss',
    'FocalLoss',
    'BoundaryLoss',
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
