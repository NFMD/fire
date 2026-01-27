"""
HR-TEM Analyzer Configuration Settings
"""
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from pathlib import Path
import yaml


@dataclass
class ProcessingMethod:
    """Single processing method configuration"""
    name: str
    enabled: bool = True
    weight: float = 1.0
    params: Dict = field(default_factory=dict)


@dataclass
class MeasurementConfig:
    """Configuration for CD measurements"""
    # Measurement depths from baseline (in nm)
    depths_nm: List[float] = field(default_factory=lambda: [5.0, 10.0, 15.0, 20.0, 25.0])

    # Number of measurement lines per depth for statistical robustness
    lines_per_depth: int = 5

    # Line spacing in pixels
    line_spacing_pixels: int = 2

    # Edge detection threshold
    edge_threshold: float = 0.5

    # Minimum confidence to accept measurement
    min_confidence: float = 0.6

    # Consensus method: 'median', 'mean', 'weighted_mean', 'trimmed_mean'
    consensus_method: str = 'trimmed_mean'

    # Trim percentage for trimmed_mean (removes outliers)
    trim_percentage: float = 0.1


@dataclass
class ParallelConfig:
    """Configuration for parallel processing"""
    # Maximum workers for parallel processing
    max_workers: int = 4

    # Batch size for memory efficiency
    batch_size: int = 5

    # Maximum images in working group
    max_working_group: int = 20

    # Memory limit per worker (MB)
    memory_limit_mb: int = 2048

    # Use GPU if available
    use_gpu: bool = True

    # GPU memory fraction to use
    gpu_memory_fraction: float = 0.8


@dataclass
class MultiMethodConfig:
    """Configuration for multi-method processing"""

    # Processing methods with their weights
    methods: List[ProcessingMethod] = field(default_factory=lambda: [
        ProcessingMethod(
            name="sobel_edge",
            weight=1.0,
            params={"ksize": 3}
        ),
        ProcessingMethod(
            name="canny_edge",
            weight=1.2,
            params={"low_threshold": 50, "high_threshold": 150}
        ),
        ProcessingMethod(
            name="laplacian_edge",
            weight=0.8,
            params={"ksize": 3}
        ),
        ProcessingMethod(
            name="gradient_magnitude",
            weight=1.0,
            params={"sigma": 1.0}
        ),
        ProcessingMethod(
            name="morphological_gradient",
            weight=0.9,
            params={"kernel_size": 3}
        ),
    ])

    # Image preprocessing variants
    preprocessing_variants: List[ProcessingMethod] = field(default_factory=lambda: [
        ProcessingMethod(
            name="original",
            weight=1.0,
            params={}
        ),
        ProcessingMethod(
            name="clahe",
            weight=1.1,
            params={"clip_limit": 2.0, "grid_size": 8}
        ),
        ProcessingMethod(
            name="gaussian_blur",
            weight=0.9,
            params={"sigma": 0.5}
        ),
        ProcessingMethod(
            name="bilateral_filter",
            weight=1.0,
            params={"d": 9, "sigma_color": 75, "sigma_space": 75}
        ),
        ProcessingMethod(
            name="median_filter",
            weight=0.8,
            params={"ksize": 3}
        ),
    ])

    # Rotation angles for multi-directional analysis (degrees)
    rotation_angles: List[float] = field(default_factory=lambda: [-2.0, -1.0, 0.0, 1.0, 2.0])

    # Scale factors for multi-scale analysis
    scale_factors: List[float] = field(default_factory=lambda: [0.9, 1.0, 1.1])


@dataclass
class Settings:
    """Main settings container"""
    measurement: MeasurementConfig = field(default_factory=MeasurementConfig)
    parallel: ParallelConfig = field(default_factory=ParallelConfig)
    multi_method: MultiMethodConfig = field(default_factory=MultiMethodConfig)

    # Output settings
    output_format: str = "jpeg"
    output_quality: int = 95
    save_intermediate: bool = False

    # Logging
    log_level: str = "INFO"
    log_file: Optional[str] = None

    @classmethod
    def from_yaml(cls, path: str) -> 'Settings':
        """Load settings from YAML file"""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return cls(**data) if data else cls()

    def to_yaml(self, path: str) -> None:
        """Save settings to YAML file"""
        import dataclasses
        with open(path, 'w') as f:
            yaml.dump(dataclasses.asdict(self), f, default_flow_style=False)


# Default settings instance
DEFAULT_SETTINGS = Settings()
