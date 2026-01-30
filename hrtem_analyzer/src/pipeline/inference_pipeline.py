"""
Main Inference Pipeline for HR-TEM Analysis

Orchestrates the complete measurement workflow:
1. Image loading with scale extraction
2. Preprocessing and auto-leveling
3. Multi-method CD measurement (standard + Gatan DM-style)
4. Result aggregation and consensus
5. Visualization and export
"""
import gc
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Literal
import numpy as np
from loguru import logger

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.image_loader import TIFFLoader, ScaleInfo
from core.preprocessor import ImagePreprocessor
from core.baseline_detector import BaselineDetector, BaselineInfo
from core.thickness_measurer import (
    ThicknessMeasurer,
    MultiVariantMeasurer,
    MeasurementResult
)
from core.result_exporter import ResultExporter, ResultVisualizer, convert_numpy_types
from core.enhanced_measurer import EnhancedCDMeasurer, EnhancedMeasurementResult, HybridMeasurer
from core.advanced_analysis import FFTAnalyzer, StatisticalAnalyzer
from pipeline.batch_processor import BatchProcessor, WorkingGroup, ImageResult


@dataclass
class PipelineConfig:
    """Configuration for inference pipeline"""
    # Measurement settings
    depths_nm: List[float] = field(default_factory=lambda: [5, 10, 15, 20, 25])
    num_lines_per_depth: int = 5
    line_spacing: int = 2

    # Preprocessing variants
    preprocessing_methods: List[str] = field(
        default_factory=lambda: ['original', 'clahe', 'bilateral_filter', 'gaussian_blur']
    )
    rotation_angles: List[float] = field(
        default_factory=lambda: [-2.0, -1.0, 0.0, 1.0, 2.0]
    )

    # Edge detection methods
    edge_methods: List[str] = field(
        default_factory=lambda: ['sobel', 'canny', 'gradient', 'morphological', 'scharr']
    )

    # Consensus settings
    consensus_method: str = 'trimmed_mean'
    trim_percentage: float = 0.1
    min_confidence: float = 0.3

    # Parallel processing
    max_workers: int = 4
    batch_size: int = 10
    memory_limit_mb: int = 4096

    # Output settings
    output_format: str = 'jpeg'
    jpeg_quality: int = 95
    save_json: bool = True
    save_csv: bool = True

    # === Advanced Analysis (Gatan DM Style) ===
    use_enhanced_analysis: bool = True  # Use Gatan DM-style features

    # Line profile methods
    profile_methods: List[str] = field(
        default_factory=lambda: ['fwhm', '10-90', 'derivative']
    )

    # Background subtraction
    background_method: str = 'rolling_ball'  # 'none', 'rolling_ball', 'polynomial', 'tophat', 'gaussian'

    # FFT calibration
    fft_calibration: bool = False
    lattice_spacing_nm: Optional[float] = None  # Known lattice for calibration

    # Drift correction
    drift_correction: bool = True

    # Sub-pixel interpolation
    interpolation_factor: int = 10

    # Statistical analysis
    outlier_method: str = 'iqr'  # 'iqr', 'mad', 'zscore', 'grubbs'
    bootstrap_ci: bool = True

    # === Precision Measurement ===
    precision_mode: bool = True  # Enable sub-pixel, ESF/LSF, wavelet, Monte Carlo
    subpixel_method: str = 'gaussian'  # 'gaussian', 'parabolic', 'centroid', 'spline'
    denoise_method: str = 'nlm'  # 'nlm', 'bilateral', 'wavelet', 'anisotropic'
    denoise_strength: float = 1.0  # 0-2 (0=off, 1=normal, 2=strong)
    mc_simulations: int = 500  # Number of Monte Carlo simulations
    atomic_fitting: bool = False  # Atomic column fitting for crystalline materials


class InferencePipeline:
    """
    Main inference pipeline for HR-TEM image analysis.

    Provides both single-image and batch processing capabilities
    with memory-efficient parallel execution.

    Supports two analysis modes:
    - Standard: Multi-variant preprocessing + edge detection
    - Enhanced (Gatan DM-style): Line profile analysis, FFT, drift correction
    """

    def __init__(self, config: PipelineConfig = None, output_dir: str = None):
        """
        Initialize inference pipeline.

        Args:
            config: Pipeline configuration
            output_dir: Output directory for results
        """
        self.config = config or PipelineConfig()

        # Initialize components
        self.loader = TIFFLoader()
        self.preprocessor = ImagePreprocessor()
        self.baseline_detector = BaselineDetector()

        # Standard measurer
        self.thickness_measurer = ThicknessMeasurer(
            edge_methods=self.config.edge_methods,
            consensus_method=self.config.consensus_method,
            trim_percentage=self.config.trim_percentage,
            min_confidence=self.config.min_confidence
        )
        self.multi_measurer = MultiVariantMeasurer(self.thickness_measurer)

        # Enhanced measurer (Gatan DM-style + precision measurement)
        self.enhanced_measurer = EnhancedCDMeasurer(
            interpolation_factor=self.config.interpolation_factor,
            background_method=self.config.background_method,
            use_drift_correction=self.config.drift_correction,
            calibrate_with_fft=self.config.fft_calibration,
            known_lattice_spacing_nm=self.config.lattice_spacing_nm,
            # Precision measurement options
            use_precision_mode=self.config.precision_mode,
            subpixel_method=self.config.subpixel_method,
            denoising_method=self.config.denoise_method,
            denoising_strength=self.config.denoise_strength,
            monte_carlo_simulations=self.config.mc_simulations,
            enable_atomic_fitting=self.config.atomic_fitting
        )

        # FFT analyzer for additional analysis
        self.fft_analyzer = FFTAnalyzer()

        self.visualizer = ResultVisualizer()

        # Output handling
        if output_dir:
            self.exporter = ResultExporter(
                output_dir=output_dir,
                jpeg_quality=self.config.jpeg_quality
            )
        else:
            self.exporter = None

        # Batch processor for parallel execution
        self.batch_processor = BatchProcessor(
            max_workers=self.config.max_workers,
            memory_limit_mb=self.config.memory_limit_mb,
            preprocessing_methods=self.config.preprocessing_methods,
            rotation_angles=self.config.rotation_angles,
            edge_methods=self.config.edge_methods
        )

    def process_single(
            self,
            image_path: str,
            baseline_hint_y: Optional[int] = None,
            depths_nm: List[float] = None,
            save_result: bool = True,
            use_enhanced: Optional[bool] = None
    ) -> Dict[str, Any]:
        """
        Process a single HR-TEM image.

        Args:
            image_path: Path to TIFF image
            baseline_hint_y: Optional hint for baseline position
            depths_nm: Measurement depths (uses config default if None)
            save_result: Whether to save visualization
            use_enhanced: Use enhanced Gatan DM-style analysis (None = use config)

        Returns:
            Dictionary with all results
        """
        depths = depths_nm or self.config.depths_nm
        enhanced = use_enhanced if use_enhanced is not None else self.config.use_enhanced_analysis

        logger.info(f"Processing: {Path(image_path).name}")
        logger.info(f"Measurement depths: {depths} nm")
        logger.info(f"Analysis mode: {'Enhanced (Gatan DM-style)' if enhanced else 'Standard'}")

        # 1. Load image
        image, scale_info = self.loader.load(image_path)
        logger.info(f"Scale: {scale_info.scale_nm_per_pixel:.4f} nm/pixel")

        # 2. Auto-level image
        leveled_image, level_angle = self.preprocessor.auto_level(image)
        if abs(level_angle) > 0.1:
            logger.info(f"Auto-leveled by {level_angle:.2f}°")

        # 3. Detect baseline
        baseline_info = self.baseline_detector.detect(
            leveled_image,
            method='auto',
            hint_y=baseline_hint_y
        )
        logger.info(
            f"Baseline detected at y={baseline_info.y_position} "
            f"(confidence: {baseline_info.confidence:.2f})"
        )

        # 4. FFT analysis (if enabled)
        fft_result = None
        calibrated_scale = scale_info.scale_nm_per_pixel
        if enhanced and self.config.fft_calibration:
            fft_result = self.fft_analyzer.analyze(
                leveled_image, scale_info.scale_nm_per_pixel
            )
            if fft_result.confidence > 0.5:
                logger.info(f"FFT detected periodicity: {fft_result.periodicity_nm:.2f} nm")

            # Calibrate scale if lattice spacing is known
            if self.config.lattice_spacing_nm:
                calibrated_scale = self.enhanced_measurer.calibrate_scale(
                    leveled_image, scale_info.scale_nm_per_pixel
                )
                if calibrated_scale != scale_info.scale_nm_per_pixel:
                    logger.info(f"Scale calibrated: {calibrated_scale:.4f} nm/pixel")

        # 5. Measurement (enhanced or standard)
        enhanced_metrics = {}

        if enhanced:
            # Use enhanced Gatan DM-style analysis
            logger.info("Running enhanced line profile analysis...")
            measurements = self.enhanced_measurer.measure_all_depths(
                image=leveled_image,
                baseline_y=baseline_info.y_position,
                depths_nm=depths,
                scale_nm_per_pixel=calibrated_scale,
                preprocess=True
            )

            # Extract enhanced metrics
            for depth, m in measurements.items():
                if hasattr(m, 'fwhm_nm') and m.fwhm_nm:
                    enhanced_metrics[f'fwhm_{depth}nm'] = m.fwhm_nm
                if hasattr(m, 'confidence_interval_95'):
                    enhanced_metrics[f'ci95_{depth}nm'] = m.confidence_interval_95
                if hasattr(m, 'method_results'):
                    enhanced_metrics[f'methods_{depth}nm'] = m.method_results

        else:
            # Use standard multi-variant analysis
            total_variants = (
                    len(self.config.preprocessing_methods) *
                    len(self.config.rotation_angles)
            )
            logger.info(f"Generating {total_variants} image variants for analysis")

            variants = self.preprocessor.generate_variants(
                leveled_image,
                methods=self.config.preprocessing_methods,
                rotation_angles=self.config.rotation_angles,
                scale_factors=[1.0]
            )

            measurements = self.multi_measurer.measure_all_variants(
                image_variants=variants,
                baseline_y=baseline_info.y_position,
                depths_nm=depths,
                scale_nm_per_pixel=calibrated_scale,
                num_lines=self.config.num_lines_per_depth,
                line_spacing=self.config.line_spacing
            )

        # 6. Log results
        logger.info("=" * 60)
        logger.info("MEASUREMENT RESULTS")
        logger.info("=" * 60)
        for depth, m in sorted(measurements.items()):
            ci_str = ""
            if hasattr(m, 'confidence_interval_95') and m.confidence_interval_95[0] != m.confidence_interval_95[1]:
                ci_str = f" CI95:[{m.confidence_interval_95[0]:.2f}-{m.confidence_interval_95[1]:.2f}]"

            logger.info(
                f"  Depth {depth:5.1f} nm: "
                f"{m.thickness_nm:7.2f} ± {m.thickness_std:5.2f} nm "
                f"(n={m.num_measurements}, conf={m.confidence:.2f}){ci_str}"
            )
        logger.info("=" * 60)

        # 7. Save results if requested
        jpeg_path = None
        json_path = None
        if save_result and self.exporter:
            jpeg_path, json_path = self.exporter.export_result(
                image=leveled_image,
                baseline_info=baseline_info,
                measurements=measurements,
                scale_info=scale_info,
                source_path=image_path,
                save_json=self.config.save_json
            )

        # 8. Create result dictionary (convert all numpy types to native Python)
        result = {
            'source_path': str(image_path),
            'success': True,
            'scale_info': scale_info.to_dict(),
            'calibrated_scale': float(calibrated_scale),
            'baseline': {
                'y_position': int(baseline_info.y_position),
                'confidence': float(baseline_info.confidence),
                'method': str(baseline_info.method)
            },
            'level_angle': float(level_angle),
            'measurements': {
                str(depth): m.to_dict() for depth, m in measurements.items()
            },
            'output': {
                'jpeg_path': str(jpeg_path) if jpeg_path else None,
                'json_path': str(json_path) if json_path else None
            },
            'analysis_mode': 'enhanced' if enhanced else 'standard',
            'statistics': {
                'preprocessing_methods': list(self.config.preprocessing_methods),
                'rotation_angles': [float(a) for a in self.config.rotation_angles],
                'edge_methods': list(self.config.edge_methods),
                'profile_methods': list(self.config.profile_methods) if enhanced else [],
                'background_method': str(self.config.background_method) if enhanced else 'none',
            }
        }

        # Add enhanced metrics if available (convert numpy types)
        if enhanced_metrics:
            result['enhanced_metrics'] = convert_numpy_types(enhanced_metrics)

        # Add FFT result if available (convert numpy types)
        if fft_result:
            result['fft_analysis'] = {
                'periodicity_nm': float(fft_result.periodicity_nm) if fft_result.periodicity_nm else None,
                'orientation_deg': float(fft_result.orientation_deg) if fft_result.orientation_deg else None,
                'lattice_spacing_nm': float(fft_result.lattice_spacing_nm) if fft_result.lattice_spacing_nm else None,
                'confidence': float(fft_result.confidence) if fft_result.confidence else None
            }

        # Cleanup
        gc.collect()

        return result

    def process_batch(
            self,
            image_paths: List[str],
            baseline_hint_y: Optional[int] = None,
            depths_nm: List[float] = None,
            working_group_name: str = "batch"
    ) -> List[Dict[str, Any]]:
        """
        Process a batch of images in parallel.

        Automatically splits into working groups of max 20 images.

        Args:
            image_paths: List of image paths
            baseline_hint_y: Optional baseline hint (applied to all)
            depths_nm: Measurement depths
            working_group_name: Name for the working group

        Returns:
            List of result dictionaries
        """
        depths = depths_nm or self.config.depths_nm

        # Create working group
        working_group = WorkingGroup(
            images=image_paths,
            name=working_group_name,
            baseline_hint_y=baseline_hint_y,
            depths_nm=depths
        )

        logger.info(f"Processing batch of {len(working_group)} images")

        # Process with batch processor
        image_results = self.batch_processor.process_working_group(working_group)

        # Convert to result dictionaries and export
        results = []
        for img_result in image_results:
            if img_result.success:
                result_dict = self._image_result_to_dict(img_result)

                # Export visualization
                if self.exporter and img_result.scale_info:
                    # Reload image for visualization (memory tradeoff for quality)
                    try:
                        image, _ = self.loader.load(img_result.path)
                        leveled, _ = self.preprocessor.auto_level(image)

                        jpeg_path, json_path = self.exporter.export_result(
                            image=leveled,
                            baseline_info=img_result.baseline_info,
                            measurements=img_result.measurements,
                            scale_info=img_result.scale_info,
                            source_path=img_result.path,
                            save_json=self.config.save_json
                        )
                        result_dict['output'] = {
                            'jpeg_path': jpeg_path,
                            'json_path': json_path
                        }

                        del image, leveled
                        gc.collect()

                    except Exception as e:
                        logger.error(f"Error exporting {img_result.path}: {e}")

                results.append(result_dict)
            else:
                results.append({
                    'source_path': img_result.path,
                    'success': False,
                    'error': img_result.error_message
                })

        # Export batch summary
        if self.exporter and self.config.save_csv:
            self.exporter.export_csv(results)
            self.exporter.export_batch_summary(results)

        # Log statistics
        stats = self.batch_processor.get_statistics()
        logger.info(f"Batch processing complete:")
        logger.info(f"  Total: {stats['total_processed']}")
        logger.info(f"  Successful: {stats['successful']}")
        logger.info(f"  Failed: {stats['failed']}")
        logger.info(f"  Success rate: {stats['success_rate']:.1f}%")

        return results

    def process_directory(
            self,
            directory: str,
            pattern: str = "*.tif*",
            baseline_hint_y: Optional[int] = None,
            depths_nm: List[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Process all matching images in a directory.

        Args:
            directory: Directory path
            pattern: Glob pattern for images
            baseline_hint_y: Optional baseline hint
            depths_nm: Measurement depths

        Returns:
            List of result dictionaries
        """
        directory = Path(directory)
        image_paths = sorted(str(p) for p in directory.glob(pattern))

        if not image_paths:
            logger.warning(f"No images found matching {pattern} in {directory}")
            return []

        logger.info(f"Found {len(image_paths)} images in {directory}")

        return self.process_batch(
            image_paths=image_paths,
            baseline_hint_y=baseline_hint_y,
            depths_nm=depths_nm,
            working_group_name=directory.name
        )

    def _image_result_to_dict(self, result: ImageResult) -> Dict[str, Any]:
        """Convert ImageResult to dictionary"""
        return {
            'source_path': result.path,
            'success': result.success,
            'processing_time_ms': result.processing_time_ms,
            'scale_nm_per_pixel': (
                result.scale_info.scale_nm_per_pixel if result.scale_info else None
            ),
            'baseline_y': (
                result.baseline_info.y_position if result.baseline_info else None
            ),
            'measurements': {
                depth: m.to_dict() for depth, m in result.measurements.items()
            } if result.measurements else {}
        }


def create_pipeline(
        output_dir: str,
        depths_nm: List[float] = None,
        max_workers: int = 4,
        high_precision: bool = True,
        use_enhanced: bool = True,
        **advanced_settings
) -> InferencePipeline:
    """
    Factory function to create configured pipeline.

    Args:
        output_dir: Output directory
        depths_nm: Measurement depths
        max_workers: Number of parallel workers
        high_precision: Use high precision mode (more variants)
        use_enhanced: Use enhanced Gatan DM-style analysis
        **advanced_settings: Additional settings for enhanced analysis:
            - profile_methods: List of profile methods ['fwhm', '10-90', 'derivative', 'sigmoid']
            - background_method: Background subtraction method
            - fft_calibration: Enable FFT calibration
            - lattice_spacing_nm: Known lattice spacing for calibration
            - drift_correction: Enable drift correction
            - interpolation_factor: Sub-pixel interpolation factor
            - outlier_method: Outlier rejection method
            - bootstrap_ci: Enable bootstrap confidence intervals

    Returns:
        Configured InferencePipeline
    """
    if high_precision:
        config = PipelineConfig(
            depths_nm=depths_nm or [5, 10, 15, 20, 25],
            preprocessing_methods=['original', 'clahe', 'bilateral_filter', 'gaussian_blur', 'median_filter'],
            rotation_angles=[-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0],
            edge_methods=['sobel', 'canny', 'gradient', 'morphological', 'scharr', 'laplacian'],
            num_lines_per_depth=7,
            max_workers=max_workers,
            # Enhanced settings
            use_enhanced_analysis=use_enhanced,
            profile_methods=advanced_settings.get('profile_methods', ['fwhm', '10-90', 'derivative']),
            background_method=advanced_settings.get('background_method', 'rolling_ball'),
            fft_calibration=advanced_settings.get('fft_calibration', False),
            lattice_spacing_nm=advanced_settings.get('lattice_spacing_nm'),
            drift_correction=advanced_settings.get('drift_correction', True),
            interpolation_factor=advanced_settings.get('interpolation_factor', 10),
            outlier_method=advanced_settings.get('outlier_method', 'iqr'),
            bootstrap_ci=advanced_settings.get('bootstrap_ci', True),
        )
    else:
        config = PipelineConfig(
            depths_nm=depths_nm or [5, 10, 15, 20],
            preprocessing_methods=['original', 'clahe', 'bilateral_filter'],
            rotation_angles=[-1.0, 0.0, 1.0],
            edge_methods=['sobel', 'canny', 'gradient'],
            num_lines_per_depth=5,
            max_workers=max_workers,
            # Enhanced settings (simplified for standard mode)
            use_enhanced_analysis=use_enhanced,
            profile_methods=advanced_settings.get('profile_methods', ['fwhm', 'derivative']),
            background_method=advanced_settings.get('background_method', 'none'),
            fft_calibration=False,
            drift_correction=advanced_settings.get('drift_correction', False),
            interpolation_factor=advanced_settings.get('interpolation_factor', 5),
            outlier_method=advanced_settings.get('outlier_method', 'iqr'),
            bootstrap_ci=advanced_settings.get('bootstrap_ci', False),
        )

    return InferencePipeline(config=config, output_dir=output_dir)
