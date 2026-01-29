"""
Enhanced CD Measurer with Gatan DM-style Features

Integrates advanced analysis techniques for high-precision measurements:
- Multi-method line profile analysis
- FFT-based scale calibration
- Background subtraction
- Drift correction
- Robust statistical analysis
- Precision measurement (sub-pixel, ESF/LSF, wavelet, Monte Carlo)
"""
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any
from loguru import logger

from .advanced_analysis import (
    LineProfileAnalyzer,
    FFTAnalyzer,
    DriftCorrector,
    BackgroundSubtractor,
    IntensityCalibrator,
    StatisticalAnalyzer,
    LineProfileResult,
    FFTAnalysisResult,
    CalibratedMeasurement
)
from .thickness_measurer import (
    ThicknessMeasurer,
    MeasurementResult,
    SingleMeasurement,
    EdgePoint
)
from .precision_measurement import (
    PrecisionCDMeasurer,
    PrecisionMeasurementResult,
    SubPixelMethod,
    SubPixelEdgeDetector,
    ESFLSFAnalyzer,
    AdvancedDenoiser,
    MultiScaleWaveletAnalyzer,
    MonteCarloUncertainty,
    AtomicColumnFitter
)


@dataclass
class EnhancedMeasurementResult(MeasurementResult):
    """Extended measurement result with advanced metrics"""
    # Line profile results
    fwhm_nm: Optional[float] = None
    edge_sharpness_left: Optional[float] = None  # 10-90% width
    edge_sharpness_right: Optional[float] = None

    # Calibration info
    scale_calibrated: bool = False
    calibration_factor: float = 1.0

    # Background info
    background_corrected: bool = False
    background_level: float = 0.0

    # Statistical metrics
    measurement_uncertainty_nm: float = 0.0
    confidence_interval_95: Tuple[float, float] = (0.0, 0.0)

    # Method breakdown
    method_results: Dict[str, float] = field(default_factory=dict)

    # Precision measurement fields
    sub_pixel_positions: Optional[List[float]] = None
    esf_width_nm: Optional[float] = None
    lsf_fwhm_nm: Optional[float] = None
    snr_db: Optional[float] = None
    monte_carlo_std: Optional[float] = None
    atomic_spacing_nm: Optional[float] = None
    precision_confidence: Optional[float] = None


class EnhancedCDMeasurer:
    """
    Enhanced Critical Dimension measurer with Gatan DM-style features.

    Provides:
    - Multiple edge detection methods (FWHM, 10-90%, derivative, sigmoid)
    - FFT-based scale calibration
    - Automatic background subtraction
    - Drift correction between measurements
    - Robust statistical analysis
    - Confidence intervals
    """

    def __init__(
            self,
            interpolation_factor: int = 10,
            background_method: str = 'rolling_ball',
            use_drift_correction: bool = True,
            calibrate_with_fft: bool = True,
            known_lattice_spacing_nm: Optional[float] = None,
            # Precision measurement options
            use_precision_mode: bool = True,
            subpixel_method: str = 'gaussian',
            denoising_method: str = 'nlm',
            denoising_strength: float = 1.0,
            monte_carlo_simulations: int = 500,
            enable_atomic_fitting: bool = False
    ):
        """
        Initialize enhanced measurer.

        Args:
            interpolation_factor: Sub-pixel interpolation factor
            background_method: 'rolling_ball', 'polynomial', 'tophat', 'gaussian', 'none'
            use_drift_correction: Enable drift correction
            calibrate_with_fft: Use FFT for scale calibration
            known_lattice_spacing_nm: Known lattice spacing for calibration
            use_precision_mode: Enable precision measurement features
            subpixel_method: 'gaussian', 'parabolic', 'centroid', 'spline'
            denoising_method: 'nlm', 'bilateral', 'wavelet', 'anisotropic'
            denoising_strength: Denoising strength (0-2)
            monte_carlo_simulations: Number of MC simulations for uncertainty
            enable_atomic_fitting: Enable atomic column fitting
        """
        self.interpolation_factor = interpolation_factor
        self.background_method = background_method
        self.use_drift_correction = use_drift_correction
        self.calibrate_with_fft = calibrate_with_fft
        self.known_lattice_spacing = known_lattice_spacing_nm
        self.use_precision_mode = use_precision_mode

        # Initialize analyzers
        self.line_analyzer = LineProfileAnalyzer(interpolation_factor=interpolation_factor)
        self.fft_analyzer = FFTAnalyzer()
        self.drift_corrector = DriftCorrector()
        self.intensity_calibrator = IntensityCalibrator()

        # Initialize precision measurement components
        subpixel_enum = getattr(SubPixelMethod, subpixel_method.upper(), SubPixelMethod.GAUSSIAN)
        self.precision_measurer = PrecisionCDMeasurer(
            subpixel_method=subpixel_enum,
            denoising_method=denoising_method,
            denoising_strength=denoising_strength,
            monte_carlo_simulations=monte_carlo_simulations,
            enable_atomic_fitting=enable_atomic_fitting,
            expected_lattice_nm=known_lattice_spacing_nm
        )

        # State
        self._reference_set = False
        self._calibration_factor = 1.0

    def preprocess_image(
            self,
            image: np.ndarray,
            apply_background: bool = True,
            apply_drift: bool = True,
            apply_intensity: bool = True
    ) -> np.ndarray:
        """
        Preprocess image with corrections.

        Args:
            image: Input image
            apply_background: Apply background subtraction
            apply_drift: Apply drift correction
            apply_intensity: Apply intensity normalization

        Returns:
            Preprocessed image
        """
        processed = image.astype(np.float64)

        # Intensity calibration
        if apply_intensity:
            processed = self.intensity_calibrator.correct_intensity(processed)
            processed = IntensityCalibrator.normalize_contrast(processed)

        # Background subtraction
        if apply_background and self.background_method != 'none':
            if self.background_method == 'rolling_ball':
                processed = BackgroundSubtractor.rolling_ball(processed, radius=50)
            elif self.background_method == 'polynomial':
                processed = BackgroundSubtractor.polynomial_fit(processed, order=2)
            elif self.background_method == 'tophat':
                processed = BackgroundSubtractor.tophat(processed, size=50)
            elif self.background_method == 'gaussian':
                processed = BackgroundSubtractor.gaussian_background(processed, sigma=50)

            # Re-normalize after background subtraction
            processed = IntensityCalibrator.normalize_contrast(processed)

        # Drift correction
        if apply_drift and self.use_drift_correction:
            if not self._reference_set:
                self.drift_corrector.set_reference(processed)
                self._reference_set = True
            else:
                dx, dy = self.drift_corrector.estimate_drift(processed)
                if abs(dx) > 0.1 or abs(dy) > 0.1:
                    processed = self.drift_corrector.correct_drift(processed, dx, dy)
                    logger.debug(f"Drift corrected: dx={dx:.2f}, dy={dy:.2f} pixels")

        return processed

    def calibrate_scale(
            self,
            image: np.ndarray,
            current_scale: float
    ) -> float:
        """
        Calibrate scale using FFT if known lattice spacing is set.

        Args:
            image: Image for calibration
            current_scale: Current scale estimate

        Returns:
            Calibrated scale
        """
        if not self.calibrate_with_fft or self.known_lattice_spacing is None:
            return current_scale

        corrected_scale, factor = self.fft_analyzer.calibrate_scale(
            image, self.known_lattice_spacing, current_scale
        )
        self._calibration_factor = factor

        return corrected_scale

    def measure_cd_enhanced(
            self,
            image: np.ndarray,
            baseline_y: int,
            depth_nm: float,
            scale_nm_per_pixel: float,
            averaging_width: int = 5,
            num_profiles: int = 7,
            profile_spacing: int = 2
    ) -> EnhancedMeasurementResult:
        """
        Perform enhanced CD measurement using multiple methods.

        Args:
            image: Preprocessed image
            baseline_y: Baseline Y position
            depth_nm: Measurement depth from baseline
            scale_nm_per_pixel: Scale factor
            averaging_width: Lines to average per profile
            num_profiles: Number of profiles to measure
            profile_spacing: Spacing between profiles in pixels

        Returns:
            EnhancedMeasurementResult with detailed metrics
        """
        depth_pixels = int(depth_nm / scale_nm_per_pixel)
        center_y = baseline_y + depth_pixels

        # Generate Y positions for multiple profiles
        half_range = (num_profiles - 1) // 2
        y_positions = [
            center_y + i * profile_spacing
            for i in range(-half_range, half_range + 1)
            if 0 <= center_y + i * profile_spacing < image.shape[0]
        ]

        # Collect measurements from all methods and positions
        all_measurements = {
            'fwhm': [],
            '10-90': [],
            'derivative': [],
            'sigmoid': [],
        }

        fwhm_values = []
        edge_sharpness_left = []
        edge_sharpness_right = []

        for y in y_positions:
            result = self.line_analyzer.analyze_profile(
                image=image,
                y=y,
                scale_nm_per_pixel=scale_nm_per_pixel,
                method='all',
                averaging_width=averaging_width
            )

            if result.width_nm > 0:
                # FWHM measurement
                if result.fwhm:
                    all_measurements['fwhm'].append(result.fwhm)
                    fwhm_values.append(result.fwhm)

                # 10-90 measurement
                if result.edge_10_90_left and result.edge_10_90_right:
                    left_mid = result.edge_10_90_left[0]
                    right_mid = result.edge_10_90_right[0]
                    width_10_90 = right_mid - left_mid
                    all_measurements['10-90'].append(width_10_90)

                    # Edge sharpness (10-90% width)
                    if isinstance(result.edge_10_90_left, tuple) and len(result.edge_10_90_left) > 1:
                        left_sharpness = abs(result.edge_10_90_left[1][1] - result.edge_10_90_left[1][0])
                        edge_sharpness_left.append(left_sharpness)
                    if isinstance(result.edge_10_90_right, tuple) and len(result.edge_10_90_right) > 1:
                        right_sharpness = abs(result.edge_10_90_right[1][1] - result.edge_10_90_right[1][0])
                        edge_sharpness_right.append(right_sharpness)

                # Derivative measurement
                if result.derivative_peaks and len(result.derivative_peaks) >= 2:
                    width_deriv = result.derivative_peaks[-1] - result.derivative_peaks[0]
                    all_measurements['derivative'].append(width_deriv)

                # Sigmoid result comes from main width
                all_measurements['sigmoid'].append(result.width_nm)

        # Combine all methods with robust statistics
        all_widths = []
        method_results = {}

        for method, values in all_measurements.items():
            if len(values) >= 3:
                # Reject outliers
                filtered, mask = StatisticalAnalyzer.reject_outliers(
                    np.array(values), method='iqr', threshold=1.5
                )

                if len(filtered) > 0:
                    mean, std = StatisticalAnalyzer.robust_mean_std(filtered)
                    method_results[method] = mean
                    all_widths.extend(filtered.tolist())

        # Calculate consensus measurement
        if len(all_widths) >= 3:
            all_widths = np.array(all_widths)
            filtered_widths, _ = StatisticalAnalyzer.reject_outliers(
                all_widths, method='mad', threshold=3.0
            )

            final_mean, final_std = StatisticalAnalyzer.robust_mean_std(filtered_widths)
            mean_ci, ci_lower, ci_upper = StatisticalAnalyzer.bootstrap_ci(
                filtered_widths, confidence=0.95
            )

            # Calculate confidence
            cv = final_std / (final_mean + 1e-10)  # Coefficient of variation
            method_agreement = 1.0 - np.std(list(method_results.values())) / (final_mean + 1e-10) \
                if len(method_results) > 1 else 0.8
            confidence = min(1.0, (1.0 - cv) * method_agreement)

        else:
            final_mean = np.mean(all_widths) if all_widths else 0
            final_std = np.std(all_widths) if len(all_widths) > 1 else 0
            ci_lower = ci_upper = final_mean
            confidence = 0.3 if all_widths else 0

        # Calculate edge positions
        left_edge_x = 0
        right_edge_x = 0
        if final_mean > 0:
            # Estimate from center of image
            center_x = image.shape[1] // 2
            half_width_px = (final_mean / scale_nm_per_pixel) / 2
            left_edge_x = int(center_x - half_width_px)
            right_edge_x = int(center_x + half_width_px)

        # Precision measurement (sub-pixel, ESF/LSF, wavelet, Monte Carlo)
        precision_result = None
        if self.use_precision_mode:
            try:
                precision_result = self.precision_measurer.measure_cd(
                    image=image,
                    profile_y=center_y,
                    scale_nm_per_pixel=scale_nm_per_pixel,
                    profile_width=averaging_width * num_profiles
                )

                # Merge precision results with multi-method results
                if precision_result.confidence_level > 0.5:
                    # Weight precision measurement into final result
                    precision_weight = 0.3
                    multi_method_weight = 0.7

                    if precision_result.thickness_nm > 0:
                        final_mean = (
                            multi_method_weight * final_mean +
                            precision_weight * precision_result.thickness_nm
                        )

                    # Update uncertainty with precision estimate
                    if precision_result.uncertainty_nm < float('inf'):
                        final_std = min(final_std, precision_result.uncertainty_nm)

                    # Update confidence interval
                    if precision_result.ci_95_low > 0:
                        ci_lower = min(ci_lower, precision_result.ci_95_low)
                        ci_upper = max(ci_upper, precision_result.ci_95_high)

                    logger.debug(
                        f"Precision measurement: {precision_result.thickness_nm:.3f}nm "
                        f"± {precision_result.uncertainty_nm:.3f}nm "
                        f"(SNR: {precision_result.snr_db:.1f}dB)"
                    )

            except Exception as e:
                logger.warning(f"Precision measurement failed: {e}")

        return EnhancedMeasurementResult(
            depth_nm=depth_nm,
            thickness_nm=final_mean,
            thickness_std=final_std,
            thickness_min=np.min(all_widths) if all_widths else 0,
            thickness_max=np.max(all_widths) if all_widths else 0,
            confidence=confidence,
            y_position=center_y,
            left_edge_x=left_edge_x,
            right_edge_x=right_edge_x,
            num_measurements=len(all_widths),
            consensus_method='enhanced_multi_method',
            # Enhanced fields
            fwhm_nm=np.median(fwhm_values) if fwhm_values else None,
            edge_sharpness_left=np.median(edge_sharpness_left) if edge_sharpness_left else None,
            edge_sharpness_right=np.median(edge_sharpness_right) if edge_sharpness_right else None,
            scale_calibrated=self._calibration_factor != 1.0,
            calibration_factor=self._calibration_factor,
            background_corrected=self.background_method != 'none',
            measurement_uncertainty_nm=final_std / np.sqrt(len(all_widths)) if all_widths else 0,
            confidence_interval_95=(ci_lower, ci_upper),
            method_results=method_results,
            # Precision measurement fields
            sub_pixel_positions=precision_result.sub_pixel_positions if precision_result else None,
            esf_width_nm=precision_result.esf_width_nm if precision_result else None,
            lsf_fwhm_nm=precision_result.lsf_fwhm_nm if precision_result else None,
            snr_db=precision_result.snr_db if precision_result else None,
            monte_carlo_std=precision_result.uncertainty_nm if precision_result else None,
            atomic_spacing_nm=precision_result.atomic_spacing_nm if precision_result else None,
            precision_confidence=precision_result.confidence_level if precision_result else None
        )

    def measure_all_depths(
            self,
            image: np.ndarray,
            baseline_y: int,
            depths_nm: List[float],
            scale_nm_per_pixel: float,
            preprocess: bool = True
    ) -> Dict[float, EnhancedMeasurementResult]:
        """
        Measure CD at multiple depths.

        Args:
            image: Input image
            baseline_y: Baseline Y position
            depths_nm: List of depths to measure
            scale_nm_per_pixel: Scale factor
            preprocess: Apply preprocessing

        Returns:
            Dict mapping depth to measurement result
        """
        # Preprocess image
        if preprocess:
            processed = self.preprocess_image(image)
        else:
            processed = image

        # Calibrate scale if enabled
        calibrated_scale = self.calibrate_scale(processed, scale_nm_per_pixel)

        # Measure at each depth
        results = {}
        for depth in depths_nm:
            try:
                result = self.measure_cd_enhanced(
                    image=processed,
                    baseline_y=baseline_y,
                    depth_nm=depth,
                    scale_nm_per_pixel=calibrated_scale
                )
                results[depth] = result

                logger.info(
                    f"Depth {depth:.1f}nm: {result.thickness_nm:.2f} ± "
                    f"{result.thickness_std:.2f}nm (CI95: {result.confidence_interval_95[0]:.2f}-"
                    f"{result.confidence_interval_95[1]:.2f}nm)"
                )

            except Exception as e:
                logger.error(f"Error measuring at depth {depth}nm: {e}")

        return results


class HybridMeasurer:
    """
    Hybrid measurer combining standard and enhanced methods.

    Automatically selects best method based on image characteristics.
    """

    def __init__(self):
        self.standard_measurer = ThicknessMeasurer()
        self.enhanced_measurer = EnhancedCDMeasurer()

    def analyze_image_quality(self, image: np.ndarray) -> Dict[str, float]:
        """
        Analyze image quality metrics.

        Returns:
            Dict with quality metrics
        """
        # Calculate SNR estimate
        mean_intensity = np.mean(image)
        std_intensity = np.std(image)
        snr = mean_intensity / (std_intensity + 1e-10)

        # Calculate edge sharpness (using Laplacian variance)
        from scipy.ndimage import laplace
        laplacian = laplace(image)
        sharpness = np.var(laplacian)

        # Calculate contrast
        contrast = (np.max(image) - np.min(image)) / (np.max(image) + np.min(image) + 1e-10)

        # Detect periodicity
        fft_result = self.enhanced_measurer.fft_analyzer.analyze(image, 1.0)
        periodicity_confidence = fft_result.confidence

        return {
            'snr': snr,
            'sharpness': sharpness,
            'contrast': contrast,
            'periodicity_confidence': periodicity_confidence
        }

    def measure_adaptive(
            self,
            image: np.ndarray,
            baseline_y: int,
            depths_nm: List[float],
            scale_nm_per_pixel: float
    ) -> Dict[float, EnhancedMeasurementResult]:
        """
        Adaptively select measurement method based on image quality.

        Args:
            image: Input image
            baseline_y: Baseline Y position
            depths_nm: Measurement depths
            scale_nm_per_pixel: Scale factor

        Returns:
            Measurement results
        """
        quality = self.analyze_image_quality(image)

        # Select method based on quality
        use_enhanced = (
            quality['snr'] > 2.0 and
            quality['sharpness'] > 0.01 and
            quality['contrast'] > 0.1
        )

        if use_enhanced:
            logger.info("Using enhanced measurement (good image quality)")
            return self.enhanced_measurer.measure_all_depths(
                image, baseline_y, depths_nm, scale_nm_per_pixel
            )
        else:
            logger.info("Using standard measurement (lower image quality)")
            # Fall back to standard but convert results
            from .preprocessor import ImagePreprocessor
            preprocessor = ImagePreprocessor()

            # Simple preprocessing
            processed, _ = preprocessor.auto_level(image)

            results = {}
            for depth in depths_nm:
                measurements = self.standard_measurer.measure_thickness(
                    image=processed,
                    baseline_y=baseline_y,
                    depth_nm=depth,
                    scale_nm_per_pixel=scale_nm_per_pixel
                )

                if measurements:
                    aggregated = self.standard_measurer.aggregate_measurements(
                        measurements, depth
                    )
                    if aggregated:
                        # Convert to enhanced result format
                        results[depth] = EnhancedMeasurementResult(
                            depth_nm=aggregated.depth_nm,
                            thickness_nm=aggregated.thickness_nm,
                            thickness_std=aggregated.thickness_std,
                            thickness_min=aggregated.thickness_min,
                            thickness_max=aggregated.thickness_max,
                            confidence=aggregated.confidence,
                            y_position=aggregated.y_position,
                            left_edge_x=aggregated.left_edge_x,
                            right_edge_x=aggregated.right_edge_x,
                            num_measurements=aggregated.num_measurements,
                            consensus_method='standard'
                        )

            return results
