"""
Precision Measurement Module for HR-TEM CD Analysis

Advanced techniques for sub-pixel accuracy in Critical Dimension measurement:
- Sub-pixel edge detection with polynomial/spline fitting
- Edge Spread Function (ESF) / Line Spread Function (LSF) analysis
- Non-local means and advanced denoising
- Multi-scale wavelet analysis
- Monte Carlo uncertainty estimation
- Atomic column fitting for crystalline materials
"""
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import warnings


class SubPixelMethod(Enum):
    """Sub-pixel interpolation methods"""
    PARABOLIC = "parabolic"
    GAUSSIAN = "gaussian"
    CENTROID = "centroid"
    SPLINE = "spline"
    SINC = "sinc"


@dataclass
class PrecisionMeasurementResult:
    """Result of precision CD measurement"""
    thickness_nm: float
    uncertainty_nm: float
    ci_95_low: float
    ci_95_high: float
    sub_pixel_positions: List[float]  # Edge positions with sub-pixel accuracy
    esf_width_nm: float  # Edge spread function width
    lsf_fwhm_nm: float  # Line spread function FWHM
    snr_db: float  # Signal-to-noise ratio
    method_contributions: Dict[str, float]  # Individual method results
    monte_carlo_distribution: Optional[np.ndarray] = None
    confidence_level: float = 0.0
    atomic_spacing_nm: Optional[float] = None


class SubPixelEdgeDetector:
    """
    Sub-pixel edge detection using multiple interpolation methods.

    Achieves sub-pixel accuracy by fitting mathematical models
    to the intensity gradient around detected edge points.
    """

    def __init__(self, method: SubPixelMethod = SubPixelMethod.GAUSSIAN):
        self.method = method
        self._fit_window = 5  # Points on each side of edge

    def detect_edges_subpixel(
        self,
        profile: np.ndarray,
        coarse_edges: List[int],
        scale_nm_per_pixel: float = 1.0
    ) -> List[float]:
        """
        Refine coarse edge positions to sub-pixel accuracy.

        Args:
            profile: 1D intensity profile
            coarse_edges: Coarse edge positions (pixel indices)
            scale_nm_per_pixel: Scale factor

        Returns:
            List of sub-pixel edge positions in nm
        """
        subpixel_edges = []

        for edge_idx in coarse_edges:
            # Extract local window around edge
            start = max(0, edge_idx - self._fit_window)
            end = min(len(profile), edge_idx + self._fit_window + 1)

            if end - start < 3:
                subpixel_edges.append(edge_idx * scale_nm_per_pixel)
                continue

            local_profile = profile[start:end]
            local_x = np.arange(start, end)

            # Compute gradient for edge localization
            gradient = np.gradient(local_profile)

            # Find sub-pixel position based on method
            if self.method == SubPixelMethod.PARABOLIC:
                refined = self._parabolic_fit(local_x, np.abs(gradient))
            elif self.method == SubPixelMethod.GAUSSIAN:
                refined = self._gaussian_fit(local_x, np.abs(gradient))
            elif self.method == SubPixelMethod.CENTROID:
                refined = self._centroid_fit(local_x, np.abs(gradient))
            elif self.method == SubPixelMethod.SPLINE:
                refined = self._spline_fit(local_x, np.abs(gradient))
            else:
                refined = edge_idx

            subpixel_edges.append(refined * scale_nm_per_pixel)

        return subpixel_edges

    def _parabolic_fit(self, x: np.ndarray, y: np.ndarray) -> float:
        """Parabolic (quadratic) interpolation for sub-pixel peak"""
        peak_idx = np.argmax(y)

        if peak_idx == 0 or peak_idx == len(y) - 1:
            return x[peak_idx]

        # Three-point parabolic interpolation
        y0, y1, y2 = y[peak_idx - 1], y[peak_idx], y[peak_idx + 1]

        # Avoid division by zero
        denom = 2 * (2 * y1 - y0 - y2)
        if abs(denom) < 1e-10:
            return x[peak_idx]

        delta = (y0 - y2) / denom

        return x[peak_idx] + delta

    def _gaussian_fit(self, x: np.ndarray, y: np.ndarray) -> float:
        """Gaussian fit for sub-pixel peak location"""
        peak_idx = np.argmax(y)

        if peak_idx == 0 or peak_idx == len(y) - 1:
            return x[peak_idx]

        # Log-based Gaussian fit (faster than full optimization)
        y0, y1, y2 = y[peak_idx - 1], y[peak_idx], y[peak_idx + 1]

        # Ensure positive values for log
        eps = 1e-10
        y0, y1, y2 = max(y0, eps), max(y1, eps), max(y2, eps)

        # Gaussian peak position
        log_y0, log_y1, log_y2 = np.log(y0), np.log(y1), np.log(y2)

        denom = 2 * (log_y0 - 2 * log_y1 + log_y2)
        if abs(denom) < 1e-10:
            return x[peak_idx]

        delta = (log_y0 - log_y2) / denom

        return x[peak_idx] + delta

    def _centroid_fit(self, x: np.ndarray, y: np.ndarray) -> float:
        """Centroid (center of mass) calculation"""
        total = np.sum(y)
        if total < 1e-10:
            return x[len(x) // 2]

        return np.sum(x * y) / total

    def _spline_fit(self, x: np.ndarray, y: np.ndarray) -> float:
        """Cubic spline interpolation for peak finding"""
        try:
            from scipy.interpolate import CubicSpline

            # Create high-resolution interpolation
            cs = CubicSpline(x, y)
            x_fine = np.linspace(x[0], x[-1], len(x) * 100)
            y_fine = cs(x_fine)

            return x_fine[np.argmax(y_fine)]
        except ImportError:
            # Fallback to parabolic
            return self._parabolic_fit(x, y)


class ESFLSFAnalyzer:
    """
    Edge Spread Function (ESF) and Line Spread Function (LSF) analyzer.

    ESF represents the response of the imaging system to a step edge.
    LSF is the derivative of ESF and represents the point spread function
    projected onto one dimension.
    """

    def __init__(self, interpolation_factor: int = 20):
        self.interpolation_factor = interpolation_factor

    def analyze_edge(
        self,
        profile: np.ndarray,
        edge_position: int,
        scale_nm_per_pixel: float = 1.0,
        window_size: int = 30
    ) -> Dict[str, Any]:
        """
        Analyze edge using ESF/LSF method.

        Args:
            profile: 1D intensity profile across edge
            edge_position: Approximate edge position
            scale_nm_per_pixel: Scale factor
            window_size: Analysis window size in pixels

        Returns:
            Dictionary with ESF/LSF analysis results
        """
        # Extract window around edge
        start = max(0, edge_position - window_size)
        end = min(len(profile), edge_position + window_size)

        esf = profile[start:end].astype(float)
        x_pixels = np.arange(len(esf))

        # Normalize ESF to 0-1 range
        esf_min, esf_max = esf.min(), esf.max()
        if esf_max - esf_min > 1e-10:
            esf_normalized = (esf - esf_min) / (esf_max - esf_min)
        else:
            return self._empty_result()

        # Interpolate for higher precision
        x_fine = np.linspace(0, len(esf) - 1, len(esf) * self.interpolation_factor)

        try:
            from scipy.interpolate import CubicSpline
            cs = CubicSpline(x_pixels, esf_normalized)
            esf_fine = cs(x_fine)
        except ImportError:
            esf_fine = np.interp(x_fine, x_pixels, esf_normalized)

        # Compute LSF (derivative of ESF)
        lsf = np.gradient(esf_fine)

        # Find edge position from LSF peak
        lsf_abs = np.abs(lsf)
        peak_idx = np.argmax(lsf_abs)

        # Sub-pixel refinement using Gaussian fit
        subpixel_detector = SubPixelEdgeDetector(SubPixelMethod.GAUSSIAN)
        edge_subpixel = subpixel_detector._gaussian_fit(x_fine, lsf_abs)

        # Calculate LSF FWHM (edge sharpness indicator)
        lsf_fwhm_pixels = self._calculate_fwhm(lsf_abs)
        lsf_fwhm_nm = lsf_fwhm_pixels * scale_nm_per_pixel / self.interpolation_factor

        # Calculate ESF 10-90% width
        esf_width_pixels = self._calculate_10_90_width(esf_fine)
        esf_width_nm = esf_width_pixels * scale_nm_per_pixel / self.interpolation_factor

        # Edge position in nm (relative to window start)
        edge_position_nm = (start + edge_subpixel / self.interpolation_factor) * scale_nm_per_pixel

        return {
            'edge_position_nm': edge_position_nm,
            'esf': esf_normalized,
            'lsf': lsf,
            'esf_fine': esf_fine,
            'lsf_abs': lsf_abs,
            'lsf_fwhm_nm': lsf_fwhm_nm,
            'esf_width_nm': esf_width_nm,
            'edge_sharpness': 1.0 / max(lsf_fwhm_nm, 0.01),  # Higher is sharper
            'confidence': min(1.0, lsf_abs.max() / 0.1)  # Based on gradient strength
        }

    def _calculate_fwhm(self, data: np.ndarray) -> float:
        """Calculate Full Width at Half Maximum"""
        peak_val = data.max()
        half_max = peak_val / 2

        # Find indices where data crosses half maximum
        above_half = data >= half_max

        # Find first and last crossing
        crossings = np.where(np.diff(above_half.astype(int)))[0]

        if len(crossings) >= 2:
            return crossings[-1] - crossings[0]
        elif len(crossings) == 1:
            return 2 * abs(crossings[0] - np.argmax(data))
        else:
            return len(data) / 4  # Fallback estimate

    def _calculate_10_90_width(self, esf: np.ndarray) -> float:
        """Calculate 10-90% rise width"""
        # Find 10% and 90% crossing points
        idx_10 = np.argmax(esf >= 0.1)
        idx_90 = np.argmax(esf >= 0.9)

        if idx_90 > idx_10:
            return idx_90 - idx_10
        else:
            # Edge is decreasing
            idx_90 = np.argmax(esf <= 0.9)
            idx_10 = np.argmax(esf <= 0.1)
            return abs(idx_10 - idx_90)

    def _empty_result(self) -> Dict[str, Any]:
        """Return empty result dictionary"""
        return {
            'edge_position_nm': 0.0,
            'esf': np.array([]),
            'lsf': np.array([]),
            'lsf_fwhm_nm': float('inf'),
            'esf_width_nm': float('inf'),
            'edge_sharpness': 0.0,
            'confidence': 0.0
        }


class AdvancedDenoiser:
    """
    Advanced denoising methods for HR-TEM images.

    Includes:
    - Non-local means denoising
    - Bilateral filtering
    - Anisotropic diffusion
    - Wavelet denoising
    """

    def __init__(self):
        self._methods = ['nlm', 'bilateral', 'wavelet', 'anisotropic']

    def denoise(
        self,
        image: np.ndarray,
        method: str = 'nlm',
        strength: float = 1.0
    ) -> np.ndarray:
        """
        Apply denoising to image.

        Args:
            image: Input image
            method: Denoising method ('nlm', 'bilateral', 'wavelet', 'anisotropic')
            strength: Denoising strength (0-2, 1=normal)

        Returns:
            Denoised image
        """
        if method == 'nlm':
            return self._non_local_means(image, strength)
        elif method == 'bilateral':
            return self._bilateral_filter(image, strength)
        elif method == 'wavelet':
            return self._wavelet_denoise(image, strength)
        elif method == 'anisotropic':
            return self._anisotropic_diffusion(image, strength)
        else:
            return image

    def _non_local_means(self, image: np.ndarray, strength: float) -> np.ndarray:
        """Non-local means denoising"""
        try:
            import cv2

            # Estimate noise level
            noise_sigma = self._estimate_noise(image) * strength

            # NLM parameters
            h = noise_sigma * 10  # Filter strength
            template_size = 7
            search_size = 21

            # Ensure proper format
            if image.dtype != np.uint8:
                img_normalized = ((image - image.min()) /
                                 (image.max() - image.min() + 1e-10) * 255).astype(np.uint8)
            else:
                img_normalized = image

            denoised = cv2.fastNlMeansDenoising(
                img_normalized,
                None,
                h=h,
                templateWindowSize=template_size,
                searchWindowSize=search_size
            )

            # Scale back to original range
            if image.dtype != np.uint8:
                denoised = denoised.astype(float) / 255 * (image.max() - image.min()) + image.min()

            return denoised

        except ImportError:
            return image

    def _bilateral_filter(self, image: np.ndarray, strength: float) -> np.ndarray:
        """Bilateral filtering - edge-preserving smoothing"""
        try:
            import cv2

            d = int(9 * strength)  # Diameter
            sigma_color = 75 * strength
            sigma_space = 75 * strength

            if image.dtype != np.uint8:
                img_normalized = ((image - image.min()) /
                                 (image.max() - image.min() + 1e-10) * 255).astype(np.uint8)
            else:
                img_normalized = image

            denoised = cv2.bilateralFilter(img_normalized, d, sigma_color, sigma_space)

            if image.dtype != np.uint8:
                denoised = denoised.astype(float) / 255 * (image.max() - image.min()) + image.min()

            return denoised

        except ImportError:
            return image

    def _wavelet_denoise(self, image: np.ndarray, strength: float) -> np.ndarray:
        """Wavelet-based denoising"""
        try:
            import pywt

            # Choose wavelet
            wavelet = 'db4'
            level = min(4, int(np.log2(min(image.shape))))

            # Decompose
            coeffs = pywt.wavedec2(image, wavelet, level=level)

            # Estimate noise from finest detail coefficients
            sigma = self._estimate_noise(image) * strength
            threshold = sigma * np.sqrt(2 * np.log(image.size))

            # Threshold detail coefficients
            new_coeffs = [coeffs[0]]  # Keep approximation
            for detail_level in coeffs[1:]:
                new_detail = tuple(
                    pywt.threshold(d, threshold, mode='soft')
                    for d in detail_level
                )
                new_coeffs.append(new_detail)

            # Reconstruct
            denoised = pywt.waverec2(new_coeffs, wavelet)

            # Match original size
            denoised = denoised[:image.shape[0], :image.shape[1]]

            return denoised

        except ImportError:
            return image

    def _anisotropic_diffusion(
        self,
        image: np.ndarray,
        strength: float,
        iterations: int = 10
    ) -> np.ndarray:
        """
        Perona-Malik anisotropic diffusion.
        Smooths flat regions while preserving edges.
        """
        img = image.astype(float)

        # Parameters
        kappa = 50 * strength  # Conduction coefficient
        gamma = 0.1  # Time step

        for _ in range(iterations):
            # Compute gradients
            grad_n = np.roll(img, -1, axis=0) - img
            grad_s = np.roll(img, 1, axis=0) - img
            grad_e = np.roll(img, -1, axis=1) - img
            grad_w = np.roll(img, 1, axis=1) - img

            # Conduction coefficients (Perona-Malik)
            c_n = np.exp(-(grad_n / kappa) ** 2)
            c_s = np.exp(-(grad_s / kappa) ** 2)
            c_e = np.exp(-(grad_e / kappa) ** 2)
            c_w = np.exp(-(grad_w / kappa) ** 2)

            # Update
            img = img + gamma * (
                c_n * grad_n + c_s * grad_s + c_e * grad_e + c_w * grad_w
            )

        return img

    def _estimate_noise(self, image: np.ndarray) -> float:
        """Estimate noise level using MAD of wavelet coefficients"""
        try:
            import pywt

            # Use finest wavelet detail coefficients
            coeffs = pywt.dwt2(image, 'db1')
            detail = coeffs[1][0]  # Horizontal detail

            # MAD estimator
            sigma = np.median(np.abs(detail)) / 0.6745
            return sigma

        except ImportError:
            # Fallback: local variance method
            from scipy.ndimage import generic_filter

            def local_var(x):
                return np.var(x)

            local_vars = generic_filter(image, local_var, size=3)
            return np.sqrt(np.median(local_vars))


class MultiScaleWaveletAnalyzer:
    """
    Multi-scale wavelet analysis for edge detection.

    Detects edges at multiple scales and combines them
    for robust edge localization.
    """

    def __init__(self, wavelet: str = 'db4', max_level: int = 4):
        self.wavelet = wavelet
        self.max_level = max_level

    def analyze(
        self,
        profile: np.ndarray,
        scale_nm_per_pixel: float = 1.0
    ) -> Dict[str, Any]:
        """
        Multi-scale edge analysis of 1D profile.

        Args:
            profile: 1D intensity profile
            scale_nm_per_pixel: Scale factor

        Returns:
            Dictionary with multi-scale analysis results
        """
        try:
            import pywt
        except ImportError:
            return self._fallback_analysis(profile, scale_nm_per_pixel)

        # Compute wavelet transform
        max_level = min(self.max_level, pywt.dwt_max_level(len(profile), self.wavelet))
        coeffs = pywt.wavedec(profile, self.wavelet, level=max_level)

        # Analyze edges at each scale
        edge_positions_by_scale = []
        edge_strengths_by_scale = []

        for level, detail in enumerate(coeffs[1:], 1):
            # Find local maxima in detail coefficients (edges)
            edges = self._find_local_maxima(np.abs(detail))

            # Map back to original scale
            scale_factor = 2 ** level
            original_positions = [e * scale_factor for e in edges]

            edge_positions_by_scale.append(original_positions)
            edge_strengths_by_scale.append([np.abs(detail[e]) for e in edges])

        # Combine edges across scales
        combined_edges = self._combine_multiscale_edges(
            edge_positions_by_scale,
            edge_strengths_by_scale,
            len(profile)
        )

        return {
            'edges_nm': [e * scale_nm_per_pixel for e in combined_edges],
            'edges_by_scale': edge_positions_by_scale,
            'strengths_by_scale': edge_strengths_by_scale,
            'num_scales': max_level,
            'dominant_scale': self._find_dominant_scale(edge_strengths_by_scale)
        }

    def _find_local_maxima(self, data: np.ndarray, min_distance: int = 2) -> List[int]:
        """Find local maxima in 1D data"""
        maxima = []
        for i in range(1, len(data) - 1):
            if data[i] > data[i-1] and data[i] > data[i+1]:
                # Check minimum distance from previous maxima
                if not maxima or i - maxima[-1] >= min_distance:
                    maxima.append(i)
        return maxima

    def _combine_multiscale_edges(
        self,
        positions_by_scale: List[List[float]],
        strengths_by_scale: List[List[float]],
        profile_length: int
    ) -> List[float]:
        """Combine edge detections across scales"""
        # Weight edges by scale and strength
        edge_histogram = np.zeros(profile_length)

        for level, (positions, strengths) in enumerate(
            zip(positions_by_scale, strengths_by_scale)
        ):
            scale_weight = 1.0 / (level + 1)  # Finer scales get more weight

            for pos, strength in zip(positions, strengths):
                idx = int(min(pos, profile_length - 1))
                edge_histogram[idx] += strength * scale_weight

        # Find peaks in combined histogram
        return [float(i) for i in self._find_local_maxima(edge_histogram, min_distance=5)]

    def _find_dominant_scale(self, strengths_by_scale: List[List[float]]) -> int:
        """Find scale with strongest edge responses"""
        avg_strengths = [
            np.mean(s) if s else 0 for s in strengths_by_scale
        ]
        return int(np.argmax(avg_strengths)) + 1 if avg_strengths else 1

    def _fallback_analysis(
        self,
        profile: np.ndarray,
        scale_nm_per_pixel: float
    ) -> Dict[str, Any]:
        """Fallback when pywt not available"""
        gradient = np.abs(np.gradient(profile))
        edges = self._find_local_maxima(gradient)

        return {
            'edges_nm': [e * scale_nm_per_pixel for e in edges],
            'edges_by_scale': [edges],
            'strengths_by_scale': [[gradient[e] for e in edges]],
            'num_scales': 1,
            'dominant_scale': 1
        }


class MonteCarloUncertainty:
    """
    Monte Carlo simulation for uncertainty estimation.

    Propagates measurement uncertainty through the analysis
    pipeline to provide robust confidence intervals.
    """

    def __init__(self, n_simulations: int = 1000, random_seed: Optional[int] = None):
        self.n_simulations = n_simulations
        self.rng = np.random.default_rng(random_seed)

    def estimate_uncertainty(
        self,
        profile: np.ndarray,
        measurement_func,
        noise_level: Optional[float] = None,
        scale_nm_per_pixel: float = 1.0
    ) -> Dict[str, Any]:
        """
        Estimate measurement uncertainty using Monte Carlo simulation.

        Args:
            profile: Original intensity profile
            measurement_func: Function that takes profile and returns measurement
            noise_level: Estimated noise level (auto-estimated if None)
            scale_nm_per_pixel: Scale factor

        Returns:
            Dictionary with uncertainty estimates
        """
        # Estimate noise if not provided
        if noise_level is None:
            noise_level = self._estimate_noise(profile)

        # Run simulations
        measurements = []

        for _ in range(self.n_simulations):
            # Add random noise to profile
            noisy_profile = profile + self.rng.normal(0, noise_level, profile.shape)

            # Make measurement
            try:
                result = measurement_func(noisy_profile)
                if result is not None and np.isfinite(result):
                    measurements.append(result)
            except Exception:
                continue

        if len(measurements) < 10:
            return {
                'mean': float(measurement_func(profile)) if measurement_func(profile) else 0,
                'std': float('inf'),
                'ci_95_low': 0,
                'ci_95_high': float('inf'),
                'distribution': np.array([]),
                'n_valid': len(measurements)
            }

        measurements = np.array(measurements)

        # Calculate statistics
        mean = np.mean(measurements)
        std = np.std(measurements)
        ci_95 = np.percentile(measurements, [2.5, 97.5])

        return {
            'mean': float(mean),
            'std': float(std),
            'ci_95_low': float(ci_95[0]),
            'ci_95_high': float(ci_95[1]),
            'distribution': measurements,
            'n_valid': len(measurements),
            'noise_level': noise_level
        }

    def _estimate_noise(self, profile: np.ndarray) -> float:
        """Estimate noise level from profile"""
        # Use difference between adjacent pixels
        diff = np.diff(profile)
        # MAD estimator for noise
        return np.median(np.abs(diff - np.median(diff))) / 0.6745


class AtomicColumnFitter:
    """
    Atomic column position fitting for crystalline materials.

    Fits Gaussian peaks to atomic columns for precise
    position determination in crystalline HR-TEM images.
    """

    def __init__(self, expected_spacing_nm: Optional[float] = None):
        self.expected_spacing_nm = expected_spacing_nm

    def fit_columns(
        self,
        profile: np.ndarray,
        scale_nm_per_pixel: float = 1.0,
        min_peak_height: float = 0.1
    ) -> Dict[str, Any]:
        """
        Fit Gaussian peaks to atomic columns in profile.

        Args:
            profile: 1D intensity profile
            scale_nm_per_pixel: Scale factor
            min_peak_height: Minimum peak height (relative)

        Returns:
            Dictionary with fitted column positions
        """
        # Normalize profile
        profile_norm = (profile - profile.min()) / (profile.max() - profile.min() + 1e-10)

        # Find peaks (potential atomic columns)
        peaks = self._find_peaks(profile_norm, min_peak_height)

        if len(peaks) < 2:
            return {
                'column_positions_nm': [],
                'spacings_nm': [],
                'mean_spacing_nm': 0,
                'spacing_std_nm': 0,
                'n_columns': 0
            }

        # Fit Gaussian to each peak for sub-pixel position
        fitted_positions = []
        fitted_widths = []

        for peak_idx in peaks:
            result = self._fit_gaussian_peak(profile_norm, peak_idx)
            if result is not None:
                fitted_positions.append(result['position'] * scale_nm_per_pixel)
                fitted_widths.append(result['width'] * scale_nm_per_pixel)

        # Calculate spacings
        fitted_positions = np.array(fitted_positions)
        spacings = np.diff(fitted_positions)

        return {
            'column_positions_nm': fitted_positions.tolist(),
            'column_widths_nm': fitted_widths,
            'spacings_nm': spacings.tolist(),
            'mean_spacing_nm': float(np.mean(spacings)) if len(spacings) > 0 else 0,
            'spacing_std_nm': float(np.std(spacings)) if len(spacings) > 0 else 0,
            'n_columns': len(fitted_positions)
        }

    def _find_peaks(
        self,
        profile: np.ndarray,
        min_height: float,
        min_distance: int = 3
    ) -> List[int]:
        """Find peaks in profile"""
        peaks = []

        for i in range(1, len(profile) - 1):
            if profile[i] > profile[i-1] and profile[i] > profile[i+1]:
                if profile[i] >= min_height:
                    if not peaks or i - peaks[-1] >= min_distance:
                        peaks.append(i)

        return peaks

    def _fit_gaussian_peak(
        self,
        profile: np.ndarray,
        peak_idx: int,
        window: int = 5
    ) -> Optional[Dict[str, float]]:
        """Fit Gaussian to single peak"""
        # Extract window
        start = max(0, peak_idx - window)
        end = min(len(profile), peak_idx + window + 1)

        if end - start < 5:
            return None

        local_profile = profile[start:end]
        local_x = np.arange(len(local_profile))

        try:
            from scipy.optimize import curve_fit

            def gaussian(x, amp, center, width):
                return amp * np.exp(-(x - center) ** 2 / (2 * width ** 2))

            # Initial guess
            p0 = [local_profile.max(), len(local_profile) / 2, 2]

            popt, _ = curve_fit(
                gaussian, local_x, local_profile,
                p0=p0, maxfev=1000
            )

            return {
                'position': start + popt[1],
                'amplitude': popt[0],
                'width': abs(popt[2])
            }

        except (ImportError, RuntimeError):
            # Fallback: parabolic fit
            detector = SubPixelEdgeDetector(SubPixelMethod.PARABOLIC)
            pos = detector._parabolic_fit(local_x + start, local_profile)
            return {'position': pos, 'amplitude': local_profile.max(), 'width': 2}


class PrecisionCDMeasurer:
    """
    High-precision Critical Dimension measurement system.

    Combines all precision measurement techniques:
    - Sub-pixel edge detection
    - ESF/LSF analysis
    - Advanced denoising
    - Multi-scale wavelet analysis
    - Monte Carlo uncertainty
    - Atomic column fitting (for crystalline materials)
    """

    def __init__(
        self,
        subpixel_method: SubPixelMethod = SubPixelMethod.GAUSSIAN,
        denoising_method: str = 'nlm',
        denoising_strength: float = 1.0,
        monte_carlo_simulations: int = 500,
        enable_atomic_fitting: bool = False,
        expected_lattice_nm: Optional[float] = None
    ):
        self.subpixel_detector = SubPixelEdgeDetector(subpixel_method)
        self.esf_analyzer = ESFLSFAnalyzer()
        self.denoiser = AdvancedDenoiser()
        self.wavelet_analyzer = MultiScaleWaveletAnalyzer()
        self.monte_carlo = MonteCarloUncertainty(monte_carlo_simulations)
        self.atomic_fitter = AtomicColumnFitter(expected_lattice_nm)

        self.denoising_method = denoising_method
        self.denoising_strength = denoising_strength
        self.enable_atomic_fitting = enable_atomic_fitting

    def measure_cd(
        self,
        image: np.ndarray,
        profile_y: int,
        scale_nm_per_pixel: float,
        profile_width: int = 50
    ) -> PrecisionMeasurementResult:
        """
        Perform high-precision CD measurement.

        Args:
            image: HR-TEM image
            profile_y: Y position for profile extraction
            scale_nm_per_pixel: Scale factor
            profile_width: Width of profile for averaging

        Returns:
            PrecisionMeasurementResult with all measurements
        """
        # Step 1: Denoise image
        denoised = self.denoiser.denoise(
            image, self.denoising_method, self.denoising_strength
        )

        # Step 2: Extract profile (average over multiple lines for noise reduction)
        y_start = max(0, profile_y - profile_width // 2)
        y_end = min(image.shape[0], profile_y + profile_width // 2)
        profile = np.mean(denoised[y_start:y_end], axis=0)

        # Step 3: Multi-scale wavelet edge detection
        wavelet_result = self.wavelet_analyzer.analyze(profile, scale_nm_per_pixel)
        coarse_edges = [int(e / scale_nm_per_pixel) for e in wavelet_result['edges_nm']]

        if len(coarse_edges) < 2:
            return self._empty_result()

        # Step 4: Sub-pixel edge refinement
        subpixel_edges = self.subpixel_detector.detect_edges_subpixel(
            profile, coarse_edges, scale_nm_per_pixel
        )

        # Step 5: ESF/LSF analysis for each edge
        esf_results = []
        for edge_idx in coarse_edges[:2]:  # Analyze first two edges
            esf_result = self.esf_analyzer.analyze_edge(
                profile, edge_idx, scale_nm_per_pixel
            )
            esf_results.append(esf_result)

        # Step 6: Calculate CD from edge positions
        if len(subpixel_edges) >= 2:
            cd_nm = abs(subpixel_edges[1] - subpixel_edges[0])
        else:
            cd_nm = 0

        # Step 7: Monte Carlo uncertainty estimation
        def measure_cd_from_profile(p):
            grad = np.abs(np.gradient(p))
            peaks = np.argsort(grad)[-2:]
            return abs(peaks[1] - peaks[0]) * scale_nm_per_pixel

        mc_result = self.monte_carlo.estimate_uncertainty(
            profile, measure_cd_from_profile, scale_nm_per_pixel=scale_nm_per_pixel
        )

        # Step 8: Atomic column fitting (if enabled)
        atomic_result = None
        if self.enable_atomic_fitting:
            atomic_result = self.atomic_fitter.fit_columns(profile, scale_nm_per_pixel)

        # Combine results
        avg_esf_width = np.mean([r['esf_width_nm'] for r in esf_results]) if esf_results else 0
        avg_lsf_fwhm = np.mean([r['lsf_fwhm_nm'] for r in esf_results]) if esf_results else 0

        # Calculate SNR
        signal = profile.max() - profile.min()
        noise = mc_result.get('noise_level', 1)
        snr_db = 20 * np.log10(signal / noise) if noise > 0 else 0

        # Confidence based on multiple factors
        confidence = self._calculate_confidence(
            esf_results, mc_result, wavelet_result, snr_db
        )

        return PrecisionMeasurementResult(
            thickness_nm=cd_nm,
            uncertainty_nm=mc_result['std'],
            ci_95_low=mc_result['ci_95_low'],
            ci_95_high=mc_result['ci_95_high'],
            sub_pixel_positions=subpixel_edges,
            esf_width_nm=avg_esf_width,
            lsf_fwhm_nm=avg_lsf_fwhm,
            snr_db=snr_db,
            method_contributions={
                'wavelet_edges': len(wavelet_result['edges_nm']),
                'dominant_scale': wavelet_result['dominant_scale'],
                'mc_valid_samples': mc_result['n_valid']
            },
            monte_carlo_distribution=mc_result['distribution'],
            confidence_level=confidence,
            atomic_spacing_nm=atomic_result['mean_spacing_nm'] if atomic_result else None
        )

    def _calculate_confidence(
        self,
        esf_results: List[Dict],
        mc_result: Dict,
        wavelet_result: Dict,
        snr_db: float
    ) -> float:
        """Calculate overall confidence score"""
        scores = []

        # ESF sharpness confidence
        if esf_results:
            avg_sharpness = np.mean([r['confidence'] for r in esf_results])
            scores.append(min(avg_sharpness, 1.0))

        # Monte Carlo convergence confidence
        if mc_result['n_valid'] > 0:
            mc_confidence = min(mc_result['n_valid'] / 100, 1.0)
            scores.append(mc_confidence)

        # SNR confidence
        snr_confidence = min(max(snr_db / 30, 0), 1.0)
        scores.append(snr_confidence)

        # Multi-scale consistency
        if wavelet_result['num_scales'] > 1:
            scores.append(0.8)  # Bonus for multi-scale confirmation

        return float(np.mean(scores)) if scores else 0.0

    def _empty_result(self) -> PrecisionMeasurementResult:
        """Return empty result"""
        return PrecisionMeasurementResult(
            thickness_nm=0,
            uncertainty_nm=float('inf'),
            ci_95_low=0,
            ci_95_high=float('inf'),
            sub_pixel_positions=[],
            esf_width_nm=0,
            lsf_fwhm_nm=0,
            snr_db=0,
            method_contributions={},
            confidence_level=0
        )
