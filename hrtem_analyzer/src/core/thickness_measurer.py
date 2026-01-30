"""
Multi-Method Thickness Measurer for HR-TEM images

Uses multiple edge detection methods, preprocessing variants,
and orientations to achieve high-precision CD measurements.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Generator
import numpy as np
import cv2
from scipy import ndimage
from scipy.stats import trim_mean
from loguru import logger


@dataclass
class EdgePoint:
    """Single edge point detection"""
    x: int
    y: int
    confidence: float
    method: str
    subpixel_x: Optional[float] = None  # Sub-pixel refined position


@dataclass
class SingleMeasurement:
    """Single measurement from one method/variant combination"""
    depth_nm: float
    thickness_nm: float
    left_edge: EdgePoint
    right_edge: EdgePoint
    method_name: str
    preprocessing: str
    rotation_angle: float
    confidence: float
    y_position: int


@dataclass
class MeasurementResult:
    """Aggregated measurement result with statistics"""
    depth_nm: float
    thickness_nm: float  # Final consensus value
    thickness_std: float  # Standard deviation
    thickness_min: float
    thickness_max: float
    confidence: float
    y_position: int
    left_edge_x: int
    right_edge_x: int
    num_measurements: int
    consensus_method: str
    individual_measurements: List[SingleMeasurement] = field(default_factory=list)

    def to_dict(self) -> Dict:
        """Convert to dictionary with native Python types"""
        return {
            'depth_nm': float(self.depth_nm),
            'thickness_nm': float(self.thickness_nm),
            'thickness_std': float(self.thickness_std),
            'thickness_min': float(self.thickness_min),
            'thickness_max': float(self.thickness_max),
            'confidence': float(self.confidence),
            'y_position': int(self.y_position),
            'left_edge_x': int(self.left_edge_x),
            'right_edge_x': int(self.right_edge_x),
            'num_measurements': int(self.num_measurements),
            'consensus_method': str(self.consensus_method),
        }


class ThicknessMeasurer:
    """
    Multi-method thickness measurement engine.

    Achieves high precision by:
    1. Using multiple edge detection algorithms
    2. Applying various preprocessing methods
    3. Measuring at multiple angles
    4. Using sub-pixel edge refinement
    5. Aggregating results with robust statistics
    """

    def __init__(
            self,
            edge_methods: List[str] = None,
            consensus_method: str = 'trimmed_mean',
            trim_percentage: float = 0.1,
            min_confidence: float = 0.3
    ):
        """
        Initialize thickness measurer.

        Args:
            edge_methods: List of edge detection methods to use
            consensus_method: Method for aggregating measurements
                ('mean', 'median', 'trimmed_mean', 'weighted_mean')
            trim_percentage: Trim percentage for trimmed_mean
            min_confidence: Minimum confidence to include measurement
        """
        self.edge_methods = edge_methods or [
            'sobel', 'canny', 'laplacian', 'gradient', 'morphological'
        ]
        self.consensus_method = consensus_method
        self.trim_percentage = trim_percentage
        self.min_confidence = min_confidence

        # Edge detection implementations
        self._edge_detectors = {
            'sobel': self._detect_edges_sobel,
            'canny': self._detect_edges_canny,
            'laplacian': self._detect_edges_laplacian,
            'gradient': self._detect_edges_gradient,
            'morphological': self._detect_edges_morphological,
            'scharr': self._detect_edges_scharr,
        }

    def measure_thickness(
            self,
            image: np.ndarray,
            baseline_y: int,
            depth_nm: float,
            scale_nm_per_pixel: float,
            preprocessing_name: str = 'original',
            rotation_angle: float = 0.0,
            num_lines: int = 5,
            line_spacing: int = 2
    ) -> List[SingleMeasurement]:
        """
        Measure thickness at specified depth using all edge detection methods.

        Args:
            image: Preprocessed grayscale image (0-1 range)
            baseline_y: Y coordinate of baseline (0-point)
            depth_nm: Depth from baseline in nm
            scale_nm_per_pixel: Scale factor
            preprocessing_name: Name of preprocessing method used
            rotation_angle: Rotation angle applied
            num_lines: Number of lines to measure for averaging
            line_spacing: Spacing between measurement lines in pixels

        Returns:
            List of SingleMeasurement from all methods
        """
        measurements = []

        # Convert depth to pixels
        depth_pixels = int(depth_nm / scale_nm_per_pixel)
        center_y = baseline_y + depth_pixels

        # Generate measurement Y positions
        y_positions = self._generate_y_positions(
            center_y, num_lines, line_spacing, image.shape[0]
        )

        for method_name in self.edge_methods:
            if method_name not in self._edge_detectors:
                continue

            method_measurements = []

            for y_pos in y_positions:
                # Extract horizontal profile
                profile = image[y_pos, :].astype(np.float64)

                # Detect edges
                edges = self._edge_detectors[method_name](profile, image, y_pos)

                if len(edges) >= 2:
                    # Find left and right edges of the structure
                    left_edge, right_edge = self._find_structure_edges(
                        edges, profile, image.shape[1]
                    )

                    if left_edge is not None and right_edge is not None:
                        # Calculate thickness
                        thickness_pixels = right_edge.x - left_edge.x
                        if left_edge.subpixel_x and right_edge.subpixel_x:
                            thickness_pixels = right_edge.subpixel_x - left_edge.subpixel_x

                        thickness_nm = thickness_pixels * scale_nm_per_pixel

                        # Calculate measurement confidence
                        confidence = (left_edge.confidence + right_edge.confidence) / 2

                        if confidence >= self.min_confidence and thickness_nm > 0:
                            method_measurements.append(SingleMeasurement(
                                depth_nm=depth_nm,
                                thickness_nm=thickness_nm,
                                left_edge=left_edge,
                                right_edge=right_edge,
                                method_name=method_name,
                                preprocessing=preprocessing_name,
                                rotation_angle=rotation_angle,
                                confidence=confidence,
                                y_position=y_pos
                            ))

            measurements.extend(method_measurements)

        return measurements

    def aggregate_measurements(
            self,
            measurements: List[SingleMeasurement],
            depth_nm: float
    ) -> Optional[MeasurementResult]:
        """
        Aggregate multiple measurements into single result using consensus.

        Args:
            measurements: List of individual measurements
            depth_nm: Depth value for result

        Returns:
            Aggregated MeasurementResult or None if no valid measurements
        """
        if not measurements:
            return None

        # Extract thickness values and confidences
        thicknesses = np.array([m.thickness_nm for m in measurements])
        confidences = np.array([m.confidence for m in measurements])

        # Remove outliers (values outside 3 sigma)
        mean_t = np.mean(thicknesses)
        std_t = np.std(thicknesses)
        if std_t > 0:
            mask = np.abs(thicknesses - mean_t) < 3 * std_t
            thicknesses = thicknesses[mask]
            confidences = confidences[mask]
            measurements = [m for m, valid in zip(measurements, mask) if valid]

        if len(thicknesses) == 0:
            return None

        # Calculate consensus thickness
        if self.consensus_method == 'mean':
            consensus_thickness = np.mean(thicknesses)
        elif self.consensus_method == 'median':
            consensus_thickness = np.median(thicknesses)
        elif self.consensus_method == 'trimmed_mean':
            consensus_thickness = trim_mean(thicknesses, self.trim_percentage)
        elif self.consensus_method == 'weighted_mean':
            weights = confidences / np.sum(confidences)
            consensus_thickness = np.sum(thicknesses * weights)
        else:
            consensus_thickness = np.median(thicknesses)

        # Calculate statistics
        thickness_std = np.std(thicknesses)
        thickness_min = np.min(thicknesses)
        thickness_max = np.max(thicknesses)

        # Calculate average edge positions
        left_edges = [m.left_edge.x for m in measurements]
        right_edges = [m.right_edge.x for m in measurements]
        avg_left = int(np.median(left_edges))
        avg_right = int(np.median(right_edges))

        # Calculate y position (median)
        y_positions = [m.y_position for m in measurements]
        avg_y = int(np.median(y_positions))

        # Aggregate confidence
        aggregate_confidence = np.mean(confidences) * min(1.0, len(measurements) / 10)

        return MeasurementResult(
            depth_nm=depth_nm,
            thickness_nm=consensus_thickness,
            thickness_std=thickness_std,
            thickness_min=thickness_min,
            thickness_max=thickness_max,
            confidence=aggregate_confidence,
            y_position=avg_y,
            left_edge_x=avg_left,
            right_edge_x=avg_right,
            num_measurements=len(measurements),
            consensus_method=self.consensus_method,
            individual_measurements=measurements
        )

    def _generate_y_positions(
            self,
            center_y: int,
            num_lines: int,
            spacing: int,
            max_y: int
    ) -> List[int]:
        """Generate Y positions for measurement lines"""
        positions = []
        half = num_lines // 2

        for i in range(-half, half + 1):
            y = center_y + i * spacing
            if 0 <= y < max_y:
                positions.append(y)

        return positions

    def _find_structure_edges(
            self,
            edges: List[EdgePoint],
            profile: np.ndarray,
            width: int
    ) -> Tuple[Optional[EdgePoint], Optional[EdgePoint]]:
        """Find left and right edges of the nanostructure"""
        if len(edges) < 2:
            return None, None

        # Sort by x position
        edges = sorted(edges, key=lambda e: e.x)

        # Simple approach: find strongest pair with reasonable separation
        # More sophisticated: use intensity profile to identify structure

        # Find profile center (approximate structure center)
        center_x = width // 2

        # Find edges closest to center from each side
        left_candidates = [e for e in edges if e.x < center_x]
        right_candidates = [e for e in edges if e.x >= center_x]

        if not left_candidates or not right_candidates:
            # Try using outermost strong edges
            if len(edges) >= 2:
                return edges[0], edges[-1]
            return None, None

        # Select strongest edge from each side
        left_edge = max(left_candidates, key=lambda e: e.confidence)
        right_edge = max(right_candidates, key=lambda e: e.confidence)

        return left_edge, right_edge

    def _detect_edges_sobel(
            self,
            profile: np.ndarray,
            image: np.ndarray,
            y_pos: int
    ) -> List[EdgePoint]:
        """Detect edges using Sobel operator"""
        # Apply Sobel in x direction
        sobel = np.abs(np.convolve(profile, [-1, 0, 1], mode='same'))

        return self._extract_edge_points(sobel, y_pos, 'sobel')

    def _detect_edges_canny(
            self,
            profile: np.ndarray,
            image: np.ndarray,
            y_pos: int
    ) -> List[EdgePoint]:
        """Detect edges using Canny on the full image row region"""
        # Use a small vertical window for robustness
        y_start = max(0, y_pos - 2)
        y_end = min(image.shape[0], y_pos + 3)

        region = (image[y_start:y_end, :] * 255).astype(np.uint8)

        # Apply Canny
        edges = cv2.Canny(region, 50, 150)

        # Extract edge points from the center row
        center_row = (y_pos - y_start)
        if center_row >= edges.shape[0]:
            center_row = edges.shape[0] - 1

        edge_profile = edges[center_row, :].astype(np.float64) / 255

        return self._extract_edge_points(edge_profile, y_pos, 'canny', threshold=0.5)

    def _detect_edges_laplacian(
            self,
            profile: np.ndarray,
            image: np.ndarray,
            y_pos: int
    ) -> List[EdgePoint]:
        """Detect edges using Laplacian operator"""
        # Laplacian kernel for 1D
        laplacian = np.convolve(profile, [1, -2, 1], mode='same')
        laplacian = np.abs(laplacian)

        return self._extract_edge_points(laplacian, y_pos, 'laplacian')

    def _detect_edges_gradient(
            self,
            profile: np.ndarray,
            image: np.ndarray,
            y_pos: int
    ) -> List[EdgePoint]:
        """Detect edges using gradient magnitude"""
        gradient = np.abs(np.gradient(profile))

        return self._extract_edge_points(gradient, y_pos, 'gradient')

    def _detect_edges_morphological(
            self,
            profile: np.ndarray,
            image: np.ndarray,
            y_pos: int
    ) -> List[EdgePoint]:
        """Detect edges using morphological gradient"""
        profile_uint8 = (profile * 255).astype(np.uint8)

        # Dilation and erosion
        kernel = np.ones(3, dtype=np.uint8)
        dilated = cv2.dilate(profile_uint8.reshape(1, -1), kernel).flatten()
        eroded = cv2.erode(profile_uint8.reshape(1, -1), kernel).flatten()

        morph_gradient = (dilated - eroded).astype(np.float64) / 255

        return self._extract_edge_points(morph_gradient, y_pos, 'morphological')

    def _detect_edges_scharr(
            self,
            profile: np.ndarray,
            image: np.ndarray,
            y_pos: int
    ) -> List[EdgePoint]:
        """Detect edges using Scharr operator (more accurate than Sobel)"""
        # Scharr kernel approximation for 1D
        scharr = np.abs(np.convolve(profile, [-3, 0, 3], mode='same'))

        return self._extract_edge_points(scharr, y_pos, 'scharr')

    def _extract_edge_points(
            self,
            edge_signal: np.ndarray,
            y_pos: int,
            method: str,
            threshold: float = None
    ) -> List[EdgePoint]:
        """Extract edge points from edge signal using peak detection"""
        edges = []

        # Normalize
        if edge_signal.max() > 0:
            edge_signal = edge_signal / edge_signal.max()

        # Adaptive threshold
        if threshold is None:
            threshold = np.mean(edge_signal) + 2 * np.std(edge_signal)
            threshold = max(0.2, min(0.8, threshold))

        # Find local maxima
        from scipy.signal import find_peaks

        peaks, properties = find_peaks(
            edge_signal,
            height=threshold,
            distance=5,
            prominence=0.1
        )

        for peak_idx in peaks:
            confidence = edge_signal[peak_idx]

            # Sub-pixel refinement using parabolic interpolation
            subpixel_x = self._subpixel_refine(edge_signal, peak_idx)

            edges.append(EdgePoint(
                x=peak_idx,
                y=y_pos,
                confidence=confidence,
                method=method,
                subpixel_x=subpixel_x
            ))

        return edges

    def _subpixel_refine(self, signal: np.ndarray, peak_idx: int) -> float:
        """Refine peak position to sub-pixel accuracy using parabolic fit"""
        if peak_idx <= 0 or peak_idx >= len(signal) - 1:
            return float(peak_idx)

        # Parabolic interpolation
        y0 = signal[peak_idx - 1]
        y1 = signal[peak_idx]
        y2 = signal[peak_idx + 1]

        denominator = 2 * (2 * y1 - y0 - y2)
        if abs(denominator) < 1e-10:
            return float(peak_idx)

        offset = (y0 - y2) / denominator
        return peak_idx + offset


class MultiVariantMeasurer:
    """
    Orchestrates measurements across multiple image variants.

    Combines preprocessing variants, rotation angles, and scale factors
    for maximum precision.
    """

    def __init__(
            self,
            thickness_measurer: ThicknessMeasurer = None,
            max_variants: int = 50
    ):
        """
        Initialize multi-variant measurer.

        Args:
            thickness_measurer: ThicknessMeasurer instance
            max_variants: Maximum number of variants to process (memory limit)
        """
        self.measurer = thickness_measurer or ThicknessMeasurer()
        self.max_variants = max_variants

    def measure_all_variants(
            self,
            image_variants: Generator,
            baseline_y: int,
            depths_nm: List[float],
            scale_nm_per_pixel: float,
            num_lines: int = 5,
            line_spacing: int = 2
    ) -> Dict[float, MeasurementResult]:
        """
        Measure across all image variants and aggregate results.

        Memory efficient - processes variants one at a time.

        Args:
            image_variants: Generator of PreprocessedImage objects
            baseline_y: Baseline Y position (adjusted per variant)
            depths_nm: List of depths to measure
            scale_nm_per_pixel: Base scale factor
            num_lines: Lines per depth for measurement
            line_spacing: Spacing between lines

        Returns:
            Dict mapping depth_nm to aggregated MeasurementResult
        """
        # Collect measurements for each depth
        all_measurements: Dict[float, List[SingleMeasurement]] = {
            d: [] for d in depths_nm
        }

        variant_count = 0

        for variant in image_variants:
            if variant_count >= self.max_variants:
                logger.warning(f"Reached max variants limit ({self.max_variants})")
                break

            # Adjust baseline for rotation/scale
            adjusted_baseline_y = self._adjust_baseline(
                baseline_y,
                variant.rotation_angle,
                variant.scale_factor,
                variant.image.shape
            )

            # Adjust scale for scaled variants
            adjusted_scale = scale_nm_per_pixel / variant.scale_factor

            # Measure at each depth
            for depth_nm in depths_nm:
                measurements = self.measurer.measure_thickness(
                    image=variant.image,
                    baseline_y=adjusted_baseline_y,
                    depth_nm=depth_nm,
                    scale_nm_per_pixel=adjusted_scale,
                    preprocessing_name=variant.method_name,
                    rotation_angle=variant.rotation_angle,
                    num_lines=num_lines,
                    line_spacing=line_spacing
                )
                all_measurements[depth_nm].extend(measurements)

            variant_count += 1

            # Clear variant from memory
            del variant

        logger.info(f"Processed {variant_count} variants")

        # Aggregate measurements for each depth
        results = {}
        for depth_nm, measurements in all_measurements.items():
            result = self.measurer.aggregate_measurements(measurements, depth_nm)
            if result:
                results[depth_nm] = result
                logger.info(
                    f"Depth {depth_nm}nm: thickness={result.thickness_nm:.2f}nm "
                    f"(±{result.thickness_std:.2f}nm) from {result.num_measurements} measurements"
                )

        return results

    def _adjust_baseline(
            self,
            baseline_y: int,
            rotation_angle: float,
            scale_factor: float,
            image_shape: Tuple[int, int]
    ) -> int:
        """Adjust baseline position for transformed image"""
        h, w = image_shape

        # Adjust for scale
        adjusted_y = int(baseline_y * scale_factor)

        # For small rotations, baseline shift is minimal
        # For larger rotations, would need proper coordinate transform

        return min(adjusted_y, h - 1)
