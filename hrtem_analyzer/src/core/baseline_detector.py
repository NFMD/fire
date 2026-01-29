"""
Baseline (0-point) Detector for HR-TEM images

Detects the reference surface line for depth measurements.
"""
from dataclasses import dataclass
from typing import Optional, Tuple, List
import numpy as np
import cv2
from loguru import logger


@dataclass
class BaselineInfo:
    """Information about detected baseline"""
    y_position: int  # Y coordinate of baseline
    confidence: float  # Detection confidence (0-1)
    method: str  # Detection method used
    angle: float = 0.0  # Slight angle if not perfectly horizontal
    x_start: int = 0  # Start X coordinate
    x_end: Optional[int] = None  # End X coordinate (None = full width)


class BaselineDetector:
    """
    Detect baseline (0-point reference) in HR-TEM images.

    Supports multiple detection methods:
    - Manual: User-specified position
    - Gradient: Find maximum vertical gradient (surface)
    - Edge: Find strongest horizontal edge
    - Intensity: Find intensity transition (bright to dark or vice versa)
    """

    def __init__(self, search_region: Tuple[float, float] = (0.2, 0.5)):
        """
        Initialize baseline detector.

        Args:
            search_region: (start, end) as fraction of image height to search
        """
        self.search_region = search_region

    def detect(
            self,
            image: np.ndarray,
            method: str = 'auto',
            hint_y: Optional[int] = None
    ) -> BaselineInfo:
        """
        Detect baseline in image.

        Args:
            image: Grayscale image (0-1 range)
            method: Detection method ('auto', 'gradient', 'edge', 'intensity', 'manual')
            hint_y: Hint for approximate Y position (for 'manual' or to guide 'auto')

        Returns:
            BaselineInfo with detected baseline
        """
        h, w = image.shape[:2]

        if method == 'manual' and hint_y is not None:
            return BaselineInfo(
                y_position=int(hint_y),
                confidence=1.0,
                method='manual',
                x_end=w
            )

        # Define search region
        y_start = int(h * self.search_region[0])
        y_end = int(h * self.search_region[1])

        if method == 'auto' or method == 'gradient':
            result = self._detect_by_gradient(image, y_start, y_end)
            if result.confidence > 0.5:
                return result

        if method == 'auto' or method == 'edge':
            result = self._detect_by_edge(image, y_start, y_end)
            if result.confidence > 0.5:
                return result

        if method == 'auto' or method == 'intensity':
            result = self._detect_by_intensity(image, y_start, y_end)
            return result

        # Fallback to hint or center of search region
        if hint_y is not None:
            y_pos = hint_y
        else:
            y_pos = (y_start + y_end) // 2

        logger.warning(f"Could not reliably detect baseline, using position {y_pos}")
        return BaselineInfo(
            y_position=y_pos,
            confidence=0.3,
            method='fallback',
            x_end=w
        )

    def _detect_by_gradient(
            self,
            image: np.ndarray,
            y_start: int,
            y_end: int
    ) -> BaselineInfo:
        """Detect baseline by finding maximum vertical gradient"""
        h, w = image.shape[:2]

        # Convert to uint8 for Sobel
        img_uint8 = (image * 255).astype(np.uint8)

        # Calculate vertical gradient
        grad_y = cv2.Sobel(img_uint8, cv2.CV_64F, 0, 1, ksize=3)
        grad_magnitude = np.abs(grad_y)

        # Search in specified region
        search_region = grad_magnitude[y_start:y_end, :]

        # Calculate row-wise average gradient
        row_gradients = np.mean(search_region, axis=1)

        # Find peak
        peak_idx = np.argmax(row_gradients)
        peak_value = row_gradients[peak_idx]

        # Calculate confidence based on peak prominence
        mean_grad = np.mean(row_gradients)
        std_grad = np.std(row_gradients)
        confidence = min(1.0, (peak_value - mean_grad) / (std_grad + 1e-6) / 5)

        y_position = y_start + peak_idx

        logger.debug(f"Gradient method: y={y_position}, confidence={confidence:.3f}")

        return BaselineInfo(
            y_position=y_position,
            confidence=max(0, confidence),
            method='gradient',
            x_end=w
        )

    def _detect_by_edge(
            self,
            image: np.ndarray,
            y_start: int,
            y_end: int
    ) -> BaselineInfo:
        """Detect baseline by finding strongest horizontal edge"""
        h, w = image.shape[:2]

        # Convert to uint8
        img_uint8 = (image * 255).astype(np.uint8)

        # Edge detection
        edges = cv2.Canny(img_uint8, 50, 150)

        # Search in specified region
        search_region = edges[y_start:y_end, :]

        # Use Hough line detection for horizontal lines
        lines = cv2.HoughLinesP(
            search_region,
            rho=1,
            theta=np.pi / 180,
            threshold=50,
            minLineLength=w // 4,
            maxLineGap=20
        )

        if lines is None:
            return BaselineInfo(
                y_position=(y_start + y_end) // 2,
                confidence=0.2,
                method='edge_fallback',
                x_end=w
            )

        # Find most horizontal line
        best_line = None
        best_score = 0

        for line in lines:
            x1, y1, x2, y2 = line[0]
            length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            angle = abs(np.arctan2(y2 - y1, x2 - x1))

            # Score: prefer long, horizontal lines
            horizontalness = 1 - abs(angle) / (np.pi / 4)  # 1 for horizontal, 0 for 45deg
            score = length * horizontalness

            if score > best_score:
                best_score = score
                best_line = line[0]

        if best_line is not None:
            x1, y1, x2, y2 = best_line
            y_position = y_start + (y1 + y2) // 2
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            confidence = min(1.0, best_score / (w * 0.5))

            return BaselineInfo(
                y_position=y_position,
                confidence=confidence,
                method='edge',
                angle=angle,
                x_start=x1,
                x_end=x2 + (w - search_region.shape[1])
            )

        return BaselineInfo(
            y_position=(y_start + y_end) // 2,
            confidence=0.2,
            method='edge_fallback',
            x_end=w
        )

    def _detect_by_intensity(
            self,
            image: np.ndarray,
            y_start: int,
            y_end: int
    ) -> BaselineInfo:
        """Detect baseline by finding intensity transition"""
        h, w = image.shape[:2]

        # Calculate row-wise mean intensity
        row_means = np.mean(image[y_start:y_end, :], axis=1)

        # Calculate derivative of intensity profile
        intensity_diff = np.diff(row_means)

        # Find largest transition
        max_diff_idx = np.argmax(np.abs(intensity_diff))
        max_diff_value = np.abs(intensity_diff[max_diff_idx])

        # Confidence based on transition strength
        confidence = min(1.0, max_diff_value / 0.2)  # 0.2 is ~20% intensity change

        y_position = y_start + max_diff_idx

        logger.debug(f"Intensity method: y={y_position}, confidence={confidence:.3f}")

        return BaselineInfo(
            y_position=y_position,
            confidence=confidence,
            method='intensity',
            x_end=w
        )

    def refine_baseline(
            self,
            image: np.ndarray,
            initial: BaselineInfo,
            window_size: int = 20
    ) -> BaselineInfo:
        """
        Refine baseline detection around initial estimate.

        Args:
            image: Grayscale image
            initial: Initial baseline estimate
            window_size: Search window around initial position

        Returns:
            Refined BaselineInfo
        """
        y_start = max(0, initial.y_position - window_size)
        y_end = min(image.shape[0], initial.y_position + window_size)

        # Re-detect in narrower region
        refined = self._detect_by_gradient(image, y_start, y_end)

        # Boost confidence if close to initial
        if abs(refined.y_position - initial.y_position) < window_size // 2:
            refined.confidence = min(1.0, refined.confidence * 1.2)

        return refined
