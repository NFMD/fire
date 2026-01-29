"""
Image Preprocessor for HR-TEM images

Provides multiple preprocessing variants for robust measurement.
"""
from dataclasses import dataclass
from typing import List, Tuple, Dict, Generator, Optional
import numpy as np
import cv2
from loguru import logger


@dataclass
class PreprocessedImage:
    """Container for preprocessed image with metadata"""
    image: np.ndarray
    method_name: str
    rotation_angle: float
    scale_factor: float
    params: Dict
    transform_matrix: Optional[np.ndarray] = None

    def get_inverse_transform(self) -> Optional[np.ndarray]:
        """Get inverse transformation matrix for coordinate mapping"""
        if self.transform_matrix is not None:
            return cv2.invertAffineTransform(self.transform_matrix)
        return None


class ImagePreprocessor:
    """
    Multi-method image preprocessor.

    Generates multiple variants of the input image using different:
    - Preprocessing methods (CLAHE, filtering, etc.)
    - Rotation angles (for orientation robustness)
    - Scale factors (for multi-scale analysis)
    """

    def __init__(self):
        self.preprocessing_methods = {
            'original': self._preprocess_original,
            'clahe': self._preprocess_clahe,
            'gaussian_blur': self._preprocess_gaussian,
            'bilateral_filter': self._preprocess_bilateral,
            'median_filter': self._preprocess_median,
            'unsharp_mask': self._preprocess_unsharp,
            'denoise_nlm': self._preprocess_nlm_denoise,
        }

    def generate_variants(
            self,
            image: np.ndarray,
            methods: List[str] = None,
            rotation_angles: List[float] = None,
            scale_factors: List[float] = None,
            yield_mode: bool = True
    ) -> Generator[PreprocessedImage, None, None]:
        """
        Generate multiple preprocessed variants of the input image.

        Memory efficient - yields one variant at a time.

        Args:
            image: Input grayscale image (0-1 range)
            methods: List of preprocessing method names
            rotation_angles: List of rotation angles in degrees
            scale_factors: List of scale factors
            yield_mode: If True, yield variants one at a time (memory efficient)

        Yields:
            PreprocessedImage objects
        """
        if methods is None:
            methods = ['original', 'clahe', 'bilateral_filter']
        if rotation_angles is None:
            rotation_angles = [0.0]
        if scale_factors is None:
            scale_factors = [1.0]

        # Convert to uint8 for OpenCV operations
        img_uint8 = (image * 255).astype(np.uint8)
        h, w = img_uint8.shape[:2]
        center = (w // 2, h // 2)

        for method in methods:
            if method not in self.preprocessing_methods:
                logger.warning(f"Unknown preprocessing method: {method}")
                continue

            for angle in rotation_angles:
                for scale in scale_factors:
                    # Apply preprocessing
                    preprocess_func = self.preprocessing_methods[method]
                    processed = preprocess_func(img_uint8)

                    # Apply rotation and scaling
                    if angle != 0.0 or scale != 1.0:
                        transform_matrix = cv2.getRotationMatrix2D(center, angle, scale)

                        # Calculate new dimensions for scale != 1
                        if scale != 1.0:
                            new_w = int(w * scale)
                            new_h = int(h * scale)
                        else:
                            new_w, new_h = w, h

                        processed = cv2.warpAffine(
                            processed,
                            transform_matrix,
                            (new_w, new_h),
                            flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_REFLECT
                        )
                    else:
                        transform_matrix = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)

                    # Convert back to float32
                    processed_float = processed.astype(np.float32) / 255.0

                    yield PreprocessedImage(
                        image=processed_float,
                        method_name=method,
                        rotation_angle=angle,
                        scale_factor=scale,
                        params={},
                        transform_matrix=transform_matrix
                    )

    def auto_level(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Automatically detect and correct image tilt.

        Uses Hough line detection to find dominant orientation.

        Returns:
            Tuple of (leveled image, detected angle in degrees)
        """
        img_uint8 = (image * 255).astype(np.uint8)

        # Edge detection
        edges = cv2.Canny(img_uint8, 50, 150, apertureSize=3)

        # Hough line detection
        lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100)

        if lines is None:
            return image, 0.0

        # Calculate dominant angle
        angles = []
        for line in lines[:20]:  # Use top 20 lines
            rho, theta = line[0]
            angle = np.degrees(theta) - 90
            # Normalize to -45 to 45 range
            while angle > 45:
                angle -= 90
            while angle < -45:
                angle += 90
            angles.append(angle)

        # Use median angle (robust to outliers)
        median_angle = np.median(angles)

        # Apply rotation
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, median_angle, 1.0)
        leveled = cv2.warpAffine(
            img_uint8,
            rotation_matrix,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT
        )

        leveled_float = leveled.astype(np.float32) / 255.0

        logger.info(f"Auto-leveled image by {median_angle:.2f} degrees")
        return leveled_float, median_angle

    def _preprocess_original(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """Return original image (no processing)"""
        return image.copy()

    def _preprocess_clahe(
            self,
            image: np.ndarray,
            clip_limit: float = 2.0,
            grid_size: int = 8
    ) -> np.ndarray:
        """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)"""
        clahe = cv2.createCLAHE(
            clipLimit=clip_limit,
            tileGridSize=(grid_size, grid_size)
        )
        return clahe.apply(image)

    def _preprocess_gaussian(
            self,
            image: np.ndarray,
            sigma: float = 0.5
    ) -> np.ndarray:
        """Apply Gaussian blur for noise reduction"""
        ksize = int(sigma * 6) | 1  # Ensure odd
        return cv2.GaussianBlur(image, (ksize, ksize), sigma)

    def _preprocess_bilateral(
            self,
            image: np.ndarray,
            d: int = 9,
            sigma_color: float = 75,
            sigma_space: float = 75
    ) -> np.ndarray:
        """Apply bilateral filter (edge-preserving smoothing)"""
        return cv2.bilateralFilter(image, d, sigma_color, sigma_space)

    def _preprocess_median(self, image: np.ndarray, ksize: int = 3) -> np.ndarray:
        """Apply median filter (salt-and-pepper noise removal)"""
        return cv2.medianBlur(image, ksize)

    def _preprocess_unsharp(
            self,
            image: np.ndarray,
            sigma: float = 1.0,
            amount: float = 1.0
    ) -> np.ndarray:
        """Apply unsharp masking for edge enhancement"""
        blurred = cv2.GaussianBlur(image, (0, 0), sigma)
        sharpened = cv2.addWeighted(image, 1.0 + amount, blurred, -amount, 0)
        return np.clip(sharpened, 0, 255).astype(np.uint8)

    def _preprocess_nlm_denoise(
            self,
            image: np.ndarray,
            h: float = 10,
            template_window: int = 7,
            search_window: int = 21
    ) -> np.ndarray:
        """Apply Non-Local Means denoising"""
        return cv2.fastNlMeansDenoising(image, None, h, template_window, search_window)
