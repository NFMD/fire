"""
Scale Bar OCR Auto-Detection for TEM Images

Detects and reads scale bars from TEM images using image processing.
Inspired by nanoSAM's scale bar detection approach.

Methods:
1. Template-based: Find bright horizontal bar region at bottom of image
2. Text detection: Read scale value text near the bar
3. Pixel measurement: Measure bar length in pixels and convert to nm
"""
import re
from typing import Optional, Tuple, Dict, List
import numpy as np
import cv2
from loguru import logger


class ScaleBarDetector:
    """
    Automatic scale bar detection and reading from TEM images.

    TEM images typically have a scale bar at the bottom of the image,
    consisting of:
    - A bright horizontal bar (white on dark background)
    - A text label with the scale value (e.g., "100 nm", "1 um")
    """

    # Common scale bar values in nm
    KNOWN_SCALE_VALUES = [
        0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500,
        1000, 2000, 5000, 10000, 20000, 50000
    ]

    def __init__(self):
        self.ocr_available = False
        try:
            import pytesseract
            self.ocr_available = True
            self._pytesseract = pytesseract
        except ImportError:
            logger.debug("pytesseract not available, using pattern-based detection")

    def detect_scale_bar(self, image: np.ndarray,
                         search_region: str = 'bottom') -> Optional[Dict]:
        """
        Detect scale bar in TEM image.

        Args:
            image: Grayscale image (0-255 uint8 or 0-1 float)
            search_region: Where to search ('bottom', 'top', 'full')

        Returns:
            Dictionary with scale bar info, or None if not found:
            {
                'bar_length_px': int,
                'bar_position': (x1, y1, x2, y2),
                'scale_value_nm': float,
                'text_detected': str,
                'nm_per_pixel': float,
                'confidence': float
            }
        """
        # Normalize image to uint8
        if image.max() <= 1.0:
            img = (image * 255).astype(np.uint8)
        elif image.dtype != np.uint8:
            img = image.astype(np.uint8)
        else:
            img = image.copy()

        h, w = img.shape[:2]

        # Extract search region
        if search_region == 'bottom':
            # Bottom 20% of image (most common location for scale bars)
            roi_y_start = int(h * 0.75)
            roi = img[roi_y_start:, :]
        elif search_region == 'top':
            roi_y_start = 0
            roi = img[:int(h * 0.25), :]
        else:
            roi_y_start = 0
            roi = img

        # Step 1: Detect the bright bar
        bar_info = self._detect_bar(roi)
        if bar_info is None:
            logger.debug("No scale bar detected")
            return None

        bar_x1, bar_y1, bar_x2, bar_y2, bar_length = bar_info

        # Adjust coordinates to full image
        bar_y1 += roi_y_start
        bar_y2 += roi_y_start

        # Step 2: Try to read scale text
        text_region = self._extract_text_region(img, bar_x1, bar_y1, bar_x2, bar_y2)
        scale_value_nm, text_detected, text_confidence = self._read_scale_text(text_region)

        # Step 3: If OCR failed, try pattern matching on the info bar
        if scale_value_nm is None:
            scale_value_nm, text_detected, text_confidence = self._detect_scale_from_info_bar(
                img, roi_y_start
            )

        if scale_value_nm is None:
            logger.debug("Could not determine scale value from text")
            return None

        # Calculate nm per pixel
        nm_per_pixel = scale_value_nm / bar_length if bar_length > 0 else None

        result = {
            'bar_length_px': int(bar_length),
            'bar_position': (int(bar_x1), int(bar_y1), int(bar_x2), int(bar_y2)),
            'scale_value_nm': float(scale_value_nm),
            'text_detected': text_detected or "",
            'nm_per_pixel': float(nm_per_pixel) if nm_per_pixel else None,
            'confidence': float(text_confidence),
        }

        logger.info(f"Scale bar detected: {scale_value_nm} nm = {bar_length} px "
                    f"-> {nm_per_pixel:.4f} nm/px (confidence: {text_confidence:.2f})")

        return result

    def _detect_bar(self, roi: np.ndarray) -> Optional[Tuple[int, int, int, int, int]]:
        """
        Detect bright horizontal bar in ROI.

        Returns: (x1, y1, x2, y2, bar_length) or None
        """
        h, w = roi.shape[:2]

        # Threshold to find bright regions
        # Use adaptive thresholding since TEM images vary widely
        mean_val = roi.mean()
        threshold = max(mean_val + 50, 180)

        _, binary = cv2.threshold(roi, threshold, 255, cv2.THRESH_BINARY)

        # Morphological operations to connect bar segments
        kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 1))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_h)

        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            # Try lower threshold
            threshold = max(mean_val + 20, 150)
            _, binary = cv2.threshold(roi, threshold, 255, cv2.THRESH_BINARY)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_h)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Find horizontal bar-like contours
        candidates = []
        for cnt in contours:
            x, y, cw, ch = cv2.boundingRect(cnt)

            # Scale bar characteristics:
            # - Much wider than tall (aspect ratio > 5)
            # - Not too small (at least 30px wide)
            # - Not too large (less than 70% of image width)
            # - Relatively thin
            aspect_ratio = cw / max(ch, 1)

            if (aspect_ratio > 5 and
                cw > 30 and
                cw < w * 0.7 and
                ch < h * 0.15 and
                ch < 30):
                candidates.append((x, y, x + cw, y + ch, cw, aspect_ratio))

        if not candidates:
            return None

        # Select the best candidate (prefer wider, higher aspect ratio)
        best = max(candidates, key=lambda c: c[4] * c[5])
        return best[0], best[1], best[2], best[3], best[4]

    def _extract_text_region(self, image: np.ndarray,
                            bar_x1: int, bar_y1: int,
                            bar_x2: int, bar_y2: int) -> np.ndarray:
        """Extract region near scale bar that likely contains text"""
        h, w = image.shape[:2]

        # Text is usually below or above the bar
        # Expand search region
        text_y1 = max(0, bar_y1 - 40)
        text_y2 = min(h, bar_y2 + 40)
        text_x1 = max(0, bar_x1 - 50)
        text_x2 = min(w, bar_x2 + 100)

        return image[text_y1:text_y2, text_x1:text_x2]

    def _read_scale_text(self, text_region: np.ndarray) -> Tuple[Optional[float], Optional[str], float]:
        """
        Read scale value from text region.

        Returns: (scale_value_nm, text_detected, confidence)
        """
        if text_region.size == 0:
            return None, None, 0.0

        # Try OCR if available
        if self.ocr_available:
            try:
                result = self._ocr_read(text_region)
                if result[0] is not None:
                    return result
            except Exception as e:
                logger.debug(f"OCR failed: {e}")

        # Fallback: try pattern-based detection on the raw pixel patterns
        return None, None, 0.0

    def _ocr_read(self, region: np.ndarray) -> Tuple[Optional[float], Optional[str], float]:
        """Read text using OCR"""
        # Preprocess for better OCR
        if region.max() <= 1.0:
            region = (region * 255).astype(np.uint8)

        # Upscale for better OCR accuracy
        scale_factor = max(1, 300 // max(region.shape[:2]))
        if scale_factor > 1:
            region = cv2.resize(region, None, fx=scale_factor, fy=scale_factor,
                              interpolation=cv2.INTER_CUBIC)

        # Try both normal and inverted
        for img in [region, 255 - region]:
            # Threshold
            _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            try:
                text = self._pytesseract.image_to_string(
                    binary,
                    config='--psm 7 -c tessedit_char_whitelist=0123456789.nmumµ '
                )
                text = text.strip()

                if text:
                    parsed = self._parse_scale_text(text)
                    if parsed is not None:
                        return parsed, text, 0.8
            except Exception:
                pass

        return None, None, 0.0

    def _parse_scale_text(self, text: str) -> Optional[float]:
        """Parse scale value from text string to nm"""
        # Clean text
        text = text.strip().replace(',', '.').replace('O', '0').replace('l', '1')

        # Patterns to match scale text
        patterns = [
            # "100 nm", "50nm", "100.0 nm"
            (r'(\d+\.?\d*)\s*nm', 1.0),
            # "1 um", "2.5 um", "1 µm"
            (r'(\d+\.?\d*)\s*[uµ]m', 1000.0),
            # "0.1 um"
            (r'(\d+\.?\d*)\s*[uµ]m', 1000.0),
            # "1 mm"
            (r'(\d+\.?\d*)\s*mm', 1e6),
            # "10 A" (angstrom)
            (r'(\d+\.?\d*)\s*[AÅ]', 0.1),
        ]

        for pattern, multiplier in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                value = float(match.group(1))
                nm_value = value * multiplier

                # Validate against known scale values
                if self._is_reasonable_scale(nm_value):
                    return nm_value

        return None

    def _is_reasonable_scale(self, nm_value: float) -> bool:
        """Check if scale value is reasonable for TEM"""
        # TEM scale bars are typically 0.1 nm to 100 um
        return 0.1 <= nm_value <= 100000

    def _detect_scale_from_info_bar(self, image: np.ndarray,
                                     roi_y_start: int) -> Tuple[Optional[float], Optional[str], float]:
        """
        Try to detect scale from the TEM information bar.

        Many TEM images have a dark info bar at the bottom with bright text.
        """
        h, w = image.shape[:2]

        # Look at the bottom portion of the image
        info_region = image[int(h * 0.85):, :]

        if info_region.size == 0:
            return None, None, 0.0

        # Check if there's a distinct info bar (region with uniform dark/bright)
        row_means = info_region.mean(axis=1)

        # Find transition to info bar
        if len(row_means) < 10:
            return None, None, 0.0

        # Detect info bar as a region with different mean intensity
        diff = np.abs(np.diff(row_means))
        if diff.max() < 20:
            return None, None, 0.0

        # Try OCR on the info bar
        if self.ocr_available:
            try:
                # Preprocess
                scale_factor = max(1, 200 // max(info_region.shape[:2]))
                if scale_factor > 1:
                    upscaled = cv2.resize(info_region, None,
                                         fx=scale_factor, fy=scale_factor,
                                         interpolation=cv2.INTER_CUBIC)
                else:
                    upscaled = info_region

                # Try both normal and inverted
                for img in [upscaled, 255 - upscaled]:
                    _, binary = cv2.threshold(img, 0, 255,
                                            cv2.THRESH_BINARY + cv2.THRESH_OTSU)

                    text = self._pytesseract.image_to_string(binary)
                    text = text.strip()

                    if text:
                        parsed = self._parse_scale_text(text)
                        if parsed is not None:
                            return parsed, text, 0.6

            except Exception:
                pass

        return None, None, 0.0

    def detect_and_calibrate(self, image: np.ndarray) -> Optional[float]:
        """
        Convenience method: detect scale bar and return nm/pixel.

        Args:
            image: Grayscale TEM image

        Returns:
            nm_per_pixel value, or None if detection failed
        """
        result = self.detect_scale_bar(image)
        if result and result['nm_per_pixel']:
            return result['nm_per_pixel']
        return None
