"""
Advanced Denoising Methods for HR-TEM Images

Implements state-of-the-art denoising:
- BM3D (Block-Matching and 3D filtering)
- Non-local Means with optimal parameters
- Total Variation denoising
- Anisotropic diffusion
- Wavelet-based denoising
- Deep learning denoising (if available)

These methods are optimized for TEM images with atomic resolution features.
"""
import numpy as np
from typing import Optional, Tuple, Dict, Any
from scipy import ndimage
from loguru import logger

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    import pywt
    PYWT_AVAILABLE = True
except ImportError:
    PYWT_AVAILABLE = False

try:
    from skimage.restoration import denoise_nl_means, denoise_tv_chambolle, denoise_bilateral
    from skimage.restoration import estimate_sigma
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False

try:
    import bm3d
    BM3D_AVAILABLE = True
except ImportError:
    BM3D_AVAILABLE = False


class AdvancedDenoiser:
    """
    Advanced denoising with multiple methods.

    Provides:
    - Automatic noise estimation
    - Method selection based on image characteristics
    - Parameter optimization
    """

    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    def estimate_noise(self, image: np.ndarray) -> float:
        """
        Estimate noise level in the image.

        Uses median absolute deviation (MAD) method on wavelet coefficients.
        """
        if SKIMAGE_AVAILABLE:
            return estimate_sigma(image)

        # Fallback: MAD on high-frequency components
        if PYWT_AVAILABLE:
            coeffs = pywt.dwt2(image, 'db1')
            _, (cH, cV, cD) = coeffs
            # Use diagonal detail coefficients
            sigma = np.median(np.abs(cD)) / 0.6745
            return sigma

        # Simple fallback: local variance
        kernel = np.ones((3, 3)) / 9
        local_mean = ndimage.convolve(image.astype(float), kernel)
        local_var = ndimage.convolve(image.astype(float) ** 2, kernel) - local_mean ** 2
        return np.sqrt(np.median(local_var[local_var > 0]))

    def denoise_bm3d(
        self,
        image: np.ndarray,
        sigma: Optional[float] = None,
        profile: str = 'np'
    ) -> np.ndarray:
        """
        BM3D denoising - state-of-the-art for Gaussian noise.

        Args:
            image: Input image (normalized to [0, 1])
            sigma: Noise standard deviation (auto-estimated if None)
            profile: 'np' for normal profile, 'lc' for low complexity

        Returns:
            Denoised image
        """
        if not BM3D_AVAILABLE:
            logger.warning("BM3D not available, falling back to NLM")
            return self.denoise_nlm(image, sigma)

        # Normalize to [0, 1]
        img_min, img_max = image.min(), image.max()
        if img_max > img_min:
            image_norm = (image - img_min) / (img_max - img_min)
        else:
            return image

        if sigma is None:
            sigma = self.estimate_noise(image_norm)
            if self.verbose:
                logger.info(f"Estimated noise sigma: {sigma:.4f}")

        # BM3D expects noise in [0, 1] range
        denoised = bm3d.bm3d(image_norm, sigma_psd=sigma, stage_arg=bm3d.BM3DStages.ALL_STAGES)

        # Denormalize
        denoised = denoised * (img_max - img_min) + img_min

        return denoised

    def denoise_nlm(
        self,
        image: np.ndarray,
        sigma: Optional[float] = None,
        patch_size: int = 7,
        patch_distance: int = 11,
        h_factor: float = 1.0
    ) -> np.ndarray:
        """
        Non-local Means denoising with optimized parameters.

        Args:
            image: Input image
            sigma: Noise std (auto-estimated if None)
            patch_size: Size of patches for comparison
            patch_distance: Maximum distance for patch search
            h_factor: Multiplier for filtering strength

        Returns:
            Denoised image
        """
        if sigma is None:
            sigma = self.estimate_noise(image)

        if SKIMAGE_AVAILABLE:
            # h parameter controls filtering strength
            h = sigma * h_factor

            denoised = denoise_nl_means(
                image,
                h=h,
                sigma=sigma,
                fast_mode=True,
                patch_size=patch_size,
                patch_distance=patch_distance,
                channel_axis=None
            )
            return denoised

        elif CV2_AVAILABLE:
            # OpenCV fastNlMeansDenoising
            if image.dtype != np.uint8:
                img_min, img_max = image.min(), image.max()
                if img_max > img_min:
                    image_uint8 = ((image - img_min) / (img_max - img_min) * 255).astype(np.uint8)
                else:
                    return image
            else:
                image_uint8 = image
                img_min, img_max = 0, 255

            h = max(3, int(sigma * 255 * h_factor))
            denoised = cv2.fastNlMeansDenoising(
                image_uint8,
                None,
                h=h,
                templateWindowSize=patch_size,
                searchWindowSize=patch_distance * 2 + 1
            )

            # Convert back
            denoised = denoised.astype(float) / 255 * (img_max - img_min) + img_min
            return denoised

        else:
            logger.warning("No denoising library available")
            return image

    def denoise_tv(
        self,
        image: np.ndarray,
        weight: Optional[float] = None
    ) -> np.ndarray:
        """
        Total Variation denoising.

        Good for preserving edges while removing noise.

        Args:
            image: Input image
            weight: TV regularization weight (auto if None)

        Returns:
            Denoised image
        """
        if not SKIMAGE_AVAILABLE:
            logger.warning("skimage not available for TV denoising")
            return image

        if weight is None:
            sigma = self.estimate_noise(image)
            weight = 0.1 + sigma * 2  # Adaptive weight

        denoised = denoise_tv_chambolle(image, weight=weight)
        return denoised

    def denoise_bilateral(
        self,
        image: np.ndarray,
        sigma_spatial: Optional[float] = None,
        sigma_color: Optional[float] = None
    ) -> np.ndarray:
        """
        Bilateral filtering - edge-preserving smoothing.

        Args:
            image: Input image
            sigma_spatial: Spatial sigma (auto if None)
            sigma_color: Range/color sigma (auto if None)

        Returns:
            Denoised image
        """
        if sigma_color is None:
            sigma_color = self.estimate_noise(image) * 3

        if sigma_spatial is None:
            sigma_spatial = 5.0

        if SKIMAGE_AVAILABLE:
            denoised = denoise_bilateral(
                image,
                sigma_color=sigma_color,
                sigma_spatial=sigma_spatial
            )
            return denoised

        elif CV2_AVAILABLE:
            # Normalize for OpenCV
            img_min, img_max = image.min(), image.max()
            if img_max > img_min:
                image_norm = ((image - img_min) / (img_max - img_min) * 255).astype(np.float32)
            else:
                return image

            denoised = cv2.bilateralFilter(
                image_norm,
                d=int(sigma_spatial * 2 + 1),
                sigmaColor=sigma_color * 255,
                sigmaSpace=sigma_spatial
            )

            denoised = denoised / 255 * (img_max - img_min) + img_min
            return denoised

        else:
            return image

    def denoise_anisotropic(
        self,
        image: np.ndarray,
        n_iter: int = 10,
        kappa: float = 50,
        gamma: float = 0.1
    ) -> np.ndarray:
        """
        Anisotropic diffusion (Perona-Malik).

        Smooths homogeneous regions while preserving edges.

        Args:
            image: Input image
            n_iter: Number of iterations
            kappa: Conduction coefficient
            gamma: Step size

        Returns:
            Denoised image
        """
        img = image.astype(float)

        for _ in range(n_iter):
            # Compute gradients
            nabla_n = np.roll(img, -1, axis=0) - img
            nabla_s = np.roll(img, 1, axis=0) - img
            nabla_e = np.roll(img, -1, axis=1) - img
            nabla_w = np.roll(img, 1, axis=1) - img

            # Conduction coefficients (Perona-Malik)
            c_n = np.exp(-(nabla_n / kappa) ** 2)
            c_s = np.exp(-(nabla_s / kappa) ** 2)
            c_e = np.exp(-(nabla_e / kappa) ** 2)
            c_w = np.exp(-(nabla_w / kappa) ** 2)

            # Update
            img = img + gamma * (
                c_n * nabla_n + c_s * nabla_s + c_e * nabla_e + c_w * nabla_w
            )

        return img

    def denoise_wavelet(
        self,
        image: np.ndarray,
        wavelet: str = 'db4',
        level: Optional[int] = None,
        threshold_type: str = 'soft'
    ) -> np.ndarray:
        """
        Wavelet-based denoising.

        Uses BayesShrink or VisuShrink thresholding.

        Args:
            image: Input image
            wavelet: Wavelet type
            level: Decomposition level (auto if None)
            threshold_type: 'soft' or 'hard'

        Returns:
            Denoised image
        """
        if not PYWT_AVAILABLE:
            logger.warning("PyWavelets not available")
            return image

        # Determine level
        if level is None:
            level = pywt.dwt_max_level(min(image.shape), pywt.Wavelet(wavelet).dec_len)
            level = min(level, 4)  # Limit to 4 levels

        # 2D wavelet decomposition
        coeffs = pywt.wavedec2(image, wavelet, level=level)

        # Estimate noise from finest detail coefficients
        sigma = np.median(np.abs(coeffs[-1][-1])) / 0.6745

        # BayesShrink threshold
        def bayes_thresh(detail, sigma):
            sigma_d = np.std(detail)
            if sigma_d > sigma:
                thresh = sigma ** 2 / np.sqrt(max(sigma_d ** 2 - sigma ** 2, 1e-10))
            else:
                thresh = np.max(np.abs(detail))
            return thresh

        # Apply thresholding to detail coefficients
        new_coeffs = [coeffs[0]]  # Keep approximation
        for i in range(1, len(coeffs)):
            new_details = []
            for detail in coeffs[i]:
                thresh = bayes_thresh(detail, sigma)
                if threshold_type == 'soft':
                    denoised_detail = pywt.threshold(detail, thresh, mode='soft')
                else:
                    denoised_detail = pywt.threshold(detail, thresh, mode='hard')
                new_details.append(denoised_detail)
            new_coeffs.append(tuple(new_details))

        # Reconstruct
        denoised = pywt.waverec2(new_coeffs, wavelet)

        # Handle size mismatch
        if denoised.shape != image.shape:
            denoised = denoised[:image.shape[0], :image.shape[1]]

        return denoised

    def denoise_adaptive(
        self,
        image: np.ndarray,
        method: str = 'auto'
    ) -> Tuple[np.ndarray, str]:
        """
        Adaptive denoising with automatic method selection.

        Analyzes image characteristics to choose the best method.

        Args:
            image: Input image
            method: 'auto' for automatic selection, or specific method name

        Returns:
            Tuple of (denoised image, method used)
        """
        sigma = self.estimate_noise(image)

        if method == 'auto':
            # Choose method based on noise level and image characteristics
            if BM3D_AVAILABLE and sigma > 0.05:
                # High noise - use BM3D
                method = 'bm3d'
            elif sigma > 0.02:
                # Medium noise - use NLM
                method = 'nlm'
            else:
                # Low noise - use TV for edge preservation
                method = 'tv'

            if self.verbose:
                logger.info(f"Auto-selected method: {method} (sigma={sigma:.4f})")

        # Apply selected method
        if method == 'bm3d':
            denoised = self.denoise_bm3d(image, sigma)
        elif method == 'nlm':
            denoised = self.denoise_nlm(image, sigma)
        elif method == 'tv':
            denoised = self.denoise_tv(image)
        elif method == 'bilateral':
            denoised = self.denoise_bilateral(image)
        elif method == 'anisotropic':
            denoised = self.denoise_anisotropic(image)
        elif method == 'wavelet':
            denoised = self.denoise_wavelet(image)
        else:
            raise ValueError(f"Unknown method: {method}")

        return denoised, method

    def denoise_multistage(
        self,
        image: np.ndarray,
        stages: Optional[list] = None
    ) -> np.ndarray:
        """
        Multi-stage denoising pipeline.

        Applies multiple complementary denoising methods in sequence.

        Args:
            image: Input image
            stages: List of methods to apply in order

        Returns:
            Denoised image
        """
        if stages is None:
            # Default pipeline
            stages = ['wavelet', 'bilateral']
            if BM3D_AVAILABLE:
                stages = ['bm3d', 'bilateral']

        result = image.copy()
        for stage in stages:
            if stage == 'bm3d' and BM3D_AVAILABLE:
                result = self.denoise_bm3d(result)
            elif stage == 'nlm':
                result = self.denoise_nlm(result)
            elif stage == 'tv':
                result = self.denoise_tv(result)
            elif stage == 'bilateral':
                result = self.denoise_bilateral(result)
            elif stage == 'anisotropic':
                result = self.denoise_anisotropic(result)
            elif stage == 'wavelet':
                result = self.denoise_wavelet(result)

        return result


class PhaseCongruencyEdgeDetector:
    """
    Phase Congruency edge detection.

    Advantages over gradient-based methods:
    - Illumination invariant
    - Contrast invariant
    - Provides edge magnitude and orientation
    - Detects both step and roof edges
    """

    def __init__(
        self,
        n_scale: int = 5,
        n_orient: int = 6,
        min_wavelength: float = 3.0,
        mult: float = 2.1,
        sigma_onf: float = 0.55,
        k: float = 2.0,
        cutoff: float = 0.5,
        g: int = 10
    ):
        """
        Initialize Phase Congruency detector.

        Args:
            n_scale: Number of wavelet scales
            n_orient: Number of orientations
            min_wavelength: Minimum wavelength (pixels)
            mult: Scaling factor between wavelengths
            sigma_onf: Ratio of standard deviation of Gaussian
            k: Noise suppression factor
            cutoff: Frequency cutoff for normalization
            g: Gain for sigmoid weighting
        """
        self.n_scale = n_scale
        self.n_orient = n_orient
        self.min_wavelength = min_wavelength
        self.mult = mult
        self.sigma_onf = sigma_onf
        self.k = k
        self.cutoff = cutoff
        self.g = g

    def _lowpass_filter(self, size: Tuple[int, int], cutoff: float) -> np.ndarray:
        """Create a lowpass Butterworth filter"""
        rows, cols = size
        x = np.arange(-cols // 2, cols // 2)
        y = np.arange(-rows // 2, rows // 2)
        x, y = np.meshgrid(x, y)
        radius = np.sqrt(x ** 2 + y ** 2)
        radius[rows // 2, cols // 2] = 1  # Avoid division by zero

        # Butterworth filter
        n = 15  # Order
        f = 1.0 / (1.0 + (radius / (cutoff * cols)) ** (2 * n))
        return f

    def _log_gabor(
        self,
        size: Tuple[int, int],
        wavelength: float,
        orientation: float
    ) -> np.ndarray:
        """Create a Log-Gabor filter in frequency domain"""
        rows, cols = size

        # Frequency coordinates
        x = np.arange(-cols // 2, cols // 2)
        y = np.arange(-rows // 2, rows // 2)
        x, y = np.meshgrid(x, y)

        radius = np.sqrt(x ** 2 + y ** 2)
        radius[rows // 2, cols // 2] = 1  # Avoid log(0)

        theta = np.arctan2(-y, x)

        # Log-Gabor radial component
        fo = 1.0 / wavelength
        log_gabor = np.exp(-(np.log(radius / (fo * cols)) ** 2) / (2 * np.log(self.sigma_onf) ** 2))
        log_gabor[rows // 2, cols // 2] = 0  # DC = 0

        # Angular component
        d_theta = theta - orientation
        d_theta = np.arctan2(np.sin(d_theta), np.cos(d_theta))  # Wrap to [-pi, pi]
        spread = np.exp(-d_theta ** 2 * self.n_orient ** 2 / (2 * np.log(2)))

        return log_gabor * spread

    def detect(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Compute phase congruency.

        Args:
            image: Input grayscale image

        Returns:
            Dictionary with:
                - 'pc': Phase congruency magnitude
                - 'orientation': Edge orientation
                - 'ft': Feature type (edge vs line)
                - 'energy': Local energy
        """
        rows, cols = image.shape

        # FFT of image
        im_fft = np.fft.fftshift(np.fft.fft2(image))

        # Lowpass filter to remove high frequency noise
        lp = self._lowpass_filter((rows, cols), self.cutoff)

        # Initialize arrays
        sum_e = np.zeros((rows, cols))
        sum_o = np.zeros((rows, cols))
        sum_an = np.zeros((rows, cols))
        max_an = np.zeros((rows, cols))
        energy = np.zeros((rows, cols))

        # Store orientation responses for computing orientation
        e_orient = np.zeros((self.n_orient, rows, cols))
        o_orient = np.zeros((self.n_orient, rows, cols))

        orientations = np.linspace(0, np.pi, self.n_orient, endpoint=False)

        for o, orient in enumerate(orientations):
            sum_e_this_orient = np.zeros((rows, cols))
            sum_o_this_orient = np.zeros((rows, cols))
            sum_an_this_orient = np.zeros((rows, cols))

            for s in range(self.n_scale):
                wavelength = self.min_wavelength * (self.mult ** s)

                # Create Log-Gabor filter
                log_gabor = self._log_gabor((rows, cols), wavelength, orient)
                log_gabor = log_gabor * lp

                # Filter image
                filtered = im_fft * log_gabor
                eo = np.fft.ifft2(np.fft.ifftshift(filtered))

                # Quadrature components
                e = np.real(eo)  # Even (symmetric)
                o = np.imag(eo)  # Odd (antisymmetric)

                an = np.abs(eo)  # Amplitude

                sum_e_this_orient += e
                sum_o_this_orient += o
                sum_an_this_orient += an
                max_an = np.maximum(max_an, an)

            e_orient[o] = sum_e_this_orient
            o_orient[o] = sum_o_this_orient

            sum_e += sum_e_this_orient
            sum_o += sum_o_this_orient
            sum_an += sum_an_this_orient

        # Local energy
        energy = np.sqrt(sum_e ** 2 + sum_o ** 2)

        # Estimate noise threshold
        tau = np.median(max_an) / np.sqrt(np.log(4))
        noise_thresh = tau * self.k

        # Phase congruency with noise compensation
        pc = (energy - noise_thresh) / (sum_an + 1e-10)
        pc = np.maximum(pc, 0)

        # Compute orientation
        x_sum = np.zeros((rows, cols))
        y_sum = np.zeros((rows, cols))
        for o, orient in enumerate(orientations):
            orient_energy = np.sqrt(e_orient[o] ** 2 + o_orient[o] ** 2)
            x_sum += orient_energy * np.cos(2 * orient)
            y_sum += orient_energy * np.sin(2 * orient)

        orientation = np.arctan2(y_sum, x_sum) / 2

        # Feature type (-1 = dark line, 0 = edge, 1 = bright line)
        ft = np.arctan2(sum_e, sum_o)

        return {
            'pc': pc,
            'orientation': orientation,
            'ft': ft,
            'energy': energy
        }

    def detect_edges(
        self,
        image: np.ndarray,
        thresh: float = 0.2
    ) -> np.ndarray:
        """
        Detect edges using phase congruency.

        Args:
            image: Input image
            thresh: Phase congruency threshold

        Returns:
            Binary edge map
        """
        result = self.detect(image)
        pc = result['pc']

        # Non-maximum suppression along gradient direction
        orientation = result['orientation']

        # Quantize orientation to 4 directions
        direction = ((orientation + np.pi / 2) / (np.pi / 4)).astype(int) % 4

        rows, cols = pc.shape
        nms = np.zeros_like(pc)

        for r in range(1, rows - 1):
            for c in range(1, cols - 1):
                d = direction[r, c]
                current = pc[r, c]

                if d == 0:  # Horizontal
                    neighbors = [pc[r, c - 1], pc[r, c + 1]]
                elif d == 1:  # Diagonal /
                    neighbors = [pc[r - 1, c + 1], pc[r + 1, c - 1]]
                elif d == 2:  # Vertical
                    neighbors = [pc[r - 1, c], pc[r + 1, c]]
                else:  # Diagonal \
                    neighbors = [pc[r - 1, c - 1], pc[r + 1, c + 1]]

                if current >= max(neighbors):
                    nms[r, c] = current

        # Threshold
        edges = nms > thresh

        return edges


def denoise_tem_image(
    image: np.ndarray,
    method: str = 'auto',
    preserve_edges: bool = True,
    **kwargs
) -> np.ndarray:
    """
    High-level function to denoise TEM image.

    Args:
        image: Input TEM image
        method: Denoising method ('auto', 'bm3d', 'nlm', 'tv', 'bilateral',
                'anisotropic', 'wavelet', 'multistage')
        preserve_edges: If True, use edge-preserving method
        **kwargs: Additional method-specific parameters

    Returns:
        Denoised image
    """
    denoiser = AdvancedDenoiser(verbose=kwargs.get('verbose', False))

    if method == 'auto':
        denoised, method_used = denoiser.denoise_adaptive(image)
        return denoised
    elif method == 'multistage':
        return denoiser.denoise_multistage(image, kwargs.get('stages'))
    elif method == 'bm3d':
        return denoiser.denoise_bm3d(image, kwargs.get('sigma'))
    elif method == 'nlm':
        return denoiser.denoise_nlm(
            image,
            kwargs.get('sigma'),
            kwargs.get('patch_size', 7),
            kwargs.get('patch_distance', 11)
        )
    elif method == 'tv':
        return denoiser.denoise_tv(image, kwargs.get('weight'))
    elif method == 'bilateral':
        return denoiser.denoise_bilateral(
            image,
            kwargs.get('sigma_spatial'),
            kwargs.get('sigma_color')
        )
    elif method == 'anisotropic':
        return denoiser.denoise_anisotropic(
            image,
            kwargs.get('n_iter', 10),
            kwargs.get('kappa', 50)
        )
    elif method == 'wavelet':
        return denoiser.denoise_wavelet(
            image,
            kwargs.get('wavelet', 'db4'),
            kwargs.get('level')
        )
    else:
        raise ValueError(f"Unknown denoising method: {method}")


def detect_edges_phase_congruency(
    image: np.ndarray,
    thresh: float = 0.2,
    **kwargs
) -> np.ndarray:
    """
    Detect edges using phase congruency.

    Args:
        image: Input image
        thresh: Detection threshold (0-1)
        **kwargs: Additional parameters for PhaseCongruencyEdgeDetector

    Returns:
        Binary edge map
    """
    detector = PhaseCongruencyEdgeDetector(**kwargs)
    return detector.detect_edges(image, thresh)
