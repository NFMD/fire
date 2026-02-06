"""
Advanced Edge Fitting Methods for Sub-pixel Precision CD Measurement

Implements state-of-the-art edge localization:
- Edge Spread Function (ESF) / Line Spread Function (LSF) fitting
- Gaussian Process Regression for edge profiles
- Fermi-Dirac edge model
- Multi-scale wavelet edge detection
- MCMC-based uncertainty quantification
"""
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional, List, Dict, Any
from scipy import optimize, ndimage, signal, interpolate
from scipy.special import erf
from loguru import logger

try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, ConstantKernel
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import pywt
    PYWT_AVAILABLE = True
except ImportError:
    PYWT_AVAILABLE = False


@dataclass
class EdgeFitResult:
    """Result of edge fitting"""
    position: float  # Sub-pixel edge position
    uncertainty: float  # Standard error of position
    width: float  # Edge width (transition zone)
    amplitude: float  # Edge contrast
    fit_quality: float  # R-squared or similar metric
    method: str  # Method used
    profile: np.ndarray  # Original profile
    fit_curve: np.ndarray  # Fitted curve
    residuals: np.ndarray  # Fit residuals
    extra_info: Dict[str, Any] = None  # Additional method-specific info


class ESFLSFFitter:
    """
    Edge Spread Function (ESF) and Line Spread Function (LSF) fitting.

    The ESF is the integral of the LSF (point spread function).
    For high-precision edge localization, we fit the ESF with various
    edge models and compute the edge position from the fit parameters.
    """

    def __init__(
        self,
        interpolation_factor: int = 20,
        fit_window: int = 50
    ):
        self.interpolation_factor = interpolation_factor
        self.fit_window = fit_window

    def fit_esf_erf(
        self,
        profile: np.ndarray,
        initial_edge: Optional[int] = None
    ) -> EdgeFitResult:
        """
        Fit ESF with error function (Gaussian edge model).

        ESF(x) = A * erf((x - x0) / (sqrt(2) * sigma)) + B

        Args:
            profile: 1D intensity profile
            initial_edge: Initial estimate of edge position

        Returns:
            EdgeFitResult with fitted parameters
        """
        x = np.arange(len(profile))

        # Interpolate for sub-pixel precision
        x_fine = np.linspace(0, len(profile) - 1, len(profile) * self.interpolation_factor)
        interp_func = interpolate.interp1d(x, profile, kind='cubic')
        profile_fine = interp_func(x_fine)

        # Initial estimates
        if initial_edge is None:
            # Find edge via gradient
            gradient = np.gradient(profile_fine)
            initial_edge = x_fine[np.argmax(np.abs(gradient))]

        A_init = (profile.max() - profile.min()) / 2
        B_init = (profile.max() + profile.min()) / 2
        sigma_init = 3.0

        def erf_model(x, x0, sigma, A, B):
            return A * erf((x - x0) / (np.sqrt(2) * sigma)) + B

        try:
            popt, pcov = optimize.curve_fit(
                erf_model,
                x_fine,
                profile_fine,
                p0=[initial_edge, sigma_init, A_init, B_init],
                bounds=(
                    [0, 0.1, -np.inf, -np.inf],
                    [len(profile), 50, np.inf, np.inf]
                ),
                maxfev=5000
            )

            x0, sigma, A, B = popt
            perr = np.sqrt(np.diag(pcov))

            # Fit quality
            fitted = erf_model(x_fine, *popt)
            ss_res = np.sum((profile_fine - fitted) ** 2)
            ss_tot = np.sum((profile_fine - np.mean(profile_fine)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

            # Map back to original pixel coordinates
            position = x0 / self.interpolation_factor
            uncertainty = perr[0] / self.interpolation_factor

            return EdgeFitResult(
                position=position,
                uncertainty=uncertainty,
                width=sigma * 2.355,  # FWHM
                amplitude=2 * A,
                fit_quality=r_squared,
                method='esf_erf',
                profile=profile,
                fit_curve=erf_model(x * self.interpolation_factor, *popt),
                residuals=profile - erf_model(x * self.interpolation_factor, *popt),
                extra_info={'sigma': sigma, 'A': A, 'B': B}
            )

        except (RuntimeError, ValueError) as e:
            logger.warning(f"ESF fitting failed: {e}")
            return self._fallback_gradient(profile)

    def fit_esf_fermi(
        self,
        profile: np.ndarray,
        initial_edge: Optional[int] = None
    ) -> EdgeFitResult:
        """
        Fit ESF with Fermi-Dirac function (asymmetric edge model).

        ESF(x) = A / (1 + exp((x - x0) / T)) + B

        This model allows for asymmetric edges common in TEM images.
        """
        x = np.arange(len(profile))

        # Interpolate
        x_fine = np.linspace(0, len(profile) - 1, len(profile) * self.interpolation_factor)
        interp_func = interpolate.interp1d(x, profile, kind='cubic')
        profile_fine = interp_func(x_fine)

        if initial_edge is None:
            gradient = np.gradient(profile_fine)
            initial_edge = x_fine[np.argmax(np.abs(gradient))]

        A_init = profile.max() - profile.min()
        B_init = profile.min()
        T_init = 2.0

        def fermi_model(x, x0, T, A, B):
            return A / (1 + np.exp((x - x0) / (T + 1e-10))) + B

        try:
            popt, pcov = optimize.curve_fit(
                fermi_model,
                x_fine,
                profile_fine,
                p0=[initial_edge, T_init, A_init, B_init],
                bounds=(
                    [0, 0.1, 0, -np.inf],
                    [len(profile) * self.interpolation_factor, 50, np.inf, np.inf]
                ),
                maxfev=5000
            )

            x0, T, A, B = popt
            perr = np.sqrt(np.diag(pcov))

            fitted = fermi_model(x_fine, *popt)
            ss_res = np.sum((profile_fine - fitted) ** 2)
            ss_tot = np.sum((profile_fine - np.mean(profile_fine)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

            position = x0 / self.interpolation_factor
            uncertainty = perr[0] / self.interpolation_factor

            return EdgeFitResult(
                position=position,
                uncertainty=uncertainty,
                width=4 * T / self.interpolation_factor,  # 10-90% width
                amplitude=A,
                fit_quality=r_squared,
                method='esf_fermi',
                profile=profile,
                fit_curve=fermi_model(x * self.interpolation_factor, *popt),
                residuals=profile - fermi_model(x * self.interpolation_factor, *popt),
                extra_info={'T': T, 'A': A, 'B': B}
            )

        except (RuntimeError, ValueError) as e:
            logger.warning(f"Fermi fitting failed: {e}")
            return self._fallback_gradient(profile)

    def fit_lsf_gaussian(
        self,
        profile: np.ndarray,
        initial_edge: Optional[int] = None
    ) -> EdgeFitResult:
        """
        Fit LSF (derivative of ESF) with Gaussian.

        The edge position is the peak of the LSF.
        """
        # Compute LSF (derivative)
        lsf = np.gradient(profile)

        x = np.arange(len(lsf))
        x_fine = np.linspace(0, len(lsf) - 1, len(lsf) * self.interpolation_factor)
        interp_func = interpolate.interp1d(x, lsf, kind='cubic')
        lsf_fine = interp_func(x_fine)

        if initial_edge is None:
            initial_edge = x_fine[np.argmax(np.abs(lsf_fine))]

        # Determine if positive or negative edge
        if lsf_fine[int(initial_edge)] < 0:
            lsf_fine = -lsf_fine
            lsf = -lsf

        A_init = np.max(lsf_fine)
        sigma_init = 3.0

        def gaussian(x, x0, sigma, A, offset):
            return A * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2)) + offset

        try:
            popt, pcov = optimize.curve_fit(
                gaussian,
                x_fine,
                lsf_fine,
                p0=[initial_edge, sigma_init, A_init, 0],
                bounds=(
                    [0, 0.1, 0, -np.inf],
                    [len(lsf) * self.interpolation_factor, 50, np.inf, np.inf]
                ),
                maxfev=5000
            )

            x0, sigma, A, offset = popt
            perr = np.sqrt(np.diag(pcov))

            fitted = gaussian(x_fine, *popt)
            ss_res = np.sum((lsf_fine - fitted) ** 2)
            ss_tot = np.sum((lsf_fine - np.mean(lsf_fine)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

            position = x0 / self.interpolation_factor
            uncertainty = perr[0] / self.interpolation_factor

            return EdgeFitResult(
                position=position,
                uncertainty=uncertainty,
                width=sigma * 2.355 / self.interpolation_factor,
                amplitude=A,
                fit_quality=r_squared,
                method='lsf_gaussian',
                profile=profile,
                fit_curve=np.cumsum(gaussian(x * self.interpolation_factor, *popt)),
                residuals=lsf - gaussian(x * self.interpolation_factor, *popt),
                extra_info={'sigma': sigma, 'A': A}
            )

        except (RuntimeError, ValueError) as e:
            logger.warning(f"LSF fitting failed: {e}")
            return self._fallback_gradient(profile)

    def _fallback_gradient(self, profile: np.ndarray) -> EdgeFitResult:
        """Fallback method using simple gradient peak"""
        gradient = np.gradient(profile)
        peak_idx = np.argmax(np.abs(gradient))

        return EdgeFitResult(
            position=float(peak_idx),
            uncertainty=0.5,
            width=5.0,
            amplitude=float(np.abs(gradient[peak_idx])),
            fit_quality=0.0,
            method='gradient_fallback',
            profile=profile,
            fit_curve=profile.copy(),
            residuals=np.zeros_like(profile)
        )


class GaussianProcessEdgeFitter:
    """
    Gaussian Process Regression for edge fitting.

    GP provides:
    - Non-parametric edge profile modeling
    - Natural uncertainty quantification
    - Robust to noise
    """

    def __init__(
        self,
        kernel_type: str = 'matern',
        interpolation_factor: int = 20,
        n_restarts: int = 5
    ):
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn required for GP edge fitting")

        self.interpolation_factor = interpolation_factor
        self.n_restarts = n_restarts

        # Define kernel
        if kernel_type == 'rbf':
            self.kernel = ConstantKernel(1.0) * RBF(length_scale=5.0) + WhiteKernel(noise_level=0.1)
        elif kernel_type == 'matern':
            self.kernel = ConstantKernel(1.0) * Matern(length_scale=5.0, nu=2.5) + WhiteKernel(noise_level=0.1)
        else:
            raise ValueError(f"Unknown kernel type: {kernel_type}")

    def fit(
        self,
        profile: np.ndarray,
        initial_edge: Optional[int] = None
    ) -> EdgeFitResult:
        """
        Fit edge profile using Gaussian Process.

        Args:
            profile: 1D intensity profile
            initial_edge: Initial estimate (not used but kept for API consistency)

        Returns:
            EdgeFitResult with GP predictions and uncertainty
        """
        x = np.arange(len(profile)).reshape(-1, 1)
        y = profile

        # Normalize
        y_mean = y.mean()
        y_std = y.std()
        y_norm = (y - y_mean) / (y_std + 1e-10)

        # Fit GP
        gp = GaussianProcessRegressor(
            kernel=self.kernel,
            n_restarts_optimizer=self.n_restarts,
            normalize_y=False
        )
        gp.fit(x, y_norm)

        # Predict on fine grid
        x_fine = np.linspace(0, len(profile) - 1, len(profile) * self.interpolation_factor).reshape(-1, 1)
        y_pred, y_std_pred = gp.predict(x_fine, return_std=True)

        # Denormalize
        y_pred = y_pred * y_std + y_mean
        y_std_pred = y_std_pred * y_std

        # Find edge as maximum gradient
        gradient = np.gradient(y_pred)
        edge_idx = np.argmax(np.abs(gradient))
        edge_position = x_fine[edge_idx, 0]

        # Uncertainty from GP
        edge_uncertainty = y_std_pred[edge_idx]

        # Compute edge width (distance between 10% and 90% of transition)
        y_min, y_max = y_pred.min(), y_pred.max()
        y_10 = y_min + 0.1 * (y_max - y_min)
        y_90 = y_min + 0.9 * (y_max - y_min)

        # Find positions
        if y_pred[0] < y_pred[-1]:  # Rising edge
            idx_10 = np.argmin(np.abs(y_pred - y_10))
            idx_90 = np.argmin(np.abs(y_pred - y_90))
        else:  # Falling edge
            idx_10 = np.argmin(np.abs(y_pred - y_90))
            idx_90 = np.argmin(np.abs(y_pred - y_10))

        edge_width = abs(x_fine[idx_90, 0] - x_fine[idx_10, 0]) / self.interpolation_factor

        # Fit quality (log marginal likelihood normalized)
        log_likelihood = gp.log_marginal_likelihood()

        return EdgeFitResult(
            position=edge_position / self.interpolation_factor,
            uncertainty=edge_uncertainty / y_std / self.interpolation_factor,
            width=edge_width,
            amplitude=float(y_max - y_min),
            fit_quality=np.exp(log_likelihood / len(profile)),  # Normalized
            method='gaussian_process',
            profile=profile,
            fit_curve=np.interp(np.arange(len(profile)), x_fine.flatten() / self.interpolation_factor, y_pred),
            residuals=profile - np.interp(np.arange(len(profile)), x_fine.flatten() / self.interpolation_factor, y_pred),
            extra_info={
                'gp_kernel': str(gp.kernel_),
                'log_likelihood': log_likelihood,
                'prediction_std': y_std_pred
            }
        )


class WaveletEdgeFitter:
    """
    Multi-scale wavelet edge detection for robust edge localization.

    Uses wavelet transform to detect edges at multiple scales
    and combines them for robust sub-pixel edge localization.
    """

    def __init__(
        self,
        wavelet: str = 'gaus1',
        scales: List[int] = None,
        interpolation_factor: int = 20
    ):
        if not PYWT_AVAILABLE:
            raise ImportError("PyWavelets required for wavelet edge fitting")

        self.wavelet = wavelet
        self.scales = scales or [1, 2, 4, 8, 16]
        self.interpolation_factor = interpolation_factor

    def fit(
        self,
        profile: np.ndarray,
        initial_edge: Optional[int] = None
    ) -> EdgeFitResult:
        """
        Fit edge using multi-scale wavelet analysis.

        Args:
            profile: 1D intensity profile
            initial_edge: Initial estimate (used for search region)

        Returns:
            EdgeFitResult with wavelet-based edge position
        """
        # Interpolate for sub-pixel precision
        x = np.arange(len(profile))
        x_fine = np.linspace(0, len(profile) - 1, len(profile) * self.interpolation_factor)
        interp_func = interpolate.interp1d(x, profile, kind='cubic')
        profile_fine = interp_func(x_fine)

        # Continuous wavelet transform
        scales_fine = [s * self.interpolation_factor for s in self.scales]
        coef, freqs = pywt.cwt(profile_fine, scales_fine, self.wavelet)

        # Find edge positions at each scale
        edge_positions = []
        edge_strengths = []

        for i, scale in enumerate(self.scales):
            # Wavelet coefficients at this scale
            cwt_coef = coef[i]

            # Find local maxima of absolute coefficients
            abs_coef = np.abs(cwt_coef)

            # Find the strongest edge
            max_idx = np.argmax(abs_coef)
            edge_positions.append(max_idx / self.interpolation_factor)
            edge_strengths.append(abs_coef[max_idx])

        # Weighted average of edge positions (weight by strength)
        weights = np.array(edge_strengths)
        weights = weights / weights.sum()
        final_position = np.sum(np.array(edge_positions) * weights)

        # Uncertainty from spread across scales
        position_std = np.sqrt(np.sum(weights * (np.array(edge_positions) - final_position) ** 2))

        # Compute edge width from finest scale
        finest_coef = np.abs(coef[0])
        peak_idx = np.argmax(finest_coef)

        # FWHM of wavelet response
        half_max = finest_coef[peak_idx] / 2
        above_half = finest_coef > half_max
        if above_half.any():
            indices = np.where(above_half)[0]
            width = (indices[-1] - indices[0]) / self.interpolation_factor
        else:
            width = self.scales[0]

        return EdgeFitResult(
            position=final_position,
            uncertainty=max(position_std, 0.1),
            width=width,
            amplitude=float(np.max(edge_strengths)),
            fit_quality=float(np.mean(edge_strengths) / (np.std(edge_strengths) + 1e-10)),
            method='wavelet',
            profile=profile,
            fit_curve=profile.copy(),  # Wavelet doesn't produce a fit curve
            residuals=np.zeros_like(profile),
            extra_info={
                'scales': self.scales,
                'edge_positions': edge_positions,
                'edge_strengths': edge_strengths,
                'wavelet': self.wavelet
            }
        )


class MCMCEdgeFitter:
    """
    MCMC-based edge fitting for robust uncertainty quantification.

    Uses Markov Chain Monte Carlo to sample from the posterior
    distribution of edge parameters.
    """

    def __init__(
        self,
        n_samples: int = 2000,
        n_burn: int = 500,
        n_chains: int = 4
    ):
        self.n_samples = n_samples
        self.n_burn = n_burn
        self.n_chains = n_chains

    def fit(
        self,
        profile: np.ndarray,
        initial_edge: Optional[int] = None
    ) -> EdgeFitResult:
        """
        Fit edge using MCMC sampling.

        Uses Metropolis-Hastings algorithm with error function model.
        """
        x = np.arange(len(profile))

        # Initial parameter estimates
        if initial_edge is None:
            gradient = np.gradient(profile)
            initial_edge = np.argmax(np.abs(gradient))

        A_init = (profile.max() - profile.min()) / 2
        B_init = (profile.max() + profile.min()) / 2
        sigma_init = 3.0

        def log_likelihood(params):
            x0, sigma, A, B = params
            if sigma <= 0:
                return -np.inf
            model = A * erf((x - x0) / (np.sqrt(2) * sigma)) + B
            residuals = profile - model
            return -0.5 * np.sum(residuals ** 2)

        def log_prior(params):
            x0, sigma, A, B = params
            if x0 < 0 or x0 > len(profile):
                return -np.inf
            if sigma <= 0 or sigma > 50:
                return -np.inf
            return 0.0  # Flat prior

        def log_posterior(params):
            lp = log_prior(params)
            if not np.isfinite(lp):
                return -np.inf
            return lp + log_likelihood(params)

        # Run MCMC
        all_samples = []
        for chain in range(self.n_chains):
            # Initialize with small random perturbation
            current = np.array([
                initial_edge + np.random.randn() * 2,
                sigma_init + np.random.rand() * 2,
                A_init + np.random.randn() * 10,
                B_init + np.random.randn() * 10
            ])
            current_log_prob = log_posterior(current)

            # Proposal scales
            proposal_scale = np.array([1.0, 0.5, 5.0, 5.0])

            samples = []
            for i in range(self.n_samples + self.n_burn):
                # Propose
                proposal = current + np.random.randn(4) * proposal_scale

                proposal_log_prob = log_posterior(proposal)

                # Accept/reject
                if np.log(np.random.rand()) < proposal_log_prob - current_log_prob:
                    current = proposal
                    current_log_prob = proposal_log_prob

                if i >= self.n_burn:
                    samples.append(current.copy())

            all_samples.extend(samples)

        samples = np.array(all_samples)

        # Extract statistics
        x0_samples = samples[:, 0]
        sigma_samples = samples[:, 1]
        A_samples = samples[:, 2]
        B_samples = samples[:, 3]

        position = np.median(x0_samples)
        uncertainty = np.std(x0_samples)
        width = np.median(sigma_samples) * 2.355  # FWHM

        # Best fit curve (using median parameters)
        best_params = np.median(samples, axis=0)
        fit_curve = best_params[2] * erf((x - best_params[0]) / (np.sqrt(2) * best_params[1])) + best_params[3]

        # Fit quality (Gelman-Rubin diagnostic approximation)
        # Lower is better, < 1.1 is good
        chain_means = []
        chain_size = len(all_samples) // self.n_chains
        for i in range(self.n_chains):
            chain_means.append(np.mean(all_samples[i * chain_size:(i + 1) * chain_size], axis=0))
        between_chain_var = np.var(chain_means, axis=0)
        within_chain_var = np.mean([np.var(all_samples[i * chain_size:(i + 1) * chain_size], axis=0)
                                    for i in range(self.n_chains)], axis=0)
        r_hat = np.sqrt((within_chain_var + between_chain_var) / (within_chain_var + 1e-10))
        fit_quality = 1.0 / (np.mean(r_hat) + 1e-10)

        return EdgeFitResult(
            position=position,
            uncertainty=uncertainty,
            width=width,
            amplitude=float(2 * np.median(A_samples)),
            fit_quality=min(fit_quality, 1.0),
            method='mcmc',
            profile=profile,
            fit_curve=fit_curve,
            residuals=profile - fit_curve,
            extra_info={
                'n_samples': len(samples),
                'n_chains': self.n_chains,
                'samples': samples,
                'r_hat': r_hat
            }
        )


class EnsembleEdgeFitter:
    """
    Ensemble of edge fitting methods for robust edge localization.

    Combines multiple methods and uses weighted voting for final position.
    """

    def __init__(
        self,
        methods: List[str] = None,
        interpolation_factor: int = 20
    ):
        if methods is None:
            methods = ['esf_erf', 'esf_fermi', 'lsf_gaussian']
            if SKLEARN_AVAILABLE:
                methods.append('gaussian_process')
            if PYWT_AVAILABLE:
                methods.append('wavelet')

        self.methods = methods
        self.interpolation_factor = interpolation_factor

        # Initialize fitters
        self.esf_fitter = ESFLSFFitter(interpolation_factor=interpolation_factor)
        self.gp_fitter = GaussianProcessEdgeFitter(interpolation_factor=interpolation_factor) if SKLEARN_AVAILABLE else None
        self.wavelet_fitter = WaveletEdgeFitter(interpolation_factor=interpolation_factor) if PYWT_AVAILABLE else None

    def fit(
        self,
        profile: np.ndarray,
        initial_edge: Optional[int] = None
    ) -> EdgeFitResult:
        """
        Fit edge using ensemble of methods.

        Returns the weighted average position with combined uncertainty.
        """
        results = []
        weights = []

        for method in self.methods:
            try:
                if method == 'esf_erf':
                    result = self.esf_fitter.fit_esf_erf(profile, initial_edge)
                elif method == 'esf_fermi':
                    result = self.esf_fitter.fit_esf_fermi(profile, initial_edge)
                elif method == 'lsf_gaussian':
                    result = self.esf_fitter.fit_lsf_gaussian(profile, initial_edge)
                elif method == 'gaussian_process' and self.gp_fitter:
                    result = self.gp_fitter.fit(profile, initial_edge)
                elif method == 'wavelet' and self.wavelet_fitter:
                    result = self.wavelet_fitter.fit(profile, initial_edge)
                else:
                    continue

                results.append(result)
                # Weight by fit quality and inverse uncertainty
                weight = result.fit_quality / (result.uncertainty + 0.01)
                weights.append(weight)

            except Exception as e:
                logger.warning(f"Method {method} failed: {e}")
                continue

        if not results:
            return self.esf_fitter._fallback_gradient(profile)

        # Normalize weights
        weights = np.array(weights)
        weights = weights / weights.sum()

        # Weighted average position
        positions = np.array([r.position for r in results])
        final_position = np.sum(positions * weights)

        # Combined uncertainty (includes both individual uncertainties and method disagreement)
        uncertainties = np.array([r.uncertainty for r in results])
        method_disagreement = np.sqrt(np.sum(weights * (positions - final_position) ** 2))
        combined_uncertainty = np.sqrt(
            np.sum(weights * uncertainties ** 2) + method_disagreement ** 2
        )

        # Weighted average of other parameters
        widths = np.array([r.width for r in results])
        final_width = np.sum(widths * weights)

        amplitudes = np.array([r.amplitude for r in results])
        final_amplitude = np.sum(amplitudes * weights)

        # Best individual fit curve (highest quality)
        best_idx = np.argmax([r.fit_quality for r in results])
        best_result = results[best_idx]

        return EdgeFitResult(
            position=final_position,
            uncertainty=combined_uncertainty,
            width=final_width,
            amplitude=final_amplitude,
            fit_quality=float(np.max([r.fit_quality for r in results])),
            method='ensemble',
            profile=profile,
            fit_curve=best_result.fit_curve,
            residuals=best_result.residuals,
            extra_info={
                'methods_used': [r.method for r in results],
                'individual_positions': positions.tolist(),
                'individual_uncertainties': uncertainties.tolist(),
                'weights': weights.tolist(),
                'method_disagreement': method_disagreement
            }
        )


def fit_edge_profile(
    profile: np.ndarray,
    method: str = 'ensemble',
    initial_edge: Optional[int] = None,
    **kwargs
) -> EdgeFitResult:
    """
    High-level function to fit edge profile.

    Args:
        profile: 1D intensity profile
        method: Fitting method ('esf_erf', 'esf_fermi', 'lsf_gaussian',
                'gaussian_process', 'wavelet', 'mcmc', 'ensemble')
        initial_edge: Initial estimate of edge position
        **kwargs: Additional arguments for specific methods

    Returns:
        EdgeFitResult with fitted parameters
    """
    interpolation_factor = kwargs.get('interpolation_factor', 20)

    if method == 'ensemble':
        fitter = EnsembleEdgeFitter(interpolation_factor=interpolation_factor)
        return fitter.fit(profile, initial_edge)

    elif method in ('esf_erf', 'esf_fermi', 'lsf_gaussian'):
        fitter = ESFLSFFitter(interpolation_factor=interpolation_factor)
        if method == 'esf_erf':
            return fitter.fit_esf_erf(profile, initial_edge)
        elif method == 'esf_fermi':
            return fitter.fit_esf_fermi(profile, initial_edge)
        else:
            return fitter.fit_lsf_gaussian(profile, initial_edge)

    elif method == 'gaussian_process':
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn required for GP fitting")
        fitter = GaussianProcessEdgeFitter(interpolation_factor=interpolation_factor)
        return fitter.fit(profile, initial_edge)

    elif method == 'wavelet':
        if not PYWT_AVAILABLE:
            raise ImportError("PyWavelets required for wavelet fitting")
        fitter = WaveletEdgeFitter(interpolation_factor=interpolation_factor)
        return fitter.fit(profile, initial_edge)

    elif method == 'mcmc':
        n_samples = kwargs.get('n_samples', 2000)
        fitter = MCMCEdgeFitter(n_samples=n_samples)
        return fitter.fit(profile, initial_edge)

    else:
        raise ValueError(f"Unknown method: {method}")
