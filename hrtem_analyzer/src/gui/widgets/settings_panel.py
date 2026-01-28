"""
Settings Panel Widget for HR-TEM Analyzer

Provides controls for analysis configuration.
"""
from pathlib import Path
from typing import Dict, Any, List

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QGroupBox, QFormLayout, QSpinBox, QDoubleSpinBox,
    QComboBox, QCheckBox, QLineEdit, QPushButton,
    QFileDialog, QScrollArea, QFrame
)
from PyQt6.QtCore import Qt, pyqtSignal


class SettingsPanel(QWidget):
    """
    Settings panel for analysis configuration.

    Provides controls for:
    - Measurement depths
    - Preprocessing methods
    - Edge detection methods
    - Consensus algorithm
    - Output settings
    - Parallel processing
    """

    settings_changed = pyqtSignal()

    def __init__(self):
        super().__init__()
        self._setup_ui()

    def _setup_ui(self):
        """Setup the widget UI"""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # Title
        title = QLabel("Settings")
        title.setObjectName("titleLabel")
        main_layout.addWidget(title)

        # Scroll area for settings
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setSpacing(12)

        # Measurement settings
        measurement_group = QGroupBox("Measurement")
        measurement_layout = QFormLayout(measurement_group)

        self.depths_edit = QLineEdit("5, 10, 15, 20, 25")
        self.depths_edit.setPlaceholderText("Comma-separated depths in nm")
        self.depths_edit.setToolTip("Measurement depths from baseline (nm)")
        measurement_layout.addRow("Depths (nm):", self.depths_edit)

        self.baseline_spin = QSpinBox()
        self.baseline_spin.setRange(0, 10000)
        self.baseline_spin.setValue(0)
        self.baseline_spin.setToolTip("Baseline Y position (0 = auto-detect)")
        measurement_layout.addRow("Baseline Y:", self.baseline_spin)

        self.lines_spin = QSpinBox()
        self.lines_spin.setRange(1, 20)
        self.lines_spin.setValue(5)
        self.lines_spin.setToolTip("Number of measurement lines per depth")
        measurement_layout.addRow("Lines/Depth:", self.lines_spin)

        scroll_layout.addWidget(measurement_group)

        # Preprocessing settings
        preprocess_group = QGroupBox("Preprocessing")
        preprocess_layout = QVBoxLayout(preprocess_group)

        self.preprocess_checks = {}
        preprocess_methods = [
            ('original', 'Original', True),
            ('clahe', 'CLAHE', True),
            ('bilateral_filter', 'Bilateral Filter', True),
            ('gaussian_blur', 'Gaussian Blur', False),
            ('median_filter', 'Median Filter', False),
        ]

        for key, label, default in preprocess_methods:
            cb = QCheckBox(label)
            cb.setChecked(default)
            cb.stateChanged.connect(self._on_settings_changed)
            self.preprocess_checks[key] = cb
            preprocess_layout.addWidget(cb)

        scroll_layout.addWidget(preprocess_group)

        # Edge detection settings
        edge_group = QGroupBox("Edge Detection")
        edge_layout = QVBoxLayout(edge_group)

        self.edge_checks = {}
        edge_methods = [
            ('sobel', 'Sobel', True),
            ('canny', 'Canny', True),
            ('gradient', 'Gradient', True),
            ('morphological', 'Morphological', True),
            ('scharr', 'Scharr', False),
            ('laplacian', 'Laplacian', False),
        ]

        for key, label, default in edge_methods:
            cb = QCheckBox(label)
            cb.setChecked(default)
            cb.stateChanged.connect(self._on_settings_changed)
            self.edge_checks[key] = cb
            edge_layout.addWidget(cb)

        scroll_layout.addWidget(edge_group)

        # Rotation angles
        rotation_group = QGroupBox("Multi-Angle Analysis")
        rotation_layout = QFormLayout(rotation_group)

        self.rotation_edit = QLineEdit("-2, -1, 0, 1, 2")
        self.rotation_edit.setPlaceholderText("Comma-separated angles in degrees")
        self.rotation_edit.setToolTip("Rotation angles for analysis")
        rotation_layout.addRow("Angles (°):", self.rotation_edit)

        scroll_layout.addWidget(rotation_group)

        # === Advanced Analysis (Gatan DM Style) ===
        advanced_group = QGroupBox("Advanced Analysis (Gatan DM Style)")
        advanced_layout = QVBoxLayout(advanced_group)

        # Line Profile Methods
        profile_label = QLabel("Line Profile Methods:")
        profile_label.setStyleSheet("font-weight: bold; margin-top: 5px;")
        advanced_layout.addWidget(profile_label)

        self.profile_checks = {}
        profile_methods = [
            ('fwhm', 'FWHM (Full Width Half Max)', True),
            ('10-90', '10-90% Threshold', True),
            ('derivative', 'Derivative Peak', True),
            ('sigmoid', 'Sigmoid Fitting', False),
        ]

        for key, label, default in profile_methods:
            cb = QCheckBox(label)
            cb.setChecked(default)
            cb.stateChanged.connect(self._on_settings_changed)
            self.profile_checks[key] = cb
            advanced_layout.addWidget(cb)

        # Background Subtraction
        bg_label = QLabel("Background Subtraction:")
        bg_label.setStyleSheet("font-weight: bold; margin-top: 10px;")
        advanced_layout.addWidget(bg_label)

        bg_form = QFormLayout()
        self.background_combo = QComboBox()
        self.background_combo.addItems([
            'None',
            'Rolling Ball',
            'Polynomial Fit',
            'Top-Hat',
            'Gaussian'
        ])
        self.background_combo.setCurrentIndex(1)  # Rolling Ball default
        self.background_combo.setToolTip("Background subtraction method")
        bg_form.addRow("Method:", self.background_combo)
        advanced_layout.addLayout(bg_form)

        # Calibration
        cal_label = QLabel("Scale Calibration:")
        cal_label.setStyleSheet("font-weight: bold; margin-top: 10px;")
        advanced_layout.addWidget(cal_label)

        self.fft_calibration_cb = QCheckBox("FFT-based scale calibration")
        self.fft_calibration_cb.setChecked(False)
        self.fft_calibration_cb.setToolTip("Calibrate scale using known lattice spacing")
        advanced_layout.addWidget(self.fft_calibration_cb)

        cal_form = QFormLayout()
        self.lattice_spacing_spin = QDoubleSpinBox()
        self.lattice_spacing_spin.setRange(0.0, 10.0)
        self.lattice_spacing_spin.setValue(0.0)
        self.lattice_spacing_spin.setSingleStep(0.01)
        self.lattice_spacing_spin.setDecimals(3)
        self.lattice_spacing_spin.setToolTip("Known lattice spacing for calibration (nm). 0 = disabled")
        cal_form.addRow("Lattice (nm):", self.lattice_spacing_spin)
        advanced_layout.addLayout(cal_form)

        # Drift Correction
        self.drift_correction_cb = QCheckBox("Drift correction")
        self.drift_correction_cb.setChecked(True)
        self.drift_correction_cb.setToolTip("Correct for sample drift between measurements")
        advanced_layout.addWidget(self.drift_correction_cb)

        # Sub-pixel Interpolation
        interp_form = QFormLayout()
        self.interpolation_spin = QSpinBox()
        self.interpolation_spin.setRange(1, 20)
        self.interpolation_spin.setValue(10)
        self.interpolation_spin.setToolTip("Sub-pixel interpolation factor (higher = more precise)")
        interp_form.addRow("Interpolation:", self.interpolation_spin)
        advanced_layout.addLayout(interp_form)

        # Statistical Analysis
        stats_label = QLabel("Statistical Analysis:")
        stats_label.setStyleSheet("font-weight: bold; margin-top: 10px;")
        advanced_layout.addWidget(stats_label)

        self.outlier_combo = QComboBox()
        self.outlier_combo.addItems([
            'IQR (Interquartile)',
            'MAD (Median Absolute)',
            'Z-Score',
            'Grubbs Test'
        ])
        self.outlier_combo.setToolTip("Outlier rejection method")
        stats_form = QFormLayout()
        stats_form.addRow("Outlier Reject:", self.outlier_combo)
        advanced_layout.addLayout(stats_form)

        self.bootstrap_ci_cb = QCheckBox("Bootstrap 95% CI")
        self.bootstrap_ci_cb.setChecked(True)
        self.bootstrap_ci_cb.setToolTip("Calculate 95% confidence interval using bootstrap")
        advanced_layout.addWidget(self.bootstrap_ci_cb)

        scroll_layout.addWidget(advanced_group)

        # === Precision Measurement ===
        precision_group = QGroupBox("Precision Measurement")
        precision_layout = QVBoxLayout(precision_group)

        self.precision_mode_cb = QCheckBox("Enable Precision Mode")
        self.precision_mode_cb.setChecked(True)
        self.precision_mode_cb.setToolTip("Enable sub-pixel, ESF/LSF, wavelet, Monte Carlo analysis")
        precision_layout.addWidget(self.precision_mode_cb)

        # Sub-pixel method
        subpixel_form = QFormLayout()
        self.subpixel_combo = QComboBox()
        self.subpixel_combo.addItems([
            'Gaussian',
            'Parabolic',
            'Centroid',
            'Spline'
        ])
        self.subpixel_combo.setToolTip("Sub-pixel edge localization method")
        subpixel_form.addRow("Sub-pixel:", self.subpixel_combo)
        precision_layout.addLayout(subpixel_form)

        # Denoising
        denoise_label = QLabel("Advanced Denoising:")
        denoise_label.setStyleSheet("font-weight: bold; margin-top: 5px;")
        precision_layout.addWidget(denoise_label)

        denoise_form = QFormLayout()
        self.denoise_combo = QComboBox()
        self.denoise_combo.addItems([
            'Non-local Means',
            'Bilateral',
            'Wavelet',
            'Anisotropic Diffusion'
        ])
        self.denoise_combo.setToolTip("Advanced denoising method for noise reduction")
        denoise_form.addRow("Method:", self.denoise_combo)

        self.denoise_strength_spin = QDoubleSpinBox()
        self.denoise_strength_spin.setRange(0.0, 2.0)
        self.denoise_strength_spin.setValue(1.0)
        self.denoise_strength_spin.setSingleStep(0.1)
        self.denoise_strength_spin.setToolTip("Denoising strength (0=off, 1=normal, 2=strong)")
        denoise_form.addRow("Strength:", self.denoise_strength_spin)
        precision_layout.addLayout(denoise_form)

        # Monte Carlo
        mc_label = QLabel("Monte Carlo Uncertainty:")
        mc_label.setStyleSheet("font-weight: bold; margin-top: 5px;")
        precision_layout.addWidget(mc_label)

        mc_form = QFormLayout()
        self.mc_simulations_spin = QSpinBox()
        self.mc_simulations_spin.setRange(100, 5000)
        self.mc_simulations_spin.setValue(500)
        self.mc_simulations_spin.setSingleStep(100)
        self.mc_simulations_spin.setToolTip("Number of Monte Carlo simulations for uncertainty estimation")
        mc_form.addRow("Simulations:", self.mc_simulations_spin)
        precision_layout.addLayout(mc_form)

        # Atomic column fitting (for crystalline materials)
        self.atomic_fitting_cb = QCheckBox("Atomic column fitting")
        self.atomic_fitting_cb.setChecked(False)
        self.atomic_fitting_cb.setToolTip("Fit Gaussian to atomic columns (for crystalline materials)")
        precision_layout.addWidget(self.atomic_fitting_cb)

        scroll_layout.addWidget(precision_group)

        # Consensus settings
        consensus_group = QGroupBox("Consensus")
        consensus_layout = QFormLayout(consensus_group)

        self.consensus_combo = QComboBox()
        self.consensus_combo.addItems([
            'Trimmed Mean',
            'Median',
            'Mean',
            'Weighted Mean'
        ])
        self.consensus_combo.setToolTip("Method for combining measurements")
        consensus_layout.addRow("Method:", self.consensus_combo)

        self.trim_spin = QDoubleSpinBox()
        self.trim_spin.setRange(0.0, 0.4)
        self.trim_spin.setValue(0.1)
        self.trim_spin.setSingleStep(0.05)
        self.trim_spin.setToolTip("Trim percentage for trimmed mean")
        consensus_layout.addRow("Trim %:", self.trim_spin)

        scroll_layout.addWidget(consensus_group)

        # Processing settings
        processing_group = QGroupBox("Processing")
        processing_layout = QFormLayout(processing_group)

        self.workers_spin = QSpinBox()
        self.workers_spin.setRange(1, 16)
        self.workers_spin.setValue(4)
        self.workers_spin.setToolTip("Number of parallel workers")
        processing_layout.addRow("Workers:", self.workers_spin)

        self.precision_combo = QComboBox()
        self.precision_combo.addItems(['Standard', 'High Precision'])
        self.precision_combo.setToolTip("Analysis precision mode")
        processing_layout.addRow("Precision:", self.precision_combo)

        scroll_layout.addWidget(processing_group)

        # Output settings
        output_group = QGroupBox("Output")
        output_layout = QFormLayout(output_group)

        output_dir_layout = QHBoxLayout()
        self.output_edit = QLineEdit()
        self.output_edit.setPlaceholderText("Select output directory...")
        output_dir_layout.addWidget(self.output_edit)

        self.browse_btn = QPushButton("...")
        self.browse_btn.setFixedWidth(30)
        self.browse_btn.clicked.connect(self._on_browse_output)
        output_dir_layout.addWidget(self.browse_btn)

        output_layout.addRow("Directory:", output_dir_layout)

        self.save_json_cb = QCheckBox("Save JSON data")
        self.save_json_cb.setChecked(True)
        output_layout.addRow("", self.save_json_cb)

        self.save_csv_cb = QCheckBox("Save CSV summary")
        self.save_csv_cb.setChecked(True)
        output_layout.addRow("", self.save_csv_cb)

        scroll_layout.addWidget(output_group)

        scroll_layout.addStretch()
        scroll.setWidget(scroll_content)
        main_layout.addWidget(scroll)

    def get_settings(self) -> Dict[str, Any]:
        """Get current settings as dictionary"""
        # Parse depths
        depths_text = self.depths_edit.text()
        try:
            depths = [float(d.strip()) for d in depths_text.split(',') if d.strip()]
        except ValueError:
            depths = [5, 10, 15, 20]

        # Parse rotation angles
        rotation_text = self.rotation_edit.text()
        try:
            rotations = [float(r.strip()) for r in rotation_text.split(',') if r.strip()]
        except ValueError:
            rotations = [-1, 0, 1]

        # Get enabled preprocessing methods
        preprocess = [k for k, cb in self.preprocess_checks.items() if cb.isChecked()]

        # Get enabled edge methods
        edge_methods = [k for k, cb in self.edge_checks.items() if cb.isChecked()]

        # Map consensus method
        consensus_map = {
            'Trimmed Mean': 'trimmed_mean',
            'Median': 'median',
            'Mean': 'mean',
            'Weighted Mean': 'weighted_mean',
        }
        consensus = consensus_map.get(
            self.consensus_combo.currentText(),
            'trimmed_mean'
        )

        # Get enabled profile methods
        profile_methods = [k for k, cb in self.profile_checks.items() if cb.isChecked()]

        # Map background method
        background_map = {
            'None': 'none',
            'Rolling Ball': 'rolling_ball',
            'Polynomial Fit': 'polynomial',
            'Top-Hat': 'tophat',
            'Gaussian': 'gaussian',
        }
        background_method = background_map.get(
            self.background_combo.currentText(),
            'rolling_ball'
        )

        # Map outlier method
        outlier_map = {
            'IQR (Interquartile)': 'iqr',
            'MAD (Median Absolute)': 'mad',
            'Z-Score': 'zscore',
            'Grubbs Test': 'grubbs',
        }
        outlier_method = outlier_map.get(
            self.outlier_combo.currentText(),
            'iqr'
        )

        # Map subpixel method
        subpixel_map = {
            'Gaussian': 'gaussian',
            'Parabolic': 'parabolic',
            'Centroid': 'centroid',
            'Spline': 'spline',
        }
        subpixel_method = subpixel_map.get(
            self.subpixel_combo.currentText(),
            'gaussian'
        )

        # Map denoising method
        denoise_map = {
            'Non-local Means': 'nlm',
            'Bilateral': 'bilateral',
            'Wavelet': 'wavelet',
            'Anisotropic Diffusion': 'anisotropic',
        }
        denoise_method = denoise_map.get(
            self.denoise_combo.currentText(),
            'nlm'
        )

        return {
            'depths_nm': depths,
            'baseline_y': self.baseline_spin.value() if self.baseline_spin.value() > 0 else None,
            'num_lines_per_depth': self.lines_spin.value(),
            'preprocessing_methods': preprocess,
            'edge_methods': edge_methods,
            'rotation_angles': rotations,
            'consensus_method': consensus,
            'trim_percentage': self.trim_spin.value(),
            'max_workers': self.workers_spin.value(),
            'high_precision': self.precision_combo.currentText() == 'High Precision',
            'output_dir': self.output_edit.text() or None,
            'save_json': self.save_json_cb.isChecked(),
            'save_csv': self.save_csv_cb.isChecked(),
            # Advanced settings (Gatan DM style)
            'profile_methods': profile_methods,
            'background_method': background_method,
            'fft_calibration': self.fft_calibration_cb.isChecked(),
            'lattice_spacing_nm': self.lattice_spacing_spin.value() if self.lattice_spacing_spin.value() > 0 else None,
            'drift_correction': self.drift_correction_cb.isChecked(),
            'interpolation_factor': self.interpolation_spin.value(),
            'outlier_method': outlier_method,
            'bootstrap_ci': self.bootstrap_ci_cb.isChecked(),
            # Precision measurement settings
            'precision_mode': self.precision_mode_cb.isChecked(),
            'subpixel_method': subpixel_method,
            'denoise_method': denoise_method,
            'denoise_strength': self.denoise_strength_spin.value(),
            'mc_simulations': self.mc_simulations_spin.value(),
            'atomic_fitting': self.atomic_fitting_cb.isChecked(),
        }

    def set_settings(self, settings: Dict[str, Any]):
        """Set settings from dictionary"""
        if 'depths_nm' in settings:
            self.depths_edit.setText(', '.join(str(d) for d in settings['depths_nm']))

        if 'baseline_y' in settings and settings['baseline_y']:
            self.baseline_spin.setValue(settings['baseline_y'])

        if 'rotation_angles' in settings:
            self.rotation_edit.setText(', '.join(str(r) for r in settings['rotation_angles']))

        if 'max_workers' in settings:
            self.workers_spin.setValue(settings['max_workers'])

        if 'output_dir' in settings and settings['output_dir']:
            self.output_edit.setText(settings['output_dir'])

    def set_baseline_y(self, y: int):
        """Set baseline Y value"""
        self.baseline_spin.setValue(y)

    def set_output_dir(self, path: str):
        """Set output directory"""
        self.output_edit.setText(path)

    def _on_browse_output(self):
        """Browse for output directory"""
        directory = QFileDialog.getExistingDirectory(
            self,
            "Select Output Directory"
        )
        if directory:
            self.output_edit.setText(directory)
            self._on_settings_changed()

    def _on_settings_changed(self):
        """Handle settings change"""
        self.settings_changed.emit()
