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
