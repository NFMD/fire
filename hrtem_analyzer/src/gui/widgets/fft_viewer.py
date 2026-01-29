"""
FFT Visualization Widget for HR-TEM Analyzer

Displays FFT analysis results including power spectrum and detected periodicities.
"""
from typing import Optional, Dict, Any, List, Tuple

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QFrame, QGroupBox, QFormLayout, QSplitter,
    QPushButton, QSlider, QComboBox
)
from PyQt6.QtCore import Qt, pyqtSignal, QRectF
from PyQt6.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QFont

try:
    import numpy as np
except ImportError:
    np = None


class FFTDisplayWidget(QLabel):
    """Widget for displaying FFT power spectrum with interactive features"""

    peak_selected = pyqtSignal(float, float)  # (frequency, spacing_nm)

    def __init__(self):
        super().__init__()
        self._fft_data: Optional[np.ndarray] = None
        self._peaks: List[Tuple[float, float, float]] = []  # (x, y, intensity)
        self._scale_nm_per_pixel: float = 1.0
        self._show_peaks = True
        self._log_scale = True

        self.setMinimumSize(256, 256)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("""
            background-color: #1e1e1e;
            border: 1px solid #454545;
            border-radius: 4px;
        """)
        self.setText("No FFT data")

    def set_fft_data(self, fft_magnitude: np.ndarray, scale_nm_per_pixel: float = 1.0):
        """Set FFT magnitude data for display"""
        if np is None:
            return

        self._fft_data = fft_magnitude
        self._scale_nm_per_pixel = scale_nm_per_pixel
        self._update_display()

    def set_peaks(self, peaks: List[Tuple[float, float, float]]):
        """Set detected peaks to highlight"""
        self._peaks = peaks
        self._update_display()

    def set_log_scale(self, enabled: bool):
        """Toggle log scale display"""
        self._log_scale = enabled
        self._update_display()

    def set_show_peaks(self, enabled: bool):
        """Toggle peak markers"""
        self._show_peaks = enabled
        self._update_display()

    def _update_display(self):
        """Update the FFT display"""
        if self._fft_data is None or np is None:
            self.setText("No FFT data")
            return

        # Apply log scale if enabled
        if self._log_scale:
            display_data = np.log1p(self._fft_data)
        else:
            display_data = self._fft_data.copy()

        # Normalize to 0-255
        data_min = display_data.min()
        data_max = display_data.max()
        if data_max > data_min:
            normalized = ((display_data - data_min) / (data_max - data_min) * 255).astype(np.uint8)
        else:
            normalized = np.zeros_like(display_data, dtype=np.uint8)

        # Create QImage
        h, w = normalized.shape
        bytes_per_line = w
        image = QImage(normalized.data, w, h, bytes_per_line, QImage.Format.Format_Grayscale8)

        # Convert to pixmap
        pixmap = QPixmap.fromImage(image)

        # Draw peak markers if enabled
        if self._show_peaks and self._peaks:
            painter = QPainter(pixmap)
            pen = QPen(QColor('#ff5555'))
            pen.setWidth(2)
            painter.setPen(pen)

            center_x, center_y = w // 2, h // 2

            for px, py, intensity in self._peaks:
                # Convert to pixel coordinates relative to center
                x = int(center_x + px)
                y = int(center_y + py)

                # Draw circle marker
                painter.drawEllipse(x - 5, y - 5, 10, 10)

                # Draw cross
                painter.drawLine(x - 7, y, x + 7, y)
                painter.drawLine(x, y - 7, x, y + 7)

            painter.end()

        # Scale to widget size while maintaining aspect ratio
        scaled = pixmap.scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )

        self.setPixmap(scaled)


class FFTProfileWidget(QWidget):
    """Widget for displaying radial FFT profile"""

    def __init__(self):
        super().__init__()
        self._profile_data: Optional[np.ndarray] = None
        self._frequencies: Optional[np.ndarray] = None
        self._peaks: List[int] = []
        self._scale_nm_per_pixel: float = 1.0

        self.setMinimumHeight(150)
        self.setStyleSheet("background-color: #1e1e1e;")

    def set_profile(self, profile: np.ndarray, frequencies: np.ndarray,
                    scale_nm_per_pixel: float = 1.0):
        """Set radial profile data"""
        self._profile_data = profile
        self._frequencies = frequencies
        self._scale_nm_per_pixel = scale_nm_per_pixel
        self.update()

    def set_peaks(self, peak_indices: List[int]):
        """Set peak indices to highlight"""
        self._peaks = peak_indices
        self.update()

    def paintEvent(self, event):
        """Paint the profile plot"""
        super().paintEvent(event)

        if self._profile_data is None or np is None:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Drawing area
        margin = 40
        w = self.width() - 2 * margin
        h = self.height() - 2 * margin

        if w <= 0 or h <= 0:
            return

        # Draw axes
        axis_pen = QPen(QColor('#666666'))
        painter.setPen(axis_pen)
        painter.drawLine(margin, self.height() - margin,
                        self.width() - margin, self.height() - margin)
        painter.drawLine(margin, margin, margin, self.height() - margin)

        # Draw axis labels
        painter.setPen(QColor('#aaaaaa'))
        font = QFont()
        font.setPointSize(8)
        painter.setFont(font)
        painter.drawText(self.width() // 2 - 30, self.height() - 5, "Frequency (1/nm)")
        painter.save()
        painter.translate(10, self.height() // 2)
        painter.rotate(-90)
        painter.drawText(0, 0, "Intensity")
        painter.restore()

        # Normalize data for plotting
        profile = self._profile_data
        profile_max = profile.max()
        if profile_max == 0:
            return

        normalized = profile / profile_max

        # Draw profile line
        profile_pen = QPen(QColor('#00aaff'))
        profile_pen.setWidth(2)
        painter.setPen(profile_pen)

        points = []
        n_points = len(normalized)
        for i, val in enumerate(normalized):
            x = margin + int(i / n_points * w)
            y = self.height() - margin - int(val * h)
            points.append((x, y))

        for i in range(len(points) - 1):
            painter.drawLine(points[i][0], points[i][1],
                           points[i + 1][0], points[i + 1][1])

        # Draw peak markers
        peak_pen = QPen(QColor('#ff5555'))
        peak_pen.setWidth(2)
        painter.setPen(peak_pen)

        for peak_idx in self._peaks:
            if 0 <= peak_idx < len(points):
                x, y = points[peak_idx]
                painter.drawLine(x, y - 10, x, y + 10)

                # Draw spacing value if we have frequencies
                if self._frequencies is not None and peak_idx < len(self._frequencies):
                    freq = self._frequencies[peak_idx]
                    if freq > 0:
                        spacing = 1.0 / freq
                        painter.drawText(x - 15, y - 15, f"{spacing:.2f}nm")


class FFTViewerWidget(QWidget):
    """
    Complete FFT viewer widget with power spectrum and analysis results.

    Features:
    - FFT power spectrum display with log scale option
    - Radial profile with peak detection
    - Detected periodicities list
    - Scale calibration suggestions
    """

    calibration_requested = pyqtSignal(float)  # spacing_nm

    def __init__(self):
        super().__init__()
        self._fft_results: Optional[Dict[str, Any]] = None
        self._setup_ui()

    def _setup_ui(self):
        """Setup the widget UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        # Title
        title = QLabel("FFT Analysis")
        title.setObjectName("titleLabel")
        title.setStyleSheet("font-weight: bold; font-size: 14px;")
        layout.addWidget(title)

        # Main content with splitter
        splitter = QSplitter(Qt.Orientation.Vertical)

        # Top: FFT display and controls
        top_widget = QWidget()
        top_layout = QHBoxLayout(top_widget)
        top_layout.setContentsMargins(0, 0, 0, 0)

        # FFT display
        self.fft_display = FFTDisplayWidget()
        top_layout.addWidget(self.fft_display, stretch=2)

        # Controls and info
        controls_widget = QWidget()
        controls_layout = QVBoxLayout(controls_widget)
        controls_layout.setContentsMargins(0, 0, 0, 0)

        # Display options
        options_group = QGroupBox("Display Options")
        options_layout = QVBoxLayout(options_group)

        from PyQt6.QtWidgets import QCheckBox
        self.log_scale_cb = QCheckBox("Log Scale")
        self.log_scale_cb.setChecked(True)
        self.log_scale_cb.toggled.connect(self.fft_display.set_log_scale)
        options_layout.addWidget(self.log_scale_cb)

        self.show_peaks_cb = QCheckBox("Show Peaks")
        self.show_peaks_cb.setChecked(True)
        self.show_peaks_cb.toggled.connect(self.fft_display.set_show_peaks)
        options_layout.addWidget(self.show_peaks_cb)

        controls_layout.addWidget(options_group)

        # Detected periodicities
        periods_group = QGroupBox("Detected Periodicities")
        periods_layout = QVBoxLayout(periods_group)

        self.periods_label = QLabel("No data")
        self.periods_label.setWordWrap(True)
        self.periods_label.setStyleSheet("color: #aaaaaa;")
        periods_layout.addWidget(self.periods_label)

        # Use for calibration button
        self.calibrate_btn = QPushButton("Use for Calibration")
        self.calibrate_btn.setEnabled(False)
        self.calibrate_btn.clicked.connect(self._on_calibrate_clicked)
        periods_layout.addWidget(self.calibrate_btn)

        controls_layout.addWidget(periods_group)

        # Scale info
        scale_group = QGroupBox("Scale Info")
        scale_layout = QFormLayout(scale_group)

        self.scale_label = QLabel("-")
        scale_layout.addRow("Current:", self.scale_label)

        self.suggested_label = QLabel("-")
        self.suggested_label.setStyleSheet("color: #17a2b8;")
        scale_layout.addRow("Suggested:", self.suggested_label)

        controls_layout.addWidget(scale_group)
        controls_layout.addStretch()

        top_layout.addWidget(controls_widget, stretch=1)
        splitter.addWidget(top_widget)

        # Bottom: Radial profile
        profile_widget = QWidget()
        profile_layout = QVBoxLayout(profile_widget)
        profile_layout.setContentsMargins(0, 0, 0, 0)

        profile_label = QLabel("Radial Profile")
        profile_label.setStyleSheet("font-weight: bold;")
        profile_layout.addWidget(profile_label)

        self.profile_plot = FFTProfileWidget()
        profile_layout.addWidget(self.profile_plot)

        splitter.addWidget(profile_widget)
        splitter.setSizes([300, 150])

        layout.addWidget(splitter)

    def set_fft_results(self, results: Dict[str, Any], scale_nm_per_pixel: float = 1.0):
        """Set FFT analysis results for display"""
        self._fft_results = results

        # Update FFT display
        if 'power_spectrum' in results and np is not None:
            self.fft_display.set_fft_data(results['power_spectrum'], scale_nm_per_pixel)

        # Update peaks
        if 'peaks' in results:
            self.fft_display.set_peaks(results['peaks'])

        # Update radial profile
        if 'radial_profile' in results and 'frequencies' in results:
            self.profile_plot.set_profile(
                results['radial_profile'],
                results['frequencies'],
                scale_nm_per_pixel
            )

        if 'peak_indices' in results:
            self.profile_plot.set_peaks(results['peak_indices'])

        # Update periodicities list
        periods_text = []
        if 'periodicities_nm' in results:
            for i, spacing in enumerate(results['periodicities_nm'][:5]):  # Top 5
                periods_text.append(f"{i+1}. {spacing:.3f} nm")

        if periods_text:
            self.periods_label.setText('\n'.join(periods_text))
            self.calibrate_btn.setEnabled(True)
        else:
            self.periods_label.setText("No periodicities detected")
            self.calibrate_btn.setEnabled(False)

        # Update scale info
        self.scale_label.setText(f"{scale_nm_per_pixel:.4f} nm/px")

        if 'suggested_scale' in results:
            self.suggested_label.setText(f"{results['suggested_scale']:.4f} nm/px")
        elif 'dominant_spacing_nm' in results:
            # If we know expected lattice, suggest calibration
            self.suggested_label.setText(f"Dominant: {results['dominant_spacing_nm']:.3f} nm")

    def _on_calibrate_clicked(self):
        """Handle calibration button click"""
        if self._fft_results and 'dominant_spacing_nm' in self._fft_results:
            self.calibration_requested.emit(self._fft_results['dominant_spacing_nm'])

    def clear(self):
        """Clear all FFT data"""
        self._fft_results = None
        self.fft_display.setText("No FFT data")
        self.profile_plot._profile_data = None
        self.profile_plot.update()
        self.periods_label.setText("No data")
        self.scale_label.setText("-")
        self.suggested_label.setText("-")
        self.calibrate_btn.setEnabled(False)
