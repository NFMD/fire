"""
Manual CD Measurement Widget

Provides interactive tools for:
- Manual horizontal line (baseline) selection
- Auto-leveling based on detected wafer surface
- Interactive CD measurement with mouse
- Real-time distance calculation in nm
"""

from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict
import numpy as np

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
    QPushButton, QLabel, QSpinBox, QDoubleSpinBox,
    QTableWidget, QTableWidgetItem, QHeaderView,
    QComboBox, QCheckBox, QMessageBox, QToolButton,
    QButtonGroup, QFrame, QSplitter
)
from PyQt6.QtCore import Qt, pyqtSignal, QPointF, QLineF
from PyQt6.QtGui import QColor, QPen, QFont


@dataclass
class Measurement:
    """Single measurement data"""
    id: int
    start_x: float
    start_y: float
    end_x: float
    end_y: float
    distance_px: float
    distance_nm: float
    angle_deg: float
    label: str = ""
    color: Tuple[int, int, int] = (0, 255, 0)


@dataclass
class MeasurementSession:
    """Collection of measurements for an image"""
    image_path: str
    scale_nm_per_pixel: float
    baseline_y: Optional[int] = None
    baseline_angle: float = 0.0
    measurements: List[Measurement] = field(default_factory=list)
    next_id: int = 1


class ManualMeasurementWidget(QWidget):
    """
    Widget for manual CD measurement controls.

    Features:
    - Tool selection (baseline, horizontal measure, free measure)
    - Auto-level detection and correction
    - Measurement table with results
    - Export capabilities
    """

    # Signals
    tool_changed = pyqtSignal(str)  # 'baseline', 'horizontal', 'free', 'none'
    baseline_set = pyqtSignal(int, float)  # y_position, angle
    auto_level_requested = pyqtSignal()
    measurement_added = pyqtSignal(Measurement)
    measurement_deleted = pyqtSignal(int)  # measurement id
    clear_measurements = pyqtSignal()

    # Tool modes
    TOOL_NONE = 'none'
    TOOL_BASELINE = 'baseline'
    TOOL_HORIZONTAL = 'horizontal'
    TOOL_FREE = 'free'
    TOOL_VERTICAL = 'vertical'

    def __init__(self, parent=None):
        super().__init__(parent)
        self.session: Optional[MeasurementSession] = None
        self.current_tool = self.TOOL_NONE
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)

        # === Tool Selection ===
        tools_group = QGroupBox("Measurement Tools")
        tools_layout = QVBoxLayout(tools_group)

        # Tool buttons
        btn_layout = QHBoxLayout()

        self.btn_group = QButtonGroup(self)
        self.btn_group.setExclusive(True)

        self.baseline_btn = QToolButton()
        self.baseline_btn.setText("━ Baseline")
        self.baseline_btn.setCheckable(True)
        self.baseline_btn.setToolTip("Set horizontal reference line (0nm depth)")
        self.btn_group.addButton(self.baseline_btn)
        btn_layout.addWidget(self.baseline_btn)

        self.horizontal_btn = QToolButton()
        self.horizontal_btn.setText("↔ Horizontal")
        self.horizontal_btn.setCheckable(True)
        self.horizontal_btn.setToolTip("Measure horizontal distance (CD)")
        self.btn_group.addButton(self.horizontal_btn)
        btn_layout.addWidget(self.horizontal_btn)

        self.vertical_btn = QToolButton()
        self.vertical_btn.setText("↕ Vertical")
        self.vertical_btn.setCheckable(True)
        self.vertical_btn.setToolTip("Measure vertical distance (depth)")
        self.btn_group.addButton(self.vertical_btn)
        btn_layout.addWidget(self.vertical_btn)

        self.free_btn = QToolButton()
        self.free_btn.setText("↗ Free")
        self.free_btn.setCheckable(True)
        self.free_btn.setToolTip("Measure any direction")
        self.btn_group.addButton(self.free_btn)
        btn_layout.addWidget(self.free_btn)

        tools_layout.addLayout(btn_layout)

        # Connect tool buttons
        self.baseline_btn.toggled.connect(lambda c: self._on_tool_selected(self.TOOL_BASELINE) if c else None)
        self.horizontal_btn.toggled.connect(lambda c: self._on_tool_selected(self.TOOL_HORIZONTAL) if c else None)
        self.vertical_btn.toggled.connect(lambda c: self._on_tool_selected(self.TOOL_VERTICAL) if c else None)
        self.free_btn.toggled.connect(lambda c: self._on_tool_selected(self.TOOL_FREE) if c else None)

        layout.addWidget(tools_group)

        # === Auto-Level Section ===
        level_group = QGroupBox("Auto Leveling")
        level_layout = QVBoxLayout(level_group)

        level_btn_layout = QHBoxLayout()

        self.auto_level_btn = QPushButton("🔄 Auto Detect Level")
        self.auto_level_btn.setToolTip("Automatically detect and correct image tilt")
        self.auto_level_btn.clicked.connect(self._on_auto_level)
        level_btn_layout.addWidget(self.auto_level_btn)

        self.apply_level_btn = QPushButton("✓ Apply")
        self.apply_level_btn.setEnabled(False)
        self.apply_level_btn.clicked.connect(self._on_apply_level)
        level_btn_layout.addWidget(self.apply_level_btn)

        level_layout.addLayout(level_btn_layout)

        # Level info
        self.level_info = QLabel("Detected angle: --")
        level_layout.addWidget(self.level_info)

        layout.addWidget(level_group)

        # === Baseline Section ===
        baseline_group = QGroupBox("Baseline (0nm Reference)")
        baseline_layout = QVBoxLayout(baseline_group)

        baseline_pos_layout = QHBoxLayout()
        baseline_pos_layout.addWidget(QLabel("Y Position:"))
        self.baseline_spin = QSpinBox()
        self.baseline_spin.setRange(0, 10000)
        self.baseline_spin.setSuffix(" px")
        self.baseline_spin.valueChanged.connect(self._on_baseline_changed)
        baseline_pos_layout.addWidget(self.baseline_spin)

        self.set_baseline_btn = QPushButton("Set")
        self.set_baseline_btn.clicked.connect(self._on_set_baseline)
        baseline_pos_layout.addWidget(self.set_baseline_btn)

        baseline_layout.addLayout(baseline_pos_layout)

        layout.addWidget(baseline_group)

        # === Scale Info ===
        scale_group = QGroupBox("Scale Information")
        scale_layout = QVBoxLayout(scale_group)

        self.scale_label = QLabel("Scale: -- nm/pixel")
        self.scale_label.setFont(QFont("Monospace", 10))
        scale_layout.addWidget(self.scale_label)

        # Manual scale override
        scale_override_layout = QHBoxLayout()
        self.scale_override_check = QCheckBox("Override:")
        self.scale_override_check.toggled.connect(self._on_scale_override)
        scale_override_layout.addWidget(self.scale_override_check)

        self.scale_override_spin = QDoubleSpinBox()
        self.scale_override_spin.setRange(0.001, 1000)
        self.scale_override_spin.setDecimals(4)
        self.scale_override_spin.setSuffix(" nm/px")
        self.scale_override_spin.setEnabled(False)
        self.scale_override_spin.valueChanged.connect(self._on_scale_value_changed)
        scale_override_layout.addWidget(self.scale_override_spin)

        scale_layout.addLayout(scale_override_layout)

        layout.addWidget(scale_group)

        # === Measurements Table ===
        table_group = QGroupBox("Measurements")
        table_layout = QVBoxLayout(table_group)

        self.measurements_table = QTableWidget()
        self.measurements_table.setColumnCount(5)
        self.measurements_table.setHorizontalHeaderLabels([
            "ID", "Type", "Distance (nm)", "Distance (px)", "Angle (°)"
        ])
        self.measurements_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.measurements_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.measurements_table.setAlternatingRowColors(True)
        table_layout.addWidget(self.measurements_table)

        # Table controls
        table_btn_layout = QHBoxLayout()

        self.delete_btn = QPushButton("Delete Selected")
        self.delete_btn.clicked.connect(self._on_delete_measurement)
        table_btn_layout.addWidget(self.delete_btn)

        self.clear_btn = QPushButton("Clear All")
        self.clear_btn.clicked.connect(self._on_clear_all)
        table_btn_layout.addWidget(self.clear_btn)

        table_layout.addLayout(table_btn_layout)

        layout.addWidget(table_group)

        # === Summary ===
        self.summary_label = QLabel("")
        self.summary_label.setStyleSheet("font-weight: bold; padding: 5px;")
        layout.addWidget(self.summary_label)

        layout.addStretch()

    def _on_tool_selected(self, tool: str):
        """Handle tool selection"""
        self.current_tool = tool
        self.tool_changed.emit(tool)

    def _on_auto_level(self):
        """Request auto-leveling"""
        self.auto_level_requested.emit()

    def _on_apply_level(self):
        """Apply detected level correction"""
        if self.session and self.session.baseline_angle != 0:
            self.baseline_set.emit(
                self.session.baseline_y or 0,
                self.session.baseline_angle
            )

    def _on_baseline_changed(self, value: int):
        """Handle baseline position change"""
        pass  # Will be applied when Set is clicked

    def _on_set_baseline(self):
        """Set baseline from spinbox value"""
        y = self.baseline_spin.value()
        if self.session:
            self.session.baseline_y = y
        self.baseline_set.emit(y, 0.0)

    def _on_scale_override(self, checked: bool):
        """Handle scale override checkbox"""
        self.scale_override_spin.setEnabled(checked)
        if checked and self.session:
            self.session.scale_nm_per_pixel = self.scale_override_spin.value()
            self._update_measurements()

    def _on_scale_value_changed(self, value: float):
        """Handle manual scale value change"""
        if self.scale_override_check.isChecked() and self.session:
            self.session.scale_nm_per_pixel = value
            self._update_measurements()

    def _on_delete_measurement(self):
        """Delete selected measurement"""
        rows = self.measurements_table.selectionModel().selectedRows()
        if rows and self.session:
            row = rows[0].row()
            if row < len(self.session.measurements):
                m = self.session.measurements[row]
                self.session.measurements.remove(m)
                self.measurement_deleted.emit(m.id)
                self._update_table()

    def _on_clear_all(self):
        """Clear all measurements"""
        if self.session:
            reply = QMessageBox.question(
                self, "Clear Measurements",
                "Are you sure you want to clear all measurements?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.Yes:
                self.session.measurements.clear()
                self.session.next_id = 1
                self.clear_measurements.emit()
                self._update_table()

    def set_session(self, image_path: str, scale_nm_per_pixel: float):
        """Start a new measurement session"""
        self.session = MeasurementSession(
            image_path=image_path,
            scale_nm_per_pixel=scale_nm_per_pixel
        )
        self.scale_label.setText(f"Scale: {scale_nm_per_pixel:.4f} nm/pixel")
        self.scale_override_spin.setValue(scale_nm_per_pixel)
        self._update_table()

    def set_detected_angle(self, angle: float):
        """Set detected tilt angle"""
        self.level_info.setText(f"Detected angle: {angle:.2f}°")
        self.apply_level_btn.setEnabled(abs(angle) > 0.01)
        if self.session:
            self.session.baseline_angle = angle

    def set_baseline(self, y: int, angle: float = 0.0):
        """Set baseline position"""
        self.baseline_spin.setValue(y)
        if self.session:
            self.session.baseline_y = y
            self.session.baseline_angle = angle

    def add_measurement(self, start: QPointF, end: QPointF,
                       measure_type: str = 'free') -> Optional[Measurement]:
        """Add a new measurement"""
        if not self.session:
            return None

        # Calculate distances
        dx = end.x() - start.x()
        dy = end.y() - start.y()

        if measure_type == 'horizontal':
            distance_px = abs(dx)
        elif measure_type == 'vertical':
            distance_px = abs(dy)
        else:
            distance_px = np.sqrt(dx**2 + dy**2)

        distance_nm = distance_px * self.session.scale_nm_per_pixel

        # Calculate angle
        if dx != 0:
            angle_deg = np.degrees(np.arctan2(dy, dx))
        else:
            angle_deg = 90.0 if dy > 0 else -90.0

        # Create measurement
        measurement = Measurement(
            id=self.session.next_id,
            start_x=start.x(),
            start_y=start.y(),
            end_x=end.x(),
            end_y=end.y(),
            distance_px=distance_px,
            distance_nm=distance_nm,
            angle_deg=angle_deg,
            label=f"M{self.session.next_id}"
        )

        self.session.measurements.append(measurement)
        self.session.next_id += 1

        self._update_table()
        self.measurement_added.emit(measurement)

        return measurement

    def _update_table(self):
        """Update measurements table"""
        self.measurements_table.setRowCount(0)

        if not self.session:
            self.summary_label.setText("")
            return

        for m in self.session.measurements:
            row = self.measurements_table.rowCount()
            self.measurements_table.insertRow(row)

            # Determine type
            if abs(m.angle_deg) < 5 or abs(m.angle_deg - 180) < 5:
                m_type = "Horizontal"
            elif abs(abs(m.angle_deg) - 90) < 5:
                m_type = "Vertical"
            else:
                m_type = "Free"

            self.measurements_table.setItem(row, 0, QTableWidgetItem(str(m.id)))
            self.measurements_table.setItem(row, 1, QTableWidgetItem(m_type))
            self.measurements_table.setItem(row, 2, QTableWidgetItem(f"{m.distance_nm:.2f}"))
            self.measurements_table.setItem(row, 3, QTableWidgetItem(f"{m.distance_px:.1f}"))
            self.measurements_table.setItem(row, 4, QTableWidgetItem(f"{m.angle_deg:.1f}"))

        # Update summary
        if self.session.measurements:
            horizontal = [m for m in self.session.measurements
                         if abs(m.angle_deg) < 5 or abs(m.angle_deg - 180) < 5]
            if horizontal:
                avg_cd = np.mean([m.distance_nm for m in horizontal])
                std_cd = np.std([m.distance_nm for m in horizontal]) if len(horizontal) > 1 else 0
                self.summary_label.setText(
                    f"Horizontal CD: {avg_cd:.2f} ± {std_cd:.2f} nm (n={len(horizontal)})"
                )
            else:
                self.summary_label.setText(f"Total measurements: {len(self.session.measurements)}")
        else:
            self.summary_label.setText("")

    def _update_measurements(self):
        """Recalculate all measurements with new scale"""
        if not self.session:
            return

        for m in self.session.measurements:
            dx = m.end_x - m.start_x
            dy = m.end_y - m.start_y

            if abs(m.angle_deg) < 5 or abs(m.angle_deg - 180) < 5:
                m.distance_px = abs(dx)
            elif abs(abs(m.angle_deg) - 90) < 5:
                m.distance_px = abs(dy)
            else:
                m.distance_px = np.sqrt(dx**2 + dy**2)

            m.distance_nm = m.distance_px * self.session.scale_nm_per_pixel

        self._update_table()

    def get_measurements(self) -> List[Measurement]:
        """Get all measurements"""
        if self.session:
            return self.session.measurements
        return []

    def get_session(self) -> Optional[MeasurementSession]:
        """Get current session"""
        return self.session


class AutoLeveler:
    """
    Automatic image leveling based on wafer surface detection.

    Methods:
    - Edge-based: Detect strong horizontal edges
    - Gradient-based: Find dominant horizontal gradient
    - FFT-based: Detect orientation from frequency domain
    """

    def __init__(self):
        self.detected_angle = 0.0
        self.confidence = 0.0
        self.method_used = ""

    def detect_level(self, image: np.ndarray,
                     method: str = 'auto') -> Tuple[float, float, str]:
        """
        Detect image tilt angle.

        Args:
            image: Grayscale image (0-255 or 0-1)
            method: 'auto', 'edge', 'gradient', 'fft', 'hough'

        Returns:
            (angle_degrees, confidence, method_used)
        """
        import cv2

        # Normalize image
        if image.max() <= 1.0:
            img = (image * 255).astype(np.uint8)
        else:
            img = image.astype(np.uint8)

        if method == 'auto':
            # Try multiple methods and use best result
            results = []

            for m in ['hough', 'gradient', 'edge']:
                try:
                    angle, conf, _ = self._detect_by_method(img, m)
                    results.append((angle, conf, m))
                except Exception:
                    pass

            if results:
                # Use result with highest confidence
                best = max(results, key=lambda x: x[1])
                self.detected_angle, self.confidence, self.method_used = best
                return best
            else:
                return 0.0, 0.0, 'none'
        else:
            return self._detect_by_method(img, method)

    def _detect_by_method(self, img: np.ndarray,
                          method: str) -> Tuple[float, float, str]:
        """Detect using specific method"""
        import cv2

        if method == 'hough':
            return self._detect_hough(img)
        elif method == 'gradient':
            return self._detect_gradient(img)
        elif method == 'edge':
            return self._detect_edge(img)
        elif method == 'fft':
            return self._detect_fft(img)
        else:
            raise ValueError(f"Unknown method: {method}")

    def _detect_hough(self, img: np.ndarray) -> Tuple[float, float, str]:
        """Detect using Hough line transform"""
        import cv2

        # Edge detection
        edges = cv2.Canny(img, 50, 150, apertureSize=3)

        # Hough transform
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)

        if lines is None or len(lines) == 0:
            return 0.0, 0.0, 'hough'

        # Find near-horizontal lines
        horizontal_angles = []
        for line in lines:
            rho, theta = line[0]
            # theta is in radians, 0 is vertical, pi/2 is horizontal
            angle_from_horizontal = np.degrees(theta) - 90

            # Only consider nearly horizontal lines (within 15 degrees)
            if abs(angle_from_horizontal) < 15:
                horizontal_angles.append(angle_from_horizontal)

        if not horizontal_angles:
            return 0.0, 0.0, 'hough'

        # Use median angle
        angle = np.median(horizontal_angles)
        confidence = min(1.0, len(horizontal_angles) / 10)

        return angle, confidence, 'hough'

    def _detect_gradient(self, img: np.ndarray) -> Tuple[float, float, str]:
        """Detect using gradient analysis"""
        import cv2

        # Compute gradients
        gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
        gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)

        # Compute gradient magnitude and angle
        magnitude = np.sqrt(gx**2 + gy**2)
        angle = np.arctan2(gy, gx)

        # Weight angles by magnitude
        threshold = np.percentile(magnitude, 90)
        mask = magnitude > threshold

        if not np.any(mask):
            return 0.0, 0.0, 'gradient'

        # Find dominant horizontal angle
        weighted_angles = angle[mask]

        # Convert to degrees and shift to horizontal reference
        angles_deg = np.degrees(weighted_angles)

        # Find angles close to horizontal (edges are perpendicular to gradient)
        horizontal_component = angles_deg - 90
        horizontal_component = np.mod(horizontal_component + 90, 180) - 90

        # Use robust median
        angle = np.median(horizontal_component)
        confidence = 0.7  # Moderate confidence

        return angle, confidence, 'gradient'

    def _detect_edge(self, img: np.ndarray) -> Tuple[float, float, str]:
        """Detect using edge profile analysis"""
        import cv2

        # Find strong horizontal edges
        gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)

        # Sum across columns to find strongest horizontal line
        row_sums = np.sum(np.abs(gy), axis=1)

        # Find peak (strongest horizontal edge)
        peak_y = np.argmax(row_sums)

        # Analyze edge at this row for angle
        edge_row = gy[peak_y, :]

        # Find zero crossings
        crossings = np.where(np.diff(np.sign(edge_row)))[0]

        if len(crossings) < 2:
            return 0.0, 0.0, 'edge'

        # Fit line to crossings
        x = crossings
        y = np.full_like(crossings, peak_y, dtype=float)

        # Add adjacent rows for more robust fit
        for dy in [-2, -1, 1, 2]:
            if 0 <= peak_y + dy < len(row_sums):
                edge_row_adj = gy[peak_y + dy, :]
                crossings_adj = np.where(np.diff(np.sign(edge_row_adj)))[0]
                if len(crossings_adj) > 0:
                    x = np.concatenate([x, crossings_adj])
                    y = np.concatenate([y, np.full_like(crossings_adj, peak_y + dy, dtype=float)])

        if len(x) < 3:
            return 0.0, 0.0, 'edge'

        # Linear fit
        coeffs = np.polyfit(x, y, 1)
        angle = np.degrees(np.arctan(coeffs[0]))

        confidence = 0.6

        return angle, confidence, 'edge'

    def _detect_fft(self, img: np.ndarray) -> Tuple[float, float, str]:
        """Detect using FFT orientation analysis"""
        # Compute 2D FFT
        f = np.fft.fft2(img)
        fshift = np.fft.fftshift(f)
        magnitude = np.abs(fshift)

        # Log transform for visualization
        log_magnitude = np.log1p(magnitude)

        h, w = img.shape
        cy, cx = h // 2, w // 2

        # Find dominant orientation
        y, x = np.ogrid[:h, :w]
        r = np.sqrt((x - cx)**2 + (y - cy)**2)

        # Consider only middle frequencies
        mask = (r > 10) & (r < min(h, w) // 4)

        # Find angle of maximum power
        angles = np.arctan2(y - cy, x - cx)

        # Weight by magnitude in mask
        weighted_angles = angles[mask]
        weights = log_magnitude[mask]

        if len(weighted_angles) == 0:
            return 0.0, 0.0, 'fft'

        # Find dominant angle (looking for horizontal features)
        # Horizontal features appear as vertical in FFT
        angle_deg = np.average(np.degrees(weighted_angles), weights=weights)

        # Convert to image space angle
        image_angle = 90 - angle_deg
        image_angle = np.mod(image_angle + 90, 180) - 90

        return image_angle, 0.5, 'fft'

    def apply_correction(self, image: np.ndarray,
                        angle: Optional[float] = None) -> np.ndarray:
        """
        Apply rotation correction to image.

        Args:
            image: Input image
            angle: Angle to correct (uses detected if None)

        Returns:
            Corrected image
        """
        import cv2

        if angle is None:
            angle = self.detected_angle

        if abs(angle) < 0.01:
            return image

        h, w = image.shape[:2]
        center = (w // 2, h // 2)

        # Create rotation matrix
        matrix = cv2.getRotationMatrix2D(center, -angle, 1.0)

        # Apply rotation
        corrected = cv2.warpAffine(
            image, matrix, (w, h),
            borderMode=cv2.BORDER_REFLECT
        )

        return corrected
