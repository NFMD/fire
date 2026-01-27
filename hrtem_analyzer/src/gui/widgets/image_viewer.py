"""
Image Viewer Widget with Baseline Selection

Provides interactive image viewing with:
- Zoom and pan
- Baseline (0-point) selection via click
- Measurement overlay
- Scale bar display
"""
from pathlib import Path
from typing import Optional, Tuple
import numpy as np

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QScrollArea, QFrame, QPushButton, QSlider,
    QCheckBox, QGraphicsView, QGraphicsScene,
    QGraphicsPixmapItem, QGraphicsLineItem, QGraphicsTextItem
)
from PyQt6.QtCore import Qt, pyqtSignal, QPointF, QRectF
from PyQt6.QtGui import (
    QPixmap, QImage, QPainter, QPen, QColor,
    QWheelEvent, QMouseEvent, QFont, QBrush
)

try:
    import tifffile
except ImportError:
    tifffile = None


class ImageGraphicsView(QGraphicsView):
    """Custom graphics view with zoom and pan support"""

    position_changed = pyqtSignal(int, int, float)  # x, y, pixel value
    clicked = pyqtSignal(int, int)  # x, y in image coordinates
    baseline_set = pyqtSignal(int)  # y position

    def __init__(self):
        super().__init__()
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setBackgroundBrush(QBrush(QColor("#1e1e1e")))

        self._zoom = 1.0
        self._image_data: Optional[np.ndarray] = None
        self._setting_baseline = False

    def set_image_data(self, data: np.ndarray):
        """Store image data for pixel value queries"""
        self._image_data = data

    def enable_baseline_mode(self, enabled: bool):
        """Enable/disable baseline setting mode"""
        self._setting_baseline = enabled
        if enabled:
            self.setDragMode(QGraphicsView.DragMode.NoDrag)
            self.setCursor(Qt.CursorShape.CrossCursor)
        else:
            self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
            self.setCursor(Qt.CursorShape.ArrowCursor)

    def wheelEvent(self, event: QWheelEvent):
        """Handle mouse wheel for zoom"""
        if event.modifiers() == Qt.KeyboardModifier.ControlModifier:
            # Zoom
            factor = 1.15
            if event.angleDelta().y() > 0:
                self._zoom *= factor
                self.scale(factor, factor)
            else:
                self._zoom /= factor
                self.scale(1 / factor, 1 / factor)
        else:
            super().wheelEvent(event)

    def mousePressEvent(self, event: QMouseEvent):
        """Handle mouse press"""
        if self._setting_baseline and event.button() == Qt.MouseButton.LeftButton:
            pos = self.mapToScene(event.pos())
            self.baseline_set.emit(int(pos.y()))
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent):
        """Handle mouse move for position tracking"""
        pos = self.mapToScene(event.pos())
        x, y = int(pos.x()), int(pos.y())

        if self._image_data is not None:
            h, w = self._image_data.shape[:2]
            if 0 <= x < w and 0 <= y < h:
                value = float(self._image_data[y, x])
                self.position_changed.emit(x, y, value)

        super().mouseMoveEvent(event)

    def zoom_in(self):
        """Zoom in"""
        factor = 1.25
        self._zoom *= factor
        self.scale(factor, factor)

    def zoom_out(self):
        """Zoom out"""
        factor = 1.25
        self._zoom /= factor
        self.scale(1 / factor, 1 / factor)

    def fit_to_window(self):
        """Fit image to window"""
        self.fitInView(self.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
        self._zoom = 1.0

    def reset_zoom(self):
        """Reset to 100% zoom"""
        self.resetTransform()
        self._zoom = 1.0


class ImageViewerWidget(QWidget):
    """
    Image viewer widget with baseline selection capability.

    Features:
    - Load and display TIFF images
    - Zoom and pan
    - Click to set baseline (0-point)
    - Show measurement overlays
    - Display scale information
    """

    baseline_changed = pyqtSignal(int)  # y position
    position_changed = pyqtSignal(int, int, float)  # x, y, value

    def __init__(self):
        super().__init__()
        self._image_path: Optional[str] = None
        self._image_data: Optional[np.ndarray] = None
        self._scale_nm_per_pixel: float = 1.0
        self._baseline_y: Optional[int] = None
        self._baseline_line: Optional[QGraphicsLineItem] = None
        self._measurement_items = []

        self._setup_ui()

    def _setup_ui(self):
        """Setup the widget UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        # Toolbar
        toolbar = QHBoxLayout()
        toolbar.setContentsMargins(8, 4, 8, 4)

        # Zoom controls
        self.zoom_in_btn = QPushButton("+")
        self.zoom_in_btn.setFixedSize(30, 30)
        self.zoom_in_btn.clicked.connect(self._on_zoom_in)
        toolbar.addWidget(self.zoom_in_btn)

        self.zoom_out_btn = QPushButton("-")
        self.zoom_out_btn.setFixedSize(30, 30)
        self.zoom_out_btn.clicked.connect(self._on_zoom_out)
        toolbar.addWidget(self.zoom_out_btn)

        self.fit_btn = QPushButton("Fit")
        self.fit_btn.setFixedSize(50, 30)
        self.fit_btn.clicked.connect(self._on_fit)
        toolbar.addWidget(self.fit_btn)

        toolbar.addSpacing(20)

        # Baseline mode toggle
        self.baseline_btn = QPushButton("Set Baseline")
        self.baseline_btn.setCheckable(True)
        self.baseline_btn.setFixedWidth(100)
        self.baseline_btn.toggled.connect(self._on_baseline_mode)
        toolbar.addWidget(self.baseline_btn)

        # Show measurements toggle
        self.show_measurements_cb = QCheckBox("Show Measurements")
        self.show_measurements_cb.setChecked(True)
        self.show_measurements_cb.toggled.connect(self._on_toggle_measurements)
        toolbar.addWidget(self.show_measurements_cb)

        toolbar.addStretch()

        # Info label
        self.info_label = QLabel()
        self.info_label.setObjectName("subtitleLabel")
        toolbar.addWidget(self.info_label)

        layout.addLayout(toolbar)

        # Graphics view
        self.scene = QGraphicsScene()
        self.view = ImageGraphicsView()
        self.view.setScene(self.scene)
        self.view.position_changed.connect(self._on_position_changed)
        self.view.baseline_set.connect(self._on_baseline_set)

        layout.addWidget(self.view)

        # Position label
        self.position_label = QLabel("Position: -")
        self.position_label.setObjectName("subtitleLabel")
        layout.addWidget(self.position_label)

    def load_image(self, path: str) -> bool:
        """
        Load an image from file.

        Args:
            path: Path to TIFF image

        Returns:
            True if successful
        """
        try:
            if tifffile is None:
                raise ImportError("tifffile not installed")

            # Load image
            with tifffile.TiffFile(path) as tif:
                self._image_data = tif.asarray()

                # Try to extract scale from metadata
                self._scale_nm_per_pixel = self._extract_scale(tif)

            # Convert to display format
            if len(self._image_data.shape) == 3:
                # RGB or RGBA
                if self._image_data.shape[2] >= 3:
                    display_data = self._image_data[:, :, :3]
                else:
                    display_data = self._image_data[:, :, 0]
            else:
                display_data = self._image_data

            # Normalize to 0-255
            if display_data.dtype != np.uint8:
                dmin, dmax = display_data.min(), display_data.max()
                if dmax > dmin:
                    display_data = ((display_data - dmin) / (dmax - dmin) * 255).astype(np.uint8)
                else:
                    display_data = np.zeros_like(display_data, dtype=np.uint8)

            # Create QImage
            h, w = display_data.shape[:2]
            if len(display_data.shape) == 2:
                # Grayscale
                qimage = QImage(
                    display_data.data, w, h, w,
                    QImage.Format.Format_Grayscale8
                )
            else:
                # RGB
                bytes_per_line = 3 * w
                qimage = QImage(
                    display_data.data, w, h, bytes_per_line,
                    QImage.Format.Format_RGB888
                )

            # Clear scene and add image
            self.scene.clear()
            self._baseline_line = None
            self._measurement_items = []

            pixmap = QPixmap.fromImage(qimage)
            self.scene.addPixmap(pixmap)
            self.scene.setSceneRect(QRectF(pixmap.rect()))

            # Store data for pixel queries
            self.view.set_image_data(self._image_data)

            # Update info
            self._image_path = path
            self.info_label.setText(
                f"{Path(path).name} | {w}x{h} | {self._scale_nm_per_pixel:.4f} nm/px"
            )

            # Fit to view
            self.view.fit_to_window()

            # Restore baseline if set
            if self._baseline_y is not None:
                self._draw_baseline(self._baseline_y)

            return True

        except Exception as e:
            self.info_label.setText(f"Error: {e}")
            return False

    def _extract_scale(self, tif) -> float:
        """Extract scale from TIFF metadata"""
        try:
            # Try FEI metadata
            if hasattr(tif, 'fei_metadata') and tif.fei_metadata:
                fei = tif.fei_metadata
                if 'Scan' in fei and 'PixelWidth' in fei['Scan']:
                    return fei['Scan']['PixelWidth'] * 1e9

            # Try ImageJ metadata
            if hasattr(tif, 'imagej_metadata') and tif.imagej_metadata:
                ij = tif.imagej_metadata
                if 'spacing' in ij:
                    unit = ij.get('unit', 'pixel').lower()
                    if unit == 'nm':
                        return ij['spacing']
                    elif unit in ('um', 'µm'):
                        return ij['spacing'] * 1000

        except Exception:
            pass

        return 1.0  # Default

    @property
    def baseline_y(self) -> Optional[int]:
        """Get current baseline Y position"""
        return self._baseline_y

    def set_baseline(self, y: int):
        """Set baseline position programmatically"""
        self._baseline_y = y
        if self._image_data is not None:
            self._draw_baseline(y)
            self.baseline_changed.emit(y)

    def _draw_baseline(self, y: int):
        """Draw baseline on the image"""
        if self._image_data is None:
            return

        h, w = self._image_data.shape[:2]

        # Remove old baseline
        if self._baseline_line:
            self.scene.removeItem(self._baseline_line)

        # Draw new baseline
        pen = QPen(QColor("#ff0000"))
        pen.setWidth(2)
        self._baseline_line = self.scene.addLine(0, y, w, y, pen)

        # Add label
        text = self.scene.addText("0 nm (Baseline)", QFont("Arial", 10))
        text.setDefaultTextColor(QColor("#ff0000"))
        text.setPos(10, y - 20)

    def _on_baseline_set(self, y: int):
        """Handle baseline set from click"""
        self._baseline_y = y
        self._draw_baseline(y)
        self.baseline_btn.setChecked(False)
        self.baseline_changed.emit(y)

    def _on_baseline_mode(self, enabled: bool):
        """Toggle baseline setting mode"""
        self.view.enable_baseline_mode(enabled)
        if enabled:
            self.baseline_btn.setText("Click Image...")
        else:
            self.baseline_btn.setText("Set Baseline")

    def _on_position_changed(self, x: int, y: int, value: float):
        """Handle mouse position change"""
        depth_info = ""
        if self._baseline_y is not None:
            depth_px = y - self._baseline_y
            depth_nm = depth_px * self._scale_nm_per_pixel
            depth_info = f" | Depth: {depth_nm:.1f} nm"

        self.position_label.setText(
            f"Position: ({x}, {y}) | Value: {value:.3f}{depth_info}"
        )
        self.position_changed.emit(x, y, value)

    def _on_toggle_measurements(self, show: bool):
        """Toggle measurement overlay visibility"""
        for item in self._measurement_items:
            item.setVisible(show)

    def _on_zoom_in(self):
        """Zoom in button clicked"""
        self.view.zoom_in()

    def _on_zoom_out(self):
        """Zoom out button clicked"""
        self.view.zoom_out()

    def _on_fit(self):
        """Fit button clicked"""
        self.view.fit_to_window()

    def zoom_in(self):
        """Public zoom in method"""
        self.view.zoom_in()

    def zoom_out(self):
        """Public zoom out method"""
        self.view.zoom_out()

    def fit_to_window(self):
        """Public fit to window method"""
        self.view.fit_to_window()

    def add_measurement_overlay(
            self,
            y: int,
            left_x: int,
            right_x: int,
            thickness_nm: float,
            depth_nm: float,
            color: str = "#00ff00"
    ):
        """Add measurement visualization overlay"""
        if self._image_data is None:
            return

        w = self._image_data.shape[1]
        qcolor = QColor(color)

        # Measurement line
        pen = QPen(qcolor)
        pen.setWidth(2)
        line = self.scene.addLine(left_x, y, right_x, y, pen)
        self._measurement_items.append(line)

        # Edge markers
        for x in [left_x, right_x]:
            marker = self.scene.addEllipse(x - 4, y - 4, 8, 8, pen, QBrush(qcolor))
            self._measurement_items.append(marker)

        # Label
        text = self.scene.addText(
            f"{depth_nm:.0f}nm: {thickness_nm:.2f}nm",
            QFont("Arial", 9)
        )
        text.setDefaultTextColor(qcolor)
        text.setPos(right_x + 10, y - 8)
        self._measurement_items.append(text)

    def clear_measurement_overlays(self):
        """Clear all measurement overlays"""
        for item in self._measurement_items:
            self.scene.removeItem(item)
        self._measurement_items = []
