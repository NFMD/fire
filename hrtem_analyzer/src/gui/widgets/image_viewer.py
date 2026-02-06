"""
Image Viewer Widget with Baseline Selection and Manual Measurement

Provides interactive image viewing with:
- Zoom and pan
- Baseline (0-point) selection via click
- Manual measurement drawing
- Measurement overlay
- Grid/ruler overlay
- Scale bar display
- Auto-leveling support
- Undo support
- DM3/DM4 file support
- Analysis result overlay
"""
from pathlib import Path
from typing import Optional, Tuple, List
import numpy as np

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QScrollArea, QFrame, QPushButton, QSlider,
    QCheckBox, QGraphicsView, QGraphicsScene,
    QGraphicsPixmapItem, QGraphicsLineItem, QGraphicsTextItem,
    QGraphicsEllipseItem, QGraphicsRectItem
)
from PyQt6.QtCore import Qt, pyqtSignal, QPointF, QRectF, QLineF
from PyQt6.QtGui import (
    QPixmap, QImage, QPainter, QPen, QColor,
    QWheelEvent, QMouseEvent, QFont, QBrush, QKeySequence,
    QShortcut
)

try:
    import tifffile
except ImportError:
    tifffile = None


class ImageGraphicsView(QGraphicsView):
    """Custom graphics view with zoom, pan, and measurement support"""

    position_changed = pyqtSignal(int, int, float)  # x, y, pixel value
    clicked = pyqtSignal(int, int)  # x, y in image coordinates
    baseline_set = pyqtSignal(int)  # y position

    # Measurement signals
    measurement_started = pyqtSignal(QPointF)  # start point
    measurement_ended = pyqtSignal(QPointF, QPointF)  # start, end
    measurement_preview = pyqtSignal(QPointF, QPointF)  # current preview line

    # Measurement modes
    MODE_PAN = 'pan'
    MODE_BASELINE = 'baseline'
    MODE_MEASURE_HORIZONTAL = 'horizontal'
    MODE_MEASURE_VERTICAL = 'vertical'
    MODE_MEASURE_FREE = 'free'

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
        self._mode = self.MODE_PAN

        # Measurement state
        self._measuring = False
        self._measure_start: Optional[QPointF] = None
        self._preview_line: Optional[QGraphicsLineItem] = None

    def set_image_data(self, data: np.ndarray):
        """Store image data for pixel value queries"""
        self._image_data = data

    def set_mode(self, mode: str):
        """Set interaction mode"""
        self._mode = mode
        self._measuring = False
        self._measure_start = None

        if self._preview_line:
            self.scene().removeItem(self._preview_line)
            self._preview_line = None

        if mode == self.MODE_PAN:
            self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
            self.setCursor(Qt.CursorShape.ArrowCursor)
        else:
            self.setDragMode(QGraphicsView.DragMode.NoDrag)
            self.setCursor(Qt.CursorShape.CrossCursor)

    def enable_baseline_mode(self, enabled: bool):
        """Enable/disable baseline setting mode"""
        if enabled:
            self.set_mode(self.MODE_BASELINE)
        else:
            self.set_mode(self.MODE_PAN)

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
        pos = self.mapToScene(event.pos())

        if event.button() == Qt.MouseButton.LeftButton:
            if self._mode == self.MODE_BASELINE:
                self.baseline_set.emit(int(pos.y()))
            elif self._mode in [self.MODE_MEASURE_HORIZONTAL,
                               self.MODE_MEASURE_VERTICAL,
                               self.MODE_MEASURE_FREE]:
                # Start measurement
                self._measuring = True
                self._measure_start = pos
                self.measurement_started.emit(pos)

                # Create preview line
                pen = QPen(QColor("#00ff00"))
                pen.setWidth(2)
                pen.setStyle(Qt.PenStyle.DashLine)
                self._preview_line = self.scene().addLine(
                    pos.x(), pos.y(), pos.x(), pos.y(), pen
                )
            else:
                super().mousePressEvent(event)
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent):
        """Handle mouse move"""
        pos = self.mapToScene(event.pos())
        x, y = int(pos.x()), int(pos.y())

        # Update position info
        if self._image_data is not None:
            h, w = self._image_data.shape[:2]
            if 0 <= x < w and 0 <= y < h:
                value = float(self._image_data[y, x])
                self.position_changed.emit(x, y, value)

        # Update measurement preview
        if self._measuring and self._measure_start and self._preview_line:
            start = self._measure_start
            end = pos

            # Constrain based on mode
            if self._mode == self.MODE_MEASURE_HORIZONTAL:
                end = QPointF(pos.x(), start.y())
            elif self._mode == self.MODE_MEASURE_VERTICAL:
                end = QPointF(start.x(), pos.y())

            self._preview_line.setLine(QLineF(start, end))
            self.measurement_preview.emit(start, end)

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent):
        """Handle mouse release"""
        if event.button() == Qt.MouseButton.LeftButton and self._measuring:
            pos = self.mapToScene(event.pos())
            start = self._measure_start
            end = pos

            # Constrain based on mode
            if self._mode == self.MODE_MEASURE_HORIZONTAL:
                end = QPointF(pos.x(), start.y())
            elif self._mode == self.MODE_MEASURE_VERTICAL:
                end = QPointF(start.x(), pos.y())

            # Remove preview
            if self._preview_line:
                self.scene().removeItem(self._preview_line)
                self._preview_line = None

            # Emit measurement
            self.measurement_ended.emit(start, end)

            self._measuring = False
            self._measure_start = None
        else:
            super().mouseReleaseEvent(event)

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
    Image viewer widget with baseline selection and manual measurement.

    Features:
    - Load and display TIFF/DM3/DM4 images
    - Zoom and pan
    - Click to set baseline (0-point)
    - Interactive manual CD measurement
    - Show measurement overlays
    - Grid/ruler overlay
    - Display scale information
    - Auto-leveling support
    - Undo (Ctrl+Z)
    - Analysis result overlay
    """

    baseline_changed = pyqtSignal(int)  # y position
    position_changed = pyqtSignal(int, int, float)  # x, y, value
    measurement_completed = pyqtSignal(QPointF, QPointF, str)  # start, end, mode
    image_rotated = pyqtSignal(float)  # rotation angle applied

    def __init__(self):
        super().__init__()
        self._image_path: Optional[str] = None
        self._image_data: Optional[np.ndarray] = None
        self._display_data: Optional[np.ndarray] = None
        self._scale_nm_per_pixel: float = 1.0
        self._baseline_y: Optional[int] = None
        self._baseline_line: Optional[QGraphicsLineItem] = None
        self._baseline_text: Optional[QGraphicsTextItem] = None
        self._measurement_items: List = []
        self._manual_measurement_items: List = []
        self._result_overlay_items: List = []
        self._grid_items: List = []
        self._undo_stack: List = []  # Stack for undo operations
        self._current_rotation: float = 0.0
        self._show_grid = False

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

        toolbar.addSpacing(10)

        # Baseline mode toggle
        self.baseline_btn = QPushButton("Set Baseline")
        self.baseline_btn.setCheckable(True)
        self.baseline_btn.setFixedWidth(100)
        self.baseline_btn.toggled.connect(self._on_baseline_mode)
        toolbar.addWidget(self.baseline_btn)

        # Delete baseline button
        self.delete_baseline_btn = QPushButton("Del BL")
        self.delete_baseline_btn.setToolTip("Delete baseline")
        self.delete_baseline_btn.setFixedWidth(55)
        self.delete_baseline_btn.clicked.connect(self._on_delete_baseline)
        toolbar.addWidget(self.delete_baseline_btn)

        toolbar.addSpacing(10)

        # Show measurements toggle
        self.show_measurements_cb = QCheckBox("Overlay")
        self.show_measurements_cb.setChecked(True)
        self.show_measurements_cb.setToolTip("Show measurement overlays")
        self.show_measurements_cb.toggled.connect(self._on_toggle_measurements)
        toolbar.addWidget(self.show_measurements_cb)

        # Grid toggle
        self.show_grid_cb = QCheckBox("Grid")
        self.show_grid_cb.setToolTip("Show grid/ruler overlay")
        self.show_grid_cb.toggled.connect(self._on_toggle_grid)
        toolbar.addWidget(self.show_grid_cb)

        toolbar.addSpacing(10)

        # Undo button
        self.undo_btn = QPushButton("Undo")
        self.undo_btn.setFixedWidth(55)
        self.undo_btn.setToolTip("Undo last action (Ctrl+Z)")
        self.undo_btn.clicked.connect(self.undo)
        toolbar.addWidget(self.undo_btn)

        # Scale bar OCR detect
        self.ocr_btn = QPushButton("OCR Scale")
        self.ocr_btn.setFixedWidth(75)
        self.ocr_btn.setToolTip("Auto-detect scale bar from image")
        self.ocr_btn.clicked.connect(self._on_ocr_scale)
        toolbar.addWidget(self.ocr_btn)

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
        self.view.measurement_ended.connect(self._on_measurement_ended)

        layout.addWidget(self.view)

        # Position label
        self.position_label = QLabel("Position: -")
        self.position_label.setObjectName("subtitleLabel")
        layout.addWidget(self.position_label)

        # Setup keyboard shortcut for undo
        undo_shortcut = QShortcut(QKeySequence("Ctrl+Z"), self)
        undo_shortcut.activated.connect(self.undo)

    def load_image(self, path: str) -> bool:
        """
        Load an image from file (TIFF, DM3, DM4).

        Args:
            path: Path to image file

        Returns:
            True if successful
        """
        try:
            ext = Path(path).suffix.lower()

            if ext in ('.dm3', '.dm4'):
                return self._load_dm_image(path)
            else:
                return self._load_tiff_image(path)

        except Exception as e:
            self.info_label.setText(f"Error: {e}")
            return False

    def _load_tiff_image(self, path: str) -> bool:
        """Load TIFF image"""
        if tifffile is None:
            raise ImportError("tifffile not installed")

        with tifffile.TiffFile(path) as tif:
            self._image_data = tif.asarray()
            self._scale_nm_per_pixel = self._extract_scale(tif)

        return self._finish_load(path)

    def _load_dm_image(self, path: str) -> bool:
        """Load DM3/DM4 image"""
        from ...core.dm_loader import DMFileLoader
        dm_loader = DMFileLoader()
        image, dm_scale = dm_loader.load(path, normalize=False)
        self._image_data = image
        self._scale_nm_per_pixel = dm_loader.get_scale_nm_per_pixel(dm_scale)
        return self._finish_load(path)

    def _finish_load(self, path: str) -> bool:
        """Common logic after loading image data"""
        # Convert to display format
        if len(self._image_data.shape) == 3:
            if self._image_data.shape[2] >= 3:
                display_data = self._image_data[:, :, :3]
            else:
                display_data = self._image_data[:, :, 0]
        else:
            display_data = self._image_data

        # Normalize to 0-255
        if display_data.dtype != np.uint8:
            dmin, dmax = float(display_data.min()), float(display_data.max())
            if dmax > dmin:
                display_data = ((display_data - dmin) / (dmax - dmin) * 255).astype(np.uint8)
            else:
                display_data = np.zeros_like(display_data, dtype=np.uint8)

        # Store for display updates
        self._display_data = display_data

        # Create QImage
        h, w = display_data.shape[:2]
        if len(display_data.shape) == 2:
            qimage = QImage(
                display_data.data, w, h, w,
                QImage.Format.Format_Grayscale8
            )
        else:
            bytes_per_line = 3 * w
            qimage = QImage(
                display_data.data, w, h, bytes_per_line,
                QImage.Format.Format_RGB888
            )

        # Clear scene and add image
        self.scene.clear()
        self._baseline_line = None
        self._baseline_text = None
        self._measurement_items = []
        self._manual_measurement_items = []
        self._result_overlay_items = []
        self._grid_items = []
        self._undo_stack = []

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

        # Show grid if enabled
        if self._show_grid:
            self._draw_grid()

        return True

    def _extract_scale(self, tif) -> float:
        """Extract scale from TIFF metadata"""
        try:
            if hasattr(tif, 'fei_metadata') and tif.fei_metadata:
                fei = tif.fei_metadata
                if 'Scan' in fei and 'PixelWidth' in fei['Scan']:
                    return fei['Scan']['PixelWidth'] * 1e9

            if hasattr(tif, 'imagej_metadata') and tif.imagej_metadata:
                ij = tif.imagej_metadata
                if 'spacing' in ij:
                    unit = ij.get('unit', 'pixel').lower()
                    if unit == 'nm':
                        return ij['spacing']
                    elif unit in ('um', '\u00b5m'):
                        return ij['spacing'] * 1000
        except Exception:
            pass
        return 1.0

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

    def delete_baseline(self):
        """Delete the baseline"""
        if self._baseline_line:
            self.scene.removeItem(self._baseline_line)
            self._baseline_line = None
        if self._baseline_text:
            self.scene.removeItem(self._baseline_text)
            self._baseline_text = None
        old_y = self._baseline_y
        self._baseline_y = None
        if old_y is not None:
            self._undo_stack.append(('baseline', old_y))

    def _draw_baseline(self, y: int):
        """Draw baseline on the image"""
        if self._image_data is None:
            return

        h, w = self._image_data.shape[:2]

        # Remove old baseline
        if self._baseline_line:
            self.scene.removeItem(self._baseline_line)
        if self._baseline_text:
            self.scene.removeItem(self._baseline_text)

        # Draw new baseline
        pen = QPen(QColor("#ff0000"))
        pen.setWidth(2)
        self._baseline_line = self.scene.addLine(0, y, w, y, pen)

        # Add label
        self._baseline_text = self.scene.addText("0 nm (Baseline)", QFont("Arial", 10))
        self._baseline_text.setDefaultTextColor(QColor("#ff0000"))
        self._baseline_text.setPos(10, y - 20)

    def _on_baseline_set(self, y: int):
        """Handle baseline set from click"""
        old_y = self._baseline_y
        self._baseline_y = y
        self._draw_baseline(y)
        self.baseline_btn.setChecked(False)
        self.baseline_changed.emit(y)
        # Save for undo
        self._undo_stack.append(('baseline_set', old_y))

    def _on_baseline_mode(self, enabled: bool):
        """Toggle baseline setting mode"""
        self.view.enable_baseline_mode(enabled)
        if enabled:
            self.baseline_btn.setText("Click Image...")
        else:
            self.baseline_btn.setText("Set Baseline")

    def _on_delete_baseline(self):
        """Delete baseline button handler"""
        self.delete_baseline()

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
        for item in self._result_overlay_items:
            item.setVisible(show)

    def _on_toggle_grid(self, show: bool):
        """Toggle grid overlay"""
        self._show_grid = show
        if show:
            self._draw_grid()
        else:
            self._clear_grid()

    def _draw_grid(self):
        """Draw grid/ruler overlay on the image"""
        if self._image_data is None:
            return

        self._clear_grid()

        h, w = self._image_data.shape[:2]
        scale = self._scale_nm_per_pixel

        # Determine grid spacing based on image size and scale
        image_width_nm = w * scale
        image_height_nm = h * scale

        # Choose nice grid spacing
        grid_spacing_nm = self._get_nice_grid_spacing(image_width_nm / 5)
        grid_spacing_px = grid_spacing_nm / scale

        if grid_spacing_px < 20:
            return  # Grid too dense

        # Grid pen (semi-transparent)
        grid_pen = QPen(QColor(100, 200, 255, 60))
        grid_pen.setWidth(1)
        grid_pen.setStyle(Qt.PenStyle.DotLine)

        # Ruler pen
        ruler_pen = QPen(QColor(100, 200, 255, 120))
        ruler_pen.setWidth(1)

        # Ruler text color
        text_color = QColor(100, 200, 255, 180)

        # Vertical grid lines
        x = 0.0
        while x < w:
            line = self.scene.addLine(x, 0, x, h, grid_pen)
            self._grid_items.append(line)

            # Label
            nm_val = x * scale
            if nm_val >= 1000:
                label = f"{nm_val/1000:.1f}um"
            else:
                label = f"{nm_val:.0f}nm"
            text = self.scene.addText(label, QFont("Arial", 7))
            text.setDefaultTextColor(text_color)
            text.setPos(x + 2, 2)
            self._grid_items.append(text)

            x += grid_spacing_px

        # Horizontal grid lines
        y = 0.0
        while y < h:
            line = self.scene.addLine(0, y, w, y, grid_pen)
            self._grid_items.append(line)

            nm_val = y * scale
            if nm_val >= 1000:
                label = f"{nm_val/1000:.1f}um"
            else:
                label = f"{nm_val:.0f}nm"
            text = self.scene.addText(label, QFont("Arial", 7))
            text.setDefaultTextColor(text_color)
            text.setPos(2, y + 2)
            self._grid_items.append(text)

            y += grid_spacing_px

        # Draw ruler bars on edges
        # Top ruler
        ruler_y = 0
        tick_x = 0.0
        while tick_x < w:
            tick_line = self.scene.addLine(tick_x, ruler_y, tick_x, ruler_y + 8, ruler_pen)
            self._grid_items.append(tick_line)
            tick_x += grid_spacing_px / 5

        # Left ruler
        ruler_x = 0
        tick_y = 0.0
        while tick_y < h:
            tick_line = self.scene.addLine(ruler_x, tick_y, ruler_x + 8, tick_y, ruler_pen)
            self._grid_items.append(tick_line)
            tick_y += grid_spacing_px / 5

    def _get_nice_grid_spacing(self, approximate_nm: float) -> float:
        """Get a 'nice' grid spacing value"""
        nice_values = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]
        for value in nice_values:
            if value >= approximate_nm * 0.5:
                return value
        return nice_values[-1]

    def _clear_grid(self):
        """Remove all grid items"""
        for item in self._grid_items:
            self.scene.removeItem(item)
        self._grid_items = []

    def _on_zoom_in(self):
        self.view.zoom_in()

    def _on_zoom_out(self):
        self.view.zoom_out()

    def _on_fit(self):
        self.view.fit_to_window()

    def zoom_in(self):
        self.view.zoom_in()

    def zoom_out(self):
        self.view.zoom_out()

    def fit_to_window(self):
        self.view.fit_to_window()

    def _on_ocr_scale(self):
        """Auto-detect scale bar from image using OCR"""
        if self._image_data is None:
            return

        try:
            from ...core.scale_bar_detector import ScaleBarDetector
            detector = ScaleBarDetector()

            # Prepare image for detection
            img = self._image_data
            if img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)
            elif img.dtype != np.uint8:
                dmin, dmax = float(img.min()), float(img.max())
                if dmax > dmin:
                    img = ((img - dmin) / (dmax - dmin) * 255).astype(np.uint8)
                else:
                    img = np.zeros_like(img, dtype=np.uint8)

            result = detector.detect_scale_bar(img)
            if result and result['nm_per_pixel']:
                self._scale_nm_per_pixel = result['nm_per_pixel']
                self.info_label.setText(
                    f"{Path(self._image_path).name} | "
                    f"{self._image_data.shape[1]}x{self._image_data.shape[0]} | "
                    f"{self._scale_nm_per_pixel:.4f} nm/px (OCR)"
                )
            else:
                self.info_label.setText(
                    f"{Path(self._image_path).name} | Scale bar not detected"
                )
        except Exception as e:
            self.info_label.setText(f"OCR failed: {e}")

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

        pen = QPen(qcolor)
        pen.setWidth(2)
        line = self.scene.addLine(left_x, y, right_x, y, pen)
        self._measurement_items.append(line)

        for x in [left_x, right_x]:
            marker = self.scene.addEllipse(x - 4, y - 4, 8, 8, pen, QBrush(qcolor))
            self._measurement_items.append(marker)

        text = self.scene.addText(
            f"{depth_nm:.0f}nm: {thickness_nm:.2f}nm",
            QFont("Arial", 9)
        )
        text.setDefaultTextColor(qcolor)
        text.setPos(right_x + 10, y - 8)
        self._measurement_items.append(text)

    def add_result_overlay(self, result: dict):
        """
        Add analysis result as editable overlay on the image.

        Draws measurement lines, edge markers, rulers, and info panel.
        """
        if self._image_data is None:
            return

        self.clear_result_overlays()

        h, w = self._image_data.shape[:2]
        measurements = result.get('measurements', {})
        baseline = result.get('baseline', {})
        baseline_y = baseline.get('y_position', 0)

        # Draw baseline
        if baseline_y > 0:
            pen = QPen(QColor("#ff4444"))
            pen.setWidth(2)
            line = self.scene.addLine(0, baseline_y, w, baseline_y, pen)
            self._result_overlay_items.append(line)

            text = self.scene.addText("Baseline (0nm)", QFont("Arial", 9))
            text.setDefaultTextColor(QColor("#ff4444"))
            text.setPos(10, baseline_y - 18)
            self._result_overlay_items.append(text)

        # Color palette for measurements
        colors = ["#00ff00", "#00ffff", "#ffff00", "#ff00ff", "#ff8800",
                  "#88ff00", "#0088ff", "#ff0088"]

        for i, (depth_key, data) in enumerate(sorted(measurements.items(),
                                                       key=lambda x: float(x[0]))):
            color = QColor(colors[i % len(colors)])
            pen = QPen(color)
            pen.setWidth(2)

            y_pos = data.get('y_position', 0)
            left_x = data.get('left_edge_x', 0)
            right_x = data.get('right_edge_x', 0)
            thickness = data.get('thickness_nm', 0)
            depth = data.get('depth_nm', float(depth_key))

            if y_pos == 0 or left_x == right_x:
                continue

            # Dashed guide line across full width
            dash_pen = QPen(color)
            dash_pen.setWidth(1)
            dash_pen.setStyle(Qt.PenStyle.DashLine)
            guide = self.scene.addLine(0, y_pos, w, y_pos, dash_pen)
            self._result_overlay_items.append(guide)

            # Solid measurement line between edges
            thick_pen = QPen(color)
            thick_pen.setWidth(3)
            mline = self.scene.addLine(left_x, y_pos, right_x, y_pos, thick_pen)
            self._result_overlay_items.append(mline)

            # Edge markers (circles)
            for ex in [left_x, right_x]:
                marker = self.scene.addEllipse(
                    ex - 5, y_pos - 5, 10, 10,
                    pen, QBrush(color)
                )
                self._result_overlay_items.append(marker)

                # Vertical edge indicator
                edge_line = self.scene.addLine(ex, y_pos - 12, ex, y_pos + 12, pen)
                self._result_overlay_items.append(edge_line)

            # Ruler ticks between edges
            num_ticks = 5
            if right_x > left_x:
                tick_spacing = (right_x - left_x) / num_ticks
                thin_pen = QPen(color)
                thin_pen.setWidth(1)
                for t in range(num_ticks + 1):
                    tx = left_x + t * tick_spacing
                    tick = self.scene.addLine(tx, y_pos - 4, tx, y_pos + 4, thin_pen)
                    self._result_overlay_items.append(tick)

            # Label with measurement value
            label_text = f"@{depth:.0f}nm: {thickness:.2f}nm"
            std = data.get('thickness_std', 0)
            if std > 0:
                label_text += f" \u00b1{std:.2f}"

            text = self.scene.addText(label_text, QFont("Arial", 9))
            text.setDefaultTextColor(color)
            text_x = min(right_x + 15, w - 200)
            text.setPos(text_x, y_pos - 8)
            self._result_overlay_items.append(text)

            # Depth ruler from baseline (vertical line)
            if baseline_y > 0 and y_pos > baseline_y:
                depth_pen = QPen(color)
                depth_pen.setWidth(1)
                depth_pen.setStyle(Qt.PenStyle.DashDotLine)
                vline = self.scene.addLine(left_x - 20, baseline_y, left_x - 20, y_pos, depth_pen)
                self._result_overlay_items.append(vline)

                # Depth label
                depth_label = self.scene.addText(f"{depth:.0f}nm", QFont("Arial", 7))
                depth_label.setDefaultTextColor(color)
                depth_label.setPos(left_x - 55, (baseline_y + y_pos) / 2 - 8)
                self._result_overlay_items.append(depth_label)

    def clear_result_overlays(self):
        """Clear all analysis result overlay items"""
        for item in self._result_overlay_items:
            self.scene.removeItem(item)
        self._result_overlay_items = []

    def clear_measurement_overlays(self):
        """Clear all measurement overlays"""
        for item in self._measurement_items:
            self.scene.removeItem(item)
        self._measurement_items = []

    def set_measurement_mode(self, mode: str):
        """Set measurement mode."""
        mode_map = {
            'none': ImageGraphicsView.MODE_PAN,
            'baseline': ImageGraphicsView.MODE_BASELINE,
            'horizontal': ImageGraphicsView.MODE_MEASURE_HORIZONTAL,
            'vertical': ImageGraphicsView.MODE_MEASURE_VERTICAL,
            'free': ImageGraphicsView.MODE_MEASURE_FREE,
        }
        self.view.set_mode(mode_map.get(mode, ImageGraphicsView.MODE_PAN))

    def _on_measurement_ended(self, start: QPointF, end: QPointF):
        """Handle completed measurement"""
        mode_map = {
            ImageGraphicsView.MODE_MEASURE_HORIZONTAL: 'horizontal',
            ImageGraphicsView.MODE_MEASURE_VERTICAL: 'vertical',
            ImageGraphicsView.MODE_MEASURE_FREE: 'free',
        }
        mode = mode_map.get(self.view._mode, 'free')

        # Draw the measurement
        self._draw_manual_measurement(start, end, mode)

        # Save for undo
        self._undo_stack.append(('measurement', len(self._manual_measurement_items)))

        # Emit signal
        self.measurement_completed.emit(start, end, mode)

    def _draw_manual_measurement(self, start: QPointF, end: QPointF,
                                  mode: str, color: str = "#00ff00"):
        """Draw a manual measurement line"""
        qcolor = QColor(color)
        pen = QPen(qcolor)
        pen.setWidth(2)

        # Main line
        line = self.scene.addLine(start.x(), start.y(), end.x(), end.y(), pen)
        self._manual_measurement_items.append(line)

        # End markers
        marker_size = 6
        for point in [start, end]:
            marker = self.scene.addEllipse(
                point.x() - marker_size/2,
                point.y() - marker_size/2,
                marker_size, marker_size,
                pen, QBrush(qcolor)
            )
            self._manual_measurement_items.append(marker)

        # Calculate distance
        dx = end.x() - start.x()
        dy = end.y() - start.y()

        if mode == 'horizontal':
            distance_px = abs(dx)
        elif mode == 'vertical':
            distance_px = abs(dy)
        else:
            distance_px = np.sqrt(dx**2 + dy**2)

        distance_nm = distance_px * self._scale_nm_per_pixel

        # Label
        mid_x = (start.x() + end.x()) / 2
        mid_y = (start.y() + end.y()) / 2

        text = self.scene.addText(f"{distance_nm:.2f} nm", QFont("Arial", 9))
        text.setDefaultTextColor(qcolor)
        text.setPos(mid_x + 5, mid_y - 15)
        self._manual_measurement_items.append(text)

    def add_manual_measurement(self, start_x: float, start_y: float,
                               end_x: float, end_y: float,
                               label: str = "", color: str = "#00ff00"):
        """Add a manual measurement visualization"""
        start = QPointF(start_x, start_y)
        end = QPointF(end_x, end_y)
        self._draw_manual_measurement(start, end, 'free', color)

    def clear_manual_measurements(self):
        """Clear all manual measurement overlays"""
        for item in self._manual_measurement_items:
            self.scene.removeItem(item)
        self._manual_measurement_items = []

    def undo(self):
        """Undo last action"""
        if not self._undo_stack:
            return

        action = self._undo_stack.pop()

        if action[0] == 'measurement':
            # Remove last measurement items (4 items per measurement: line + 2 markers + text)
            items_to_remove = min(4, len(self._manual_measurement_items))
            for _ in range(items_to_remove):
                if self._manual_measurement_items:
                    item = self._manual_measurement_items.pop()
                    self.scene.removeItem(item)

        elif action[0] == 'baseline_set':
            old_y = action[1]
            if old_y is not None:
                self._baseline_y = old_y
                self._draw_baseline(old_y)
            else:
                self.delete_baseline()

        elif action[0] == 'baseline':
            # Restore deleted baseline
            old_y = action[1]
            if old_y is not None:
                self._baseline_y = old_y
                self._draw_baseline(old_y)

    def rotate_image(self, angle: float):
        """Rotate the displayed image."""
        if self._image_data is None:
            return

        import cv2

        self._current_rotation = angle

        h, w = self._image_data.shape[:2]
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

        rotated = cv2.warpAffine(
            self._image_data, matrix, (w, h),
            borderMode=cv2.BORDER_REFLECT
        )

        self._update_display(rotated)
        self.image_rotated.emit(angle)

    def _update_display(self, data: np.ndarray):
        """Update the displayed image"""
        self._display_data = data

        if len(data.shape) == 3:
            display_data = data[:, :, :3] if data.shape[2] >= 3 else data[:, :, 0]
        else:
            display_data = data

        if display_data.dtype != np.uint8:
            dmin, dmax = float(display_data.min()), float(display_data.max())
            if dmax > dmin:
                display_data = ((display_data - dmin) / (dmax - dmin) * 255).astype(np.uint8)
            else:
                display_data = np.zeros_like(display_data, dtype=np.uint8)

        h, w = display_data.shape[:2]
        if len(display_data.shape) == 2:
            qimage = QImage(
                display_data.data, w, h, w,
                QImage.Format.Format_Grayscale8
            )
        else:
            bytes_per_line = 3 * w
            qimage = QImage(
                display_data.data, w, h, bytes_per_line,
                QImage.Format.Format_RGB888
            )

        # Update scene
        for item in self.scene.items():
            if isinstance(item, QGraphicsPixmapItem):
                item.setPixmap(QPixmap.fromImage(qimage))
                break
        else:
            self.scene.clear()
            self._baseline_line = None
            self._baseline_text = None
            self._measurement_items = []
            self._manual_measurement_items = []
            self._result_overlay_items = []
            self._grid_items = []

            pixmap = QPixmap.fromImage(qimage)
            self.scene.addPixmap(pixmap)
            self.scene.setSceneRect(QRectF(pixmap.rect()))

        if self._baseline_y is not None:
            self._draw_baseline(self._baseline_y)

    def get_scale_nm_per_pixel(self) -> float:
        return self._scale_nm_per_pixel

    def set_scale_nm_per_pixel(self, scale: float):
        self._scale_nm_per_pixel = scale
        if self._image_path:
            self.info_label.setText(
                f"{Path(self._image_path).name} | "
                f"{self._image_data.shape[1]}x{self._image_data.shape[0]} | "
                f"{scale:.4f} nm/px"
            )

    def get_image_data(self) -> Optional[np.ndarray]:
        return self._display_data if self._display_data is not None else self._image_data

    def get_current_rotation(self) -> float:
        return self._current_rotation
