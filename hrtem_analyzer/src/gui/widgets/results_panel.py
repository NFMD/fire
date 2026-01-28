"""
Results Panel Widget for HR-TEM Analyzer

Displays analysis results with visualization.
Enhanced to support Gatan DM-style metrics.
"""
from pathlib import Path
from typing import List, Dict, Any, Optional

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QTableWidget, QTableWidgetItem, QHeaderView,
    QPushButton, QTabWidget, QScrollArea, QFrame,
    QGridLayout, QSplitter, QGroupBox, QFormLayout
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor, QPixmap, QImage, QPainter, QPen

try:
    import numpy as np
except ImportError:
    np = None


class ResultCard(QFrame):
    """Single result card widget"""

    clicked = pyqtSignal(dict)

    def __init__(self, result: Dict[str, Any]):
        super().__init__()
        self.result = result
        self._setup_ui()

    def _setup_ui(self):
        """Setup the card UI"""
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setStyleSheet("""
            ResultCard {
                background-color: #2d2d2d;
                border: 1px solid #454545;
                border-radius: 8px;
                padding: 8px;
            }
            ResultCard:hover {
                border: 1px solid #007acc;
            }
        """)

        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        # Filename
        filename = Path(self.result.get('source_path', 'Unknown')).name
        name_label = QLabel(filename)
        name_label.setStyleSheet("font-weight: bold; font-size: 12px;")
        layout.addWidget(name_label)

        # Status
        success = self.result.get('success', False)
        status_text = "✓ Success" if success else "✗ Failed"
        status_color = "#5cb85c" if success else "#d9534f"
        status_label = QLabel(status_text)
        status_label.setStyleSheet(f"color: {status_color};")
        layout.addWidget(status_label)

        # Analysis mode indicator
        if success:
            enhanced = self.result.get('enhanced_analysis', False)
            mode_text = "Enhanced" if enhanced else "Standard"
            mode_color = "#17a2b8" if enhanced else "#6c757d"
            mode_label = QLabel(f"[{mode_text}]")
            mode_label.setStyleSheet(f"color: {mode_color}; font-size: 10px;")
            layout.addWidget(mode_label)

        # Measurements summary
        if success and 'measurements' in self.result:
            measurements = self.result['measurements']
            for depth, m in sorted(measurements.items()):
                depth_val = float(depth) if isinstance(depth, str) else depth

                # Basic measurement
                text = f"@{depth_val:.0f}nm: {m['thickness_nm']:.2f} ± {m['thickness_std']:.2f} nm"
                m_label = QLabel(text)
                m_label.setStyleSheet("color: #888888; font-size: 11px;")
                layout.addWidget(m_label)

                # Show CI95 if available (enhanced mode)
                if 'ci_95_low' in m and 'ci_95_high' in m:
                    ci_text = f"  CI95: [{m['ci_95_low']:.2f}, {m['ci_95_high']:.2f}]"
                    ci_label = QLabel(ci_text)
                    ci_label.setStyleSheet("color: #17a2b8; font-size: 10px;")
                    layout.addWidget(ci_label)

        elif not success:
            error = self.result.get('error', 'Unknown error')
            error_label = QLabel(error[:50] + '...' if len(error) > 50 else error)
            error_label.setStyleSheet("color: #d9534f; font-size: 11px;")
            layout.addWidget(error_label)

    def mousePressEvent(self, event):
        """Handle mouse press"""
        self.clicked.emit(self.result)
        super().mousePressEvent(event)


class ResultsPanel(QWidget):
    """
    Results panel showing analysis results.

    Features:
    - Result cards grid
    - Detailed view for selected result
    - Summary statistics
    - Export options
    """

    result_selected = pyqtSignal(dict)

    def __init__(self):
        super().__init__()
        self._results: List[Dict] = []
        self._setup_ui()

    def _setup_ui(self):
        """Setup the widget UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        # Header with summary
        header = QHBoxLayout()

        self.summary_label = QLabel("No results")
        self.summary_label.setObjectName("titleLabel")
        header.addWidget(self.summary_label)

        header.addStretch()

        self.export_btn = QPushButton("Export All")
        self.export_btn.setEnabled(False)
        header.addWidget(self.export_btn)

        layout.addLayout(header)

        # Main content with splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left: Results list
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)

        # Scroll area for result cards
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        self.cards_container = QWidget()
        self.cards_layout = QGridLayout(self.cards_container)
        self.cards_layout.setSpacing(8)

        scroll.setWidget(self.cards_container)
        left_layout.addWidget(scroll)

        splitter.addWidget(left_panel)

        # Right: Detail view
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)

        detail_label = QLabel("Result Details")
        detail_label.setObjectName("titleLabel")
        right_layout.addWidget(detail_label)

        # Image preview
        self.preview_label = QLabel("Select a result to view details")
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setMinimumHeight(200)
        self.preview_label.setStyleSheet("""
            background-color: #1e1e1e;
            border: 1px solid #454545;
            border-radius: 4px;
        """)
        right_layout.addWidget(self.preview_label)

        # Detail table
        self.detail_table = QTableWidget()
        self.detail_table.setColumnCount(2)
        self.detail_table.setHorizontalHeaderLabels(['Property', 'Value'])
        self.detail_table.horizontalHeader().setStretchLastSection(True)
        self.detail_table.verticalHeader().setVisible(False)
        right_layout.addWidget(self.detail_table)

        splitter.addWidget(right_panel)
        splitter.setSizes([400, 300])

        layout.addWidget(splitter)

        # Statistics
        stats_group = QWidget()
        stats_layout = QHBoxLayout(stats_group)
        stats_layout.setContentsMargins(0, 8, 0, 0)

        self.stats_labels = {}
        for name in ['Total', 'Success', 'Failed', 'Avg Thickness']:
            label = QLabel(f"{name}: -")
            label.setStyleSheet("color: #888888;")
            self.stats_labels[name] = label
            stats_layout.addWidget(label)

        stats_layout.addStretch()
        layout.addWidget(stats_group)

    def set_results(self, results: List[Dict]):
        """Set results to display"""
        self._results = results

        # Clear existing cards
        while self.cards_layout.count():
            child = self.cards_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        # Add result cards
        cols = 2
        for i, result in enumerate(results):
            card = ResultCard(result)
            card.clicked.connect(self._on_result_clicked)
            self.cards_layout.addWidget(card, i // cols, i % cols)

        # Update summary
        successful = sum(1 for r in results if r.get('success', False))
        self.summary_label.setText(f"Results: {successful}/{len(results)} successful")
        self.export_btn.setEnabled(len(results) > 0)

        # Update statistics
        self._update_statistics()

    def _update_statistics(self):
        """Update statistics labels"""
        total = len(self._results)
        successful = sum(1 for r in self._results if r.get('success', False))
        failed = total - successful

        self.stats_labels['Total'].setText(f"Total: {total}")
        self.stats_labels['Success'].setText(f"Success: {successful}")
        self.stats_labels['Failed'].setText(f"Failed: {failed}")

        # Calculate average thickness at first depth
        if successful > 0:
            thicknesses = []
            for r in self._results:
                if r.get('success') and r.get('measurements'):
                    first_depth = sorted(r['measurements'].keys())[0]
                    thicknesses.append(r['measurements'][first_depth]['thickness_nm'])

            if thicknesses and np is not None:
                avg = np.mean(thicknesses)
                std = np.std(thicknesses)
                self.stats_labels['Avg Thickness'].setText(
                    f"Avg Thickness: {avg:.2f} ± {std:.2f} nm"
                )

    def _on_result_clicked(self, result: Dict):
        """Handle result card click"""
        self._show_detail(result)
        self.result_selected.emit(result)

    def _show_detail(self, result: Dict):
        """Show detail view for result"""
        # Update preview
        output = result.get('output', {})
        jpeg_path = output.get('jpeg_path')

        if jpeg_path and Path(jpeg_path).exists():
            pixmap = QPixmap(jpeg_path)
            scaled = pixmap.scaled(
                self.preview_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            self.preview_label.setPixmap(scaled)
        else:
            self.preview_label.setText("No preview available")

        # Update detail table
        self.detail_table.setRowCount(0)

        rows = [
            ('File', Path(result.get('source_path', '')).name),
            ('Status', 'Success' if result.get('success') else 'Failed'),
        ]

        if result.get('success'):
            # Analysis mode
            enhanced = result.get('enhanced_analysis', False)
            rows.append(('Analysis Mode', 'Enhanced (Gatan DM)' if enhanced else 'Standard'))

            scale_info = result.get('scale_info', {})
            rows.append(('Scale', f"{scale_info.get('scale_nm_per_pixel', 0):.4f} nm/px"))

            # FFT calibration info if available
            if result.get('fft_calibration_applied'):
                fft_scale = result.get('fft_calibrated_scale', 0)
                rows.append(('FFT Calibrated Scale', f"{fft_scale:.4f} nm/px"))

            baseline = result.get('baseline', {})
            rows.append(('Baseline Y', str(baseline.get('y_position', '-'))))

            # Measurements
            if 'measurements' in result:
                for depth, m in sorted(result['measurements'].items()):
                    depth_val = float(depth) if isinstance(depth, str) else depth

                    # Basic measurement
                    rows.append((
                        f'Thickness @{depth_val:.0f}nm',
                        f"{m['thickness_nm']:.2f} ± {m['thickness_std']:.2f} nm"
                    ))

                    # Enhanced metrics
                    if 'ci_95_low' in m and 'ci_95_high' in m:
                        rows.append((
                            f'  95% CI @{depth_val:.0f}nm',
                            f"[{m['ci_95_low']:.2f}, {m['ci_95_high']:.2f}] nm"
                        ))

                    if 'fwhm_nm' in m:
                        rows.append((
                            f'  FWHM @{depth_val:.0f}nm',
                            f"{m['fwhm_nm']:.2f} nm"
                        ))

                    # Method breakdown if available
                    if 'method_results' in m:
                        method_str = ', '.join(
                            f"{k}: {v:.2f}" for k, v in m['method_results'].items()
                        )
                        rows.append((f'  Methods @{depth_val:.0f}nm', method_str))

            # FFT analysis results
            fft_analysis = result.get('fft_analysis', {})
            if fft_analysis:
                if 'dominant_spacing_nm' in fft_analysis:
                    rows.append((
                        'FFT Dominant Spacing',
                        f"{fft_analysis['dominant_spacing_nm']:.3f} nm"
                    ))
                if 'lattice_constant_nm' in fft_analysis:
                    rows.append((
                        'FFT Lattice Constant',
                        f"{fft_analysis['lattice_constant_nm']:.3f} nm"
                    ))

            # Precision measurement results
            if 'measurements' in result:
                for depth, m in sorted(result['measurements'].items()):
                    depth_val = float(depth) if isinstance(depth, str) else depth

                    # ESF/LSF analysis
                    if 'esf_width_nm' in m and m['esf_width_nm']:
                        rows.append((
                            f'  ESF Width @{depth_val:.0f}nm',
                            f"{m['esf_width_nm']:.3f} nm"
                        ))

                    if 'lsf_fwhm_nm' in m and m['lsf_fwhm_nm']:
                        rows.append((
                            f'  LSF FWHM @{depth_val:.0f}nm',
                            f"{m['lsf_fwhm_nm']:.3f} nm"
                        ))

                    # SNR
                    if 'snr_db' in m and m['snr_db']:
                        rows.append((
                            f'  SNR @{depth_val:.0f}nm',
                            f"{m['snr_db']:.1f} dB"
                        ))

                    # Monte Carlo uncertainty
                    if 'monte_carlo_std' in m and m['monte_carlo_std']:
                        rows.append((
                            f'  MC Uncertainty @{depth_val:.0f}nm',
                            f"± {m['monte_carlo_std']:.3f} nm"
                        ))

                    # Atomic spacing (crystalline materials)
                    if 'atomic_spacing_nm' in m and m['atomic_spacing_nm']:
                        rows.append((
                            f'  Atomic Spacing @{depth_val:.0f}nm',
                            f"{m['atomic_spacing_nm']:.3f} nm"
                        ))

                    # Precision confidence
                    if 'precision_confidence' in m and m['precision_confidence']:
                        rows.append((
                            f'  Precision Conf @{depth_val:.0f}nm',
                            f"{m['precision_confidence']:.1%}"
                        ))

        else:
            rows.append(('Error', result.get('error', 'Unknown')))

        self.detail_table.setRowCount(len(rows))
        for i, (prop, value) in enumerate(rows):
            self.detail_table.setItem(i, 0, QTableWidgetItem(prop))
            self.detail_table.setItem(i, 1, QTableWidgetItem(str(value)))

    def clear(self):
        """Clear all results"""
        self._results = []
        while self.cards_layout.count():
            child = self.cards_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        self.summary_label.setText("No results")
        self.export_btn.setEnabled(False)
        self.preview_label.setText("Select a result to view details")
        self.detail_table.setRowCount(0)
