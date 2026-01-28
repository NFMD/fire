"""
Measurement Table Widget for HR-TEM Analyzer

Displays measurement results in tabular format.
Enhanced to support Gatan DM-style metrics.
"""
from pathlib import Path
from typing import List, Dict, Any, Optional

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QTableWidget, QTableWidgetItem, QHeaderView,
    QPushButton, QComboBox, QCheckBox, QMenu
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor, QAction


class MeasurementTableWidget(QWidget):
    """
    Table widget for displaying measurements.

    Features:
    - Sortable columns
    - Filter by depth
    - Export selected
    - Color coding for confidence
    """

    row_selected = pyqtSignal(dict)

    def __init__(self):
        super().__init__()
        self._results: List[Dict] = []
        self._depths: List[float] = []
        self._setup_ui()

    def _setup_ui(self):
        """Setup the widget UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Controls
        controls = QHBoxLayout()

        # Depth filter
        controls.addWidget(QLabel("Depth:"))
        self.depth_combo = QComboBox()
        self.depth_combo.addItem("All")
        self.depth_combo.currentIndexChanged.connect(self._on_filter_changed)
        controls.addWidget(self.depth_combo)

        controls.addStretch()

        # Show statistics
        self.stats_cb = QCheckBox("Show Stats")
        self.stats_cb.setChecked(True)
        self.stats_cb.toggled.connect(self._on_filter_changed)
        controls.addWidget(self.stats_cb)

        # Enhanced mode toggle
        self.enhanced_cb = QCheckBox("Show Enhanced")
        self.enhanced_cb.setChecked(True)
        self.enhanced_cb.setToolTip("Show enhanced metrics (CI95, FWHM)")
        self.enhanced_cb.toggled.connect(self._on_enhanced_toggled)
        controls.addWidget(self.enhanced_cb)

        layout.addLayout(controls)

        # Table - extended columns for enhanced metrics
        self.table = QTableWidget()
        self._base_columns = ['File', 'Depth (nm)', 'Thickness (nm)', 'Std (nm)', 'Confidence', 'N']
        self._enhanced_columns = ['CI95 Low', 'CI95 High', 'Mode']
        self._all_columns = self._base_columns + self._enhanced_columns

        self.table.setColumnCount(len(self._all_columns))
        self.table.setHorizontalHeaderLabels(self._all_columns)

        # Configure header
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for i in range(1, len(self._all_columns)):
            header.setSectionResizeMode(i, QHeaderView.ResizeMode.ResizeToContents)

        self.table.setSortingEnabled(True)
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self._show_context_menu)
        self.table.itemClicked.connect(self._on_item_clicked)

        layout.addWidget(self.table)

        # Initially show enhanced columns
        self._on_enhanced_toggled(True)

        # Summary
        self.summary_label = QLabel("No measurements")
        self.summary_label.setObjectName("subtitleLabel")
        layout.addWidget(self.summary_label)

    def add_result(self, result: Dict):
        """Add a single result to the table"""
        if not result.get('success'):
            return

        self._results.append(result)
        measurements = result.get('measurements', {})
        enhanced_mode = result.get('enhanced_analysis', False)

        for depth, m in measurements.items():
            depth_val = float(depth) if isinstance(depth, str) else depth

            # Track depths for filter
            if depth_val not in self._depths:
                self._depths.append(depth_val)
                self._depths.sort()
                self._update_depth_filter()

            # Add row
            row = self.table.rowCount()
            self.table.insertRow(row)

            # File
            filename = Path(result.get('source_path', '')).name
            item = QTableWidgetItem(filename)
            item.setData(Qt.ItemDataRole.UserRole, result)
            self.table.setItem(row, 0, item)

            # Depth
            self.table.setItem(row, 1, self._create_number_item(depth_val, '{:.1f}'))

            # Thickness
            self.table.setItem(row, 2, self._create_number_item(m['thickness_nm'], '{:.2f}'))

            # Std
            std_item = self._create_number_item(m['thickness_std'], '{:.2f}')
            if m['thickness_std'] > m['thickness_nm'] * 0.1:  # High variance warning
                std_item.setForeground(QColor('#f0ad4e'))
            self.table.setItem(row, 3, std_item)

            # Confidence
            conf_item = self._create_number_item(m['confidence'], '{:.2f}')
            if m['confidence'] < 0.5:
                conf_item.setForeground(QColor('#d9534f'))
            elif m['confidence'] > 0.8:
                conf_item.setForeground(QColor('#5cb85c'))
            self.table.setItem(row, 4, conf_item)

            # N (number of measurements)
            self.table.setItem(row, 5, self._create_number_item(m['num_measurements'], '{:.0f}'))

            # Enhanced metrics: CI95
            if 'ci_95_low' in m:
                ci_low_item = self._create_number_item(m['ci_95_low'], '{:.2f}')
                ci_low_item.setForeground(QColor('#17a2b8'))
                self.table.setItem(row, 6, ci_low_item)
            else:
                self.table.setItem(row, 6, QTableWidgetItem('-'))

            if 'ci_95_high' in m:
                ci_high_item = self._create_number_item(m['ci_95_high'], '{:.2f}')
                ci_high_item.setForeground(QColor('#17a2b8'))
                self.table.setItem(row, 7, ci_high_item)
            else:
                self.table.setItem(row, 7, QTableWidgetItem('-'))

            # Analysis mode
            mode_item = QTableWidgetItem('Enhanced' if enhanced_mode else 'Standard')
            mode_item.setForeground(QColor('#17a2b8') if enhanced_mode else QColor('#6c757d'))
            self.table.setItem(row, 8, mode_item)

        self._update_summary()

    def _create_number_item(self, value: float, fmt: str) -> QTableWidgetItem:
        """Create a right-aligned number item"""
        item = QTableWidgetItem(fmt.format(value))
        item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        item.setData(Qt.ItemDataRole.UserRole + 1, value)  # Store raw value for sorting
        return item

    def _update_depth_filter(self):
        """Update depth filter combo box"""
        current = self.depth_combo.currentText()
        self.depth_combo.clear()
        self.depth_combo.addItem("All")
        for d in self._depths:
            self.depth_combo.addItem(f"{d:.1f}")

        # Restore selection
        idx = self.depth_combo.findText(current)
        if idx >= 0:
            self.depth_combo.setCurrentIndex(idx)

    def _on_filter_changed(self):
        """Handle filter change"""
        selected_depth = self.depth_combo.currentText()

        for row in range(self.table.rowCount()):
            depth_item = self.table.item(row, 1)
            if depth_item:
                show = (selected_depth == "All" or
                        depth_item.text() == selected_depth)
                self.table.setRowHidden(row, not show)

        self._update_summary()

    def _on_item_clicked(self, item: QTableWidgetItem):
        """Handle item click"""
        row = item.row()
        first_item = self.table.item(row, 0)
        if first_item:
            result = first_item.data(Qt.ItemDataRole.UserRole)
            if result:
                self.row_selected.emit(result)

    def _update_summary(self):
        """Update summary label"""
        visible_rows = sum(
            1 for row in range(self.table.rowCount())
            if not self.table.isRowHidden(row)
        )

        if visible_rows == 0:
            self.summary_label.setText("No measurements")
            return

        # Calculate statistics from visible rows
        thicknesses = []
        for row in range(self.table.rowCount()):
            if not self.table.isRowHidden(row):
                item = self.table.item(row, 2)
                if item:
                    thicknesses.append(item.data(Qt.ItemDataRole.UserRole + 1))

        if thicknesses:
            import statistics
            mean = statistics.mean(thicknesses)
            std = statistics.stdev(thicknesses) if len(thicknesses) > 1 else 0

            self.summary_label.setText(
                f"{visible_rows} measurements | "
                f"Mean: {mean:.2f} nm | Std: {std:.2f} nm"
            )

    def clear(self):
        """Clear all measurements"""
        self.table.setRowCount(0)
        self._results = []
        self._depths = []
        self.depth_combo.clear()
        self.depth_combo.addItem("All")
        self.summary_label.setText("No measurements")

    def get_all_data(self) -> List[Dict]:
        """Get all measurement data"""
        data = []
        for row in range(self.table.rowCount()):
            row_data = {
                'file': self.table.item(row, 0).text() if self.table.item(row, 0) else '',
                'depth_nm': self.table.item(row, 1).data(Qt.ItemDataRole.UserRole + 1) if self.table.item(row, 1) else 0,
                'thickness_nm': self.table.item(row, 2).data(Qt.ItemDataRole.UserRole + 1) if self.table.item(row, 2) else 0,
                'std_nm': self.table.item(row, 3).data(Qt.ItemDataRole.UserRole + 1) if self.table.item(row, 3) else 0,
                'confidence': self.table.item(row, 4).data(Qt.ItemDataRole.UserRole + 1) if self.table.item(row, 4) else 0,
                'n_measurements': self.table.item(row, 5).data(Qt.ItemDataRole.UserRole + 1) if self.table.item(row, 5) else 0,
            }

            # Add enhanced metrics if available
            ci_low_item = self.table.item(row, 6)
            if ci_low_item and ci_low_item.text() != '-':
                row_data['ci_95_low'] = ci_low_item.data(Qt.ItemDataRole.UserRole + 1)

            ci_high_item = self.table.item(row, 7)
            if ci_high_item and ci_high_item.text() != '-':
                row_data['ci_95_high'] = ci_high_item.data(Qt.ItemDataRole.UserRole + 1)

            mode_item = self.table.item(row, 8)
            if mode_item:
                row_data['analysis_mode'] = mode_item.text()

            data.append(row_data)
        return data

    def _on_enhanced_toggled(self, checked: bool):
        """Toggle enhanced column visibility"""
        for i in range(6, len(self._all_columns)):
            self.table.setColumnHidden(i, not checked)

    def _show_context_menu(self, position):
        """Show context menu for table"""
        menu = QMenu(self)

        # Export selected action
        export_action = menu.addAction("Export Selected")
        export_action.triggered.connect(self._export_selected)

        # Copy to clipboard action
        copy_action = menu.addAction("Copy to Clipboard")
        copy_action.triggered.connect(self._copy_to_clipboard)

        menu.exec(self.table.viewport().mapToGlobal(position))

    def _export_selected(self):
        """Export selected rows"""
        selected_rows = set()
        for item in self.table.selectedItems():
            selected_rows.add(item.row())

        if not selected_rows:
            return

        # Collect data from selected rows
        data = []
        for row in sorted(selected_rows):
            row_data = {}
            for col in range(self.table.columnCount()):
                if not self.table.isColumnHidden(col):
                    item = self.table.item(row, col)
                    if item:
                        row_data[self._all_columns[col]] = item.text()
            data.append(row_data)

        # Signal that export is requested (parent can handle)
        print(f"Export requested for {len(data)} rows")

    def _copy_to_clipboard(self):
        """Copy selected data to clipboard"""
        from PyQt6.QtWidgets import QApplication

        selected_rows = set()
        for item in self.table.selectedItems():
            selected_rows.add(item.row())

        if not selected_rows:
            return

        # Build tab-separated text
        lines = []

        # Header
        headers = [self._all_columns[i] for i in range(len(self._all_columns))
                   if not self.table.isColumnHidden(i)]
        lines.append('\t'.join(headers))

        # Data
        for row in sorted(selected_rows):
            row_data = []
            for col in range(self.table.columnCount()):
                if not self.table.isColumnHidden(col):
                    item = self.table.item(row, col)
                    row_data.append(item.text() if item else '')
            lines.append('\t'.join(row_data))

        clipboard = QApplication.clipboard()
        clipboard.setText('\n'.join(lines))
