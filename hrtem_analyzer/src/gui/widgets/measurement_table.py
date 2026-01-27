"""
Measurement Table Widget for HR-TEM Analyzer

Displays measurement results in tabular format.
"""
from pathlib import Path
from typing import List, Dict, Any, Optional

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QTableWidget, QTableWidgetItem, QHeaderView,
    QPushButton, QComboBox, QCheckBox
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor


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

        layout.addLayout(controls)

        # Table
        self.table = QTableWidget()
        self.table.setColumnCount(6)
        self.table.setHorizontalHeaderLabels([
            'File', 'Depth (nm)', 'Thickness (nm)', 'Std (nm)', 'Confidence', 'N'
        ])

        # Configure header
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(5, QHeaderView.ResizeMode.ResizeToContents)

        self.table.setSortingEnabled(True)
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.itemClicked.connect(self._on_item_clicked)

        layout.addWidget(self.table)

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
            data.append(row_data)
        return data
