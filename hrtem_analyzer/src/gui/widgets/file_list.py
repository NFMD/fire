"""
File List Widget for HR-TEM Analyzer

Displays list of loaded images with drag-and-drop support.
"""
from pathlib import Path
from typing import List, Dict, Optional

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QListWidget, QListWidgetItem, QPushButton,
    QAbstractItemView, QMenu
)
from PyQt6.QtCore import Qt, pyqtSignal, QMimeData
from PyQt6.QtGui import QDragEnterEvent, QDropEvent, QColor, QIcon


class FileListWidget(QWidget):
    """
    File list widget with drag-and-drop support.

    Features:
    - Display loaded image files
    - Drag and drop files
    - Select multiple files
    - Show processing status
    - Context menu for file operations
    """

    file_selected = pyqtSignal(str)  # path
    files_dropped = pyqtSignal(list)  # list of paths
    files_removed = pyqtSignal(list)  # list of paths

    STATUS_COLORS = {
        'pending': '#888888',
        'processing': '#f0ad4e',
        'completed': '#5cb85c',
        'failed': '#d9534f',
    }

    def __init__(self):
        super().__init__()
        self._files: Dict[str, Dict] = {}  # path -> {status, ...}
        self._setup_ui()

    def _setup_ui(self):
        """Setup the widget UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Header
        header = QHBoxLayout()

        title = QLabel("Images")
        title.setObjectName("titleLabel")
        header.addWidget(title)

        header.addStretch()

        self.count_label = QLabel("0 files")
        self.count_label.setObjectName("subtitleLabel")
        header.addWidget(self.count_label)

        layout.addLayout(header)

        # File list
        self.list_widget = QListWidget()
        self.list_widget.setSelectionMode(
            QAbstractItemView.SelectionMode.ExtendedSelection
        )
        self.list_widget.setDragDropMode(
            QAbstractItemView.DragDropMode.DropOnly
        )
        self.list_widget.setAcceptDrops(True)
        self.list_widget.itemClicked.connect(self._on_item_clicked)
        self.list_widget.itemDoubleClicked.connect(self._on_item_double_clicked)
        self.list_widget.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.list_widget.customContextMenuRequested.connect(self._show_context_menu)

        # Enable drag and drop
        self.list_widget.dragEnterEvent = self._drag_enter_event
        self.list_widget.dropEvent = self._drop_event

        layout.addWidget(self.list_widget)

        # Buttons
        btn_layout = QHBoxLayout()

        self.clear_btn = QPushButton("Clear All")
        self.clear_btn.setObjectName("secondaryButton")
        self.clear_btn.clicked.connect(self._on_clear_all)
        btn_layout.addWidget(self.clear_btn)

        self.remove_btn = QPushButton("Remove Selected")
        self.remove_btn.setObjectName("secondaryButton")
        self.remove_btn.clicked.connect(self._on_remove_selected)
        btn_layout.addWidget(self.remove_btn)

        layout.addLayout(btn_layout)

    def add_files(self, paths: List[str]):
        """Add files to the list"""
        for path in paths:
            if path not in self._files:
                self._files[path] = {'status': 'pending'}

                item = QListWidgetItem(Path(path).name)
                item.setData(Qt.ItemDataRole.UserRole, path)
                item.setToolTip(path)
                self._update_item_status(item, 'pending')

                self.list_widget.addItem(item)

        self._update_count()

    def remove_files(self, paths: List[str]):
        """Remove files from the list"""
        for path in paths:
            if path in self._files:
                del self._files[path]

                # Find and remove item
                for i in range(self.list_widget.count()):
                    item = self.list_widget.item(i)
                    if item.data(Qt.ItemDataRole.UserRole) == path:
                        self.list_widget.takeItem(i)
                        break

        self._update_count()
        self.files_removed.emit(paths)

    def clear_all(self):
        """Clear all files"""
        paths = list(self._files.keys())
        self._files.clear()
        self.list_widget.clear()
        self._update_count()
        self.files_removed.emit(paths)

    def get_all_files(self) -> List[str]:
        """Get all file paths"""
        return list(self._files.keys())

    def get_selected_files(self) -> List[str]:
        """Get selected file paths"""
        return [
            item.data(Qt.ItemDataRole.UserRole)
            for item in self.list_widget.selectedItems()
        ]

    def set_file_status(self, path: str, status: str):
        """Set status for a file"""
        if path in self._files:
            self._files[path]['status'] = status

            # Update item appearance
            for i in range(self.list_widget.count()):
                item = self.list_widget.item(i)
                if item.data(Qt.ItemDataRole.UserRole) == path:
                    self._update_item_status(item, status)
                    break

    def _update_item_status(self, item: QListWidgetItem, status: str):
        """Update item appearance based on status"""
        color = self.STATUS_COLORS.get(status, '#888888')

        # Status indicator prefix
        prefix = {
            'pending': '○',
            'processing': '◐',
            'completed': '●',
            'failed': '✕',
        }.get(status, '○')

        path = item.data(Qt.ItemDataRole.UserRole)
        item.setText(f"{prefix} {Path(path).name}")
        item.setForeground(QColor(color))

    def _update_count(self):
        """Update file count label"""
        count = len(self._files)
        self.count_label.setText(f"{count} file{'s' if count != 1 else ''}")

    def _on_item_clicked(self, item: QListWidgetItem):
        """Handle item click"""
        path = item.data(Qt.ItemDataRole.UserRole)
        self.file_selected.emit(path)

    def _on_item_double_clicked(self, item: QListWidgetItem):
        """Handle item double click"""
        path = item.data(Qt.ItemDataRole.UserRole)
        self.file_selected.emit(path)

    def _on_clear_all(self):
        """Clear all button clicked"""
        self.clear_all()

    def _on_remove_selected(self):
        """Remove selected button clicked"""
        selected = self.get_selected_files()
        if selected:
            self.remove_files(selected)

    def _show_context_menu(self, pos):
        """Show context menu"""
        item = self.list_widget.itemAt(pos)
        if not item:
            return

        menu = QMenu(self)

        open_action = menu.addAction("Open")
        open_action.triggered.connect(
            lambda: self.file_selected.emit(item.data(Qt.ItemDataRole.UserRole))
        )

        menu.addSeparator()

        remove_action = menu.addAction("Remove")
        remove_action.triggered.connect(
            lambda: self.remove_files([item.data(Qt.ItemDataRole.UserRole)])
        )

        menu.exec(self.list_widget.mapToGlobal(pos))

    def _drag_enter_event(self, event: QDragEnterEvent):
        """Handle drag enter"""
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def _drop_event(self, event: QDropEvent):
        """Handle drop"""
        if event.mimeData().hasUrls():
            paths = []
            for url in event.mimeData().urls():
                path = url.toLocalFile()
                if path.lower().endswith(('.tif', '.tiff')):
                    paths.append(path)

            if paths:
                self.files_dropped.emit(paths)
                event.acceptProposedAction()
            else:
                event.ignore()
        else:
            event.ignore()
