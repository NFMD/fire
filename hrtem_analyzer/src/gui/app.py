"""
HR-TEM Analyzer Application Entry Point
"""
import sys
from pathlib import Path

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont

from .main_window import MainWindow


def run_app():
    """Run the HR-TEM Analyzer GUI application"""
    # Enable high DPI scaling
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    app = QApplication(sys.argv)

    # Set application info
    app.setApplicationName("HR-TEM Analyzer")
    app.setApplicationVersion("0.1.0")
    app.setOrganizationName("HR-TEM Analysis")

    # Set default font
    font = QFont("Segoe UI", 10)
    app.setFont(font)

    # Apply modern stylesheet
    app.setStyleSheet(get_stylesheet())

    # Create and show main window
    window = MainWindow()
    window.show()

    sys.exit(app.exec())


def get_stylesheet() -> str:
    """Get application stylesheet"""
    return """
    QMainWindow {
        background-color: #1e1e1e;
    }

    QWidget {
        background-color: #252526;
        color: #cccccc;
    }

    QMenuBar {
        background-color: #333333;
        color: #cccccc;
        padding: 2px;
    }

    QMenuBar::item:selected {
        background-color: #094771;
    }

    QMenu {
        background-color: #252526;
        border: 1px solid #454545;
    }

    QMenu::item:selected {
        background-color: #094771;
    }

    QToolBar {
        background-color: #333333;
        border: none;
        spacing: 3px;
        padding: 3px;
    }

    QToolButton {
        background-color: transparent;
        border: none;
        padding: 5px;
        border-radius: 3px;
    }

    QToolButton:hover {
        background-color: #3e3e3e;
    }

    QToolButton:pressed {
        background-color: #094771;
    }

    QPushButton {
        background-color: #0e639c;
        color: white;
        border: none;
        padding: 8px 16px;
        border-radius: 4px;
        font-weight: bold;
    }

    QPushButton:hover {
        background-color: #1177bb;
    }

    QPushButton:pressed {
        background-color: #094771;
    }

    QPushButton:disabled {
        background-color: #3e3e3e;
        color: #6e6e6e;
    }

    QPushButton#secondaryButton {
        background-color: #3e3e3e;
        color: #cccccc;
    }

    QPushButton#secondaryButton:hover {
        background-color: #4e4e4e;
    }

    QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
        background-color: #3c3c3c;
        border: 1px solid #3c3c3c;
        border-radius: 4px;
        padding: 6px;
        color: #cccccc;
    }

    QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {
        border: 1px solid #007acc;
    }

    QComboBox::drop-down {
        border: none;
        padding-right: 8px;
    }

    QComboBox::down-arrow {
        width: 12px;
        height: 12px;
    }

    QGroupBox {
        font-weight: bold;
        border: 1px solid #454545;
        border-radius: 6px;
        margin-top: 12px;
        padding-top: 10px;
    }

    QGroupBox::title {
        subcontrol-origin: margin;
        left: 10px;
        padding: 0 5px;
    }

    QTabWidget::pane {
        border: 1px solid #454545;
        border-radius: 4px;
    }

    QTabBar::tab {
        background-color: #2d2d2d;
        color: #cccccc;
        padding: 8px 16px;
        border-top-left-radius: 4px;
        border-top-right-radius: 4px;
    }

    QTabBar::tab:selected {
        background-color: #1e1e1e;
        border-bottom: 2px solid #007acc;
    }

    QTabBar::tab:hover:!selected {
        background-color: #3e3e3e;
    }

    QTableWidget {
        background-color: #1e1e1e;
        gridline-color: #454545;
        border: 1px solid #454545;
    }

    QTableWidget::item {
        padding: 5px;
    }

    QTableWidget::item:selected {
        background-color: #094771;
    }

    QHeaderView::section {
        background-color: #333333;
        color: #cccccc;
        padding: 8px;
        border: none;
        border-right: 1px solid #454545;
        border-bottom: 1px solid #454545;
    }

    QScrollBar:vertical {
        background-color: #1e1e1e;
        width: 12px;
        border: none;
    }

    QScrollBar::handle:vertical {
        background-color: #5a5a5a;
        border-radius: 6px;
        min-height: 20px;
    }

    QScrollBar::handle:vertical:hover {
        background-color: #6e6e6e;
    }

    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
        height: 0px;
    }

    QScrollBar:horizontal {
        background-color: #1e1e1e;
        height: 12px;
        border: none;
    }

    QScrollBar::handle:horizontal {
        background-color: #5a5a5a;
        border-radius: 6px;
        min-width: 20px;
    }

    QProgressBar {
        background-color: #3c3c3c;
        border: none;
        border-radius: 4px;
        text-align: center;
        height: 20px;
    }

    QProgressBar::chunk {
        background-color: #0e639c;
        border-radius: 4px;
    }

    QStatusBar {
        background-color: #007acc;
        color: white;
    }

    QLabel#titleLabel {
        font-size: 14px;
        font-weight: bold;
        color: #ffffff;
    }

    QLabel#subtitleLabel {
        color: #888888;
    }

    QListWidget {
        background-color: #1e1e1e;
        border: 1px solid #454545;
        border-radius: 4px;
    }

    QListWidget::item {
        padding: 8px;
        border-bottom: 1px solid #333333;
    }

    QListWidget::item:selected {
        background-color: #094771;
    }

    QListWidget::item:hover:!selected {
        background-color: #2a2d2e;
    }

    QSplitter::handle {
        background-color: #454545;
    }

    QSplitter::handle:horizontal {
        width: 2px;
    }

    QSplitter::handle:vertical {
        height: 2px;
    }
    """


if __name__ == '__main__':
    run_app()
