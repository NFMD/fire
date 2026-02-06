"""
Main Window for HR-TEM Analyzer GUI
"""
from pathlib import Path
from typing import Optional, List, Dict, Any
import json

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QSplitter, QTabWidget, QMenuBar, QMenu, QToolBar,
    QStatusBar, QFileDialog, QMessageBox, QLabel,
    QProgressBar, QApplication
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QSize
from PyQt6.QtGui import QAction, QIcon, QKeySequence

from .widgets import (
    ImageViewerWidget,
    FileListWidget,
    SettingsPanel,
    ResultsPanel,
    MeasurementTableWidget,
    FFTViewerWidget,
    TrainingPanel,
    ManualMeasurementWidget,
    AutoLeveler
)
from ..core.result_exporter import NumpyEncoder, convert_numpy_types


class AnalysisWorker(QThread):
    """Background worker for running analysis"""
    progress = pyqtSignal(int, int, str)  # current, total, message
    finished = pyqtSignal(list)  # results
    error = pyqtSignal(str)  # error message
    image_completed = pyqtSignal(dict)  # single image result

    def __init__(self, image_paths: List[str], settings: Dict[str, Any]):
        super().__init__()
        self.image_paths = image_paths
        self.settings = settings
        self._is_cancelled = False

    def cancel(self):
        """Cancel the analysis"""
        self._is_cancelled = True

    def run(self):
        """Run the analysis in background thread"""
        try:
            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent))

            from pipeline.inference_pipeline import InferencePipeline, PipelineConfig

            # Determine if using enhanced analysis
            use_enhanced = bool(self.settings.get('profile_methods'))

            # Create pipeline config from settings
            config = PipelineConfig(
                depths_nm=self.settings.get('depths_nm', [5, 10, 15, 20]),
                preprocessing_methods=self.settings.get('preprocessing_methods',
                    ['original', 'clahe', 'bilateral_filter']),
                rotation_angles=self.settings.get('rotation_angles', [-1, 0, 1]),
                edge_methods=self.settings.get('edge_methods',
                    ['sobel', 'canny', 'gradient']),
                max_workers=self.settings.get('max_workers', 4),
                consensus_method=self.settings.get('consensus_method', 'trimmed_mean'),
                # Advanced Gatan DM-style settings
                use_enhanced_analysis=use_enhanced,
                profile_methods=self.settings.get('profile_methods', ['fwhm', '10-90', 'derivative']),
                background_method=self.settings.get('background_method', 'rolling_ball'),
                fft_calibration=self.settings.get('fft_calibration', False),
                lattice_spacing_nm=self.settings.get('lattice_spacing_nm'),
                drift_correction=self.settings.get('drift_correction', True),
                interpolation_factor=self.settings.get('interpolation_factor', 10),
                outlier_method=self.settings.get('outlier_method', 'iqr'),
                bootstrap_ci=self.settings.get('bootstrap_ci', True),
                # Precision measurement settings
                precision_mode=self.settings.get('precision_mode', True),
                subpixel_method=self.settings.get('subpixel_method', 'gaussian'),
                denoise_method=self.settings.get('denoise_method', 'nlm'),
                denoise_strength=self.settings.get('denoise_strength', 1.0),
                mc_simulations=self.settings.get('mc_simulations', 500),
                atomic_fitting=self.settings.get('atomic_fitting', False),
            )

            # Create output directory
            output_dir = self.settings.get('output_dir', './results')
            Path(output_dir).mkdir(parents=True, exist_ok=True)

            pipeline = InferencePipeline(config=config, output_dir=output_dir)

            results = []
            total = len(self.image_paths)

            for i, path in enumerate(self.image_paths):
                if self._is_cancelled:
                    break

                mode_str = "(Enhanced)" if use_enhanced else "(Standard)"
                self.progress.emit(i, total, f"Processing {Path(path).name} {mode_str}...")

                try:
                    from core.result_exporter import convert_numpy_types
                    result = pipeline.process_single(
                        image_path=path,
                        baseline_hint_y=self.settings.get('baseline_y'),
                        depths_nm=self.settings.get('depths_nm', [5, 10, 15, 20]),
                        save_result=True,
                        use_enhanced=use_enhanced
                    )
                    result = convert_numpy_types(result)
                    results.append(result)
                    self.image_completed.emit(result)
                except Exception as e:
                    results.append({
                        'source_path': path,
                        'success': False,
                        'error': str(e)
                    })

            self.progress.emit(total, total, "Complete!")
            self.finished.emit(results)

        except Exception as e:
            self.error.emit(str(e))


class MainWindow(QMainWindow):
    """Main application window"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("HR-TEM Analyzer")
        self.setMinimumSize(1400, 900)

        # State
        self.current_image_path: Optional[str] = None
        self.image_paths: List[str] = []
        self.results: List[Dict] = []
        self.worker: Optional[AnalysisWorker] = None

        # Setup UI
        self._setup_ui()
        self._setup_menus()
        self._setup_toolbar()
        self._setup_statusbar()
        self._connect_signals()

    def _setup_ui(self):
        """Setup the main UI layout"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Main splitter
        self.main_splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left panel: File list and settings
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(8, 8, 8, 8)

        # File list
        self.file_list = FileListWidget()
        left_layout.addWidget(self.file_list, stretch=1)

        # Settings panel
        self.settings_panel = SettingsPanel()
        left_layout.addWidget(self.settings_panel, stretch=1)

        left_panel.setMinimumWidth(300)
        left_panel.setMaximumWidth(400)

        # Center panel: Image viewer with tabs
        center_panel = QWidget()
        center_layout = QVBoxLayout(center_panel)
        center_layout.setContentsMargins(0, 0, 0, 0)

        self.view_tabs = QTabWidget()

        # Image viewer tab with manual measurement panel
        image_tab = QWidget()
        image_tab_layout = QHBoxLayout(image_tab)
        image_tab_layout.setContentsMargins(0, 0, 0, 0)

        # Image viewer
        self.image_viewer = ImageViewerWidget()
        image_tab_layout.addWidget(self.image_viewer, stretch=3)

        # Manual measurement panel (collapsible right side)
        self.manual_measurement = ManualMeasurementWidget()
        self.manual_measurement.setMaximumWidth(320)
        self.manual_measurement.setMinimumWidth(280)
        image_tab_layout.addWidget(self.manual_measurement, stretch=1)

        # Auto-leveler instance
        self.auto_leveler = AutoLeveler()

        self.view_tabs.addTab(image_tab, "Image View")

        # FFT Analysis tab
        self.fft_viewer = FFTViewerWidget()
        self.view_tabs.addTab(self.fft_viewer, "FFT Analysis")

        # Results tab
        self.results_panel = ResultsPanel()
        self.view_tabs.addTab(self.results_panel, "Results")

        # Deep Learning Training tab
        self.training_panel = TrainingPanel()
        self.view_tabs.addTab(self.training_panel, "DL Training")

        center_layout.addWidget(self.view_tabs)

        # Right panel: Measurements table
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(8, 8, 8, 8)

        right_label = QLabel("Measurements")
        right_label.setObjectName("titleLabel")
        right_layout.addWidget(right_label)

        self.measurement_table = MeasurementTableWidget()
        right_layout.addWidget(self.measurement_table)

        right_panel.setMinimumWidth(350)
        right_panel.setMaximumWidth(450)

        # Add panels to splitter
        self.main_splitter.addWidget(left_panel)
        self.main_splitter.addWidget(center_panel)
        self.main_splitter.addWidget(right_panel)

        # Set splitter sizes
        self.main_splitter.setSizes([300, 700, 400])

        main_layout.addWidget(self.main_splitter)

    def _setup_menus(self):
        """Setup menu bar"""
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("&File")

        open_action = QAction("&Open Images...", self)
        open_action.setShortcut(QKeySequence.StandardKey.Open)
        open_action.triggered.connect(self._on_open_images)
        file_menu.addAction(open_action)

        open_folder_action = QAction("Open &Folder...", self)
        open_folder_action.setShortcut("Ctrl+Shift+O")
        open_folder_action.triggered.connect(self._on_open_folder)
        file_menu.addAction(open_folder_action)

        file_menu.addSeparator()

        export_action = QAction("&Export Results...", self)
        export_action.setShortcut("Ctrl+E")
        export_action.triggered.connect(self._on_export_results)
        file_menu.addAction(export_action)

        file_menu.addSeparator()

        exit_action = QAction("E&xit", self)
        exit_action.setShortcut(QKeySequence.StandardKey.Quit)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # Analysis menu
        analysis_menu = menubar.addMenu("&Analysis")

        run_action = QAction("&Run Analysis", self)
        run_action.setShortcut("F5")
        run_action.triggered.connect(self._on_run_analysis)
        analysis_menu.addAction(run_action)

        run_selected_action = QAction("Run &Selected", self)
        run_selected_action.setShortcut("F6")
        run_selected_action.triggered.connect(self._on_run_selected)
        analysis_menu.addAction(run_selected_action)

        analysis_menu.addSeparator()

        stop_action = QAction("&Stop", self)
        stop_action.setShortcut("Escape")
        stop_action.triggered.connect(self._on_stop_analysis)
        analysis_menu.addAction(stop_action)

        analysis_menu.addSeparator()

        add_to_training_action = QAction("Add Results to &Training Data", self)
        add_to_training_action.setShortcut("Ctrl+T")
        add_to_training_action.triggered.connect(self._on_add_to_training)
        analysis_menu.addAction(add_to_training_action)

        # View menu
        view_menu = menubar.addMenu("&View")

        zoom_in_action = QAction("Zoom &In", self)
        zoom_in_action.setShortcut(QKeySequence.StandardKey.ZoomIn)
        zoom_in_action.triggered.connect(lambda: self.image_viewer.zoom_in())
        view_menu.addAction(zoom_in_action)

        zoom_out_action = QAction("Zoom &Out", self)
        zoom_out_action.setShortcut(QKeySequence.StandardKey.ZoomOut)
        zoom_out_action.triggered.connect(lambda: self.image_viewer.zoom_out())
        view_menu.addAction(zoom_out_action)

        fit_action = QAction("&Fit to Window", self)
        fit_action.setShortcut("Ctrl+0")
        fit_action.triggered.connect(lambda: self.image_viewer.fit_to_window())
        view_menu.addAction(fit_action)

        # Help menu
        help_menu = menubar.addMenu("&Help")

        about_action = QAction("&About", self)
        about_action.triggered.connect(self._on_about)
        help_menu.addAction(about_action)

    def _setup_toolbar(self):
        """Setup toolbar"""
        toolbar = QToolBar("Main Toolbar")
        toolbar.setMovable(False)
        toolbar.setIconSize(QSize(24, 24))
        self.addToolBar(toolbar)

        # Open
        open_btn = QAction("Open", self)
        open_btn.setToolTip("Open Images (Ctrl+O)")
        open_btn.triggered.connect(self._on_open_images)
        toolbar.addAction(open_btn)

        # Open Folder
        folder_btn = QAction("Folder", self)
        folder_btn.setToolTip("Open Folder (Ctrl+Shift+O)")
        folder_btn.triggered.connect(self._on_open_folder)
        toolbar.addAction(folder_btn)

        toolbar.addSeparator()

        # Run
        self.run_btn = QAction("▶ Run", self)
        self.run_btn.setToolTip("Run Analysis (F5)")
        self.run_btn.triggered.connect(self._on_run_analysis)
        toolbar.addAction(self.run_btn)

        # Stop
        self.stop_btn = QAction("■ Stop", self)
        self.stop_btn.setToolTip("Stop Analysis (Escape)")
        self.stop_btn.setEnabled(False)
        self.stop_btn.triggered.connect(self._on_stop_analysis)
        toolbar.addAction(self.stop_btn)

        toolbar.addSeparator()

        # Zoom
        zoom_in_btn = QAction("🔍+", self)
        zoom_in_btn.setToolTip("Zoom In")
        zoom_in_btn.triggered.connect(lambda: self.image_viewer.zoom_in())
        toolbar.addAction(zoom_in_btn)

        zoom_out_btn = QAction("🔍-", self)
        zoom_out_btn.setToolTip("Zoom Out")
        zoom_out_btn.triggered.connect(lambda: self.image_viewer.zoom_out())
        toolbar.addAction(zoom_out_btn)

        fit_btn = QAction("Fit", self)
        fit_btn.setToolTip("Fit to Window")
        fit_btn.triggered.connect(lambda: self.image_viewer.fit_to_window())
        toolbar.addAction(fit_btn)

    def _setup_statusbar(self):
        """Setup status bar"""
        self.statusbar = QStatusBar()
        self.setStatusBar(self.statusbar)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximumWidth(200)
        self.progress_bar.setVisible(False)
        self.statusbar.addPermanentWidget(self.progress_bar)

        # Status label
        self.status_label = QLabel("Ready")
        self.statusbar.addWidget(self.status_label)

    def _connect_signals(self):
        """Connect widget signals"""
        # File list signals
        self.file_list.file_selected.connect(self._on_file_selected)
        self.file_list.files_dropped.connect(self._on_files_dropped)

        # Image viewer signals
        self.image_viewer.baseline_changed.connect(self._on_baseline_changed)
        self.image_viewer.position_changed.connect(self._on_position_changed)
        self.image_viewer.measurement_completed.connect(self._on_manual_measurement)

        # Manual measurement widget signals
        self.manual_measurement.tool_changed.connect(self._on_measurement_tool_changed)
        self.manual_measurement.baseline_set.connect(self._on_manual_baseline_set)
        self.manual_measurement.auto_level_requested.connect(self._on_auto_level_requested)
        self.manual_measurement.clear_measurements.connect(self._on_clear_manual_measurements)

        # FFT viewer signals
        self.fft_viewer.calibration_requested.connect(self._on_fft_calibration)

    def _on_fft_calibration(self, spacing_nm: float):
        """Handle FFT calibration request"""
        self.settings_panel.lattice_spacing_spin.setValue(spacing_nm)
        self.settings_panel.fft_calibration_cb.setChecked(True)
        self.status_label.setText(f"FFT calibration set: {spacing_nm:.3f} nm")

    def _on_measurement_tool_changed(self, tool: str):
        """Handle measurement tool change"""
        self.image_viewer.set_measurement_mode(tool)
        tool_names = {
            'none': 'Pan/Zoom',
            'baseline': 'Set Baseline',
            'horizontal': 'Horizontal CD',
            'vertical': 'Depth',
            'free': 'Free Measure',
        }
        self.status_label.setText(f"Tool: {tool_names.get(tool, tool)}")

    def _on_manual_baseline_set(self, y: int, angle: float):
        """Handle manual baseline setting"""
        self.image_viewer.set_baseline(y)
        self.settings_panel.baseline_spin.setValue(y)
        if abs(angle) > 0.01:
            self.image_viewer.rotate_image(-angle)
            self.status_label.setText(f"Baseline set at y={y}, rotated {-angle:.2f}°")
        else:
            self.status_label.setText(f"Baseline set at y={y}")

    def _on_auto_level_requested(self):
        """Handle auto-level detection request"""
        image_data = self.image_viewer.get_image_data()
        if image_data is None:
            QMessageBox.warning(self, "No Image", "Please load an image first.")
            return

        self.status_label.setText("Detecting image tilt...")
        QApplication.processEvents()

        # Run auto-leveling detection
        angle, confidence, method = self.auto_leveler.detect_level(image_data)

        # Update manual measurement widget
        self.manual_measurement.set_detected_angle(angle)

        self.status_label.setText(
            f"Detected tilt: {angle:.2f}° (confidence: {confidence:.0%}, method: {method})"
        )

        if abs(angle) > 0.1:
            # Ask if user wants to apply correction
            reply = QMessageBox.question(
                self, "Apply Level Correction?",
                f"Detected tilt angle: {angle:.2f}°\n\n"
                f"Method: {method}\n"
                f"Confidence: {confidence:.0%}\n\n"
                "Apply rotation correction?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )

            if reply == QMessageBox.StandardButton.Yes:
                self.image_viewer.rotate_image(-angle)
                self.status_label.setText(f"Applied rotation correction: {-angle:.2f}°")

    def _on_manual_measurement(self, start, end, mode: str):
        """Handle completed manual measurement"""
        if not self.manual_measurement.session:
            # Initialize session if needed
            scale = self.image_viewer.get_scale_nm_per_pixel()
            path = self.current_image_path or "unknown"
            self.manual_measurement.set_session(path, scale)

        # Add measurement to widget
        measurement = self.manual_measurement.add_measurement(start, end, mode)

        if measurement:
            self.status_label.setText(
                f"Measurement: {measurement.distance_nm:.2f} nm "
                f"({measurement.distance_px:.1f} px)"
            )

    def _on_clear_manual_measurements(self):
        """Clear all manual measurements from viewer"""
        self.image_viewer.clear_manual_measurements()

    def _on_open_images(self):
        """Open image files dialog"""
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Open HR-TEM Images",
            "",
            "TEM Images (*.tif *.tiff *.dm3 *.dm4);;TIFF Images (*.tif *.tiff);;Gatan DM (*.dm3 *.dm4);;All Files (*.*)"
        )
        if files:
            self._add_images(files)

    def _on_open_folder(self):
        """Open folder dialog"""
        folder = QFileDialog.getExistingDirectory(
            self,
            "Open Folder with HR-TEM Images"
        )
        if folder:
            folder_path = Path(folder)
            files = (list(folder_path.glob("*.tif")) + list(folder_path.glob("*.tiff"))
                    + list(folder_path.glob("*.dm3")) + list(folder_path.glob("*.dm4")))
            if files:
                self._add_images([str(f) for f in sorted(files)])
            else:
                QMessageBox.warning(
                    self,
                    "No Images Found",
                    "No supported images (TIFF/DM3/DM4) found in the selected folder."
                )

    def _add_images(self, paths: List[str]):
        """Add images to the file list"""
        self.image_paths.extend(paths)
        self.file_list.add_files(paths)
        self.status_label.setText(f"Loaded {len(self.image_paths)} images")

        # Load first image
        if paths:
            self._load_image(paths[0])

    def _on_files_dropped(self, paths: List[str]):
        """Handle files dropped onto the file list"""
        tiff_files = [p for p in paths if p.lower().endswith(('.tif', '.tiff', '.dm3', '.dm4'))]
        if tiff_files:
            self._add_images(tiff_files)

    def _on_file_selected(self, path: str):
        """Handle file selection in list"""
        self._load_image(path)

    def _load_image(self, path: str):
        """Load an image into the viewer"""
        self.current_image_path = path
        success = self.image_viewer.load_image(path)

        if success:
            # Initialize manual measurement session with scale from image
            scale = self.image_viewer.get_scale_nm_per_pixel()
            self.manual_measurement.set_session(path, scale)

            # Clear previous measurements
            self.image_viewer.clear_manual_measurements()

            self.status_label.setText(f"Loaded: {Path(path).name}")
        else:
            self.status_label.setText(f"Failed to load: {Path(path).name}")

    def _on_baseline_changed(self, y: int):
        """Handle baseline position change"""
        self.settings_panel.set_baseline_y(y)
        self.status_label.setText(f"Baseline set to y={y}")

    def _on_position_changed(self, x: int, y: int, value: float):
        """Handle mouse position change in viewer"""
        self.statusbar.showMessage(f"Position: ({x}, {y})  Value: {value:.3f}", 2000)

    def _on_run_analysis(self):
        """Run analysis on all images"""
        if not self.image_paths:
            QMessageBox.warning(self, "No Images", "Please load images first.")
            return

        self._start_analysis(self.image_paths)

    def _on_run_selected(self):
        """Run analysis on selected images"""
        selected = self.file_list.get_selected_files()
        if not selected:
            QMessageBox.warning(self, "No Selection", "Please select images to analyze.")
            return

        self._start_analysis(selected)

    def _start_analysis(self, paths: List[str]):
        """Start the analysis worker"""
        # Get settings
        settings = self.settings_panel.get_settings()
        settings['baseline_y'] = self.image_viewer.baseline_y

        # Check output directory
        if not settings.get('output_dir'):
            output_dir = QFileDialog.getExistingDirectory(
                self, "Select Output Directory"
            )
            if not output_dir:
                return
            settings['output_dir'] = output_dir
            self.settings_panel.set_output_dir(output_dir)

        # Create and start worker
        self.worker = AnalysisWorker(paths, settings)
        self.worker.progress.connect(self._on_analysis_progress)
        self.worker.finished.connect(self._on_analysis_finished)
        self.worker.error.connect(self._on_analysis_error)
        self.worker.image_completed.connect(self._on_image_completed)

        # Update UI
        self.run_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)

        self.worker.start()

    def _on_stop_analysis(self):
        """Stop the running analysis"""
        if self.worker:
            self.worker.cancel()
            self.status_label.setText("Cancelling...")

    def _on_analysis_progress(self, current: int, total: int, message: str):
        """Handle analysis progress update"""
        progress = int((current / total) * 100) if total > 0 else 0
        self.progress_bar.setValue(progress)
        self.status_label.setText(message)

    def _on_analysis_finished(self, results: List[Dict]):
        """Handle analysis completion"""
        self.results = results
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.progress_bar.setVisible(False)

        # Update results panel
        self.results_panel.set_results(results)

        # Show summary
        successful = sum(1 for r in results if r.get('success', False))
        self.status_label.setText(
            f"Analysis complete: {successful}/{len(results)} successful"
        )

        # Switch to results tab
        self.view_tabs.setCurrentIndex(1)

    def _on_analysis_error(self, error: str):
        """Handle analysis error"""
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.progress_bar.setVisible(False)

        QMessageBox.critical(self, "Analysis Error", f"An error occurred:\n{error}")
        self.status_label.setText("Analysis failed")

    def _on_image_completed(self, result: Dict):
        """Handle single image completion"""
        if result.get('success'):
            # Update measurements table
            self.measurement_table.add_result(result)

            # Update file list status
            self.file_list.set_file_status(
                result['source_path'],
                'completed'
            )

            # Update FFT viewer if FFT analysis available
            if 'fft_analysis' in result:
                scale = result.get('scale_info', {}).get('scale_nm_per_pixel', 1.0)
                self.fft_viewer.set_fft_results(result['fft_analysis'], scale)

            # Show result overlay on current image if it matches
            if self.current_image_path and result['source_path'] == self.current_image_path:
                self.image_viewer.add_result_overlay(result)

    def _on_export_results(self):
        """Export results to file"""
        if not self.results:
            QMessageBox.warning(self, "No Results", "No results to export.")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Results",
            "measurements.json",
            "JSON Files (*.json);;CSV Files (*.csv)"
        )

        if file_path:
            try:
                if file_path.endswith('.json'):
                    with open(file_path, 'w') as f:
                        json.dump(convert_numpy_types(self.results), f, indent=2, cls=NumpyEncoder)
                elif file_path.endswith('.csv'):
                    self._export_csv(file_path)

                self.status_label.setText(f"Exported to {Path(file_path).name}")
            except Exception as e:
                QMessageBox.critical(self, "Export Error", str(e))

    def _export_csv(self, file_path: str):
        """Export results to CSV"""
        import csv

        with open(file_path, 'w', newline='') as f:
            writer = csv.writer(f)

            # Header
            header = ['Filename', 'Scale (nm/px)', 'Baseline Y']
            depths = set()
            for r in self.results:
                if r.get('measurements'):
                    depths.update(r['measurements'].keys())

            for d in sorted(depths):
                header.extend([f'Thickness@{d}nm', f'Std@{d}nm'])

            writer.writerow(header)

            # Data
            for r in self.results:
                if r.get('success'):
                    row = [
                        Path(r['source_path']).name,
                        r.get('scale_info', {}).get('scale_nm_per_pixel', ''),
                        r.get('baseline', {}).get('y_position', '')
                    ]

                    measurements = r.get('measurements', {})
                    for d in sorted(depths):
                        if d in measurements:
                            m = measurements[d]
                            row.extend([m['thickness_nm'], m['thickness_std']])
                        else:
                            row.extend(['', ''])

                    writer.writerow(row)

    def _on_add_to_training(self):
        """Add current results to training data"""
        if not self.results:
            QMessageBox.warning(self, "No Results", "No analysis results to add to training data.")
            return

        # Check if deep learning module is available
        try:
            from ..deep_learning import PYTORCH_AVAILABLE
            if not PYTORCH_AVAILABLE:
                QMessageBox.warning(
                    self,
                    "PyTorch Not Available",
                    "PyTorch is not installed. Install with:\npip install torch"
                )
                return
        except ImportError:
            QMessageBox.warning(self, "Module Error", "Deep learning module not found.")
            return

        # Get training data directory from training panel
        data_dir = self.training_panel.data_widget.dir_edit.text()
        if not data_dir:
            QMessageBox.information(
                self,
                "Set Training Directory",
                "Please set a training data directory in the DL Training tab first."
            )
            self.view_tabs.setCurrentWidget(self.training_panel)
            return

        # Add results to training data
        try:
            from ..deep_learning import TrainingDataManager, ImageAnnotation, CDAnnotation

            manager = TrainingDataManager(data_dir)
            added_count = 0

            for result in self.results:
                if not result.get('success'):
                    continue

                image_path = result['source_path']
                scale = result.get('scale_info', {}).get('scale_nm_per_pixel', 1.0)
                baseline_y = result.get('baseline', {}).get('y_position', 0)

                # Create measurements list
                measurements = []
                for depth_str, data in result.get('measurements', {}).items():
                    depth = float(depth_str)
                    thickness = data.get('thickness_nm', 0)
                    left_x = data.get('left_edge', 0)
                    right_x = data.get('right_edge', 0)

                    if thickness > 0:
                        measurements.append(CDAnnotation(
                            depth_nm=depth,
                            thickness_nm=thickness,
                            left_edge_x=int(left_x),
                            right_edge_x=int(right_x),
                            y_position=baseline_y + int(depth / scale),
                            confidence=data.get('confidence', 1.0),
                            annotator="auto"
                        ))

                if measurements:
                    # Copy image to training directory
                    new_image_path = manager.add_image(image_path, copy=True)

                    # Create annotation
                    annotation = ImageAnnotation(
                        image_path=new_image_path,
                        scale_nm_per_pixel=scale,
                        baseline_y=baseline_y,
                        measurements=measurements
                    )

                    manager.add_annotation(annotation)
                    added_count += 1

            if added_count > 0:
                QMessageBox.information(
                    self,
                    "Training Data Added",
                    f"Successfully added {added_count} images to training data.\n"
                    f"Directory: {data_dir}"
                )
                # Refresh training panel statistics
                self.training_panel.data_widget._refresh_stats()
            else:
                QMessageBox.warning(
                    self,
                    "No Data Added",
                    "No valid measurement results found to add."
                )

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to add training data:\n{str(e)}")

    def _on_about(self):
        """Show about dialog"""
        # Check deep learning availability
        try:
            from ..deep_learning import PYTORCH_AVAILABLE, PYTORCH_VERSION
            dl_status = f"PyTorch {PYTORCH_VERSION}" if PYTORCH_AVAILABLE else "Not installed"
        except ImportError:
            dl_status = "Not available"

        QMessageBox.about(
            self,
            "About HR-TEM Analyzer",
            f"""<h2>HR-TEM Analyzer</h2>
            <p>Version 1.0.0</p>
            <p>Automated Critical Dimension (CD) measurement system
            for High-Resolution Transmission Electron Microscopy images.</p>
            <p><b>Features:</b></p>
            <ul>
                <li>Multi-method edge detection with consensus</li>
                <li>Automatic scale extraction from TIFF metadata</li>
                <li>FFT-based calibration and analysis</li>
                <li>Sub-pixel precision measurement (ESF/LSF)</li>
                <li>Monte Carlo uncertainty estimation</li>
                <li>Deep learning CD measurement (CPU/GPU)</li>
                <li>Parallel batch processing</li>
            </ul>
            <p><b>Deep Learning:</b> {dl_status}</p>
            """
        )

    def closeEvent(self, event):
        """Handle window close"""
        if self.worker and self.worker.isRunning():
            reply = QMessageBox.question(
                self,
                "Analysis Running",
                "Analysis is still running. Do you want to stop and exit?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.Yes:
                self.worker.cancel()
                self.worker.wait()
            else:
                event.ignore()
                return

        event.accept()
