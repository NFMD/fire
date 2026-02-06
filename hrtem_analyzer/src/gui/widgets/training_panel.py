"""
Training Panel Widget for HR-TEM Analyzer

Provides GUI for:
- Training data upload and management
- Annotation of measurement results
- Model training with progress visualization
- Model evaluation and export
"""

from pathlib import Path
from typing import Dict, Any, Optional, List
import json

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QGroupBox, QFormLayout, QSpinBox, QDoubleSpinBox,
    QComboBox, QCheckBox, QLineEdit, QPushButton,
    QFileDialog, QScrollArea, QFrame, QProgressBar,
    QTableWidget, QTableWidgetItem, QHeaderView,
    QMessageBox, QTabWidget, QTextEdit, QSplitter
)
from PyQt6.QtCore import Qt, pyqtSignal, QThread
from PyQt6.QtGui import QColor

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TrainingWorker(QThread):
    """Background worker for model training"""

    progress = pyqtSignal(dict)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, data_dir: str, config: Dict[str, Any]):
        super().__init__()
        self.data_dir = data_dir
        self.config = config
        self._is_cancelled = False

    def cancel(self):
        self._is_cancelled = True

    def run(self):
        try:
            use_advanced = self.config.get('use_advanced_trainer', False)
            model_type = self.config.get('model_type', 'cd_measurement')

            if use_advanced:
                # Use advanced trainer for SOTA models
                from deep_learning import (
                    ADVANCED_MODELS_AVAILABLE,
                    create_advanced_model,
                    AdvancedTrainer,
                    AdvancedTrainerConfig,
                    EnsembleTrainer,
                    TrainingDataManager,
                )
                from deep_learning.dataset import CDDataset
                from torch.utils.data import DataLoader, random_split
                import torch

                if not ADVANCED_MODELS_AVAILABLE:
                    raise ImportError("Advanced models not available. Check PyTorch installation.")

                # Load data
                manager = TrainingDataManager(self.data_dir)
                annotations = manager.get_all_annotations()

                if len(annotations) < 5:
                    raise ValueError(f"Need at least 5 training samples, found {len(annotations)}")

                dataset = CDDataset(self.data_dir, annotations)

                # Split into train/val
                train_size = int(0.8 * len(dataset))
                val_size = len(dataset) - train_size
                train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

                # Create data loaders
                train_loader = DataLoader(train_dataset, batch_size=self.config.get('batch_size', 8),
                                          shuffle=True, num_workers=0)
                val_loader = DataLoader(val_dataset, batch_size=self.config.get('batch_size', 8),
                                        shuffle=False, num_workers=0)

                # Training config
                trainer_config = AdvancedTrainerConfig(
                    model_type=model_type,
                    epochs=self.config.get('epochs', 100),
                    batch_size=self.config.get('batch_size', 8),
                    learning_rate=self.config.get('learning_rate', 0.0001),
                    use_amp=True,
                    deep_supervision=True,
                    patience=20,
                    log_dir=self.config.get('output_dir', 'trained_models'),
                    experiment_name=f'hrtem_{model_type}',
                )

                # Check for ensemble models
                if model_type.startswith('ensemble_'):
                    base_type = model_type.replace('ensemble_', '')
                    trainer_config.num_ensemble = 5

                    ensemble_trainer = EnsembleTrainer(
                        model_class=lambda **kw: create_advanced_model(base_type, **kw),
                        config=trainer_config,
                        train_dataset=train_dataset,
                        val_dataset=val_dataset,
                        in_channels=1,
                        out_channels=1
                    )

                    # Train with progress
                    for epoch in range(trainer_config.epochs):
                        if self._is_cancelled:
                            raise InterruptedError("Training cancelled")
                        self.progress.emit({
                            'epoch': epoch + 1,
                            'total_epochs': trainer_config.epochs,
                            'status': f'Training ensemble (member updates)...'
                        })

                    histories = ensemble_trainer.train()
                    ensemble_trainer.save(trainer_config.log_dir + '/' + trainer_config.experiment_name)

                    results = {
                        'model_type': model_type,
                        'histories': [h for h in histories],
                        'model_path': trainer_config.log_dir + '/' + trainer_config.experiment_name
                    }
                else:
                    # Single advanced model
                    model = create_advanced_model(model_type, in_channels=1, out_channels=1)

                    trainer = AdvancedTrainer(
                        model=model,
                        config=trainer_config,
                        train_loader=train_loader,
                        val_loader=val_loader
                    )

                    def progress_callback(trainer_obj, epoch, train_metrics, val_metrics):
                        if self._is_cancelled:
                            raise InterruptedError("Training cancelled")
                        self.progress.emit({
                            'epoch': epoch + 1,
                            'total_epochs': trainer_config.epochs,
                            'train_loss': train_metrics['loss'],
                            'val_loss': val_metrics['loss'],
                            'lr': trainer_obj.scheduler.get_lr()[0]
                        })

                    history = trainer.train(callbacks=[progress_callback])

                    results = {
                        'model_type': model_type,
                        'history': history,
                        'best_val_loss': trainer.best_val_loss,
                        'model_path': trainer_config.log_dir + '/' + trainer_config.experiment_name
                    }

            else:
                # Use standard trainer
                from deep_learning import train_model, TrainingConfig

                config = TrainingConfig(
                    epochs=self.config.get('epochs', 50),
                    batch_size=self.config.get('batch_size', 8),
                    learning_rate=self.config.get('learning_rate', 0.001),
                    model_type=model_type,
                    checkpoint_dir=self.config.get('output_dir', 'trained_models'),
                )

                def progress_callback(info):
                    if self._is_cancelled:
                        raise InterruptedError("Training cancelled")
                    self.progress.emit(info)

                results = train_model(
                    self.data_dir,
                    config.checkpoint_dir,
                    config,
                    progress_callback
                )

            self.finished.emit(results)

        except InterruptedError:
            self.error.emit("Training cancelled by user")
        except Exception as e:
            import traceback
            self.error.emit(f"{str(e)}\n\n{traceback.format_exc()}")


class TrainingDataWidget(QWidget):
    """Widget for managing training data"""

    data_updated = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.data_dir = None
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # Data directory selection
        dir_group = QGroupBox("Training Data Directory")
        dir_layout = QHBoxLayout(dir_group)

        self.dir_edit = QLineEdit()
        self.dir_edit.setPlaceholderText("Select training data directory...")
        self.dir_edit.setReadOnly(True)
        dir_layout.addWidget(self.dir_edit)

        self.browse_btn = QPushButton("Browse...")
        self.browse_btn.clicked.connect(self._on_browse)
        dir_layout.addWidget(self.browse_btn)

        self.create_btn = QPushButton("Create New")
        self.create_btn.clicked.connect(self._on_create_new)
        dir_layout.addWidget(self.create_btn)

        layout.addWidget(dir_group)

        # Upload buttons
        upload_group = QGroupBox("Add Training Data")
        upload_layout = QHBoxLayout(upload_group)

        self.upload_images_btn = QPushButton("Upload Images...")
        self.upload_images_btn.clicked.connect(self._on_upload_images)
        self.upload_images_btn.setEnabled(False)
        upload_layout.addWidget(self.upload_images_btn)

        self.upload_annotated_btn = QPushButton("Upload Annotated Results...")
        self.upload_annotated_btn.clicked.connect(self._on_upload_annotated)
        self.upload_annotated_btn.setEnabled(False)
        upload_layout.addWidget(self.upload_annotated_btn)

        layout.addWidget(upload_group)

        # Statistics
        stats_group = QGroupBox("Dataset Statistics")
        stats_layout = QFormLayout(stats_group)

        self.total_images_label = QLabel("0")
        stats_layout.addRow("Total Images:", self.total_images_label)

        self.annotated_label = QLabel("0")
        stats_layout.addRow("Annotated:", self.annotated_label)

        self.measurements_label = QLabel("0")
        stats_layout.addRow("Total Measurements:", self.measurements_label)

        self.depths_label = QLabel("-")
        stats_layout.addRow("Depth Range:", self.depths_label)

        layout.addWidget(stats_group)

        # Data table
        table_group = QGroupBox("Training Samples")
        table_layout = QVBoxLayout(table_group)

        self.data_table = QTableWidget()
        self.data_table.setColumnCount(4)
        self.data_table.setHorizontalHeaderLabels([
            'Filename', 'Annotations', 'Quality', 'Added'
        ])
        self.data_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Stretch
        )
        table_layout.addWidget(self.data_table)

        layout.addWidget(table_group)

    def _on_browse(self):
        directory = QFileDialog.getExistingDirectory(
            self, "Select Training Data Directory"
        )
        if directory:
            self._set_data_dir(directory)

    def _on_create_new(self):
        directory = QFileDialog.getExistingDirectory(
            self, "Create Training Data Directory"
        )
        if directory:
            # Create structure
            from deep_learning import TrainingDataManager
            manager = TrainingDataManager(directory)
            self._set_data_dir(directory)

    def _set_data_dir(self, directory: str):
        self.data_dir = directory
        self.dir_edit.setText(directory)
        self.upload_images_btn.setEnabled(True)
        self.upload_annotated_btn.setEnabled(True)
        self._refresh_stats()

    def _on_upload_images(self):
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select HR-TEM Images",
            "",
            "TIFF Images (*.tif *.tiff);;All Files (*.*)"
        )

        if files and self.data_dir:
            from deep_learning import TrainingDataManager
            manager = TrainingDataManager(self.data_dir)

            for f in files:
                try:
                    manager.add_image(f, copy=True)
                except Exception as e:
                    QMessageBox.warning(
                        self, "Upload Error",
                        f"Failed to add {Path(f).name}: {e}"
                    )

            self._refresh_stats()
            self.data_updated.emit()

    def _on_upload_annotated(self):
        """Upload already analyzed results as training data"""
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Analysis Results (JSON)",
            "",
            "JSON Files (*.json)"
        )

        if files and self.data_dir:
            from deep_learning import TrainingDataManager, ImageAnnotation, CDAnnotation
            manager = TrainingDataManager(self.data_dir)

            for f in files:
                try:
                    with open(f, 'r') as fp:
                        result = json.load(fp)

                    # Convert analysis result to annotation
                    if result.get('success'):
                        # Copy source image
                        src_path = result.get('source_path', '')
                        if Path(src_path).exists():
                            img_path = manager.add_image(src_path, copy=True)
                        else:
                            continue

                        # Create annotation
                        measurements = []
                        for depth, m in result.get('measurements', {}).items():
                            depth_val = float(depth) if isinstance(depth, str) else depth
                            measurements.append(CDAnnotation(
                                depth_nm=depth_val,
                                thickness_nm=m['thickness_nm'],
                                left_edge_x=m.get('left_edge_x', 0),
                                right_edge_x=m.get('right_edge_x', 0),
                                y_position=m.get('y_position', 0),
                                confidence=m.get('confidence', 1.0),
                                annotator='auto'
                            ))

                        annotation = ImageAnnotation(
                            image_path=img_path,
                            scale_nm_per_pixel=result.get('scale_info', {}).get('scale_nm_per_pixel', 1.0),
                            baseline_y=result.get('baseline', {}).get('y_position', 0),
                            measurements=measurements,
                            quality_score=min(m.confidence for m in measurements) if measurements else 0.5,
                        )

                        manager.add_annotation(annotation)

                except Exception as e:
                    QMessageBox.warning(
                        self, "Upload Error",
                        f"Failed to process {Path(f).name}: {e}"
                    )

            self._refresh_stats()
            self.data_updated.emit()

    def _refresh_stats(self):
        if not self.data_dir:
            return

        try:
            from deep_learning import TrainingDataManager
            manager = TrainingDataManager(self.data_dir)
            stats = manager.get_statistics()

            self.total_images_label.setText(str(stats['total_images']))
            self.annotated_label.setText(str(stats['annotated_images']))
            self.measurements_label.setText(str(stats['total_measurements']))

            depth_range = stats.get('depth_range', (0, 0))
            self.depths_label.setText(f"{depth_range[0]:.1f} - {depth_range[1]:.1f} nm")

            # Update table
            self._update_table(manager)

        except Exception as e:
            print(f"Error refreshing stats: {e}")

    def _update_table(self, manager):
        samples = manager.list_annotated_images()
        self.data_table.setRowCount(len(samples))

        for i, (img_path, ann) in enumerate(samples):
            self.data_table.setItem(i, 0, QTableWidgetItem(Path(img_path).name))
            self.data_table.setItem(i, 1, QTableWidgetItem(str(len(ann.measurements))))
            self.data_table.setItem(i, 2, QTableWidgetItem(f"{ann.quality_score:.2f}"))
            self.data_table.setItem(i, 3, QTableWidgetItem(ann.created_at[:10]))


class ModelTrainingWidget(QWidget):
    """Widget for model training controls"""

    training_complete = pyqtSignal(dict)

    def __init__(self):
        super().__init__()
        self.worker = None
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # Training configuration
        config_group = QGroupBox("Training Configuration")
        config_layout = QFormLayout(config_group)

        self.model_combo = QComboBox()
        self.model_combo.addItems([
            'CD Measurement (Recommended)',
            'Edge Segmentation',
            'EfficientNet CD (Transfer Learning)',
            'EfficientNet Edge (Transfer Learning)',
            'Ensemble (3 models)',
            '--- Advanced (High Accuracy) ---',
            'Attention U-Net (SOTA)',
            'Swin Transformer U-Net (SOTA)',
            'Deep Ensemble - Attention (5 models)',
            'Deep Ensemble - Swin (5 models)',
            'Hybrid CNN-Transformer'
        ])
        config_layout.addRow("Model Type:", self.model_combo)

        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(10, 500)
        self.epochs_spin.setValue(100)  # Increased default for better training
        config_layout.addRow("Epochs:", self.epochs_spin)

        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 32)
        self.batch_spin.setValue(8)
        config_layout.addRow("Batch Size:", self.batch_spin)

        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setRange(0.00001, 0.1)
        self.lr_spin.setValue(0.001)
        self.lr_spin.setDecimals(5)
        self.lr_spin.setSingleStep(0.0001)
        config_layout.addRow("Learning Rate:", self.lr_spin)

        layout.addWidget(config_group)

        # Output directory
        output_group = QGroupBox("Output")
        output_layout = QHBoxLayout(output_group)

        self.output_edit = QLineEdit()
        self.output_edit.setText("trained_models")
        output_layout.addWidget(self.output_edit)

        self.output_browse_btn = QPushButton("...")
        self.output_browse_btn.setFixedWidth(30)
        self.output_browse_btn.clicked.connect(self._on_browse_output)
        output_layout.addWidget(self.output_browse_btn)

        layout.addWidget(output_group)

        # Training controls
        controls_layout = QHBoxLayout()

        self.train_btn = QPushButton("Start Training")
        self.train_btn.clicked.connect(self._on_start_training)
        controls_layout.addWidget(self.train_btn)

        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._on_stop_training)
        controls_layout.addWidget(self.stop_btn)

        layout.addLayout(controls_layout)

        # Progress
        progress_group = QGroupBox("Training Progress")
        progress_layout = QVBoxLayout(progress_group)

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        progress_layout.addWidget(self.progress_bar)

        self.status_label = QLabel("Ready to train")
        progress_layout.addWidget(self.status_label)

        # Metrics display
        metrics_layout = QHBoxLayout()

        self.epoch_label = QLabel("Epoch: -")
        metrics_layout.addWidget(self.epoch_label)

        self.train_loss_label = QLabel("Train Loss: -")
        metrics_layout.addWidget(self.train_loss_label)

        self.val_loss_label = QLabel("Val Loss: -")
        metrics_layout.addWidget(self.val_loss_label)

        progress_layout.addLayout(metrics_layout)

        layout.addWidget(progress_group)

        # Training log
        log_group = QGroupBox("Training Log")
        log_layout = QVBoxLayout(log_group)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(150)
        log_layout.addWidget(self.log_text)

        layout.addWidget(log_group)

    def _on_browse_output(self):
        directory = QFileDialog.getExistingDirectory(
            self, "Select Output Directory"
        )
        if directory:
            self.output_edit.setText(directory)

    def set_data_dir(self, data_dir: str):
        self.data_dir = data_dir

    def _on_start_training(self):
        if not hasattr(self, 'data_dir') or not self.data_dir:
            QMessageBox.warning(
                self, "No Data",
                "Please select training data directory first."
            )
            return

        # Check for PyTorch
        try:
            from deep_learning import PYTORCH_AVAILABLE
            if not PYTORCH_AVAILABLE:
                QMessageBox.critical(
                    self, "PyTorch Required",
                    "PyTorch is required for training.\n\n"
                    "Install with: pip install torch"
                )
                return
        except ImportError:
            QMessageBox.critical(
                self, "Module Error",
                "Deep learning module not properly installed."
            )
            return

        model_map = {
            'CD Measurement (Recommended)': 'cd_measurement',
            'Edge Segmentation': 'edge_segmentation',
            'EfficientNet CD (Transfer Learning)': 'efficientnet_cd',
            'EfficientNet Edge (Transfer Learning)': 'efficientnet_edge',
            'Ensemble (3 models)': 'ensemble',
            # Advanced models (SOTA)
            'Attention U-Net (SOTA)': 'attention_unet',
            'Swin Transformer U-Net (SOTA)': 'swin_unet',
            'Deep Ensemble - Attention (5 models)': 'ensemble_attention',
            'Deep Ensemble - Swin (5 models)': 'ensemble_swin',
            'Hybrid CNN-Transformer': 'hybrid_cd',
        }

        selected_model = self.model_combo.currentText()

        # Skip separator items
        if selected_model.startswith('---'):
            QMessageBox.warning(self, "Invalid Selection", "Please select a valid model type.")
            return

        config = {
            'epochs': self.epochs_spin.value(),
            'batch_size': self.batch_spin.value(),
            'learning_rate': self.lr_spin.value(),
            'model_type': model_map.get(selected_model, 'cd_measurement'),
            'output_dir': self.output_edit.text(),
            # Advanced training options
            'use_advanced_trainer': selected_model in [
                'Attention U-Net (SOTA)', 'Swin Transformer U-Net (SOTA)',
                'Deep Ensemble - Attention (5 models)', 'Deep Ensemble - Swin (5 models)',
                'Hybrid CNN-Transformer'
            ],
        }

        self.worker = TrainingWorker(self.data_dir, config)
        self.worker.progress.connect(self._on_progress)
        self.worker.finished.connect(self._on_finished)
        self.worker.error.connect(self._on_error)

        self.train_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.progress_bar.setValue(0)
        self.log_text.clear()
        self._log("Starting training...")

        self.worker.start()

    def _on_stop_training(self):
        if self.worker:
            self.worker.cancel()
            self._log("Stopping training...")

    def _on_progress(self, info: dict):
        epoch = info['epoch']
        total = info['total_epochs']
        progress = int(100 * epoch / total)

        self.progress_bar.setValue(progress)
        self.epoch_label.setText(f"Epoch: {epoch}/{total}")
        self.train_loss_label.setText(f"Train: {info['train_loss']:.4f}")
        self.val_loss_label.setText(f"Val: {info['val_loss']:.4f}")
        self.status_label.setText(f"Training... ({progress}%)")

        self._log(f"Epoch {epoch}: train={info['train_loss']:.4f}, val={info['val_loss']:.4f}")

    def _on_finished(self, results: dict):
        self.train_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.progress_bar.setValue(100)
        self.status_label.setText("Training complete!")

        self._log(f"\nTraining complete!")
        self._log(f"Best validation loss: {results['best_val_loss']:.4f}")
        self._log(f"Epochs trained: {results['epochs_trained']}")
        self._log(f"Total time: {results['total_time']:.1f}s")

        self.training_complete.emit(results)

        QMessageBox.information(
            self, "Training Complete",
            f"Model trained successfully!\n\n"
            f"Best validation loss: {results['best_val_loss']:.4f}\n"
            f"Model saved to: {self.output_edit.text()}"
        )

    def _on_error(self, error: str):
        self.train_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.status_label.setText("Training failed")

        self._log(f"\nError: {error}")

        QMessageBox.critical(self, "Training Error", error)

    def _log(self, message: str):
        self.log_text.append(message)


class TrainingPanel(QWidget):
    """Complete training panel with data management and training"""

    def __init__(self):
        super().__init__()
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        # Title
        title = QLabel("Deep Learning Training")
        title.setStyleSheet("font-weight: bold; font-size: 14px;")
        layout.addWidget(title)

        # Check PyTorch availability
        try:
            from deep_learning import PYTORCH_AVAILABLE, PYTORCH_VERSION
            if PYTORCH_AVAILABLE:
                status = QLabel(f"PyTorch {PYTORCH_VERSION} available")
                status.setStyleSheet("color: #5cb85c;")
            else:
                status = QLabel("PyTorch not installed - Install with: pip install torch")
                status.setStyleSheet("color: #d9534f;")
            layout.addWidget(status)
        except ImportError:
            status = QLabel("Deep learning module not available")
            status.setStyleSheet("color: #d9534f;")
            layout.addWidget(status)

        # Tabs
        tabs = QTabWidget()

        # Training data tab
        self.data_widget = TrainingDataWidget()
        tabs.addTab(self.data_widget, "Training Data")

        # Model training tab
        self.training_widget = ModelTrainingWidget()
        tabs.addTab(self.training_widget, "Model Training")

        # Connect signals
        self.data_widget.data_updated.connect(self._on_data_updated)

        layout.addWidget(tabs)

    def _on_data_updated(self):
        if self.data_widget.data_dir:
            self.training_widget.set_data_dir(self.data_widget.data_dir)
