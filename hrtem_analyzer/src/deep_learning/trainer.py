"""
Training Pipeline for HR-TEM Deep Learning Models

Features:
- CPU and integrated GPU optimized training
- Mixed precision for faster training
- Early stopping and learning rate scheduling
- Progress callbacks for GUI integration
- Model checkpointing
"""

import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Any
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from .models import CDMeasurementNet, get_device, create_model
from .dataset import TrainingDataManager, create_dataloader


@dataclass
class TrainingConfig:
    """Configuration for training"""
    # Model
    model_type: str = 'cd_measurement'
    num_depths: int = 5

    # Training
    epochs: int = 100
    batch_size: int = 8
    learning_rate: float = 0.001
    weight_decay: float = 0.0001

    # Image
    image_size: Tuple[int, int] = (256, 256)

    # Loss weights
    edge_loss_weight: float = 1.0
    cd_loss_weight: float = 1.0

    # Early stopping
    patience: int = 10
    min_delta: float = 0.001

    # Checkpointing
    checkpoint_dir: str = 'checkpoints'
    save_best_only: bool = True

    # Hardware
    num_workers: int = 0  # 0 for CPU
    use_amp: bool = False  # Automatic mixed precision


class CombinedLoss(nn.Module):
    """Combined loss for edge segmentation and CD regression"""

    def __init__(self, edge_weight: float = 1.0, cd_weight: float = 1.0):
        super().__init__()
        self.edge_weight = edge_weight
        self.cd_weight = cd_weight

        self.bce_loss = nn.BCELoss()
        self.mse_loss = nn.MSELoss()
        self.smooth_l1 = nn.SmoothL1Loss()

    def forward(self, predictions: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                targets: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Calculate combined loss.

        Args:
            predictions: (edge_map, cd_values, confidence)
            targets: Dict with 'edge_map' and 'cd_values'

        Returns:
            total_loss, loss_dict
        """
        pred_edge, pred_cd, pred_conf = predictions
        target_edge = targets['edge_map']
        target_cd = targets['cd_values']

        # Edge segmentation loss (BCE)
        edge_loss = self.bce_loss(pred_edge, target_edge)

        # CD regression loss (Smooth L1 for robustness)
        # Only compute loss for non-zero targets (annotated depths)
        mask = target_cd != 0
        if mask.any():
            cd_loss = self.smooth_l1(pred_cd[mask], target_cd[mask])
        else:
            cd_loss = torch.tensor(0.0, device=pred_cd.device)

        # Total loss
        total_loss = self.edge_weight * edge_loss + self.cd_weight * cd_loss

        return total_loss, {
            'total': total_loss.item(),
            'edge': edge_loss.item(),
            'cd': cd_loss.item(),
        }


class Trainer:
    """
    Trainer for HR-TEM deep learning models.

    Optimized for CPU and integrated GPU.
    """

    def __init__(self, config: TrainingConfig,
                 progress_callback: Optional[Callable[[Dict], None]] = None):
        """
        Initialize trainer.

        Args:
            config: Training configuration
            progress_callback: Callback for progress updates (for GUI)
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required for training")

        self.config = config
        self.progress_callback = progress_callback

        self.device = get_device()
        print(f"Using device: {self.device}")

        # Create model
        self.model = create_model(
            config.model_type,
            num_depths=config.num_depths
        ).to(self.device)

        # Loss function
        self.criterion = CombinedLoss(
            edge_weight=config.edge_loss_weight,
            cd_weight=config.cd_loss_weight
        )

        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5,
            patience=5, verbose=True
        )

        # Mixed precision scaler (for GPU)
        self.scaler = torch.cuda.amp.GradScaler() if config.use_amp and self.device.type == 'cuda' else None

        # Tracking
        self.best_loss = float('inf')
        self.epochs_without_improvement = 0
        self.history: Dict[str, List[float]] = {
            'train_loss': [],
            'val_loss': [],
            'train_edge_loss': [],
            'train_cd_loss': [],
            'val_edge_loss': [],
            'val_cd_loss': [],
            'learning_rate': [],
        }

        # Checkpoint directory
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def train(self, data_manager: TrainingDataManager) -> Dict[str, Any]:
        """
        Train the model.

        Args:
            data_manager: TrainingDataManager with training data

        Returns:
            Training results dictionary
        """
        # Create data loaders
        train_loader = create_dataloader(
            data_manager, split='train',
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            target_size=self.config.image_size
        )

        val_loader = create_dataloader(
            data_manager, split='val',
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            target_size=self.config.image_size
        )

        print(f"Training samples: {len(train_loader.dataset)}")
        print(f"Validation samples: {len(val_loader.dataset)}")
        print(f"Model parameters: {self.model.get_num_params():,}")

        start_time = time.time()

        for epoch in range(self.config.epochs):
            epoch_start = time.time()

            # Training phase
            train_loss, train_metrics = self._train_epoch(train_loader, epoch)

            # Validation phase
            val_loss, val_metrics = self._validate_epoch(val_loader, epoch)

            # Update scheduler
            self.scheduler.step(val_loss)

            # Record history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_edge_loss'].append(train_metrics['edge'])
            self.history['train_cd_loss'].append(train_metrics['cd'])
            self.history['val_edge_loss'].append(val_metrics['edge'])
            self.history['val_cd_loss'].append(val_metrics['cd'])
            self.history['learning_rate'].append(self.optimizer.param_groups[0]['lr'])

            epoch_time = time.time() - epoch_start

            # Print progress
            print(f"Epoch {epoch+1}/{self.config.epochs} "
                  f"- Train Loss: {train_loss:.4f} "
                  f"- Val Loss: {val_loss:.4f} "
                  f"- Time: {epoch_time:.1f}s")

            # Callback for GUI
            if self.progress_callback:
                self.progress_callback({
                    'epoch': epoch + 1,
                    'total_epochs': self.config.epochs,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'train_metrics': train_metrics,
                    'val_metrics': val_metrics,
                    'learning_rate': self.optimizer.param_groups[0]['lr'],
                    'epoch_time': epoch_time,
                })

            # Check for improvement
            if val_loss < self.best_loss - self.config.min_delta:
                self.best_loss = val_loss
                self.epochs_without_improvement = 0

                if self.config.save_best_only:
                    self._save_checkpoint(epoch, val_loss, is_best=True)
            else:
                self.epochs_without_improvement += 1

            # Early stopping
            if self.epochs_without_improvement >= self.config.patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

        total_time = time.time() - start_time

        # Save final model
        self._save_checkpoint(epoch, val_loss, is_best=False)

        return {
            'epochs_trained': epoch + 1,
            'best_val_loss': self.best_loss,
            'final_val_loss': val_loss,
            'total_time': total_time,
            'history': self.history,
        }

    def _train_epoch(self, loader: DataLoader, epoch: int) -> Tuple[float, Dict[str, float]]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        total_metrics = {'edge': 0.0, 'cd': 0.0}
        num_batches = 0

        for batch in loader:
            images = batch['image'].to(self.device)
            targets = {
                'edge_map': batch['edge_map'].to(self.device),
                'cd_values': batch['cd_values'].to(self.device),
            }

            self.optimizer.zero_grad()

            # Forward pass
            if self.scaler:
                with torch.cuda.amp.autocast():
                    predictions = self.model(images)
                    loss, metrics = self.criterion(predictions, targets)

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                predictions = self.model(images)
                loss, metrics = self.criterion(predictions, targets)
                loss.backward()
                self.optimizer.step()

            total_loss += loss.item()
            for k in total_metrics:
                total_metrics[k] += metrics[k]
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        avg_metrics = {k: v / max(num_batches, 1) for k, v in total_metrics.items()}

        return avg_loss, avg_metrics

    def _validate_epoch(self, loader: DataLoader, epoch: int) -> Tuple[float, Dict[str, float]]:
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0.0
        total_metrics = {'edge': 0.0, 'cd': 0.0}
        num_batches = 0

        with torch.no_grad():
            for batch in loader:
                images = batch['image'].to(self.device)
                targets = {
                    'edge_map': batch['edge_map'].to(self.device),
                    'cd_values': batch['cd_values'].to(self.device),
                }

                predictions = self.model(images)
                loss, metrics = self.criterion(predictions, targets)

                total_loss += loss.item()
                for k in total_metrics:
                    total_metrics[k] += metrics[k]
                num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        avg_metrics = {k: v / max(num_batches, 1) for k, v in total_metrics.items()}

        return avg_loss, avg_metrics

    def _save_checkpoint(self, epoch: int, val_loss: float, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'config': self.config.__dict__,
            'history': self.history,
            'timestamp': datetime.now().isoformat(),
        }

        if is_best:
            path = self.checkpoint_dir / 'best_model.pt'
        else:
            path = self.checkpoint_dir / f'checkpoint_epoch_{epoch+1}.pt'

        torch.save(checkpoint, path)
        print(f"Saved checkpoint: {path}")

    def load_checkpoint(self, path: str):
        """Load model from checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint.get('history', self.history)
        print(f"Loaded checkpoint from {path}")

    def export_model(self, output_path: str, format: str = 'pytorch'):
        """
        Export trained model.

        Args:
            output_path: Output file path
            format: 'pytorch' or 'onnx'
        """
        if format == 'pytorch':
            torch.save(self.model.state_dict(), output_path)
        elif format == 'onnx':
            dummy_input = torch.randn(1, 1, *self.config.image_size).to(self.device)
            torch.onnx.export(
                self.model, dummy_input, output_path,
                input_names=['image'],
                output_names=['edge_map', 'cd_values', 'confidence'],
                dynamic_axes={'image': {0: 'batch'}}
            )
        else:
            raise ValueError(f"Unknown format: {format}")

        print(f"Exported model to {output_path}")


def train_model(data_dir: str,
                output_dir: str = 'trained_models',
                config: Optional[TrainingConfig] = None,
                progress_callback: Optional[Callable[[Dict], None]] = None) -> Dict[str, Any]:
    """
    Convenience function to train a model.

    Args:
        data_dir: Path to training data directory
        output_dir: Output directory for models
        config: Training configuration
        progress_callback: Progress callback for GUI

    Returns:
        Training results
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required for training")

    config = config or TrainingConfig()
    config.checkpoint_dir = output_dir

    data_manager = TrainingDataManager(data_dir)
    trainer = Trainer(config, progress_callback)

    results = trainer.train(data_manager)

    # Export best model
    best_model_path = Path(output_dir) / 'best_model.pt'
    if best_model_path.exists():
        trainer.load_checkpoint(str(best_model_path))
        trainer.export_model(str(Path(output_dir) / 'model_final.pt'), format='pytorch')

    # Save training summary
    summary_path = Path(output_dir) / 'training_summary.json'
    with open(summary_path, 'w') as f:
        # Convert numpy types for JSON serialization
        summary = {k: v if not isinstance(v, np.ndarray) else v.tolist()
                   for k, v in results.items() if k != 'history'}
        summary['history'] = {k: [float(x) for x in v] for k, v in results['history'].items()}
        json.dump(summary, f, indent=2)

    return results
