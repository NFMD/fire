"""
Advanced Training Module for HR-TEM Deep Learning Models

Features:
- Support for advanced models (Attention U-Net, Swin Transformer, Ensembles)
- Mixed precision training (FP16)
- Gradient accumulation for large batch sizes
- Advanced learning rate schedulers
- Multi-loss training (Dice, Focal, Boundary)
- Uncertainty-aware loss functions
- Early stopping with model checkpointing
- TensorBoard logging
"""
import os
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Callable, Tuple
import numpy as np
from loguru import logger

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    from torch.cuda.amp import autocast, GradScaler
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    torch = None

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False


if PYTORCH_AVAILABLE:

    # ===================== Loss Functions =====================

    class DiceLoss(nn.Module):
        """Dice loss for segmentation"""
        def __init__(self, smooth: float = 1.0):
            super().__init__()
            self.smooth = smooth

        def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            pred = torch.sigmoid(pred)
            pred_flat = pred.view(-1)
            target_flat = target.view(-1)

            intersection = (pred_flat * target_flat).sum()
            dice = (2. * intersection + self.smooth) / (
                pred_flat.sum() + target_flat.sum() + self.smooth
            )
            return 1 - dice


    class FocalLoss(nn.Module):
        """Focal loss for handling class imbalance"""
        def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
            super().__init__()
            self.alpha = alpha
            self.gamma = gamma

        def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
            pt = torch.exp(-bce)
            focal_loss = self.alpha * (1 - pt) ** self.gamma * bce
            return focal_loss.mean()


    class BoundaryLoss(nn.Module):
        """Boundary-aware loss for edge detection"""
        def __init__(self, theta0: float = 3, theta: float = 5):
            super().__init__()
            self.theta0 = theta0
            self.theta = theta

        def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            # Compute distance transform of target
            # Use morphological gradient as approximation
            pred = torch.sigmoid(pred)

            # Sobel filters for edge detection
            sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=pred.dtype, device=pred.device).view(1, 1, 3, 3)
            sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=pred.dtype, device=pred.device).view(1, 1, 3, 3)

            # Edge magnitude of target
            target_edge_x = F.conv2d(target, sobel_x, padding=1)
            target_edge_y = F.conv2d(target, sobel_y, padding=1)
            target_edge = torch.sqrt(target_edge_x ** 2 + target_edge_y ** 2)

            # Weight by edge magnitude
            weight = 1 + self.theta0 * target_edge

            bce = F.binary_cross_entropy(pred, target, reduction='none')
            weighted_bce = weight * bce

            return weighted_bce.mean()


    class CombinedLoss(nn.Module):
        """Combined loss with multiple components"""
        def __init__(
            self,
            dice_weight: float = 1.0,
            bce_weight: float = 1.0,
            focal_weight: float = 0.5,
            boundary_weight: float = 0.5
        ):
            super().__init__()
            self.dice_weight = dice_weight
            self.bce_weight = bce_weight
            self.focal_weight = focal_weight
            self.boundary_weight = boundary_weight

            self.dice = DiceLoss()
            self.focal = FocalLoss()
            self.boundary = BoundaryLoss()

        def forward(self, pred: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
            losses = {}

            if self.dice_weight > 0:
                losses['dice'] = self.dice(pred, target) * self.dice_weight

            if self.bce_weight > 0:
                losses['bce'] = F.binary_cross_entropy_with_logits(pred, target) * self.bce_weight

            if self.focal_weight > 0:
                losses['focal'] = self.focal(pred, target) * self.focal_weight

            if self.boundary_weight > 0:
                losses['boundary'] = self.boundary(pred, target) * self.boundary_weight

            losses['total'] = sum(losses.values())
            return losses


    class DeepSupervisionLoss(nn.Module):
        """Loss for deep supervision (multi-scale outputs)"""
        def __init__(self, weights: List[float] = None):
            super().__init__()
            self.weights = weights or [1.0, 0.5, 0.25, 0.125]
            self.base_loss = CombinedLoss()

        def forward(
            self,
            outputs: Tuple[torch.Tensor, ...],
            target: torch.Tensor
        ) -> Dict[str, torch.Tensor]:
            total_loss = 0
            all_losses = {}

            for i, (output, weight) in enumerate(zip(outputs, self.weights)):
                losses = self.base_loss(output, target)
                total_loss += losses['total'] * weight
                all_losses[f'scale_{i}'] = losses['total']

            all_losses['total'] = total_loss
            return all_losses


    class UncertaintyLoss(nn.Module):
        """Loss that accounts for prediction uncertainty"""
        def __init__(self):
            super().__init__()

        def forward(
            self,
            pred_mean: torch.Tensor,
            pred_var: torch.Tensor,
            target: torch.Tensor
        ) -> torch.Tensor:
            # Gaussian negative log-likelihood
            pred_var = pred_var.clamp(min=1e-6)  # Ensure positive variance
            nll = 0.5 * (torch.log(pred_var) + (target - pred_mean) ** 2 / pred_var)
            return nll.mean()


    # ===================== Learning Rate Schedulers =====================

    class WarmupCosineScheduler:
        """Cosine annealing with linear warmup"""
        def __init__(
            self,
            optimizer,
            warmup_epochs: int,
            total_epochs: int,
            min_lr: float = 1e-6
        ):
            self.optimizer = optimizer
            self.warmup_epochs = warmup_epochs
            self.total_epochs = total_epochs
            self.min_lr = min_lr
            self.base_lrs = [pg['lr'] for pg in optimizer.param_groups]

        def step(self, epoch: int):
            if epoch < self.warmup_epochs:
                # Linear warmup
                alpha = epoch / self.warmup_epochs
                for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
                    pg['lr'] = base_lr * alpha
            else:
                # Cosine annealing
                progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
                for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
                    pg['lr'] = self.min_lr + 0.5 * (base_lr - self.min_lr) * (1 + np.cos(np.pi * progress))

        def get_lr(self) -> List[float]:
            return [pg['lr'] for pg in self.optimizer.param_groups]


    # ===================== Trainer Configuration =====================

    @dataclass
    class AdvancedTrainerConfig:
        """Configuration for advanced trainer"""
        # Model
        model_type: str = 'attention_unet'
        pretrained: bool = True

        # Training
        epochs: int = 100
        batch_size: int = 8
        accumulation_steps: int = 4  # Effective batch = batch_size * accumulation_steps
        learning_rate: float = 1e-4
        weight_decay: float = 1e-5

        # Mixed precision
        use_amp: bool = True

        # Learning rate schedule
        warmup_epochs: int = 5
        min_lr: float = 1e-6

        # Loss
        dice_weight: float = 1.0
        bce_weight: float = 1.0
        focal_weight: float = 0.5
        boundary_weight: float = 0.5
        deep_supervision: bool = True

        # Early stopping
        patience: int = 15
        min_delta: float = 1e-4

        # Checkpointing
        save_best: bool = True
        save_every: int = 10

        # Data augmentation
        augment: bool = True
        aug_rotate: float = 15.0  # Max rotation degrees
        aug_scale: Tuple[float, float] = (0.8, 1.2)
        aug_flip: bool = True
        aug_noise: float = 0.02

        # Validation
        val_split: float = 0.2

        # Ensemble training
        num_ensemble: int = 1  # > 1 for ensemble

        # Logging
        log_dir: str = './logs'
        experiment_name: str = 'hrtem_training'


    # ===================== Advanced Trainer =====================

    class AdvancedTrainer:
        """
        Advanced trainer for HR-TEM deep learning models.

        Features:
        - Mixed precision training
        - Gradient accumulation
        - Advanced LR scheduling
        - Multi-loss training
        - Uncertainty estimation
        - TensorBoard logging
        """

        def __init__(
            self,
            model: nn.Module,
            config: AdvancedTrainerConfig,
            train_loader: DataLoader,
            val_loader: Optional[DataLoader] = None,
            device: Optional[torch.device] = None
        ):
            self.model = model
            self.config = config
            self.train_loader = train_loader
            self.val_loader = val_loader

            # Device
            if device is None:
                self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            else:
                self.device = device

            self.model.to(self.device)

            # Optimizer
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay
            )

            # Scheduler
            self.scheduler = WarmupCosineScheduler(
                self.optimizer,
                warmup_epochs=config.warmup_epochs,
                total_epochs=config.epochs,
                min_lr=config.min_lr
            )

            # Loss
            if config.deep_supervision:
                self.criterion = DeepSupervisionLoss()
            else:
                self.criterion = CombinedLoss(
                    dice_weight=config.dice_weight,
                    bce_weight=config.bce_weight,
                    focal_weight=config.focal_weight,
                    boundary_weight=config.boundary_weight
                )

            # Mixed precision
            self.scaler = GradScaler() if config.use_amp else None

            # TensorBoard
            if TENSORBOARD_AVAILABLE:
                log_path = Path(config.log_dir) / config.experiment_name
                log_path.mkdir(parents=True, exist_ok=True)
                self.writer = SummaryWriter(log_path)
            else:
                self.writer = None

            # Training state
            self.epoch = 0
            self.global_step = 0
            self.best_val_loss = float('inf')
            self.patience_counter = 0
            self.history = {'train_loss': [], 'val_loss': [], 'lr': []}

        def train_epoch(self) -> Dict[str, float]:
            """Train for one epoch"""
            self.model.train()
            epoch_losses = []
            self.optimizer.zero_grad()

            for batch_idx, (images, targets) in enumerate(self.train_loader):
                images = images.to(self.device)
                targets = targets.to(self.device)

                # Mixed precision forward
                with autocast(enabled=self.config.use_amp):
                    outputs = self.model(images)

                    # Handle deep supervision
                    if isinstance(outputs, tuple):
                        losses = self.criterion(outputs, targets)
                    else:
                        losses = self.criterion(outputs, targets)

                    loss = losses['total'] / self.config.accumulation_steps

                # Backward
                if self.scaler:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()

                # Gradient accumulation
                if (batch_idx + 1) % self.config.accumulation_steps == 0:
                    if self.scaler:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()

                    self.optimizer.zero_grad()
                    self.global_step += 1

                epoch_losses.append(losses['total'].item())

                # Log to TensorBoard
                if self.writer and self.global_step % 10 == 0:
                    self.writer.add_scalar('Train/Loss', losses['total'].item(), self.global_step)
                    for name, value in losses.items():
                        if name != 'total':
                            self.writer.add_scalar(f'Train/{name}', value.item(), self.global_step)

            return {'loss': np.mean(epoch_losses)}

        @torch.no_grad()
        def validate(self) -> Dict[str, float]:
            """Validate the model"""
            if self.val_loader is None:
                return {'loss': 0.0}

            self.model.eval()
            val_losses = []

            for images, targets in self.val_loader:
                images = images.to(self.device)
                targets = targets.to(self.device)

                with autocast(enabled=self.config.use_amp):
                    outputs = self.model(images)

                    if isinstance(outputs, tuple):
                        outputs = outputs[0]  # Use main output for validation

                    losses = self.criterion(outputs, targets) if not isinstance(self.criterion, DeepSupervisionLoss) else \
                             CombinedLoss()(outputs, targets)

                val_losses.append(losses['total'].item())

            return {'loss': np.mean(val_losses)}

        def train(self, callbacks: Optional[List[Callable]] = None) -> Dict[str, List[float]]:
            """
            Full training loop.

            Args:
                callbacks: Optional list of callback functions called after each epoch
                          Each callback receives (trainer, epoch, train_metrics, val_metrics)

            Returns:
                Training history
            """
            logger.info(f"Starting training on {self.device}")
            logger.info(f"Model: {self.config.model_type}")
            logger.info(f"Epochs: {self.config.epochs}, Batch size: {self.config.batch_size}")
            logger.info(f"Effective batch size: {self.config.batch_size * self.config.accumulation_steps}")

            start_time = time.time()

            for epoch in range(self.config.epochs):
                self.epoch = epoch

                # Update learning rate
                self.scheduler.step(epoch)
                current_lr = self.scheduler.get_lr()[0]
                self.history['lr'].append(current_lr)

                # Train
                train_metrics = self.train_epoch()
                self.history['train_loss'].append(train_metrics['loss'])

                # Validate
                val_metrics = self.validate()
                self.history['val_loss'].append(val_metrics['loss'])

                # Log
                logger.info(
                    f"Epoch {epoch + 1}/{self.config.epochs} | "
                    f"Train Loss: {train_metrics['loss']:.4f} | "
                    f"Val Loss: {val_metrics['loss']:.4f} | "
                    f"LR: {current_lr:.2e}"
                )

                # TensorBoard
                if self.writer:
                    self.writer.add_scalar('Epoch/Train_Loss', train_metrics['loss'], epoch)
                    self.writer.add_scalar('Epoch/Val_Loss', val_metrics['loss'], epoch)
                    self.writer.add_scalar('Epoch/LR', current_lr, epoch)

                # Early stopping check
                if val_metrics['loss'] < self.best_val_loss - self.config.min_delta:
                    self.best_val_loss = val_metrics['loss']
                    self.patience_counter = 0

                    if self.config.save_best:
                        self.save_checkpoint('best_model.pt')
                else:
                    self.patience_counter += 1

                if self.patience_counter >= self.config.patience:
                    logger.info(f"Early stopping at epoch {epoch + 1}")
                    break

                # Periodic checkpointing
                if self.config.save_every > 0 and (epoch + 1) % self.config.save_every == 0:
                    self.save_checkpoint(f'checkpoint_epoch_{epoch + 1}.pt')

                # Callbacks
                if callbacks:
                    for callback in callbacks:
                        callback(self, epoch, train_metrics, val_metrics)

            elapsed = time.time() - start_time
            logger.info(f"Training completed in {elapsed / 60:.1f} minutes")
            logger.info(f"Best validation loss: {self.best_val_loss:.4f}")

            # Save final model
            self.save_checkpoint('final_model.pt')

            if self.writer:
                self.writer.close()

            return self.history

        def save_checkpoint(self, filename: str):
            """Save model checkpoint"""
            save_path = Path(self.config.log_dir) / self.config.experiment_name / filename
            save_path.parent.mkdir(parents=True, exist_ok=True)

            checkpoint = {
                'epoch': self.epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'best_val_loss': self.best_val_loss,
                'config': self.config,
                'history': self.history
            }

            torch.save(checkpoint, save_path)
            logger.info(f"Saved checkpoint: {save_path}")

        def load_checkpoint(self, path: str):
            """Load model checkpoint"""
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)

            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epoch = checkpoint['epoch']
            self.best_val_loss = checkpoint['best_val_loss']
            self.history = checkpoint.get('history', self.history)

            logger.info(f"Loaded checkpoint from epoch {self.epoch}")

        @torch.no_grad()
        def predict(
            self,
            image: np.ndarray,
            return_uncertainty: bool = False
        ) -> Dict[str, np.ndarray]:
            """
            Make prediction on a single image.

            Args:
                image: Input image (H, W) or (H, W, C)
                return_uncertainty: If True, use MC dropout for uncertainty

            Returns:
                Dictionary with 'prediction' and optionally 'uncertainty'
            """
            self.model.eval()

            # Prepare input
            if image.ndim == 2:
                image = image[np.newaxis, np.newaxis, ...]  # Add batch and channel dims
            elif image.ndim == 3:
                image = image[np.newaxis, ...]  # Add batch dim
                if image.shape[-1] < image.shape[1]:  # Channel last
                    image = np.transpose(image, (0, 3, 1, 2))

            image_tensor = torch.from_numpy(image.astype(np.float32)).to(self.device)

            if return_uncertainty:
                # Monte Carlo Dropout
                self.model.train()  # Enable dropout
                predictions = []

                for _ in range(10):
                    with autocast(enabled=self.config.use_amp):
                        output = self.model(image_tensor)
                        if isinstance(output, tuple):
                            output = output[0]
                        predictions.append(torch.sigmoid(output).cpu().numpy())

                self.model.eval()

                predictions = np.array(predictions)
                mean_pred = predictions.mean(axis=0)
                uncertainty = predictions.std(axis=0)

                return {
                    'prediction': mean_pred[0, 0],
                    'uncertainty': uncertainty[0, 0]
                }

            else:
                with autocast(enabled=self.config.use_amp):
                    output = self.model(image_tensor)
                    if isinstance(output, tuple):
                        output = output[0]

                prediction = torch.sigmoid(output).cpu().numpy()
                return {'prediction': prediction[0, 0]}


    # ===================== Ensemble Trainer =====================

    class EnsembleTrainer:
        """
        Trainer for Deep Ensembles.

        Trains multiple models independently and combines them.
        """

        def __init__(
            self,
            model_class,
            config: AdvancedTrainerConfig,
            train_dataset,
            val_dataset=None,
            **model_kwargs
        ):
            self.model_class = model_class
            self.config = config
            self.train_dataset = train_dataset
            self.val_dataset = val_dataset
            self.model_kwargs = model_kwargs

            self.trainers = []
            self.models = []

            # Device
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        def train(self) -> List[Dict[str, List[float]]]:
            """Train all ensemble members"""
            all_histories = []

            for i in range(self.config.num_ensemble):
                logger.info(f"\n{'=' * 50}")
                logger.info(f"Training ensemble member {i + 1}/{self.config.num_ensemble}")
                logger.info(f"{'=' * 50}\n")

                # Create model
                model = self.model_class(**self.model_kwargs)

                # Create data loaders (with different random seeds for diversity)
                torch.manual_seed(i * 42)
                train_loader = DataLoader(
                    self.train_dataset,
                    batch_size=self.config.batch_size,
                    shuffle=True,
                    num_workers=0,
                    pin_memory=True
                )

                val_loader = None
                if self.val_dataset:
                    val_loader = DataLoader(
                        self.val_dataset,
                        batch_size=self.config.batch_size,
                        shuffle=False,
                        num_workers=0
                    )

                # Create config for this member
                member_config = AdvancedTrainerConfig(
                    **{**self.config.__dict__,
                       'experiment_name': f"{self.config.experiment_name}_member_{i}"}
                )

                # Train
                trainer = AdvancedTrainer(
                    model, member_config, train_loader, val_loader, self.device
                )
                history = trainer.train()

                self.trainers.append(trainer)
                self.models.append(model)
                all_histories.append(history)

            return all_histories

        @torch.no_grad()
        def predict(self, image: np.ndarray) -> Dict[str, np.ndarray]:
            """
            Ensemble prediction with uncertainty.

            Returns:
                Dictionary with 'mean', 'epistemic' (model uncertainty),
                and 'predictions' (all model outputs)
            """
            predictions = []

            for model in self.models:
                model.eval()

                # Prepare input
                if image.ndim == 2:
                    img = image[np.newaxis, np.newaxis, ...]
                else:
                    img = image[np.newaxis, ...]
                    if img.shape[-1] < img.shape[1]:
                        img = np.transpose(img, (0, 3, 1, 2))

                img_tensor = torch.from_numpy(img.astype(np.float32)).to(self.device)

                output = model(img_tensor)
                if isinstance(output, tuple):
                    output = output[0]

                pred = torch.sigmoid(output).cpu().numpy()[0, 0]
                predictions.append(pred)

            predictions = np.array(predictions)
            mean = predictions.mean(axis=0)
            epistemic = predictions.std(axis=0)

            return {
                'mean': mean,
                'epistemic': epistemic,
                'predictions': predictions
            }

        def save(self, path: str):
            """Save all ensemble members"""
            save_path = Path(path)
            save_path.mkdir(parents=True, exist_ok=True)

            for i, model in enumerate(self.models):
                model_path = save_path / f'ensemble_member_{i}.pt'
                torch.save(model.state_dict(), model_path)

            # Save config
            import json
            config_path = save_path / 'config.json'
            with open(config_path, 'w') as f:
                json.dump(self.config.__dict__, f, indent=2, default=str)

            logger.info(f"Saved ensemble to {save_path}")

        def load(self, path: str):
            """Load ensemble members"""
            load_path = Path(path)

            self.models = []
            for i in range(self.config.num_ensemble):
                model = self.model_class(**self.model_kwargs)
                model_path = load_path / f'ensemble_member_{i}.pt'
                model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
                model.to(self.device)
                self.models.append(model)

            logger.info(f"Loaded {len(self.models)} ensemble members from {load_path}")


else:
    # Dummy classes when PyTorch not available
    class AdvancedTrainerConfig:
        pass

    class AdvancedTrainer:
        pass

    class EnsembleTrainer:
        pass
