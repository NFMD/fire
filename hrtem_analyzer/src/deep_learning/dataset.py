"""
Dataset Management for HR-TEM Deep Learning

Handles:
- Training data loading and storage
- Data augmentation
- Annotation management
- Dataset splitting
"""

import json
import shutil
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    Dataset = object

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


@dataclass
class CDAnnotation:
    """Annotation for a single CD measurement"""
    depth_nm: float
    thickness_nm: float
    left_edge_x: int
    right_edge_x: int
    y_position: int
    confidence: float = 1.0
    annotator: str = "manual"
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class ImageAnnotation:
    """Complete annotation for an HR-TEM image"""
    image_path: str
    scale_nm_per_pixel: float
    baseline_y: int
    measurements: List[CDAnnotation] = field(default_factory=list)
    image_width: int = 0
    image_height: int = 0
    notes: str = ""
    quality_score: float = 1.0  # 0-1, for weighting during training
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    modified_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> 'ImageAnnotation':
        measurements = [CDAnnotation(**m) for m in data.pop('measurements', [])]
        return cls(measurements=measurements, **data)


class TrainingDataManager:
    """
    Manages training data for deep learning models.

    Directory structure:
    training_data/
    ├── images/          # Original TIFF images
    ├── annotations/     # JSON annotation files
    ├── augmented/       # Augmented training data
    └── metadata.json    # Dataset metadata
    """

    def __init__(self, data_dir: str = "training_data"):
        self.data_dir = Path(data_dir)
        self.images_dir = self.data_dir / "images"
        self.annotations_dir = self.data_dir / "annotations"
        self.augmented_dir = self.data_dir / "augmented"
        self.metadata_path = self.data_dir / "metadata.json"

        self._ensure_directories()
        self._load_metadata()

    def _ensure_directories(self):
        """Create directory structure if not exists"""
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.annotations_dir.mkdir(parents=True, exist_ok=True)
        self.augmented_dir.mkdir(parents=True, exist_ok=True)

    def _load_metadata(self):
        """Load or initialize metadata"""
        if self.metadata_path.exists():
            with open(self.metadata_path, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {
                'created_at': datetime.now().isoformat(),
                'num_images': 0,
                'num_annotations': 0,
                'depth_range': [],
                'scale_range': [],
            }
            self._save_metadata()

    def _save_metadata(self):
        """Save metadata to file"""
        self.metadata['modified_at'] = datetime.now().isoformat()
        with open(self.metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)

    def add_image(self, image_path: str, copy: bool = True) -> str:
        """
        Add an image to the training dataset.

        Args:
            image_path: Path to the source image
            copy: If True, copy the file; if False, move it

        Returns:
            Path to the image in the dataset
        """
        src = Path(image_path)
        if not src.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Generate unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dst_name = f"{timestamp}_{src.name}"
        dst = self.images_dir / dst_name

        if copy:
            shutil.copy2(src, dst)
        else:
            shutil.move(src, dst)

        self.metadata['num_images'] += 1
        self._save_metadata()

        return str(dst)

    def add_annotation(self, annotation: ImageAnnotation) -> str:
        """
        Add annotation for an image.

        Args:
            annotation: ImageAnnotation object

        Returns:
            Path to the saved annotation file
        """
        # Use image filename as base for annotation
        image_name = Path(annotation.image_path).stem
        annotation_path = self.annotations_dir / f"{image_name}.json"

        with open(annotation_path, 'w') as f:
            json.dump(annotation.to_dict(), f, indent=2)

        self.metadata['num_annotations'] += 1

        # Update depth and scale ranges
        for m in annotation.measurements:
            if m.depth_nm not in self.metadata['depth_range']:
                self.metadata['depth_range'].append(m.depth_nm)
                self.metadata['depth_range'].sort()

        if annotation.scale_nm_per_pixel not in self.metadata['scale_range']:
            self.metadata['scale_range'].append(annotation.scale_nm_per_pixel)

        self._save_metadata()

        return str(annotation_path)

    def get_annotation(self, image_path: str) -> Optional[ImageAnnotation]:
        """Get annotation for an image"""
        image_name = Path(image_path).stem
        annotation_path = self.annotations_dir / f"{image_name}.json"

        if not annotation_path.exists():
            return None

        with open(annotation_path, 'r') as f:
            data = json.load(f)

        return ImageAnnotation.from_dict(data)

    def list_images(self) -> List[str]:
        """List all images in the dataset"""
        extensions = ['.tif', '.tiff', '.png', '.jpg', '.jpeg']
        images = []

        for ext in extensions:
            images.extend(self.images_dir.glob(f"*{ext}"))
            images.extend(self.images_dir.glob(f"*{ext.upper()}"))

        return sorted([str(p) for p in images])

    def list_annotated_images(self) -> List[Tuple[str, ImageAnnotation]]:
        """List images with their annotations"""
        result = []

        for image_path in self.list_images():
            annotation = self.get_annotation(image_path)
            if annotation:
                result.append((image_path, annotation))

        return result

    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics"""
        images = self.list_images()
        annotated = self.list_annotated_images()

        all_depths = []
        all_thicknesses = []
        all_scales = []

        for _, ann in annotated:
            all_scales.append(ann.scale_nm_per_pixel)
            for m in ann.measurements:
                all_depths.append(m.depth_nm)
                all_thicknesses.append(m.thickness_nm)

        return {
            'total_images': len(images),
            'annotated_images': len(annotated),
            'total_measurements': len(all_thicknesses),
            'depth_range': (min(all_depths), max(all_depths)) if all_depths else (0, 0),
            'thickness_range': (min(all_thicknesses), max(all_thicknesses)) if all_thicknesses else (0, 0),
            'scale_range': (min(all_scales), max(all_scales)) if all_scales else (0, 0),
            'unique_depths': sorted(set(all_depths)),
        }

    def export_for_training(self, output_path: str, train_ratio: float = 0.8):
        """
        Export dataset for training.

        Args:
            output_path: Output directory
            train_ratio: Ratio of training data (rest is validation)

        Returns:
            Dict with train and val file lists
        """
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        annotated = self.list_annotated_images()

        # Shuffle and split
        np.random.shuffle(annotated)
        split_idx = int(len(annotated) * train_ratio)

        train_data = annotated[:split_idx]
        val_data = annotated[split_idx:]

        # Save split info
        split_info = {
            'train': [{'image': img, 'annotation': ann.to_dict()} for img, ann in train_data],
            'val': [{'image': img, 'annotation': ann.to_dict()} for img, ann in val_data],
        }

        with open(output_dir / 'split.json', 'w') as f:
            json.dump(split_info, f, indent=2)

        return {
            'train_count': len(train_data),
            'val_count': len(val_data),
            'output_path': str(output_dir),
        }


class DataAugmentor:
    """Data augmentation for HR-TEM images"""

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

    def augment(self, image: np.ndarray, annotation: ImageAnnotation,
                num_augmentations: int = 5) -> List[Tuple[np.ndarray, ImageAnnotation]]:
        """
        Generate augmented versions of an image.

        Args:
            image: Input image (H, W) or (H, W, C)
            annotation: Original annotation
            num_augmentations: Number of augmented versions

        Returns:
            List of (augmented_image, adjusted_annotation) tuples
        """
        results = []

        for _ in range(num_augmentations):
            aug_image = image.copy()
            aug_annotation = ImageAnnotation.from_dict(annotation.to_dict())

            # Random horizontal flip
            if self.rng.random() > 0.5:
                aug_image = np.fliplr(aug_image)
                self._flip_annotation_horizontal(aug_annotation, image.shape[1])

            # Random brightness/contrast
            alpha = self.rng.uniform(0.8, 1.2)  # Contrast
            beta = self.rng.uniform(-20, 20)  # Brightness
            aug_image = np.clip(alpha * aug_image + beta, 0, 255).astype(aug_image.dtype)

            # Random noise
            if self.rng.random() > 0.5:
                noise_level = self.rng.uniform(0, 10)
                noise = self.rng.normal(0, noise_level, aug_image.shape)
                aug_image = np.clip(aug_image + noise, 0, 255).astype(aug_image.dtype)

            # Random small rotation (±2 degrees)
            if self.rng.random() > 0.5 and CV2_AVAILABLE:
                angle = self.rng.uniform(-2, 2)
                aug_image = self._rotate_image(aug_image, angle)
                # Note: Small rotations don't significantly affect measurements

            results.append((aug_image, aug_annotation))

        return results

    def _flip_annotation_horizontal(self, annotation: ImageAnnotation, width: int):
        """Flip annotation for horizontal flip"""
        for m in annotation.measurements:
            new_left = width - m.right_edge_x
            new_right = width - m.left_edge_x
            m.left_edge_x = new_left
            m.right_edge_x = new_right

    def _rotate_image(self, image: np.ndarray, angle: float) -> np.ndarray:
        """Rotate image by small angle"""
        if not CV2_AVAILABLE:
            return image

        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(image, matrix, (w, h), borderMode=cv2.BORDER_REFLECT)


if TORCH_AVAILABLE:
    class HRTEMDataset(Dataset):
        """
        PyTorch Dataset for HR-TEM images.

        Loads images and annotations for training deep learning models.
        """

        def __init__(self, data_manager: TrainingDataManager,
                     split: str = 'train',
                     transform=None,
                     target_size: Tuple[int, int] = (256, 256),
                     depths_nm: List[float] = None):
            """
            Args:
                data_manager: TrainingDataManager instance
                split: 'train' or 'val'
                transform: Optional transforms
                target_size: Target image size (H, W)
                depths_nm: List of depth values to predict
            """
            self.data_manager = data_manager
            self.split = split
            self.transform = transform
            self.target_size = target_size
            self.depths_nm = depths_nm or [5, 10, 15, 20, 25]

            # Load data
            self.samples = self._load_samples()

        def _load_samples(self) -> List[Tuple[str, ImageAnnotation]]:
            """Load samples based on split"""
            all_samples = self.data_manager.list_annotated_images()

            # Simple split based on index
            split_idx = int(len(all_samples) * 0.8)

            if self.split == 'train':
                return all_samples[:split_idx]
            else:
                return all_samples[split_idx:]

        def __len__(self) -> int:
            return len(self.samples)

        def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
            image_path, annotation = self.samples[idx]

            # Load image
            if PIL_AVAILABLE:
                image = np.array(Image.open(image_path).convert('L'))
            elif CV2_AVAILABLE:
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            else:
                raise RuntimeError("Neither PIL nor OpenCV available")

            # Resize
            if CV2_AVAILABLE:
                image = cv2.resize(image, self.target_size[::-1])
            elif PIL_AVAILABLE:
                img_pil = Image.fromarray(image)
                img_pil = img_pil.resize(self.target_size[::-1])
                image = np.array(img_pil)

            # Normalize to [0, 1]
            image = image.astype(np.float32) / 255.0

            # Create edge map target (simplified)
            edge_map = self._create_edge_map(annotation, self.target_size)

            # Create CD targets
            cd_targets = self._create_cd_targets(annotation)

            # Apply transforms
            if self.transform:
                image = self.transform(image)

            # Convert to tensors
            image_tensor = torch.from_numpy(image).unsqueeze(0)  # (1, H, W)
            edge_tensor = torch.from_numpy(edge_map).unsqueeze(0)  # (1, H, W)
            cd_tensor = torch.tensor(cd_targets, dtype=torch.float32)

            return {
                'image': image_tensor,
                'edge_map': edge_tensor,
                'cd_values': cd_tensor,
                'scale': torch.tensor(annotation.scale_nm_per_pixel, dtype=torch.float32),
            }

        def _create_edge_map(self, annotation: ImageAnnotation,
                             target_size: Tuple[int, int]) -> np.ndarray:
            """Create edge map from annotation"""
            edge_map = np.zeros(target_size, dtype=np.float32)

            orig_h = annotation.image_height or target_size[0]
            orig_w = annotation.image_width or target_size[1]

            scale_y = target_size[0] / orig_h
            scale_x = target_size[1] / orig_w

            for m in annotation.measurements:
                y = int(m.y_position * scale_y)
                left_x = int(m.left_edge_x * scale_x)
                right_x = int(m.right_edge_x * scale_x)

                # Draw edges with Gaussian profile
                if 0 <= y < target_size[0]:
                    for x in [left_x, right_x]:
                        if 0 <= x < target_size[1]:
                            # Draw vertical line around edge
                            for dy in range(-3, 4):
                                yy = y + dy
                                if 0 <= yy < target_size[0]:
                                    weight = np.exp(-dy**2 / 2)
                                    edge_map[yy, x] = max(edge_map[yy, x], weight)

            return edge_map

        def _create_cd_targets(self, annotation: ImageAnnotation) -> List[float]:
            """Create CD targets for each depth"""
            targets = []

            # Create a mapping from depth to measurement
            depth_to_thickness = {}
            for m in annotation.measurements:
                depth_to_thickness[m.depth_nm] = m.thickness_nm

            for depth in self.depths_nm:
                if depth in depth_to_thickness:
                    targets.append(depth_to_thickness[depth])
                else:
                    # Interpolate or use 0
                    targets.append(0.0)

            return targets


def create_dataloader(data_manager: TrainingDataManager,
                      split: str = 'train',
                      batch_size: int = 8,
                      num_workers: int = 0,
                      target_size: Tuple[int, int] = (256, 256)) -> 'DataLoader':
    """
    Create a DataLoader for training.

    Args:
        data_manager: TrainingDataManager instance
        split: 'train' or 'val'
        batch_size: Batch size
        num_workers: Number of data loading workers (0 for CPU)
        target_size: Target image size

    Returns:
        DataLoader instance
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required for DataLoader")

    dataset = HRTEMDataset(data_manager, split=split, target_size=target_size)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        num_workers=num_workers,
        pin_memory=False,  # CPU-friendly
    )
