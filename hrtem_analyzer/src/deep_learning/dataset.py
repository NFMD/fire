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
    class HRTEMPatchDataset(Dataset):
        """
        Patch-based PyTorch Dataset for HR-TEM images.

        Instead of resizing the entire image, extracts patches at original resolution
        to preserve fine details critical for precise CD measurement.

        Approach:
        - Extract multiple patches from each image
        - Focus patches on edge regions (where measurements exist)
        - Include context patches for background learning
        - During inference, use sliding window with overlap
        """

        def __init__(self, data_manager: TrainingDataManager,
                     split: str = 'train',
                     transform=None,
                     patch_size: int = 256,
                     patches_per_image: int = 20,
                     edge_focus_ratio: float = 0.7,
                     overlap_ratio: float = 0.5,
                     depths_nm: List[float] = None):
            """
            Args:
                data_manager: TrainingDataManager instance
                split: 'train' or 'val'
                transform: Optional transforms
                patch_size: Size of patches to extract (square)
                patches_per_image: Number of patches to extract per image
                edge_focus_ratio: Ratio of patches centered on edges vs random
                overlap_ratio: Overlap between patches for sliding window
                depths_nm: List of depth values to predict
            """
            self.data_manager = data_manager
            self.split = split
            self.transform = transform
            self.patch_size = patch_size
            self.patches_per_image = patches_per_image
            self.edge_focus_ratio = edge_focus_ratio
            self.overlap_ratio = overlap_ratio
            self.depths_nm = depths_nm or [5, 10, 15, 20, 25]

            # Random generator for reproducibility
            self.rng = np.random.default_rng(42 if split == 'train' else 43)

            # Load image list and pre-compute patch locations
            self.image_annotations = self._load_samples()
            self.patch_index = self._build_patch_index()

        def _load_samples(self) -> List[Tuple[str, ImageAnnotation]]:
            """Load samples based on split"""
            all_samples = self.data_manager.list_annotated_images()

            # Shuffle with fixed seed for reproducibility
            rng = np.random.default_rng(42)
            indices = rng.permutation(len(all_samples))
            all_samples = [all_samples[i] for i in indices]

            split_idx = int(len(all_samples) * 0.8)

            if self.split == 'train':
                return all_samples[:split_idx]
            else:
                return all_samples[split_idx:]

        def _build_patch_index(self) -> List[Tuple[int, int, int, List]]:
            """
            Build index of all patches to extract.

            Returns:
                List of (image_idx, patch_x, patch_y, measurements_in_patch)
            """
            patch_index = []

            for img_idx, (image_path, annotation) in enumerate(self.image_annotations):
                # Get image dimensions
                if PIL_AVAILABLE:
                    with Image.open(image_path) as img:
                        img_w, img_h = img.size
                elif CV2_AVAILABLE:
                    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                    img_h, img_w = img.shape
                else:
                    continue

                # Collect edge positions for focused sampling
                edge_positions = []
                for m in annotation.measurements:
                    # Left edge
                    edge_positions.append((m.left_edge_x, m.y_position, m))
                    # Right edge
                    edge_positions.append((m.right_edge_x, m.y_position, m))

                # Number of edge-focused patches
                n_edge_patches = int(self.patches_per_image * self.edge_focus_ratio)
                n_random_patches = self.patches_per_image - n_edge_patches

                half_patch = self.patch_size // 2

                # Extract edge-focused patches
                if edge_positions:
                    for _ in range(n_edge_patches):
                        # Randomly select an edge
                        ex, ey, _ = edge_positions[self.rng.integers(len(edge_positions))]

                        # Add jitter to avoid always centering exactly on edge
                        jitter_x = self.rng.integers(-half_patch//2, half_patch//2)
                        jitter_y = self.rng.integers(-half_patch//2, half_patch//2)

                        # Calculate patch top-left corner
                        px = max(0, min(img_w - self.patch_size, ex - half_patch + jitter_x))
                        py = max(0, min(img_h - self.patch_size, ey - half_patch + jitter_y))

                        # Find measurements in this patch
                        measurements = self._get_measurements_in_patch(
                            annotation.measurements, px, py, self.patch_size
                        )

                        patch_index.append((img_idx, px, py, measurements))

                # Extract random patches for background context
                for _ in range(n_random_patches):
                    px = self.rng.integers(0, max(1, img_w - self.patch_size))
                    py = self.rng.integers(0, max(1, img_h - self.patch_size))

                    measurements = self._get_measurements_in_patch(
                        annotation.measurements, px, py, self.patch_size
                    )

                    patch_index.append((img_idx, px, py, measurements))

            return patch_index

        def _get_measurements_in_patch(self, measurements: List[CDAnnotation],
                                       px: int, py: int, size: int) -> List[CDAnnotation]:
            """Get measurements that fall within a patch"""
            result = []
            for m in measurements:
                # Check if measurement y position is in patch
                if py <= m.y_position < py + size:
                    # Check if at least one edge is in patch
                    if (px <= m.left_edge_x < px + size or
                        px <= m.right_edge_x < px + size):
                        # Create adjusted measurement relative to patch
                        adjusted = CDAnnotation(
                            depth_nm=m.depth_nm,
                            thickness_nm=m.thickness_nm,
                            left_edge_x=m.left_edge_x - px,
                            right_edge_x=m.right_edge_x - px,
                            y_position=m.y_position - py,
                            confidence=m.confidence,
                            annotator=m.annotator
                        )
                        result.append(adjusted)
            return result

        def __len__(self) -> int:
            return len(self.patch_index)

        def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
            img_idx, px, py, measurements = self.patch_index[idx]
            image_path, annotation = self.image_annotations[img_idx]

            # Load full image
            if PIL_AVAILABLE:
                full_image = np.array(Image.open(image_path).convert('L'))
            elif CV2_AVAILABLE:
                full_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            else:
                raise RuntimeError("Neither PIL nor OpenCV available")

            # Extract patch (no resize - original resolution!)
            patch = full_image[py:py+self.patch_size, px:px+self.patch_size]

            # Pad if necessary (edge cases)
            if patch.shape[0] < self.patch_size or patch.shape[1] < self.patch_size:
                padded = np.zeros((self.patch_size, self.patch_size), dtype=patch.dtype)
                padded[:patch.shape[0], :patch.shape[1]] = patch
                patch = padded

            # Normalize to [0, 1]
            patch = patch.astype(np.float32) / 255.0

            # Create edge map for this patch
            edge_map = self._create_patch_edge_map(measurements)

            # Create CD regression targets for this patch
            cd_targets, cd_mask = self._create_patch_cd_targets(measurements)

            # Apply transforms (augmentation)
            if self.transform and self.split == 'train':
                patch, edge_map = self.transform(patch, edge_map)

            # Convert to tensors
            patch_tensor = torch.from_numpy(patch).unsqueeze(0)  # (1, H, W)
            edge_tensor = torch.from_numpy(edge_map).unsqueeze(0)  # (1, H, W)
            cd_tensor = torch.tensor(cd_targets, dtype=torch.float32)
            cd_mask_tensor = torch.tensor(cd_mask, dtype=torch.float32)

            return {
                'image': patch_tensor,
                'edge_map': edge_tensor,
                'cd_values': cd_tensor,
                'cd_mask': cd_mask_tensor,  # 1 where we have valid measurements
                'scale': torch.tensor(annotation.scale_nm_per_pixel, dtype=torch.float32),
                'patch_coords': torch.tensor([px, py], dtype=torch.int32),
                'has_edges': torch.tensor(len(measurements) > 0, dtype=torch.bool),
            }

        def _create_patch_edge_map(self, measurements: List[CDAnnotation]) -> np.ndarray:
            """Create high-resolution edge map for a patch"""
            edge_map = np.zeros((self.patch_size, self.patch_size), dtype=np.float32)

            for m in measurements:
                y = m.y_position
                left_x = m.left_edge_x
                right_x = m.right_edge_x

                # Draw edges with sub-pixel Gaussian profile for precise learning
                for x in [left_x, right_x]:
                    if 0 <= x < self.patch_size:
                        # Vertical edge line with Gaussian spread
                        for dy in range(-5, 6):
                            yy = y + dy
                            if 0 <= yy < self.patch_size:
                                # Gaussian weight for vertical spread
                                y_weight = np.exp(-dy**2 / 4.0)

                                # Sub-pixel edge with horizontal Gaussian
                                for dx in range(-3, 4):
                                    xx = x + dx
                                    if 0 <= xx < self.patch_size:
                                        x_weight = np.exp(-dx**2 / 1.0)
                                        edge_map[yy, xx] = max(
                                            edge_map[yy, xx],
                                            y_weight * x_weight
                                        )

            return edge_map

        def _create_patch_cd_targets(self, measurements: List[CDAnnotation]) -> Tuple[List[float], List[float]]:
            """Create CD targets for each depth in this patch"""
            targets = []
            mask = []

            # Create a mapping from depth to measurement
            depth_to_thickness = {}
            for m in measurements:
                depth_to_thickness[m.depth_nm] = m.thickness_nm

            for depth in self.depths_nm:
                if depth in depth_to_thickness:
                    targets.append(depth_to_thickness[depth])
                    mask.append(1.0)  # Valid measurement
                else:
                    targets.append(0.0)
                    mask.append(0.0)  # No measurement at this depth

            return targets, mask

    # Keep old name for backwards compatibility
    HRTEMDataset = HRTEMPatchDataset


class SlidingWindowInference:
    """
    Sliding window inference for full-resolution HR-TEM images.

    Scans the image with overlapping patches and aggregates predictions
    for high-precision CD measurement.
    """

    def __init__(self, model, patch_size: int = 256,
                 stride: int = 128, device: str = 'cpu'):
        """
        Args:
            model: Trained PyTorch model
            patch_size: Size of patches (must match training)
            stride: Step size between patches (smaller = more overlap)
            device: 'cpu' or 'cuda'
        """
        self.model = model
        self.patch_size = patch_size
        self.stride = stride
        self.device = device

        self.model.to(device)
        self.model.eval()

    def predict_full_image(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Run inference on full resolution image.

        Args:
            image: Full resolution grayscale image (H, W)

        Returns:
            Dict with:
                'edge_map': Aggregated edge probability map
                'confidence_map': Prediction confidence at each pixel
                'cd_predictions': List of CD predictions per patch
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch required for inference")

        h, w = image.shape

        # Initialize accumulator maps
        edge_accum = np.zeros((h, w), dtype=np.float64)
        weight_accum = np.zeros((h, w), dtype=np.float64)
        cd_predictions = []

        # Create Gaussian weight kernel for smooth blending
        weight_kernel = self._create_weight_kernel()

        # Normalize image
        image_norm = image.astype(np.float32) / 255.0

        with torch.no_grad():
            # Slide window across image
            for y in range(0, h - self.patch_size + 1, self.stride):
                for x in range(0, w - self.patch_size + 1, self.stride):
                    # Extract patch
                    patch = image_norm[y:y+self.patch_size, x:x+self.patch_size]

                    # Convert to tensor
                    patch_tensor = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0)
                    patch_tensor = patch_tensor.to(self.device)

                    # Run inference
                    outputs = self.model(patch_tensor)

                    # Extract edge map prediction
                    if isinstance(outputs, dict):
                        edge_pred = outputs.get('edge_map', outputs.get('edges'))
                    else:
                        edge_pred = outputs[0]

                    edge_pred = edge_pred.squeeze().cpu().numpy()

                    # Accumulate with Gaussian weighting
                    edge_accum[y:y+self.patch_size, x:x+self.patch_size] += edge_pred * weight_kernel
                    weight_accum[y:y+self.patch_size, x:x+self.patch_size] += weight_kernel

                    # Store CD predictions if available
                    if isinstance(outputs, dict) and 'cd_values' in outputs:
                        cd_pred = outputs['cd_values'].squeeze().cpu().numpy()
                        cd_predictions.append({
                            'x': x,
                            'y': y,
                            'values': cd_pred.tolist()
                        })

            # Handle edges of image (areas not covered by full stride)
            # Process remaining right edge
            if w % self.stride != 0:
                x = w - self.patch_size
                for y in range(0, h - self.patch_size + 1, self.stride):
                    self._process_patch(image_norm, x, y, edge_accum, weight_accum,
                                       weight_kernel, cd_predictions)

            # Process remaining bottom edge
            if h % self.stride != 0:
                y = h - self.patch_size
                for x in range(0, w - self.patch_size + 1, self.stride):
                    self._process_patch(image_norm, x, y, edge_accum, weight_accum,
                                       weight_kernel, cd_predictions)

        # Normalize accumulated predictions
        edge_map = np.divide(edge_accum, weight_accum,
                            where=weight_accum > 0,
                            out=np.zeros_like(edge_accum))

        # Confidence based on number of overlapping predictions
        max_overlaps = (self.patch_size / self.stride) ** 2
        confidence_map = np.clip(weight_accum / max_overlaps, 0, 1)

        return {
            'edge_map': edge_map.astype(np.float32),
            'confidence_map': confidence_map.astype(np.float32),
            'cd_predictions': cd_predictions
        }

    def _process_patch(self, image_norm, x, y, edge_accum, weight_accum,
                       weight_kernel, cd_predictions):
        """Process a single patch"""
        patch = image_norm[y:y+self.patch_size, x:x+self.patch_size]
        patch_tensor = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0)
        patch_tensor = patch_tensor.to(self.device)

        outputs = self.model(patch_tensor)

        if isinstance(outputs, dict):
            edge_pred = outputs.get('edge_map', outputs.get('edges'))
        else:
            edge_pred = outputs[0]

        edge_pred = edge_pred.squeeze().cpu().numpy()

        edge_accum[y:y+self.patch_size, x:x+self.patch_size] += edge_pred * weight_kernel
        weight_accum[y:y+self.patch_size, x:x+self.patch_size] += weight_kernel

    def _create_weight_kernel(self) -> np.ndarray:
        """Create Gaussian weight kernel for smooth patch blending"""
        sigma = self.patch_size / 4
        center = self.patch_size // 2

        y, x = np.ogrid[:self.patch_size, :self.patch_size]
        weight = np.exp(-((x - center)**2 + (y - center)**2) / (2 * sigma**2))

        return weight

    def find_edges_at_y(self, edge_map: np.ndarray, y: int,
                        threshold: float = 0.5,
                        min_distance: int = 10) -> List[Tuple[float, float]]:
        """
        Find edge positions at a specific y coordinate with sub-pixel precision.

        Args:
            edge_map: Edge probability map
            y: Y coordinate (depth)
            threshold: Minimum edge probability
            min_distance: Minimum distance between edges

        Returns:
            List of (x_position, confidence) tuples
        """
        if y < 0 or y >= edge_map.shape[0]:
            return []

        line_profile = edge_map[y, :]

        edges = []

        # Find local maxima above threshold
        for x in range(1, len(line_profile) - 1):
            if (line_profile[x] > threshold and
                line_profile[x] > line_profile[x-1] and
                line_profile[x] > line_profile[x+1]):

                # Sub-pixel refinement using parabolic interpolation
                y0, y1, y2 = line_profile[x-1], line_profile[x], line_profile[x+1]
                denom = 2 * (2*y1 - y0 - y2)
                if abs(denom) > 1e-6:
                    x_subpixel = x + (y0 - y2) / denom
                else:
                    x_subpixel = float(x)

                edges.append((x_subpixel, float(line_profile[x])))

        # Filter by minimum distance
        if len(edges) > 1:
            filtered = [edges[0]]
            for edge in edges[1:]:
                if edge[0] - filtered[-1][0] >= min_distance:
                    filtered.append(edge)
            edges = filtered

        return edges


def create_dataloader(data_manager: TrainingDataManager,
                      split: str = 'train',
                      batch_size: int = 8,
                      num_workers: int = 0,
                      patch_size: int = 256,
                      patches_per_image: int = 20) -> 'DataLoader':
    """
    Create a DataLoader for patch-based training.

    Args:
        data_manager: TrainingDataManager instance
        split: 'train' or 'val'
        batch_size: Batch size
        num_workers: Number of data loading workers (0 for CPU)
        patch_size: Size of patches to extract
        patches_per_image: Number of patches per image

    Returns:
        DataLoader instance
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required for DataLoader")

    dataset = HRTEMPatchDataset(
        data_manager,
        split=split,
        patch_size=patch_size,
        patches_per_image=patches_per_image
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        num_workers=num_workers,
        pin_memory=False,  # CPU-friendly
    )
