"""
Inference Module for HR-TEM Deep Learning Models

Provides easy-to-use inference API for CD measurement.
Optimized for CPU and integrated GPU.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

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

from .models import create_model, get_device, CDMeasurementNet


@dataclass
class CDPrediction:
    """Result of CD prediction"""
    depth_nm: float
    thickness_nm: float
    confidence: float
    left_edge_x: Optional[int] = None
    right_edge_x: Optional[int] = None


@dataclass
class InferenceResult:
    """Complete inference result for an image"""
    predictions: List[CDPrediction]
    edge_map: Optional[np.ndarray]
    processing_time_ms: float
    model_name: str
    image_size: Tuple[int, int]

    def to_dict(self) -> dict:
        return {
            'predictions': [
                {
                    'depth_nm': p.depth_nm,
                    'thickness_nm': p.thickness_nm,
                    'confidence': p.confidence,
                    'left_edge_x': p.left_edge_x,
                    'right_edge_x': p.right_edge_x,
                }
                for p in self.predictions
            ],
            'processing_time_ms': self.processing_time_ms,
            'model_name': self.model_name,
            'image_size': self.image_size,
        }


class DeepLearningInference:
    """
    Deep Learning inference for HR-TEM CD measurement.

    Optimized for CPU and integrated GPU inference.
    """

    def __init__(self,
                 model_path: Optional[str] = None,
                 model_type: str = 'cd_measurement',
                 num_depths: int = 5,
                 depths_nm: Optional[List[float]] = None,
                 device: Optional[str] = None):
        """
        Initialize inference engine.

        Args:
            model_path: Path to trained model weights
            model_type: Model type ('cd_measurement', 'edge_segmentation', 'ensemble')
            num_depths: Number of depth measurements
            depths_nm: List of depth values (nm)
            device: Device to use ('cpu', 'cuda', 'mps', or None for auto)
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required for inference. "
                             "Install with: pip install torch")

        self.num_depths = num_depths
        self.depths_nm = depths_nm or [5.0, 10.0, 15.0, 20.0, 25.0]
        self.model_type = model_type

        # Set device
        if device:
            self.device = torch.device(device)
        else:
            self.device = get_device()

        print(f"Inference device: {self.device}")

        # Load model
        self.model = create_model(model_type, num_depths=num_depths)

        if model_path and Path(model_path).exists():
            self._load_weights(model_path)

        self.model.to(self.device)
        self.model.eval()

        # Compile model for faster inference (PyTorch 2.0+)
        if hasattr(torch, 'compile') and self.device.type != 'mps':
            try:
                self.model = torch.compile(self.model, mode='reduce-overhead')
                print("Model compiled for faster inference")
            except Exception:
                pass  # Compilation not supported

    def _load_weights(self, path: str):
        """Load model weights from file"""
        checkpoint = torch.load(path, map_location='cpu')

        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)

        print(f"Loaded model from {path}")

    def predict(self,
                image: Union[str, np.ndarray],
                scale_nm_per_pixel: float = 1.0,
                baseline_y: Optional[int] = None,
                return_edge_map: bool = True) -> InferenceResult:
        """
        Predict CD measurements for an image.

        Args:
            image: Image path or numpy array (H, W) or (H, W, C)
            scale_nm_per_pixel: Scale factor for converting pixels to nm
            baseline_y: Baseline Y position (optional, for edge position calculation)
            return_edge_map: Whether to return the edge probability map

        Returns:
            InferenceResult with predictions
        """
        import time
        start_time = time.time()

        # Load and preprocess image
        img_array = self._load_image(image)
        original_size = img_array.shape[:2]

        # Prepare input tensor
        input_tensor = self._preprocess(img_array)
        input_tensor = input_tensor.to(self.device)

        # Inference
        with torch.no_grad():
            if self.model_type == 'edge_segmentation':
                edge_map = self.model(input_tensor)
                cd_values = None
                confidence = None
            else:
                edge_map, cd_values, confidence = self.model(input_tensor)

        # Process outputs
        edge_map_np = None
        if return_edge_map and edge_map is not None:
            edge_map_np = edge_map[0, 0].cpu().numpy()
            # Resize back to original size
            edge_map_np = self._resize_array(edge_map_np, original_size)

        # Create predictions
        predictions = []
        if cd_values is not None:
            cd_np = cd_values[0].cpu().numpy()
            conf_np = confidence[0].cpu().numpy() if confidence is not None else np.ones(len(cd_np))

            for i, (depth, cd, conf) in enumerate(zip(self.depths_nm, cd_np, conf_np)):
                # Convert CD from model units to nm using scale
                thickness_nm = float(cd) * scale_nm_per_pixel

                # Calculate edge positions if baseline is provided
                left_x, right_x = None, None
                if baseline_y is not None and edge_map_np is not None:
                    left_x, right_x = self._find_edges_in_map(
                        edge_map_np, baseline_y, depth, scale_nm_per_pixel
                    )

                predictions.append(CDPrediction(
                    depth_nm=depth,
                    thickness_nm=thickness_nm,
                    confidence=float(conf),
                    left_edge_x=left_x,
                    right_edge_x=right_x,
                ))

        processing_time = (time.time() - start_time) * 1000

        return InferenceResult(
            predictions=predictions,
            edge_map=edge_map_np,
            processing_time_ms=processing_time,
            model_name=self.model_type,
            image_size=original_size,
        )

    def _load_image(self, image: Union[str, np.ndarray]) -> np.ndarray:
        """Load image from path or array"""
        if isinstance(image, str):
            if PIL_AVAILABLE:
                img = Image.open(image).convert('L')
                return np.array(img)
            elif CV2_AVAILABLE:
                return cv2.imread(image, cv2.IMREAD_GRAYSCALE)
            else:
                raise RuntimeError("Neither PIL nor OpenCV available")
        else:
            # Assume numpy array
            if len(image.shape) == 3:
                # Convert to grayscale
                if image.shape[2] == 3:
                    if CV2_AVAILABLE:
                        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    else:
                        return np.mean(image, axis=2).astype(np.uint8)
                else:
                    return image[:, :, 0]
            return image

    def _preprocess(self, image: np.ndarray, target_size: Tuple[int, int] = (256, 256)) -> torch.Tensor:
        """Preprocess image for model input"""
        # Resize
        if CV2_AVAILABLE:
            resized = cv2.resize(image, target_size[::-1])
        elif PIL_AVAILABLE:
            img_pil = Image.fromarray(image)
            img_pil = img_pil.resize(target_size[::-1])
            resized = np.array(img_pil)
        else:
            # Simple nearest neighbor resize
            resized = self._simple_resize(image, target_size)

        # Normalize to [0, 1]
        normalized = resized.astype(np.float32) / 255.0

        # Add batch and channel dimensions: (H, W) -> (1, 1, H, W)
        tensor = torch.from_numpy(normalized).unsqueeze(0).unsqueeze(0)

        return tensor

    def _simple_resize(self, image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """Simple nearest-neighbor resize without external libraries"""
        h, w = image.shape[:2]
        new_h, new_w = target_size

        row_indices = (np.arange(new_h) * h / new_h).astype(int)
        col_indices = (np.arange(new_w) * w / new_w).astype(int)

        return image[row_indices][:, col_indices]

    def _resize_array(self, array: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """Resize array to target size"""
        if CV2_AVAILABLE:
            return cv2.resize(array, target_size[::-1])
        elif PIL_AVAILABLE:
            img = Image.fromarray((array * 255).astype(np.uint8))
            img = img.resize(target_size[::-1])
            return np.array(img).astype(np.float32) / 255
        else:
            return self._simple_resize(array, target_size)

    def _find_edges_in_map(self,
                           edge_map: np.ndarray,
                           baseline_y: int,
                           depth_nm: float,
                           scale_nm_per_pixel: float) -> Tuple[Optional[int], Optional[int]]:
        """Find edge positions in edge map at given depth"""
        depth_pixels = int(depth_nm / scale_nm_per_pixel)
        y = baseline_y + depth_pixels

        if y < 0 or y >= edge_map.shape[0]:
            return None, None

        # Get profile at this y position
        profile = edge_map[y, :]

        # Find peaks (edges)
        threshold = 0.3
        above_threshold = profile > threshold

        # Find first and last edge
        edges = np.where(above_threshold)[0]

        if len(edges) >= 2:
            return int(edges[0]), int(edges[-1])
        elif len(edges) == 1:
            return int(edges[0]), None

        return None, None

    def batch_predict(self,
                      images: List[Union[str, np.ndarray]],
                      scale_nm_per_pixel: float = 1.0) -> List[InferenceResult]:
        """
        Batch prediction for multiple images.

        Args:
            images: List of image paths or arrays
            scale_nm_per_pixel: Scale factor

        Returns:
            List of InferenceResult
        """
        results = []

        for image in images:
            result = self.predict(image, scale_nm_per_pixel)
            results.append(result)

        return results

    def get_model_info(self) -> Dict:
        """Get information about the loaded model"""
        return {
            'model_type': self.model_type,
            'num_depths': self.num_depths,
            'depths_nm': self.depths_nm,
            'device': str(self.device),
            'num_parameters': sum(p.numel() for p in self.model.parameters()),
        }


class HybridMeasurer:
    """
    Combines deep learning predictions with traditional CV methods.

    Uses DL for initial prediction, then refines with traditional methods.
    """

    def __init__(self,
                 dl_model_path: Optional[str] = None,
                 use_dl: bool = True,
                 dl_weight: float = 0.5):
        """
        Args:
            dl_model_path: Path to trained deep learning model
            use_dl: Whether to use deep learning
            dl_weight: Weight for DL predictions (0-1)
        """
        self.use_dl = use_dl and TORCH_AVAILABLE
        self.dl_weight = dl_weight

        if self.use_dl:
            self.dl_inference = DeepLearningInference(model_path=dl_model_path)
        else:
            self.dl_inference = None

    def measure(self,
                image: np.ndarray,
                baseline_y: int,
                depths_nm: List[float],
                scale_nm_per_pixel: float,
                traditional_results: Optional[Dict] = None) -> Dict:
        """
        Measure CD using hybrid approach.

        Args:
            image: Input image
            baseline_y: Baseline Y position
            depths_nm: Depths to measure
            scale_nm_per_pixel: Scale factor
            traditional_results: Results from traditional CV methods

        Returns:
            Combined measurement results
        """
        results = {}

        # Get DL predictions
        dl_predictions = {}
        if self.use_dl and self.dl_inference:
            try:
                inference_result = self.dl_inference.predict(
                    image, scale_nm_per_pixel, baseline_y
                )

                for pred in inference_result.predictions:
                    dl_predictions[pred.depth_nm] = {
                        'thickness_nm': pred.thickness_nm,
                        'confidence': pred.confidence,
                    }
            except Exception as e:
                print(f"DL inference failed: {e}")

        # Combine with traditional results
        for depth in depths_nm:
            dl_result = dl_predictions.get(depth, {})
            trad_result = traditional_results.get(depth, {}) if traditional_results else {}

            dl_thickness = dl_result.get('thickness_nm', 0)
            dl_conf = dl_result.get('confidence', 0)
            trad_thickness = trad_result.get('thickness_nm', 0)
            trad_conf = trad_result.get('confidence', 0)

            # Weighted combination
            if dl_thickness > 0 and trad_thickness > 0:
                combined_thickness = (
                    self.dl_weight * dl_thickness +
                    (1 - self.dl_weight) * trad_thickness
                )
                combined_conf = (dl_conf + trad_conf) / 2
            elif dl_thickness > 0:
                combined_thickness = dl_thickness
                combined_conf = dl_conf * 0.8  # Lower confidence without validation
            elif trad_thickness > 0:
                combined_thickness = trad_thickness
                combined_conf = trad_conf
            else:
                combined_thickness = 0
                combined_conf = 0

            results[depth] = {
                'thickness_nm': combined_thickness,
                'confidence': combined_conf,
                'dl_thickness': dl_thickness,
                'trad_thickness': trad_thickness,
                'method': 'hybrid' if dl_thickness > 0 and trad_thickness > 0 else
                         ('dl' if dl_thickness > 0 else 'traditional'),
            }

        return results


def load_inference_engine(model_path: Optional[str] = None,
                         depths_nm: Optional[List[float]] = None) -> DeepLearningInference:
    """
    Convenience function to load inference engine.

    Args:
        model_path: Path to trained model (None for untrained model)
        depths_nm: Depth values in nm

    Returns:
        DeepLearningInference instance
    """
    return DeepLearningInference(
        model_path=model_path,
        depths_nm=depths_nm
    )
