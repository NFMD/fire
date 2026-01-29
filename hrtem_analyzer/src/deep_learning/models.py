"""
Lightweight Deep Learning Models for HR-TEM CD Measurement

Optimized for CPU and integrated GPU (Intel UHD, AMD Vega, etc.)
- MobileNetV3-based encoder
- Lightweight U-Net for edge segmentation
- Regression head for direct CD prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List
import math


def get_device() -> torch.device:
    """Get the best available device (CUDA > MPS > CPU)"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')  # Apple Silicon
    else:
        return torch.device('cpu')


class ConvBNReLU(nn.Module):
    """Convolution + BatchNorm + ReLU block"""

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int = 3, stride: int = 1, groups: int = 1):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride, padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU6(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class DepthwiseSeparableConv(nn.Module):
    """Depthwise Separable Convolution - efficient for mobile/edge devices"""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.depthwise = ConvBNReLU(in_channels, in_channels,
                                     kernel_size=3, stride=stride, groups=in_channels)
        self.pointwise = ConvBNReLU(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class InvertedResidual(nn.Module):
    """MobileNetV2/V3 style inverted residual block"""

    def __init__(self, in_channels: int, out_channels: int,
                 stride: int = 1, expand_ratio: int = 6):
        super().__init__()
        self.stride = stride
        self.use_residual = stride == 1 and in_channels == out_channels

        hidden_dim = in_channels * expand_ratio

        layers = []
        if expand_ratio != 1:
            layers.append(ConvBNReLU(in_channels, hidden_dim, kernel_size=1))

        layers.extend([
            # Depthwise
            ConvBNReLU(hidden_dim, hidden_dim, kernel_size=3,
                       stride=stride, groups=hidden_dim),
            # Pointwise linear
            nn.Conv2d(hidden_dim, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels),
        ])

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_residual:
            return x + self.conv(x)
        return self.conv(x)


class LightweightEncoder(nn.Module):
    """
    Lightweight encoder based on MobileNetV3 architecture.
    Suitable for CPU and integrated GPU inference.
    """

    def __init__(self, in_channels: int = 1):
        super().__init__()

        # Initial conv
        self.stem = ConvBNReLU(in_channels, 16, kernel_size=3, stride=2)

        # Encoder stages with increasing channels
        self.stage1 = nn.Sequential(
            InvertedResidual(16, 24, stride=2, expand_ratio=4),
            InvertedResidual(24, 24, stride=1, expand_ratio=3),
        )

        self.stage2 = nn.Sequential(
            InvertedResidual(24, 40, stride=2, expand_ratio=3),
            InvertedResidual(40, 40, stride=1, expand_ratio=3),
            InvertedResidual(40, 40, stride=1, expand_ratio=3),
        )

        self.stage3 = nn.Sequential(
            InvertedResidual(40, 80, stride=2, expand_ratio=6),
            InvertedResidual(80, 80, stride=1, expand_ratio=4),
            InvertedResidual(80, 80, stride=1, expand_ratio=4),
        )

        self.stage4 = nn.Sequential(
            InvertedResidual(80, 112, stride=2, expand_ratio=6),
            InvertedResidual(112, 112, stride=1, expand_ratio=6),
        )

        # Feature channels at each stage
        self.feature_channels = [16, 24, 40, 80, 112]

    def forward(self, x) -> List[torch.Tensor]:
        """Returns feature maps at multiple scales"""
        features = []

        x = self.stem(x)
        features.append(x)  # 1/2

        x = self.stage1(x)
        features.append(x)  # 1/4

        x = self.stage2(x)
        features.append(x)  # 1/8

        x = self.stage3(x)
        features.append(x)  # 1/16

        x = self.stage4(x)
        features.append(x)  # 1/32

        return features


class LightweightDecoder(nn.Module):
    """Lightweight decoder for segmentation"""

    def __init__(self, encoder_channels: List[int], out_channels: int = 1):
        super().__init__()

        # Reverse order for decoding
        channels = encoder_channels[::-1]

        self.up_blocks = nn.ModuleList()
        self.fusion_blocks = nn.ModuleList()

        for i in range(len(channels) - 1):
            in_ch = channels[i]
            skip_ch = channels[i + 1]
            out_ch = skip_ch

            self.up_blocks.append(
                nn.ConvTranspose2d(in_ch, in_ch, kernel_size=2, stride=2)
            )
            self.fusion_blocks.append(
                DepthwiseSeparableConv(in_ch + skip_ch, out_ch)
            )

        self.final = nn.Conv2d(channels[-1], out_channels, kernel_size=1)

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        features = features[::-1]  # Reverse for bottom-up

        x = features[0]

        for i, (up, fusion) in enumerate(zip(self.up_blocks, self.fusion_blocks)):
            x = up(x)
            skip = features[i + 1]

            # Handle size mismatch
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)

            x = torch.cat([x, skip], dim=1)
            x = fusion(x)

        return self.final(x)


class EdgeSegmentationNet(nn.Module):
    """
    Lightweight edge segmentation network for HR-TEM images.

    Input: Grayscale HR-TEM image
    Output: Edge probability map

    Architecture: Lightweight U-Net with MobileNetV3-style encoder
    """

    def __init__(self, in_channels: int = 1):
        super().__init__()
        self.encoder = LightweightEncoder(in_channels)
        self.decoder = LightweightDecoder(self.encoder.feature_channels, out_channels=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        edge_map = self.decoder(features)
        return torch.sigmoid(edge_map)

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


class CDRegressionHead(nn.Module):
    """
    Regression head for direct CD (Critical Dimension) prediction.

    Takes encoder features and predicts CD values at specified depths.
    """

    def __init__(self, encoder_channels: List[int], num_depths: int = 5):
        super().__init__()

        # Global average pooling + FC layers
        total_channels = sum(encoder_channels)

        self.pools = nn.ModuleList([
            nn.AdaptiveAvgPool2d(1) for _ in encoder_channels
        ])

        self.fc = nn.Sequential(
            nn.Linear(total_channels, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_depths),  # One CD prediction per depth
        )

        self.num_depths = num_depths

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        pooled = []
        for pool, feat in zip(self.pools, features):
            pooled.append(pool(feat).flatten(1))

        x = torch.cat(pooled, dim=1)
        return self.fc(x)


class CDMeasurementNet(nn.Module):
    """
    Complete CD measurement network.

    Combines:
    - Edge segmentation for visualization
    - Direct CD regression for measurement

    Input: Grayscale HR-TEM image (B, 1, H, W)
    Output:
        - edge_map: Edge probability map (B, 1, H, W)
        - cd_values: Predicted CD at each depth (B, num_depths)
        - confidence: Prediction confidence (B, num_depths)
    """

    def __init__(self, in_channels: int = 1, num_depths: int = 5):
        super().__init__()

        self.encoder = LightweightEncoder(in_channels)
        self.edge_decoder = LightweightDecoder(self.encoder.feature_channels, out_channels=1)
        self.cd_head = CDRegressionHead(self.encoder.feature_channels, num_depths)

        # Confidence head
        self.confidence_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self.encoder.feature_channels[-1], 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_depths),
            nn.Sigmoid(),
        )

        self.num_depths = num_depths

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.encoder(x)

        edge_map = torch.sigmoid(self.edge_decoder(features))
        cd_values = self.cd_head(features)
        confidence = self.confidence_head(features[-1])

        return edge_map, cd_values, confidence

    def predict(self, x: torch.Tensor) -> dict:
        """Convenience method for inference"""
        self.eval()
        with torch.no_grad():
            edge_map, cd_values, confidence = self.forward(x)

        return {
            'edge_map': edge_map,
            'cd_values': cd_values,
            'confidence': confidence
        }

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


class EnsembleModel(nn.Module):
    """
    Ensemble of multiple models for improved accuracy.
    Uses model averaging for more robust predictions.
    """

    def __init__(self, num_models: int = 3, in_channels: int = 1, num_depths: int = 5):
        super().__init__()

        self.models = nn.ModuleList([
            CDMeasurementNet(in_channels, num_depths) for _ in range(num_models)
        ])
        self.num_models = num_models

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        edge_maps = []
        cd_values = []
        confidences = []

        for model in self.models:
            edge, cd, conf = model(x)
            edge_maps.append(edge)
            cd_values.append(cd)
            confidences.append(conf)

        # Average predictions
        avg_edge = torch.stack(edge_maps).mean(dim=0)
        avg_cd = torch.stack(cd_values).mean(dim=0)
        avg_conf = torch.stack(confidences).mean(dim=0)

        # Adjust confidence based on model agreement
        cd_std = torch.stack(cd_values).std(dim=0)
        agreement_bonus = torch.exp(-cd_std / 10)  # Higher agreement = higher confidence
        final_conf = avg_conf * agreement_bonus

        return avg_edge, avg_cd, final_conf


def create_model(model_type: str = 'cd_measurement',
                 num_depths: int = 5,
                 pretrained_path: Optional[str] = None) -> nn.Module:
    """
    Factory function to create models.

    Args:
        model_type: 'edge_segmentation', 'cd_measurement', or 'ensemble'
        num_depths: Number of depth measurements
        pretrained_path: Path to pretrained weights

    Returns:
        Model instance
    """
    if model_type == 'edge_segmentation':
        model = EdgeSegmentationNet(in_channels=1)
    elif model_type == 'cd_measurement':
        model = CDMeasurementNet(in_channels=1, num_depths=num_depths)
    elif model_type == 'ensemble':
        model = EnsembleModel(num_models=3, in_channels=1, num_depths=num_depths)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    if pretrained_path:
        state_dict = torch.load(pretrained_path, map_location='cpu')
        model.load_state_dict(state_dict)

    return model


# Model information for UI
MODEL_INFO = {
    'edge_segmentation': {
        'name': 'Edge Segmentation',
        'description': 'Detects edges in HR-TEM images',
        'params': '~500K',
        'input': 'Grayscale image',
        'output': 'Edge probability map',
    },
    'cd_measurement': {
        'name': 'CD Measurement',
        'description': 'Direct Critical Dimension prediction',
        'params': '~800K',
        'input': 'Grayscale image + depth config',
        'output': 'CD values at each depth',
    },
    'ensemble': {
        'name': 'Ensemble Model',
        'description': 'Ensemble of 3 models for higher accuracy',
        'params': '~2.4M',
        'input': 'Grayscale image + depth config',
        'output': 'Averaged CD predictions with confidence',
    },
}
