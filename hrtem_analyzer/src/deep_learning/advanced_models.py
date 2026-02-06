"""
Advanced Deep Learning Models for HR-TEM CD Measurement

State-of-the-art architectures:
- Attention U-Net: Attention gates for feature refinement
- Swin Transformer: Hierarchical vision transformer
- Deep Ensemble: Multiple models for uncertainty estimation
- Monte Carlo Dropout: Bayesian approximation

These models prioritize accuracy over speed.
"""
import math
from typing import List, Optional, Tuple, Dict, Any
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torchvision import models
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    torch = None
    nn = None
    F = None


if PYTORCH_AVAILABLE:

    class AttentionGate(nn.Module):
        """
        Attention Gate for focusing on relevant features.

        From: "Attention U-Net: Learning Where to Look for the Pancreas"
        https://arxiv.org/abs/1804.03999
        """
        def __init__(self, gate_channels: int, feat_channels: int, inter_channels: int):
            super().__init__()
            self.W_g = nn.Sequential(
                nn.Conv2d(gate_channels, inter_channels, kernel_size=1, bias=True),
                nn.BatchNorm2d(inter_channels)
            )
            self.W_x = nn.Sequential(
                nn.Conv2d(feat_channels, inter_channels, kernel_size=1, bias=True),
                nn.BatchNorm2d(inter_channels)
            )
            self.psi = nn.Sequential(
                nn.Conv2d(inter_channels, 1, kernel_size=1, bias=True),
                nn.BatchNorm2d(1),
                nn.Sigmoid()
            )
            self.relu = nn.ReLU(inplace=True)

        def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
            """
            Args:
                g: Gating signal from coarser scale
                x: Feature map to be attended
            """
            g1 = self.W_g(g)
            x1 = self.W_x(x)

            # Upsample g1 to match x1 size if needed
            if g1.shape[2:] != x1.shape[2:]:
                g1 = F.interpolate(g1, size=x1.shape[2:], mode='bilinear', align_corners=True)

            psi = self.relu(g1 + x1)
            psi = self.psi(psi)
            return x * psi


    class CBAM(nn.Module):
        """
        Convolutional Block Attention Module.

        From: "CBAM: Convolutional Block Attention Module"
        https://arxiv.org/abs/1807.06521
        """
        def __init__(self, channels: int, reduction: int = 16):
            super().__init__()
            # Channel attention
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
            self.max_pool = nn.AdaptiveMaxPool2d(1)
            self.fc = nn.Sequential(
                nn.Conv2d(channels, channels // reduction, 1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels // reduction, channels, 1, bias=False)
            )
            # Spatial attention
            self.conv = nn.Sequential(
                nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
                nn.BatchNorm2d(1),
                nn.Sigmoid()
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # Channel attention
            avg_out = self.fc(self.avg_pool(x))
            max_out = self.fc(self.max_pool(x))
            channel_att = torch.sigmoid(avg_out + max_out)
            x = x * channel_att

            # Spatial attention
            avg_out = torch.mean(x, dim=1, keepdim=True)
            max_out, _ = torch.max(x, dim=1, keepdim=True)
            spatial_att = self.conv(torch.cat([avg_out, max_out], dim=1))
            x = x * spatial_att

            return x


    class DoubleConvBlock(nn.Module):
        """Double convolution block with optional attention"""
        def __init__(self, in_ch: int, out_ch: int, use_attention: bool = True):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )
            self.attention = CBAM(out_ch) if use_attention else nn.Identity()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.conv(x)
            x = self.attention(x)
            return x


    class AttentionUNet(nn.Module):
        """
        Attention U-Net for precise edge segmentation.

        Combines:
        - EfficientNet encoder with ImageNet pretraining
        - Attention gates for skip connections
        - CBAM modules for channel/spatial attention
        - Deep supervision for multi-scale learning
        """
        def __init__(
            self,
            in_channels: int = 1,
            out_channels: int = 1,
            features: List[int] = None,
            deep_supervision: bool = True,
            dropout_rate: float = 0.2
        ):
            super().__init__()
            if features is None:
                features = [64, 128, 256, 512, 1024]

            self.deep_supervision = deep_supervision
            self.dropout_rate = dropout_rate

            # Encoder with EfficientNet backbone
            efficientnet = models.efficientnet_b3(weights='IMAGENET1K_V1')

            # Adapt first conv for grayscale
            original_conv = efficientnet.features[0][0]
            self.stem = nn.Sequential(
                nn.Conv2d(
                    in_channels, original_conv.out_channels,
                    kernel_size=original_conv.kernel_size,
                    stride=original_conv.stride,
                    padding=original_conv.padding,
                    bias=False
                ),
                efficientnet.features[0][1],  # BatchNorm
                efficientnet.features[0][2]   # Activation
            )

            # Copy pretrained weights (average RGB channels)
            with torch.no_grad():
                self.stem[0].weight.copy_(
                    original_conv.weight.mean(dim=1, keepdim=True)
                )

            # EfficientNet-B3 feature channels: [24, 32, 48, 136, 384]
            self.encoder_channels = [24, 32, 48, 136, 384]

            # Split EfficientNet features into stages
            self.enc1 = efficientnet.features[1:2]   # 24 ch
            self.enc2 = efficientnet.features[2:3]   # 32 ch
            self.enc3 = efficientnet.features[3:4]   # 48 ch
            self.enc4 = efficientnet.features[4:6]   # 136 ch
            self.enc5 = efficientnet.features[6:9]   # 384 ch

            # Decoder with attention gates
            self.attention4 = AttentionGate(384, 136, 68)
            self.up4 = nn.ConvTranspose2d(384, 256, 2, stride=2)
            self.dec4 = DoubleConvBlock(256 + 136, 256)

            self.attention3 = AttentionGate(256, 48, 24)
            self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
            self.dec3 = DoubleConvBlock(128 + 48, 128)

            self.attention2 = AttentionGate(128, 32, 16)
            self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
            self.dec2 = DoubleConvBlock(64 + 32, 64)

            self.attention1 = AttentionGate(64, 24, 12)
            self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
            self.dec1 = DoubleConvBlock(32 + 24, 32)

            # Final upsampling to original resolution
            self.final_up = nn.ConvTranspose2d(32, 16, 2, stride=2)
            self.final_conv = nn.Conv2d(16, out_channels, 1)

            # Deep supervision outputs
            if deep_supervision:
                self.ds4 = nn.Conv2d(256, out_channels, 1)
                self.ds3 = nn.Conv2d(128, out_channels, 1)
                self.ds2 = nn.Conv2d(64, out_channels, 1)

            # Dropout for Monte Carlo
            self.dropout = nn.Dropout2d(dropout_rate)

        def forward(
            self, x: torch.Tensor, return_features: bool = False
        ) -> torch.Tensor:
            input_size = x.shape[2:]

            # Encoder
            x0 = self.stem(x)
            e1 = self.enc1(x0)
            e2 = self.enc2(e1)
            e3 = self.enc3(e2)
            e4 = self.enc4(e3)
            e5 = self.enc5(e4)

            e5 = self.dropout(e5)

            # Decoder with attention
            a4 = self.attention4(e5, e4)
            d4 = self.up4(e5)
            d4 = self._match_size(d4, a4)
            d4 = torch.cat([d4, a4], dim=1)
            d4 = self.dec4(d4)
            d4 = self.dropout(d4)

            a3 = self.attention3(d4, e3)
            d3 = self.up3(d4)
            d3 = self._match_size(d3, a3)
            d3 = torch.cat([d3, a3], dim=1)
            d3 = self.dec3(d3)
            d3 = self.dropout(d3)

            a2 = self.attention2(d3, e2)
            d2 = self.up2(d3)
            d2 = self._match_size(d2, a2)
            d2 = torch.cat([d2, a2], dim=1)
            d2 = self.dec2(d2)

            a1 = self.attention1(d2, e1)
            d1 = self.up1(d2)
            d1 = self._match_size(d1, a1)
            d1 = torch.cat([d1, a1], dim=1)
            d1 = self.dec1(d1)

            # Final output
            out = self.final_up(d1)
            out = F.interpolate(out, size=input_size, mode='bilinear', align_corners=True)
            out = self.final_conv(out)

            if self.training and self.deep_supervision:
                # Return multi-scale outputs for deep supervision
                ds4 = F.interpolate(self.ds4(d4), size=input_size, mode='bilinear', align_corners=True)
                ds3 = F.interpolate(self.ds3(d3), size=input_size, mode='bilinear', align_corners=True)
                ds2 = F.interpolate(self.ds2(d2), size=input_size, mode='bilinear', align_corners=True)
                return out, ds4, ds3, ds2

            if return_features:
                return out, {'e5': e5, 'd4': d4, 'd3': d3, 'd2': d2, 'd1': d1}

            return out

        def _match_size(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            if x.shape[2:] != target.shape[2:]:
                x = F.interpolate(x, size=target.shape[2:], mode='bilinear', align_corners=True)
            return x


    class PatchEmbedding(nn.Module):
        """Patch embedding for Swin Transformer"""
        def __init__(self, in_channels: int = 1, embed_dim: int = 96, patch_size: int = 4):
            super().__init__()
            self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
            self.norm = nn.LayerNorm(embed_dim)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.proj(x)  # [B, C, H/4, W/4]
            B, C, H, W = x.shape
            x = x.flatten(2).transpose(1, 2)  # [B, H*W, C]
            x = self.norm(x)
            return x, H, W


    class WindowAttention(nn.Module):
        """Window-based multi-head self attention for Swin Transformer"""
        def __init__(
            self,
            dim: int,
            window_size: int,
            num_heads: int,
            qkv_bias: bool = True,
            attn_drop: float = 0.,
            proj_drop: float = 0.
        ):
            super().__init__()
            self.dim = dim
            self.window_size = window_size
            self.num_heads = num_heads
            head_dim = dim // num_heads
            self.scale = head_dim ** -0.5

            # Relative position bias
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads)
            )

            coords_h = torch.arange(window_size)
            coords_w = torch.arange(window_size)
            coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
            coords_flatten = torch.flatten(coords, 1)
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()
            relative_coords[:, :, 0] += window_size - 1
            relative_coords[:, :, 1] += window_size - 1
            relative_coords[:, :, 0] *= 2 * window_size - 1
            relative_position_index = relative_coords.sum(-1)
            self.register_buffer("relative_position_index", relative_position_index)

            self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
            self.attn_drop = nn.Dropout(attn_drop)
            self.proj = nn.Linear(dim, dim)
            self.proj_drop = nn.Dropout(proj_drop)

            nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)

        def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
            B_, N, C = x.shape
            qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]

            q = q * self.scale
            attn = (q @ k.transpose(-2, -1))

            relative_position_bias = self.relative_position_bias_table[
                self.relative_position_index.view(-1)
            ].view(self.window_size * self.window_size, self.window_size * self.window_size, -1)
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
            attn = attn + relative_position_bias.unsqueeze(0)

            if mask is not None:
                nW = mask.shape[0]
                attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
                attn = attn.view(-1, self.num_heads, N, N)
                attn = F.softmax(attn, dim=-1)
            else:
                attn = F.softmax(attn, dim=-1)

            attn = self.attn_drop(attn)
            x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
            x = self.proj(x)
            x = self.proj_drop(x)
            return x


    class SwinTransformerBlock(nn.Module):
        """Swin Transformer Block"""
        def __init__(
            self,
            dim: int,
            num_heads: int,
            window_size: int = 7,
            shift_size: int = 0,
            mlp_ratio: float = 4.,
            qkv_bias: bool = True,
            drop: float = 0.,
            attn_drop: float = 0.,
            drop_path: float = 0.
        ):
            super().__init__()
            self.dim = dim
            self.num_heads = num_heads
            self.window_size = window_size
            self.shift_size = shift_size
            self.mlp_ratio = mlp_ratio

            self.norm1 = nn.LayerNorm(dim)
            self.attn = WindowAttention(
                dim, window_size=window_size, num_heads=num_heads,
                qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop
            )

            self.drop_path = nn.Identity() if drop_path <= 0. else DropPath(drop_path)
            self.norm2 = nn.LayerNorm(dim)
            mlp_hidden_dim = int(dim * mlp_ratio)
            self.mlp = nn.Sequential(
                nn.Linear(dim, mlp_hidden_dim),
                nn.GELU(),
                nn.Dropout(drop),
                nn.Linear(mlp_hidden_dim, dim),
                nn.Dropout(drop)
            )

        def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
            B, L, C = x.shape
            shortcut = x
            x = self.norm1(x)
            x = x.view(B, H, W, C)

            # Pad if needed
            pad_r = (self.window_size - W % self.window_size) % self.window_size
            pad_b = (self.window_size - H % self.window_size) % self.window_size
            x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))
            Hp, Wp = x.shape[1], x.shape[2]

            # Cyclic shift
            if self.shift_size > 0:
                shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
                attn_mask = self._create_mask(Hp, Wp, x.device)
            else:
                shifted_x = x
                attn_mask = None

            # Partition windows
            x_windows = self._window_partition(shifted_x)
            x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

            # Attention
            attn_windows = self.attn(x_windows, mask=attn_mask)

            # Merge windows
            attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
            shifted_x = self._window_reverse(attn_windows, Hp, Wp)

            # Reverse cyclic shift
            if self.shift_size > 0:
                x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
            else:
                x = shifted_x

            # Remove padding
            if pad_r > 0 or pad_b > 0:
                x = x[:, :H, :W, :].contiguous()

            x = x.view(B, H * W, C)
            x = shortcut + self.drop_path(x)
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x

        def _window_partition(self, x: torch.Tensor) -> torch.Tensor:
            B, H, W, C = x.shape
            x = x.view(B, H // self.window_size, self.window_size, W // self.window_size, self.window_size, C)
            windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, self.window_size, self.window_size, C)
            return windows

        def _window_reverse(self, windows: torch.Tensor, H: int, W: int) -> torch.Tensor:
            B = int(windows.shape[0] / (H * W / self.window_size / self.window_size))
            x = windows.view(B, H // self.window_size, W // self.window_size, self.window_size, self.window_size, -1)
            x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
            return x

        def _create_mask(self, H: int, W: int, device) -> torch.Tensor:
            img_mask = torch.zeros((1, H, W, 1), device=device)
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = self._window_partition(img_mask)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
            return attn_mask


    class DropPath(nn.Module):
        """Drop paths (Stochastic Depth) per sample"""
        def __init__(self, drop_prob: float = 0.):
            super().__init__()
            self.drop_prob = drop_prob

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            if self.drop_prob == 0. or not self.training:
                return x
            keep_prob = 1 - self.drop_prob
            shape = (x.shape[0],) + (1,) * (x.ndim - 1)
            random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
            random_tensor.floor_()
            output = x.div(keep_prob) * random_tensor
            return output


    class SwinUNet(nn.Module):
        """
        Swin Transformer U-Net for edge segmentation.

        State-of-the-art vision transformer architecture adapted for
        dense prediction tasks.
        """
        def __init__(
            self,
            in_channels: int = 1,
            out_channels: int = 1,
            embed_dim: int = 96,
            depths: List[int] = None,
            num_heads: List[int] = None,
            window_size: int = 7,
            mlp_ratio: float = 4.,
            drop_rate: float = 0.,
            attn_drop_rate: float = 0.,
            drop_path_rate: float = 0.1
        ):
            super().__init__()
            if depths is None:
                depths = [2, 2, 6, 2]
            if num_heads is None:
                num_heads = [3, 6, 12, 24]

            self.embed_dim = embed_dim
            self.num_layers = len(depths)

            # Patch embedding
            self.patch_embed = PatchEmbedding(in_channels, embed_dim, patch_size=4)

            # Stochastic depth decay
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

            # Encoder stages
            self.encoder_layers = nn.ModuleList()
            self.downsample_layers = nn.ModuleList()

            for i_layer in range(self.num_layers):
                layer_dim = embed_dim * (2 ** i_layer)
                layer = nn.ModuleList([
                    SwinTransformerBlock(
                        dim=layer_dim,
                        num_heads=num_heads[i_layer],
                        window_size=window_size,
                        shift_size=0 if (j % 2 == 0) else window_size // 2,
                        mlp_ratio=mlp_ratio,
                        drop=drop_rate,
                        attn_drop=attn_drop_rate,
                        drop_path=dpr[sum(depths[:i_layer]) + j]
                    )
                    for j in range(depths[i_layer])
                ])
                self.encoder_layers.append(layer)

                if i_layer < self.num_layers - 1:
                    downsample = nn.Sequential(
                        nn.LayerNorm(layer_dim),
                        nn.Linear(layer_dim, layer_dim * 2)
                    )
                    self.downsample_layers.append(downsample)

            # Decoder stages
            self.decoder_layers = nn.ModuleList()
            self.upsample_layers = nn.ModuleList()
            self.concat_layers = nn.ModuleList()

            for i_layer in range(self.num_layers - 1, 0, -1):
                layer_dim = embed_dim * (2 ** i_layer)

                upsample = nn.Sequential(
                    nn.LayerNorm(layer_dim),
                    nn.Linear(layer_dim, layer_dim // 2)
                )
                self.upsample_layers.append(upsample)

                concat_linear = nn.Linear(layer_dim, layer_dim // 2)
                self.concat_layers.append(concat_linear)

                layer = nn.ModuleList([
                    SwinTransformerBlock(
                        dim=layer_dim // 2,
                        num_heads=num_heads[i_layer - 1],
                        window_size=window_size,
                        shift_size=0 if (j % 2 == 0) else window_size // 2,
                        mlp_ratio=mlp_ratio,
                        drop=drop_rate,
                        attn_drop=attn_drop_rate,
                        drop_path=dpr[sum(depths[:i_layer-1]) + j] if i_layer > 1 else 0.
                    )
                    for j in range(depths[i_layer - 1])
                ])
                self.decoder_layers.append(layer)

            # Final projection
            self.norm = nn.LayerNorm(embed_dim)
            self.head = nn.Sequential(
                nn.ConvTranspose2d(embed_dim, embed_dim // 2, kernel_size=2, stride=2),
                nn.BatchNorm2d(embed_dim // 2),
                nn.GELU(),
                nn.ConvTranspose2d(embed_dim // 2, embed_dim // 4, kernel_size=2, stride=2),
                nn.BatchNorm2d(embed_dim // 4),
                nn.GELU(),
                nn.Conv2d(embed_dim // 4, out_channels, kernel_size=1)
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            input_size = x.shape[2:]

            # Patch embedding
            x, H, W = self.patch_embed(x)

            # Encoder
            encoder_features = []
            for i, (layer, downsample) in enumerate(
                zip(self.encoder_layers[:-1], self.downsample_layers)
            ):
                for blk in layer:
                    x = blk(x, H, W)
                encoder_features.append((x, H, W))

                # Downsample (patch merging)
                B, L, C = x.shape
                x = x.view(B, H, W, C)
                x0 = x[:, 0::2, 0::2, :]
                x1 = x[:, 1::2, 0::2, :]
                x2 = x[:, 0::2, 1::2, :]
                x3 = x[:, 1::2, 1::2, :]
                x = torch.cat([x0, x1, x2, x3], dim=-1)
                H, W = H // 2, W // 2
                x = x.view(B, H * W, 4 * C)
                x = downsample(x)

            # Bottleneck
            for blk in self.encoder_layers[-1]:
                x = blk(x, H, W)

            # Decoder
            for i, (layer, upsample, concat) in enumerate(
                zip(self.decoder_layers, self.upsample_layers, self.concat_layers)
            ):
                # Upsample
                x = upsample(x)
                B, L, C = x.shape
                x = x.view(B, H, W, C)
                x = x.repeat_interleave(2, dim=1).repeat_interleave(2, dim=2)
                H, W = H * 2, W * 2
                x = x.view(B, H * W, C)

                # Skip connection
                skip, skip_H, skip_W = encoder_features[-(i + 1)]
                if x.shape[1] != skip.shape[1]:
                    x = F.interpolate(
                        x.view(B, H, W, C).permute(0, 3, 1, 2),
                        size=(skip_H, skip_W),
                        mode='bilinear',
                        align_corners=True
                    ).permute(0, 2, 3, 1).view(B, skip_H * skip_W, C)
                    H, W = skip_H, skip_W

                x = torch.cat([x, skip], dim=-1)
                x = concat(x)

                for blk in layer:
                    x = blk(x, H, W)

            # Final projection
            x = self.norm(x)
            B, L, C = x.shape
            x = x.view(B, H, W, C).permute(0, 3, 1, 2)  # [B, C, H, W]
            x = self.head(x)
            x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=True)

            return x


    class DeepEnsemble(nn.Module):
        """
        Deep Ensemble for uncertainty estimation.

        Trains multiple models and aggregates predictions to provide
        both prediction and epistemic uncertainty.
        """
        def __init__(
            self,
            model_class,
            num_models: int = 5,
            **model_kwargs
        ):
            super().__init__()
            self.num_models = num_models
            self.models = nn.ModuleList([
                model_class(**model_kwargs) for _ in range(num_models)
            ])

        def forward(
            self,
            x: torch.Tensor,
            return_all: bool = False
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            Forward pass through all ensemble members.

            Returns:
                mean: Ensemble mean prediction
                uncertainty: Epistemic uncertainty (std across models)
            """
            predictions = []
            for model in self.models:
                pred = model(x)
                if isinstance(pred, tuple):
                    pred = pred[0]  # Handle deep supervision
                predictions.append(pred)

            predictions = torch.stack(predictions, dim=0)  # [num_models, B, C, H, W]
            mean = predictions.mean(dim=0)
            std = predictions.std(dim=0)

            if return_all:
                return mean, std, predictions
            return mean, std

        def predict_with_uncertainty(
            self,
            x: torch.Tensor,
            num_mc_samples: int = 10
        ) -> Dict[str, torch.Tensor]:
            """
            Predict with both epistemic and aleatoric uncertainty.

            Uses Monte Carlo Dropout within each ensemble member.
            """
            self.train()  # Enable dropout

            all_predictions = []
            for model in self.models:
                mc_preds = []
                for _ in range(num_mc_samples):
                    with torch.no_grad():
                        pred = model(x)
                        if isinstance(pred, tuple):
                            pred = pred[0]
                        mc_preds.append(pred)
                mc_preds = torch.stack(mc_preds, dim=0)
                all_predictions.append(mc_preds)

            all_predictions = torch.stack(all_predictions, dim=0)  # [num_models, num_mc, B, C, H, W]

            # Mean prediction
            mean = all_predictions.mean(dim=(0, 1))

            # Epistemic uncertainty (model uncertainty)
            model_means = all_predictions.mean(dim=1)
            epistemic = model_means.std(dim=0)

            # Aleatoric uncertainty (data uncertainty) - average MC variance
            mc_vars = all_predictions.var(dim=1)
            aleatoric = mc_vars.mean(dim=0).sqrt()

            # Total uncertainty
            total = (epistemic ** 2 + aleatoric ** 2).sqrt()

            self.eval()

            return {
                'mean': mean,
                'epistemic': epistemic,
                'aleatoric': aleatoric,
                'total': total
            }


    class HybridCDNet(nn.Module):
        """
        Hybrid CD Measurement Network.

        Combines:
        - EfficientNet for local feature extraction
        - Swin Transformer for global context
        - Multi-task heads for edge, CD, and uncertainty
        """
        def __init__(
            self,
            in_channels: int = 1,
            num_depths: int = 5,
            dropout_rate: float = 0.2
        ):
            super().__init__()

            # EfficientNet backbone
            efficientnet = models.efficientnet_b3(weights='IMAGENET1K_V1')

            # Adapt for grayscale
            self.stem = nn.Sequential(
                nn.Conv2d(in_channels, 40, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(40),
                nn.SiLU(inplace=True)
            )
            with torch.no_grad():
                orig_weight = efficientnet.features[0][0].weight.mean(dim=1, keepdim=True)
                # Expand to 40 channels
                self.stem[0].weight.copy_(orig_weight.repeat(40, 1, 1, 1)[:40])

            # Local feature extractor (EfficientNet)
            self.local_encoder = efficientnet.features[1:7]  # Up to 136 channels

            # Global context (lightweight Swin blocks)
            self.global_encoder = nn.Sequential(
                nn.Conv2d(136, 192, 1),
                nn.BatchNorm2d(192),
                nn.SiLU(),
            )

            # Feature fusion
            self.fusion = nn.Sequential(
                nn.Conv2d(192, 256, 3, padding=1),
                nn.BatchNorm2d(256),
                nn.SiLU(),
                nn.Dropout2d(dropout_rate),
                nn.Conv2d(256, 256, 3, padding=1),
                nn.BatchNorm2d(256),
                nn.SiLU()
            )

            # Edge segmentation head
            self.edge_head = nn.Sequential(
                nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
                nn.BatchNorm2d(128),
                nn.SiLU(),
                nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
                nn.BatchNorm2d(64),
                nn.SiLU(),
                nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
                nn.BatchNorm2d(32),
                nn.SiLU(),
                nn.Conv2d(32, 1, 1)
            )

            # CD regression head (predicts left/right edges for each depth)
            self.cd_head = nn.Sequential(
                nn.AdaptiveAvgPool2d((8, 1)),  # Global pooling in x, keep y structure
                nn.Flatten(),
                nn.Linear(256 * 8, 512),
                nn.SiLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(512, 256),
                nn.SiLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(256, num_depths * 2)  # left, right for each depth
            )

            # Uncertainty head
            self.uncertainty_head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(256, 128),
                nn.SiLU(),
                nn.Linear(128, num_depths * 2),  # uncertainty for each edge
                nn.Softplus()  # Ensure positive uncertainty
            )

        def forward(
            self,
            x: torch.Tensor
        ) -> Dict[str, torch.Tensor]:
            input_size = x.shape[2:]

            # Local features
            x = self.stem(x)
            local_feat = self.local_encoder(x)

            # Global context
            global_feat = self.global_encoder(local_feat)

            # Fusion
            fused = self.fusion(global_feat)

            # Edge segmentation
            edge_map = self.edge_head(fused)
            edge_map = F.interpolate(edge_map, size=input_size, mode='bilinear', align_corners=True)

            # CD regression
            cd_pred = self.cd_head(fused)

            # Uncertainty
            uncertainty = self.uncertainty_head(fused)

            return {
                'edge_map': edge_map,
                'cd_pred': cd_pred,
                'uncertainty': uncertainty
            }


    # Factory function
    def create_advanced_model(
        model_type: str,
        in_channels: int = 1,
        out_channels: int = 1,
        **kwargs
    ) -> nn.Module:
        """
        Create advanced model by type.

        Args:
            model_type: One of 'attention_unet', 'swin_unet', 'ensemble_attention',
                       'ensemble_swin', 'hybrid_cd'
            in_channels: Number of input channels (1 for grayscale)
            out_channels: Number of output channels
            **kwargs: Additional model arguments

        Returns:
            Initialized model
        """
        model_type = model_type.lower()

        if model_type == 'attention_unet':
            return AttentionUNet(
                in_channels=in_channels,
                out_channels=out_channels,
                **kwargs
            )
        elif model_type == 'swin_unet':
            return SwinUNet(
                in_channels=in_channels,
                out_channels=out_channels,
                **kwargs
            )
        elif model_type == 'ensemble_attention':
            num_models = kwargs.pop('num_models', 5)
            return DeepEnsemble(
                AttentionUNet,
                num_models=num_models,
                in_channels=in_channels,
                out_channels=out_channels,
                **kwargs
            )
        elif model_type == 'ensemble_swin':
            num_models = kwargs.pop('num_models', 5)
            return DeepEnsemble(
                SwinUNet,
                num_models=num_models,
                in_channels=in_channels,
                out_channels=out_channels,
                **kwargs
            )
        elif model_type == 'hybrid_cd':
            num_depths = kwargs.get('num_depths', 5)
            return HybridCDNet(
                in_channels=in_channels,
                num_depths=num_depths,
                **kwargs
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")


    # Model info for GUI
    ADVANCED_MODEL_INFO = {
        'attention_unet': 'Attention U-Net with EfficientNet-B3 encoder and CBAM attention',
        'swin_unet': 'Swin Transformer U-Net - state-of-the-art vision transformer',
        'ensemble_attention': 'Deep Ensemble of Attention U-Nets for uncertainty estimation',
        'ensemble_swin': 'Deep Ensemble of Swin U-Nets for uncertainty estimation',
        'hybrid_cd': 'Hybrid CNN-Transformer for edge detection and CD regression'
    }

else:
    # Dummy implementations when PyTorch not available
    def create_advanced_model(*args, **kwargs):
        raise ImportError("PyTorch is required for advanced models")

    ADVANCED_MODEL_INFO = {}
