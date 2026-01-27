"""
TIFF Image Loader with Scale Information Extraction

Supports various TEM manufacturers:
- FEI/Thermo Fisher
- JEOL
- Hitachi
- Zeiss
"""
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import numpy as np

try:
    import tifffile
except ImportError:
    tifffile = None

from loguru import logger


@dataclass
class ScaleInfo:
    """Scale information extracted from TEM image metadata"""
    scale_nm_per_pixel: float  # nm per pixel
    scale_bar_length_nm: Optional[float] = None  # Scale bar length in nm
    magnification: Optional[int] = None  # Magnification
    accelerating_voltage_kv: Optional[float] = None  # Accelerating voltage in kV
    pixel_size_x: Optional[float] = None  # X pixel size
    pixel_size_y: Optional[float] = None  # Y pixel size
    unit: str = "nm"  # Unit of measurement
    source: str = "unknown"  # Source of scale info (metadata, ocr, manual)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'scale_nm_per_pixel': self.scale_nm_per_pixel,
            'scale_bar_length_nm': self.scale_bar_length_nm,
            'magnification': self.magnification,
            'accelerating_voltage_kv': self.accelerating_voltage_kv,
            'unit': self.unit,
            'source': self.source,
        }


class TIFFLoader:
    """
    TIFF loader with automatic scale extraction from metadata.

    Memory-efficient loading with optional downsampling for preview.
    """

    # Known TEM metadata tags
    SCALE_TAGS = {
        'FEI': ['EScan', 'Scan', 'PixelSize'],
        'JEOL': ['JEOL-SEM', 'PixelSize'],
        'Hitachi': ['Hitachi', 'PixelSize'],
        'ImageJ': ['ImageDescription'],
        'Generic': ['XResolution', 'YResolution', 'ResolutionUnit'],
    }

    def __init__(self, default_scale_nm: float = 1.0):
        """
        Initialize TIFF loader.

        Args:
            default_scale_nm: Default scale (nm/pixel) if not found in metadata
        """
        self.default_scale_nm = default_scale_nm
        if tifffile is None:
            raise ImportError("tifffile is required. Install with: pip install tifffile")

    def load(
            self,
            path: str,
            normalize: bool = True,
            target_dtype: np.dtype = np.float32
    ) -> Tuple[np.ndarray, ScaleInfo]:
        """
        Load TIFF image and extract scale information.

        Args:
            path: Path to TIFF file
            normalize: Normalize image to 0-1 range
            target_dtype: Target data type for image

        Returns:
            Tuple of (image array, scale info)
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")

        logger.info(f"Loading image: {path.name}")

        # Load image data
        with tifffile.TiffFile(str(path)) as tif:
            image = tif.asarray()
            metadata = self._extract_metadata(tif)

        # Convert to grayscale if needed
        if len(image.shape) == 3:
            if image.shape[2] == 3:
                # RGB to grayscale
                image = np.dot(image[..., :3], [0.299, 0.587, 0.114])
            elif image.shape[2] == 4:
                # RGBA to grayscale
                image = np.dot(image[..., :3], [0.299, 0.587, 0.114])
            else:
                # Take first channel
                image = image[..., 0]

        # Convert dtype
        image = image.astype(target_dtype)

        # Normalize
        if normalize:
            img_min, img_max = image.min(), image.max()
            if img_max > img_min:
                image = (image - img_min) / (img_max - img_min)

        # Extract scale info
        scale_info = self._parse_scale_info(metadata, image.shape)

        logger.info(f"Loaded image: {image.shape}, scale: {scale_info.scale_nm_per_pixel:.4f} nm/pixel")

        return image, scale_info

    def load_lazy(self, path: str) -> Tuple[Any, ScaleInfo]:
        """
        Lazy load - returns memmap for large files.

        Memory efficient for large images.
        """
        path = Path(path)
        with tifffile.TiffFile(str(path)) as tif:
            metadata = self._extract_metadata(tif)
            # Get shape without loading full image
            shape = tif.pages[0].shape

        scale_info = self._parse_scale_info(metadata, shape)

        # Return memmap
        memmap = tifffile.memmap(str(path))
        return memmap, scale_info

    def _extract_metadata(self, tif: 'tifffile.TiffFile') -> Dict[str, Any]:
        """Extract all metadata from TIFF file"""
        metadata = {}

        # Basic TIFF tags
        page = tif.pages[0]
        for tag in page.tags.values():
            try:
                metadata[tag.name] = tag.value
            except Exception:
                pass

        # FEI/Thermo Fisher metadata
        if hasattr(tif, 'fei_metadata') and tif.fei_metadata:
            metadata['fei'] = tif.fei_metadata

        # SEM/TEM metadata
        if hasattr(tif, 'sem_metadata') and tif.sem_metadata:
            metadata['sem'] = tif.sem_metadata

        # ImageJ metadata
        if hasattr(tif, 'imagej_metadata') and tif.imagej_metadata:
            metadata['imagej'] = tif.imagej_metadata

        # OME metadata
        if hasattr(tif, 'ome_metadata') and tif.ome_metadata:
            metadata['ome'] = tif.ome_metadata

        return metadata

    def _parse_scale_info(self, metadata: Dict[str, Any], shape: Tuple) -> ScaleInfo:
        """Parse scale information from metadata"""

        scale_nm = self.default_scale_nm
        magnification = None
        voltage = None
        source = "default"

        # Try FEI metadata
        if 'fei' in metadata:
            fei = metadata['fei']
            if 'Scan' in fei and 'PixelWidth' in fei['Scan']:
                # FEI stores in meters
                scale_nm = fei['Scan']['PixelWidth'] * 1e9
                source = "fei_metadata"
            if 'EBeam' in fei:
                if 'HV' in fei['EBeam']:
                    voltage = fei['EBeam']['HV'] / 1000  # V to kV

        # Try ImageJ metadata
        elif 'imagej' in metadata:
            ij = metadata['imagej']
            if 'unit' in ij and 'spacing' in ij:
                unit = ij['unit'].lower()
                spacing = ij['spacing']
                if unit == 'nm':
                    scale_nm = spacing
                elif unit == 'um' or unit == 'µm':
                    scale_nm = spacing * 1000
                elif unit == 'm':
                    scale_nm = spacing * 1e9
                source = "imagej_metadata"

        # Try generic TIFF resolution tags
        elif 'XResolution' in metadata:
            x_res = metadata['XResolution']
            if isinstance(x_res, tuple):
                x_res = x_res[0] / x_res[1]

            unit = metadata.get('ResolutionUnit', 2)
            if unit == 2:  # inches
                # pixels per inch -> nm per pixel
                scale_nm = 25400000 / x_res  # 1 inch = 25.4mm = 25400000 nm
            elif unit == 3:  # centimeters
                scale_nm = 10000000 / x_res  # 1 cm = 10000000 nm
            source = "tiff_resolution"

        # Try to parse from ImageDescription
        elif 'ImageDescription' in metadata:
            desc = str(metadata['ImageDescription'])
            scale_nm, found_source = self._parse_description(desc)
            if found_source:
                source = found_source

        return ScaleInfo(
            scale_nm_per_pixel=scale_nm,
            magnification=magnification,
            accelerating_voltage_kv=voltage,
            source=source
        )

    def _parse_description(self, description: str) -> Tuple[float, Optional[str]]:
        """Parse scale from image description text"""
        # Pattern: number followed by nm, um, etc.
        patterns = [
            (r'(\d+\.?\d*)\s*nm\s*/\s*pixel', 1.0, 'description_nm'),
            (r'(\d+\.?\d*)\s*nm/px', 1.0, 'description_nm'),
            (r'pixel\s*size\s*[=:]\s*(\d+\.?\d*)\s*nm', 1.0, 'description_pixel_size'),
            (r'scale\s*[=:]\s*(\d+\.?\d*)\s*nm', 1.0, 'description_scale'),
            (r'(\d+\.?\d*)\s*um\s*/\s*pixel', 1000.0, 'description_um'),
            (r'(\d+\.?\d*)\s*µm\s*/\s*pixel', 1000.0, 'description_um'),
        ]

        for pattern, multiplier, source in patterns:
            match = re.search(pattern, description, re.IGNORECASE)
            if match:
                return float(match.group(1)) * multiplier, source

        return self.default_scale_nm, None

    def get_image_info(self, path: str) -> Dict[str, Any]:
        """Get image information without loading full image"""
        path = Path(path)
        with tifffile.TiffFile(str(path)) as tif:
            page = tif.pages[0]
            metadata = self._extract_metadata(tif)
            scale_info = self._parse_scale_info(metadata, page.shape)

            return {
                'path': str(path),
                'shape': page.shape,
                'dtype': str(page.dtype),
                'size_mb': path.stat().st_size / (1024 * 1024),
                'scale': scale_info.to_dict(),
            }
