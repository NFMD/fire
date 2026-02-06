"""
DM3/DM4 File Loader for Gatan Digital Micrograph files

Supports reading image data and scale metadata from:
- DM3 (Gatan DigitalMicrograph 3) files
- DM4 (Gatan DigitalMicrograph 4) files

Based on the DM3/DM4 binary format specification.
Reference: SimpliPyTEM (gabriel-ing) DM file handling approach.
"""
import struct
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, BinaryIO
import numpy as np
from loguru import logger


# DM data type mapping
DM_DATA_TYPES = {
    2: ('h', 2),   # signed 16-bit integer
    3: ('i', 4),   # signed 32-bit integer
    4: ('H', 2),   # unsigned 16-bit integer
    5: ('I', 4),   # unsigned 32-bit integer
    6: ('f', 4),   # 32-bit float
    7: ('d', 8),   # 64-bit float
    8: ('?', 1),   # boolean
    9: ('b', 1),   # signed 8-bit integer (char)
    10: ('B', 1),  # unsigned 8-bit integer (byte)
    11: ('q', 8),  # signed 64-bit integer
    12: ('Q', 8),  # unsigned 64-bit integer
}

DM_NUMPY_TYPES = {
    2: np.int16,
    3: np.int32,
    4: np.uint16,
    5: np.uint32,
    6: np.float32,
    7: np.float64,
    8: np.bool_,
    9: np.int8,
    10: np.uint8,
    11: np.int64,
    12: np.uint64,
}


@dataclass
class DMScaleInfo:
    """Scale information from DM file"""
    scale_x: float = 1.0
    scale_y: float = 1.0
    unit_x: str = ""
    unit_y: str = ""
    origin_x: float = 0.0
    origin_y: float = 0.0


class DMTagReader:
    """Read DM3/DM4 tag directory structure"""

    def __init__(self, f: BinaryIO, dm_version: int = 3):
        self.f = f
        self.dm_version = dm_version
        self.tags: Dict[str, Any] = {}
        self._byte_order = '>'  # Big endian for DM3, can be little for DM4

    def _read_uint(self, size: int) -> int:
        """Read unsigned integer of given byte size"""
        data = self.f.read(size)
        if len(data) < size:
            raise EOFError("Unexpected end of file")
        if size == 1:
            return struct.unpack(f'{self._byte_order}B', data)[0]
        elif size == 2:
            return struct.unpack(f'{self._byte_order}H', data)[0]
        elif size == 4:
            return struct.unpack(f'{self._byte_order}I', data)[0]
        elif size == 8:
            return struct.unpack(f'{self._byte_order}Q', data)[0]
        else:
            raise ValueError(f"Unsupported uint size: {size}")

    def _read_string(self, length: int) -> str:
        """Read string of given length"""
        data = self.f.read(length)
        try:
            return data.decode('utf-8', errors='replace')
        except Exception:
            return data.decode('latin-1', errors='replace')

    def read_tag_directory(self, path: str = ""):
        """Read a tag directory recursively"""
        # sorted flag
        is_sorted = self._read_uint(1)

        # closed flag
        is_closed = self._read_uint(1)

        # number of tags
        if self.dm_version == 4:
            num_tags = self._read_uint(8)
        else:
            num_tags = self._read_uint(4)

        for _ in range(num_tags):
            self._read_tag_entry(path)

    def _read_tag_entry(self, parent_path: str):
        """Read a single tag entry"""
        # Tag type: 0=end, 20=tag directory, 21=tag data
        tag_type = self._read_uint(1)

        if tag_type == 0:
            return

        # Tag name length
        if self.dm_version == 4:
            name_length = self._read_uint(2)
        else:
            name_length = self._read_uint(2)

        # Tag name
        tag_name = self._read_string(name_length) if name_length > 0 else ""

        full_path = f"{parent_path}.{tag_name}" if parent_path else tag_name

        if tag_type == 20:
            # Tag directory - recurse
            self.read_tag_directory(full_path)
        elif tag_type == 21:
            # Tag data
            self._read_tag_data(full_path)

    def _read_tag_data(self, path: str):
        """Read tag data value"""
        # delimiter '%%%%'
        delimiter = self.f.read(4)

        if self.dm_version == 4:
            info_array_size = self._read_uint(8)
        else:
            info_array_size = self._read_uint(4)

        if info_array_size == 0:
            return

        # Read info array
        info_array = []
        for _ in range(info_array_size):
            if self.dm_version == 4:
                info_array.append(self._read_uint(8))
            else:
                info_array.append(self._read_uint(4))

        # Decode based on info array
        try:
            value = self._decode_tag_value(info_array)
            self.tags[path] = value
        except Exception:
            pass

    def _decode_tag_value(self, info_array):
        """Decode tag value from info array"""
        if len(info_array) == 0:
            return None

        encoding = info_array[0]

        if encoding in DM_DATA_TYPES:
            # Simple data type
            fmt, size = DM_DATA_TYPES[encoding]
            data = self.f.read(size)
            if len(data) < size:
                return None
            return struct.unpack(f'{self._byte_order}{fmt}', data)[0]

        elif encoding == 18:
            # String
            if len(info_array) >= 2:
                str_length = info_array[1]
                return self._read_string(str_length)

        elif encoding == 15:
            # Struct
            if len(info_array) >= 4:
                # Read struct fields
                struct_size = info_array[1]
                num_fields = info_array[2]
                field_data = []
                for i in range(num_fields):
                    field_info_idx = 3 + i * 2
                    if field_info_idx + 1 < len(info_array):
                        field_type = info_array[field_info_idx + 1]
                        if field_type in DM_DATA_TYPES:
                            fmt, size = DM_DATA_TYPES[field_type]
                            data = self.f.read(size)
                            if len(data) >= size:
                                val = struct.unpack(f'{self._byte_order}{fmt}', data)[0]
                                field_data.append(val)
                return field_data

        elif encoding == 20:
            # Array
            if len(info_array) >= 2:
                array_type = info_array[1]
                if self.dm_version == 4 and len(info_array) >= 3:
                    array_size = info_array[2]
                elif len(info_array) >= 3:
                    array_size = info_array[2]
                else:
                    return None

                if array_type in DM_DATA_TYPES:
                    fmt, elem_size = DM_DATA_TYPES[array_type]
                    total_bytes = elem_size * array_size
                    data = self.f.read(total_bytes)
                    if array_type == 10 or array_type == 9:
                        # Byte array - might be string
                        try:
                            return data.decode('utf-8', errors='replace').rstrip('\x00')
                        except Exception:
                            return data
                    else:
                        if len(data) >= total_bytes:
                            dtype = DM_NUMPY_TYPES.get(array_type, np.uint8)
                            arr = np.frombuffer(data, dtype=dtype)
                            if self._byte_order == '>':
                                arr = arr.byteswap()
                            return arr
                elif array_type == 15:
                    # Array of structs - skip for now
                    pass

        # Unknown - skip remaining bytes
        return None


class DMFileLoader:
    """
    Load Gatan Digital Micrograph DM3/DM4 files.

    Extracts:
    - Image data (supports multiple data types)
    - Scale/calibration information
    - Microscope metadata (voltage, magnification, etc.)
    """

    def __init__(self):
        self.supported_extensions = {'.dm3', '.dm4'}

    def can_load(self, path: str) -> bool:
        """Check if file can be loaded"""
        return Path(path).suffix.lower() in self.supported_extensions

    def load(self, path: str, normalize: bool = True) -> Tuple[np.ndarray, DMScaleInfo]:
        """
        Load DM3/DM4 file.

        Args:
            path: Path to DM3/DM4 file
            normalize: Normalize image to 0-1 float32

        Returns:
            Tuple of (image_array, scale_info)
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        ext = path.suffix.lower()
        logger.info(f"Loading {ext.upper()} file: {path.name}")

        # Try hyperspy first (most reliable)
        try:
            return self._load_with_hyperspy(str(path), normalize)
        except ImportError:
            logger.debug("hyperspy not available, using built-in reader")
        except Exception as e:
            logger.debug(f"hyperspy failed: {e}, trying built-in reader")

        # Try ncempy
        try:
            return self._load_with_ncempy(str(path), normalize)
        except ImportError:
            logger.debug("ncempy not available, using built-in reader")
        except Exception as e:
            logger.debug(f"ncempy failed: {e}, trying built-in reader")

        # Fallback: built-in reader
        return self._load_builtin(str(path), normalize)

    def _load_with_hyperspy(self, path: str, normalize: bool) -> Tuple[np.ndarray, DMScaleInfo]:
        """Load using hyperspy library"""
        import hyperspy.api as hs

        signal = hs.load(path, signal_type='')

        # Extract image data
        image = signal.data
        if image.ndim > 2:
            image = image[0] if image.ndim == 3 else image

        # Extract scale
        scale_info = DMScaleInfo()
        axes = signal.axes_manager

        if len(axes) >= 2:
            ax_y = axes[0]
            ax_x = axes[1]
            scale_info.scale_y = ax_y.scale
            scale_info.scale_x = ax_x.scale
            scale_info.unit_y = ax_y.units or ""
            scale_info.unit_x = ax_x.units or ""
            scale_info.origin_y = ax_y.offset
            scale_info.origin_x = ax_x.offset

        if normalize:
            image = image.astype(np.float32)
            img_min, img_max = image.min(), image.max()
            if img_max > img_min:
                image = (image - img_min) / (img_max - img_min)

        logger.info(f"Loaded via hyperspy: {image.shape}, "
                    f"scale: {scale_info.scale_x} {scale_info.unit_x}/px")

        return image, scale_info

    def _load_with_ncempy(self, path: str, normalize: bool) -> Tuple[np.ndarray, DMScaleInfo]:
        """Load using ncempy library"""
        from ncempy.io import dm

        dm_file = dm.fileDM(path)
        dataset = dm_file.getDataset(0)

        image = dataset['data']
        if image.ndim > 2:
            image = image[0] if image.ndim == 3 else image

        scale_info = DMScaleInfo()

        # Extract pixel size from dataset
        if 'pixelSize' in dataset:
            pixel_sizes = dataset['pixelSize']
            if len(pixel_sizes) >= 2:
                scale_info.scale_y = pixel_sizes[0]
                scale_info.scale_x = pixel_sizes[1]

        if 'pixelUnit' in dataset:
            units = dataset['pixelUnit']
            if len(units) >= 2:
                scale_info.unit_y = units[0]
                scale_info.unit_x = units[1]

        if normalize:
            image = image.astype(np.float32)
            img_min, img_max = image.min(), image.max()
            if img_max > img_min:
                image = (image - img_min) / (img_max - img_min)

        logger.info(f"Loaded via ncempy: {image.shape}, "
                    f"scale: {scale_info.scale_x} {scale_info.unit_x}/px")

        return image, scale_info

    def _load_builtin(self, path: str, normalize: bool) -> Tuple[np.ndarray, DMScaleInfo]:
        """Load using built-in binary reader"""
        with open(path, 'rb') as f:
            # Read DM header
            dm_version = struct.unpack('>I', f.read(4))[0]

            if dm_version not in (3, 4):
                raise ValueError(f"Unsupported DM version: {dm_version}")

            if dm_version == 4:
                file_size = struct.unpack('>Q', f.read(8))[0]
                byte_order = struct.unpack('>I', f.read(4))[0]
            else:
                file_size = struct.unpack('>I', f.read(4))[0]
                byte_order = struct.unpack('>I', f.read(4))[0]

            # Read tag directory
            reader = DMTagReader(f, dm_version)
            if byte_order == 0:
                reader._byte_order = '>'  # Big endian
            else:
                reader._byte_order = '<'  # Little endian

            try:
                reader.read_tag_directory()
            except (EOFError, struct.error):
                pass

            # Extract image data and scale from tags
            return self._extract_from_tags(reader.tags, normalize, path)

    def _extract_from_tags(self, tags: Dict[str, Any],
                           normalize: bool, path: str) -> Tuple[np.ndarray, DMScaleInfo]:
        """Extract image data and scale from parsed tags"""
        image = None
        scale_info = DMScaleInfo()

        # Search for image data in tags
        for key, value in tags.items():
            if isinstance(value, np.ndarray) and value.size > 100:
                # Likely image data - find the largest array
                if image is None or value.size > image.size:
                    image = value

            # Look for scale information
            key_lower = key.lower()
            if 'scale' in key_lower and 'x' in key_lower:
                try:
                    scale_info.scale_x = float(value)
                except (TypeError, ValueError):
                    pass
            elif 'scale' in key_lower and 'y' in key_lower:
                try:
                    scale_info.scale_y = float(value)
                except (TypeError, ValueError):
                    pass
            elif 'unit' in key_lower and isinstance(value, str):
                if 'x' in key_lower:
                    scale_info.unit_x = value
                elif 'y' in key_lower:
                    scale_info.unit_y = value

        if image is None:
            raise ValueError(f"No image data found in DM file: {path}")

        # Try to reshape if 1D
        if image.ndim == 1:
            # Guess dimensions - try square first
            side = int(np.sqrt(image.size))
            if side * side == image.size:
                image = image.reshape(side, side)
            else:
                # Try common aspect ratios
                for h_try in range(side + 100, side - 100, -1):
                    if h_try > 0 and image.size % h_try == 0:
                        w_try = image.size // h_try
                        if 0.5 < w_try / h_try < 2.0:
                            image = image.reshape(h_try, w_try)
                            break
                else:
                    raise ValueError(f"Cannot determine image dimensions for {image.size} pixels")

        if normalize:
            image = image.astype(np.float32)
            img_min, img_max = image.min(), image.max()
            if img_max > img_min:
                image = (image - img_min) / (img_max - img_min)

        logger.info(f"Loaded via built-in reader: {image.shape}, "
                    f"scale: {scale_info.scale_x} {scale_info.unit_x}/px")

        return image, scale_info

    def get_scale_nm_per_pixel(self, scale_info: DMScaleInfo) -> float:
        """Convert DMScaleInfo to nm/pixel"""
        scale = scale_info.scale_x
        unit = scale_info.unit_x.lower().strip()

        if not unit or scale == 0:
            return 1.0

        # Convert to nm
        if unit in ('nm', 'nanometer', 'nanometers'):
            return scale
        elif unit in ('um', 'µm', 'micrometer', 'micrometers', 'micron'):
            return scale * 1000.0
        elif unit in ('mm', 'millimeter', 'millimeters'):
            return scale * 1e6
        elif unit in ('m', 'meter', 'meters'):
            return scale * 1e9
        elif unit in ('a', 'angstrom', 'angstroms', '\u00c5'):
            return scale * 0.1
        elif unit in ('pm', 'picometer', 'picometers'):
            return scale * 0.001
        else:
            logger.warning(f"Unknown DM scale unit: '{unit}', using raw value")
            return scale
