"""
Result Exporter for HR-TEM Analysis

Exports measurement results with visualization to JPEG files.
"""
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import cv2
from loguru import logger

from .image_loader import ScaleInfo
from .baseline_detector import BaselineInfo
from .thickness_measurer import MeasurementResult


class ResultVisualizer:
    """
    Visualize measurement results on HR-TEM images.

    Creates annotated images with:
    - Baseline (0-point) marker
    - Measurement lines at each depth
    - Edge markers
    - Thickness values
    - Scale bar
    - Information panel
    """

    # Color scheme (BGR for OpenCV)
    COLORS = {
        'baseline': (0, 0, 255),      # Red
        'measurement': (0, 255, 0),    # Green
        'edge': (255, 255, 0),         # Cyan
        'text': (255, 255, 255),       # White
        'text_bg': (0, 0, 0),          # Black
        'scale_bar': (255, 255, 255),  # White
        'info_bg': (40, 40, 40),       # Dark gray
    }

    # Depth-specific colors for multiple measurements
    DEPTH_COLORS = [
        (0, 255, 0),    # Green
        (255, 255, 0),  # Cyan
        (0, 255, 255),  # Yellow
        (255, 0, 255),  # Magenta
        (255, 128, 0),  # Orange
    ]

    def __init__(
            self,
            font_scale: float = 0.5,
            line_thickness: int = 2,
            marker_size: int = 6
    ):
        """
        Initialize visualizer.

        Args:
            font_scale: Font scale for text
            line_thickness: Line thickness in pixels
            marker_size: Size of edge markers
        """
        self.font_scale = font_scale
        self.line_thickness = line_thickness
        self.marker_size = marker_size
        self.font = cv2.FONT_HERSHEY_SIMPLEX

    def create_visualization(
            self,
            image: np.ndarray,
            baseline_info: BaselineInfo,
            measurements: Dict[float, MeasurementResult],
            scale_info: ScaleInfo,
            title: Optional[str] = None
    ) -> np.ndarray:
        """
        Create visualization with all annotations.

        Args:
            image: Grayscale image (0-1 range)
            baseline_info: Baseline detection info
            measurements: Dict of depth -> MeasurementResult
            scale_info: Scale information
            title: Optional title for the image

        Returns:
            Annotated BGR image
        """
        # Convert to BGR
        if len(image.shape) == 2:
            if image.max() <= 1.0:
                img_uint8 = (image * 255).astype(np.uint8)
            else:
                img_uint8 = image.astype(np.uint8)
            result = cv2.cvtColor(img_uint8, cv2.COLOR_GRAY2BGR)
        else:
            result = image.copy()

        h, w = result.shape[:2]

        # 1. Draw baseline
        self._draw_baseline(result, baseline_info, w)

        # 2. Draw measurements
        sorted_depths = sorted(measurements.keys())
        for i, depth in enumerate(sorted_depths):
            measurement = measurements[depth]
            color = self.DEPTH_COLORS[i % len(self.DEPTH_COLORS)]
            self._draw_measurement(result, measurement, color, scale_info)

        # 3. Draw scale bar
        self._draw_scale_bar(result, scale_info)

        # 4. Draw info panel
        self._draw_info_panel(result, measurements, scale_info, title)

        return result

    def _draw_baseline(
            self,
            image: np.ndarray,
            baseline_info: BaselineInfo,
            width: int
    ):
        """Draw baseline (0-point) marker"""
        y = baseline_info.y_position

        # Draw baseline
        cv2.line(
            image,
            (0, y),
            (width, y),
            self.COLORS['baseline'],
            self.line_thickness
        )

        # Draw label
        label = "0 nm (Baseline)"
        label_size = cv2.getTextSize(label, self.font, self.font_scale, 1)[0]

        # Background rectangle
        cv2.rectangle(
            image,
            (5, y - label_size[1] - 8),
            (5 + label_size[0] + 4, y - 4),
            self.COLORS['text_bg'],
            -1
        )

        # Text
        cv2.putText(
            image,
            label,
            (7, y - 7),
            self.font,
            self.font_scale,
            self.COLORS['baseline'],
            1,
            cv2.LINE_AA
        )

    def _draw_measurement(
            self,
            image: np.ndarray,
            measurement: MeasurementResult,
            color: Tuple[int, int, int],
            scale_info: ScaleInfo
    ):
        """Draw single measurement with edges and value"""
        y = measurement.y_position
        left_x = measurement.left_edge_x
        right_x = measurement.right_edge_x
        h, w = image.shape[:2]

        # Draw measurement line (dashed)
        self._draw_dashed_line(image, (0, y), (w, y), color, 1)

        # Draw thickness line (solid, thicker)
        cv2.line(
            image,
            (left_x, y),
            (right_x, y),
            color,
            self.line_thickness + 1
        )

        # Draw edge markers
        cv2.circle(image, (left_x, y), self.marker_size, color, -1)
        cv2.circle(image, (right_x, y), self.marker_size, color, -1)

        # Draw edge lines (vertical)
        cv2.line(image, (left_x, y - 10), (left_x, y + 10), color, 1)
        cv2.line(image, (right_x, y - 10), (right_x, y + 10), color, 1)

        # Draw measurement text
        text = f"{measurement.depth_nm:.1f}nm: {measurement.thickness_nm:.2f}nm"
        if measurement.thickness_std > 0:
            text += f" (±{measurement.thickness_std:.2f})"

        text_x = min(right_x + 15, w - 200)
        text_y = y + 5

        # Text background
        text_size = cv2.getTextSize(text, self.font, self.font_scale * 0.9, 1)[0]
        cv2.rectangle(
            image,
            (text_x - 2, text_y - text_size[1] - 2),
            (text_x + text_size[0] + 2, text_y + 4),
            self.COLORS['text_bg'],
            -1
        )

        cv2.putText(
            image,
            text,
            (text_x, text_y),
            self.font,
            self.font_scale * 0.9,
            color,
            1,
            cv2.LINE_AA
        )

    def _draw_dashed_line(
            self,
            image: np.ndarray,
            pt1: Tuple[int, int],
            pt2: Tuple[int, int],
            color: Tuple[int, int, int],
            thickness: int,
            dash_length: int = 10
    ):
        """Draw a dashed line"""
        x1, y1 = pt1
        x2, y2 = pt2

        # Calculate line length and direction
        length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        dx = (x2 - x1) / length
        dy = (y2 - y1) / length

        # Draw dashes
        pos = 0
        drawing = True
        while pos < length:
            if drawing:
                start_x = int(x1 + pos * dx)
                start_y = int(y1 + pos * dy)
                end_x = int(x1 + min(pos + dash_length, length) * dx)
                end_y = int(y1 + min(pos + dash_length, length) * dy)
                cv2.line(image, (start_x, start_y), (end_x, end_y), color, thickness)
            pos += dash_length
            drawing = not drawing

    def _draw_scale_bar(self, image: np.ndarray, scale_info: ScaleInfo):
        """Draw scale bar in bottom right corner"""
        h, w = image.shape[:2]

        # Determine appropriate scale bar length
        target_length_nm = self._get_nice_scale_length(
            scale_info.scale_nm_per_pixel * w * 0.15
        )
        scale_bar_pixels = int(target_length_nm / scale_info.scale_nm_per_pixel)

        # Position
        margin = 20
        bar_height = 8
        x_end = w - margin
        x_start = x_end - scale_bar_pixels
        y = h - margin - bar_height

        # Draw background
        cv2.rectangle(
            image,
            (x_start - 10, y - 25),
            (x_end + 10, h - margin + 5),
            self.COLORS['info_bg'],
            -1
        )

        # Draw scale bar
        cv2.rectangle(
            image,
            (x_start, y),
            (x_end, y + bar_height),
            self.COLORS['scale_bar'],
            -1
        )

        # Draw end caps
        cv2.line(
            image,
            (x_start, y - 3),
            (x_start, y + bar_height + 3),
            self.COLORS['scale_bar'],
            2
        )
        cv2.line(
            image,
            (x_end, y - 3),
            (x_end, y + bar_height + 3),
            self.COLORS['scale_bar'],
            2
        )

        # Draw label
        if target_length_nm >= 1000:
            label = f"{target_length_nm / 1000:.0f} µm"
        else:
            label = f"{target_length_nm:.0f} nm"

        text_size = cv2.getTextSize(label, self.font, self.font_scale, 1)[0]
        text_x = x_start + (scale_bar_pixels - text_size[0]) // 2
        text_y = y - 5

        cv2.putText(
            image,
            label,
            (text_x, text_y),
            self.font,
            self.font_scale,
            self.COLORS['scale_bar'],
            1,
            cv2.LINE_AA
        )

    def _get_nice_scale_length(self, approximate_nm: float) -> float:
        """Get a 'nice' scale bar length (1, 2, 5, 10, 20, 50, 100, etc.)"""
        nice_values = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000]

        for value in nice_values:
            if value >= approximate_nm * 0.5:
                return value

        return nice_values[-1]

    def _draw_info_panel(
            self,
            image: np.ndarray,
            measurements: Dict[float, MeasurementResult],
            scale_info: ScaleInfo,
            title: Optional[str]
    ):
        """Draw information panel in top left corner"""
        h, w = image.shape[:2]
        margin = 10
        line_height = 20
        panel_width = 280

        # Calculate panel height
        num_lines = 3 + len(measurements)  # Title, scale, blank, measurements
        if title:
            num_lines += 1
        panel_height = num_lines * line_height + 2 * margin

        # Draw panel background
        cv2.rectangle(
            image,
            (margin, margin),
            (margin + panel_width, margin + panel_height),
            self.COLORS['info_bg'],
            -1
        )
        cv2.rectangle(
            image,
            (margin, margin),
            (margin + panel_width, margin + panel_height),
            self.COLORS['text'],
            1
        )

        # Draw content
        y = margin + line_height
        x = margin + 10

        # Title
        if title:
            cv2.putText(
                image, title[:35], (x, y),
                self.font, self.font_scale, self.COLORS['text'], 1, cv2.LINE_AA
            )
            y += line_height

        # Scale info
        scale_text = f"Scale: {scale_info.scale_nm_per_pixel:.4f} nm/px"
        cv2.putText(
            image, scale_text, (x, y),
            self.font, self.font_scale * 0.9, self.COLORS['text'], 1, cv2.LINE_AA
        )
        y += line_height

        # Separator
        y += 5
        cv2.line(image, (x, y), (margin + panel_width - 10, y), self.COLORS['text'], 1)
        y += line_height

        # Measurements
        for depth in sorted(measurements.keys()):
            m = measurements[depth]
            text = f"@{depth:.0f}nm: {m.thickness_nm:.2f} ± {m.thickness_std:.2f} nm"
            cv2.putText(
                image, text, (x, y),
                self.font, self.font_scale * 0.85, self.COLORS['text'], 1, cv2.LINE_AA
            )
            y += line_height


class ResultExporter:
    """
    Export measurement results to various formats.

    Supports:
    - JPEG with visualization
    - JSON data export
    - CSV summary
    """

    def __init__(
            self,
            output_dir: str,
            jpeg_quality: int = 95,
            create_subdirs: bool = True
    ):
        """
        Initialize exporter.

        Args:
            output_dir: Output directory path
            jpeg_quality: JPEG quality (0-100)
            create_subdirs: Create subdirectories for organization
        """
        self.output_dir = Path(output_dir)
        self.jpeg_quality = jpeg_quality
        self.create_subdirs = create_subdirs

        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if create_subdirs:
            (self.output_dir / 'images').mkdir(exist_ok=True)
            (self.output_dir / 'data').mkdir(exist_ok=True)

        self.visualizer = ResultVisualizer()

    def export_result(
            self,
            image: np.ndarray,
            baseline_info: BaselineInfo,
            measurements: Dict[float, MeasurementResult],
            scale_info: ScaleInfo,
            source_path: str,
            save_json: bool = True
    ) -> Tuple[str, Optional[str]]:
        """
        Export single result with visualization.

        Args:
            image: Original grayscale image
            baseline_info: Baseline information
            measurements: Measurement results
            scale_info: Scale information
            source_path: Original image path
            save_json: Also save JSON data

        Returns:
            Tuple of (jpeg_path, json_path or None)
        """
        source_name = Path(source_path).stem

        # Create visualization
        vis_image = self.visualizer.create_visualization(
            image=image,
            baseline_info=baseline_info,
            measurements=measurements,
            scale_info=scale_info,
            title=source_name
        )

        # Save JPEG
        if self.create_subdirs:
            jpeg_path = self.output_dir / 'images' / f"{source_name}_measured.jpg"
        else:
            jpeg_path = self.output_dir / f"{source_name}_measured.jpg"

        cv2.imwrite(
            str(jpeg_path),
            vis_image,
            [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
        )
        logger.info(f"Saved visualization: {jpeg_path}")

        # Save JSON
        json_path = None
        if save_json:
            if self.create_subdirs:
                json_path = self.output_dir / 'data' / f"{source_name}_data.json"
            else:
                json_path = self.output_dir / f"{source_name}_data.json"

            data = self._create_json_data(
                measurements, scale_info, baseline_info, source_path
            )
            with open(json_path, 'w') as f:
                json.dump(data, f, indent=2)
            logger.info(f"Saved data: {json_path}")

        return str(jpeg_path), str(json_path) if json_path else None

    def export_batch_summary(
            self,
            results: List[Dict],
            filename: str = "batch_summary.json"
    ) -> str:
        """
        Export batch processing summary.

        Args:
            results: List of result dictionaries
            filename: Output filename

        Returns:
            Path to summary file
        """
        summary_path = self.output_dir / filename

        summary = {
            'timestamp': datetime.now().isoformat(),
            'total_images': len(results),
            'successful': sum(1 for r in results if r.get('success', False)),
            'results': results
        }

        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Saved batch summary: {summary_path}")
        return str(summary_path)

    def export_csv(
            self,
            results: List[Dict],
            filename: str = "measurements.csv"
    ) -> str:
        """
        Export measurements to CSV format.

        Args:
            results: List of result dictionaries
            filename: Output filename

        Returns:
            Path to CSV file
        """
        csv_path = self.output_dir / filename

        # Collect all depths
        all_depths = set()
        for result in results:
            if 'measurements' in result:
                all_depths.update(result['measurements'].keys())
        sorted_depths = sorted(all_depths)

        # Write CSV
        with open(csv_path, 'w') as f:
            # Header
            header = ['filename', 'scale_nm_per_pixel', 'baseline_y']
            for depth in sorted_depths:
                header.extend([
                    f'thickness_{depth}nm',
                    f'std_{depth}nm',
                    f'confidence_{depth}nm'
                ])
            f.write(','.join(header) + '\n')

            # Data rows
            for result in results:
                row = [
                    result.get('source_path', ''),
                    str(result.get('scale_nm_per_pixel', '')),
                    str(result.get('baseline_y', ''))
                ]

                measurements = result.get('measurements', {})
                for depth in sorted_depths:
                    if depth in measurements:
                        m = measurements[depth]
                        row.extend([
                            f"{m['thickness_nm']:.3f}",
                            f"{m['thickness_std']:.3f}",
                            f"{m['confidence']:.3f}"
                        ])
                    else:
                        row.extend(['', '', ''])

                f.write(','.join(row) + '\n')

        logger.info(f"Saved CSV: {csv_path}")
        return str(csv_path)

    def _create_json_data(
            self,
            measurements: Dict[float, MeasurementResult],
            scale_info: ScaleInfo,
            baseline_info: BaselineInfo,
            source_path: str
    ) -> Dict:
        """Create JSON-serializable data dictionary"""
        return {
            'source_path': source_path,
            'timestamp': datetime.now().isoformat(),
            'scale': scale_info.to_dict(),
            'baseline': {
                'y_position': baseline_info.y_position,
                'confidence': baseline_info.confidence,
                'method': baseline_info.method
            },
            'measurements': {
                str(depth): m.to_dict()
                for depth, m in measurements.items()
            }
        }
