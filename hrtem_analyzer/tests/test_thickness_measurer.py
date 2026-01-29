"""
Tests for thickness measurement module
"""
import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from core.thickness_measurer import (
    ThicknessMeasurer,
    MultiVariantMeasurer,
    EdgePoint,
    SingleMeasurement,
    MeasurementResult
)


class TestThicknessMeasurer:
    """Tests for ThicknessMeasurer class"""

    def setup_method(self):
        """Setup test fixtures"""
        self.measurer = ThicknessMeasurer()

    def test_init_default_methods(self):
        """Test default edge detection methods"""
        assert 'sobel' in self.measurer.edge_methods
        assert 'canny' in self.measurer.edge_methods
        assert len(self.measurer.edge_methods) >= 3

    def test_init_custom_methods(self):
        """Test custom edge detection methods"""
        custom = ThicknessMeasurer(edge_methods=['sobel', 'canny'])
        assert custom.edge_methods == ['sobel', 'canny']

    def test_measure_synthetic_image(self):
        """Test measurement on synthetic image with known thickness"""
        # Create synthetic image with a vertical structure
        image = np.zeros((100, 200), dtype=np.float32)

        # Create structure: dark background, bright structure at x=80-120
        image[:, 80:120] = 1.0

        # Known parameters
        baseline_y = 20
        depth_nm = 30
        scale = 1.0  # 1 nm/pixel

        measurements = self.measurer.measure_thickness(
            image=image,
            baseline_y=baseline_y,
            depth_nm=depth_nm,
            scale_nm_per_pixel=scale,
            num_lines=3
        )

        # Should detect edges around x=80 and x=120
        assert len(measurements) > 0

        # Check that at least one measurement is close to expected
        thicknesses = [m.thickness_nm for m in measurements]
        # Expected: 40 pixels = 40 nm (120-80)
        assert any(35 < t < 45 for t in thicknesses)

    def test_aggregate_measurements(self):
        """Test measurement aggregation"""
        # Create mock measurements
        measurements = [
            SingleMeasurement(
                depth_nm=10,
                thickness_nm=40.0,
                left_edge=EdgePoint(x=80, y=30, confidence=0.9, method='sobel'),
                right_edge=EdgePoint(x=120, y=30, confidence=0.9, method='sobel'),
                method_name='sobel',
                preprocessing='original',
                rotation_angle=0.0,
                confidence=0.9,
                y_position=30
            ),
            SingleMeasurement(
                depth_nm=10,
                thickness_nm=41.0,
                left_edge=EdgePoint(x=79, y=30, confidence=0.85, method='canny'),
                right_edge=EdgePoint(x=120, y=30, confidence=0.85, method='canny'),
                method_name='canny',
                preprocessing='original',
                rotation_angle=0.0,
                confidence=0.85,
                y_position=30
            ),
            SingleMeasurement(
                depth_nm=10,
                thickness_nm=39.5,
                left_edge=EdgePoint(x=81, y=30, confidence=0.8, method='gradient'),
                right_edge=EdgePoint(x=121, y=30, confidence=0.8, method='gradient'),
                method_name='gradient',
                preprocessing='original',
                rotation_angle=0.0,
                confidence=0.8,
                y_position=30
            ),
        ]

        result = self.measurer.aggregate_measurements(measurements, depth_nm=10)

        assert result is not None
        assert result.depth_nm == 10
        assert 39 < result.thickness_nm < 42
        assert result.num_measurements == 3
        assert result.thickness_std < 1.0  # Low variance expected

    def test_subpixel_refinement(self):
        """Test sub-pixel edge refinement"""
        # Create signal with peak at index 50
        signal = np.zeros(100)
        signal[49] = 0.3
        signal[50] = 1.0
        signal[51] = 0.5

        # Peak should be refined to slightly left of 50
        refined = self.measurer._subpixel_refine(signal, 50)

        assert 49.5 < refined < 50.5

    def test_empty_measurements(self):
        """Test aggregation of empty measurements"""
        result = self.measurer.aggregate_measurements([], depth_nm=10)
        assert result is None


class TestMultiVariantMeasurer:
    """Tests for MultiVariantMeasurer class"""

    def test_init(self):
        """Test initialization"""
        measurer = MultiVariantMeasurer()
        assert measurer.measurer is not None
        assert measurer.max_variants > 0

    def test_adjust_baseline(self):
        """Test baseline adjustment for transformations"""
        measurer = MultiVariantMeasurer()

        # No change for identity transform
        adjusted = measurer._adjust_baseline(100, 0.0, 1.0, (200, 200))
        assert adjusted == 100

        # Scale adjustment
        adjusted = measurer._adjust_baseline(100, 0.0, 2.0, (400, 200))
        assert adjusted == 200


class TestEdgePoint:
    """Tests for EdgePoint dataclass"""

    def test_creation(self):
        """Test EdgePoint creation"""
        edge = EdgePoint(x=100, y=50, confidence=0.95, method='sobel')
        assert edge.x == 100
        assert edge.y == 50
        assert edge.confidence == 0.95
        assert edge.method == 'sobel'
        assert edge.subpixel_x is None

    def test_with_subpixel(self):
        """Test EdgePoint with sub-pixel position"""
        edge = EdgePoint(
            x=100, y=50, confidence=0.95,
            method='sobel', subpixel_x=100.35
        )
        assert edge.subpixel_x == 100.35


class TestMeasurementResult:
    """Tests for MeasurementResult dataclass"""

    def test_to_dict(self):
        """Test conversion to dictionary"""
        result = MeasurementResult(
            depth_nm=10.0,
            thickness_nm=40.5,
            thickness_std=0.5,
            thickness_min=39.8,
            thickness_max=41.2,
            confidence=0.9,
            y_position=30,
            left_edge_x=80,
            right_edge_x=120,
            num_measurements=5,
            consensus_method='trimmed_mean'
        )

        d = result.to_dict()

        assert d['depth_nm'] == 10.0
        assert d['thickness_nm'] == 40.5
        assert d['num_measurements'] == 5
        assert 'individual_measurements' not in d  # Should not include full list


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
