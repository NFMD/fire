"""
Parallel Batch Processor for HR-TEM Image Analysis

Memory-efficient parallel processing of image working groups.
Supports up to 20 images per working group with automatic
memory management and resource cleanup.
"""
import gc
import os
import psutil
import queue
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional, Callable, Any, Tuple
import numpy as np
from loguru import logger

# Import core modules
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.image_loader import TIFFLoader, ScaleInfo
from core.preprocessor import ImagePreprocessor, PreprocessedImage
from core.baseline_detector import BaselineDetector, BaselineInfo
from core.thickness_measurer import (
    ThicknessMeasurer,
    MultiVariantMeasurer,
    MeasurementResult
)


@dataclass
class ImageTask:
    """Single image processing task"""
    path: str
    index: int
    baseline_hint_y: Optional[int] = None
    depths_nm: List[float] = field(default_factory=list)


@dataclass
class ImageResult:
    """Result from processing single image"""
    path: str
    index: int
    success: bool
    measurements: Dict[float, MeasurementResult] = field(default_factory=dict)
    baseline_info: Optional[BaselineInfo] = None
    scale_info: Optional[ScaleInfo] = None
    error_message: Optional[str] = None
    processing_time_ms: float = 0.0


@dataclass
class WorkingGroup:
    """
    Working group of images for batch processing.

    Maximum 20 images per group for memory efficiency.
    """
    images: List[str]
    name: str = "default"
    baseline_hint_y: Optional[int] = None
    depths_nm: List[float] = field(default_factory=lambda: [5, 10, 15, 20])

    def __post_init__(self):
        if len(self.images) > 20:
            logger.warning(
                f"Working group '{self.name}' has {len(self.images)} images, "
                f"truncating to 20 for memory efficiency"
            )
            self.images = self.images[:20]

    def __len__(self):
        return len(self.images)


class MemoryMonitor:
    """Monitor and manage memory usage"""

    def __init__(self, limit_mb: int = 4096):
        self.limit_mb = limit_mb
        self.process = psutil.Process(os.getpid())

    def get_memory_usage_mb(self) -> float:
        """Get current memory usage in MB"""
        return self.process.memory_info().rss / (1024 * 1024)

    def is_memory_available(self, required_mb: float = 500) -> bool:
        """Check if required memory is available"""
        current = self.get_memory_usage_mb()
        return (current + required_mb) < self.limit_mb

    def cleanup(self):
        """Force garbage collection"""
        gc.collect()


class BatchProcessor:
    """
    Parallel batch processor for HR-TEM images.

    Features:
    - Parallel processing with configurable workers
    - Memory-efficient processing with automatic cleanup
    - Progress tracking and logging
    - Error handling and recovery
    """

    def __init__(
            self,
            max_workers: int = 4,
            memory_limit_mb: int = 4096,
            use_multiprocessing: bool = False,
            preprocessing_methods: List[str] = None,
            rotation_angles: List[float] = None,
            edge_methods: List[str] = None
    ):
        """
        Initialize batch processor.

        Args:
            max_workers: Maximum parallel workers
            memory_limit_mb: Memory limit in MB
            use_multiprocessing: Use process pool instead of thread pool
            preprocessing_methods: List of preprocessing methods to use
            rotation_angles: List of rotation angles for multi-angle analysis
            edge_methods: List of edge detection methods
        """
        self.max_workers = max_workers
        self.memory_limit_mb = memory_limit_mb
        self.use_multiprocessing = use_multiprocessing

        # Processing configuration
        self.preprocessing_methods = preprocessing_methods or [
            'original', 'clahe', 'bilateral_filter'
        ]
        self.rotation_angles = rotation_angles or [-1.0, 0.0, 1.0]
        self.edge_methods = edge_methods or [
            'sobel', 'canny', 'gradient', 'morphological'
        ]

        # Initialize components
        self.loader = TIFFLoader()
        self.preprocessor = ImagePreprocessor()
        self.baseline_detector = BaselineDetector()
        self.thickness_measurer = ThicknessMeasurer(
            edge_methods=self.edge_methods,
            consensus_method='trimmed_mean'
        )
        self.multi_measurer = MultiVariantMeasurer(self.thickness_measurer)
        self.memory_monitor = MemoryMonitor(memory_limit_mb)

        # Statistics
        self.stats = {
            'total_processed': 0,
            'successful': 0,
            'failed': 0,
            'total_time_ms': 0
        }

    def process_working_group(
            self,
            working_group: WorkingGroup,
            progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> List[ImageResult]:
        """
        Process a working group of images in parallel.

        Args:
            working_group: WorkingGroup containing images to process
            progress_callback: Callback for progress updates (current, total, message)

        Returns:
            List of ImageResult objects
        """
        logger.info(
            f"Processing working group '{working_group.name}' "
            f"with {len(working_group)} images"
        )

        # Create tasks
        tasks = [
            ImageTask(
                path=path,
                index=i,
                baseline_hint_y=working_group.baseline_hint_y,
                depths_nm=working_group.depths_nm
            )
            for i, path in enumerate(working_group.images)
        ]

        results = []
        completed = 0

        # Choose executor based on configuration
        executor_class = (
            ProcessPoolExecutor if self.use_multiprocessing
            else ThreadPoolExecutor
        )

        with executor_class(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_task = {
                executor.submit(self._process_single_image, task): task
                for task in tasks
            }

            # Process completed tasks
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                completed += 1

                try:
                    result = future.result()
                    results.append(result)

                    if result.success:
                        self.stats['successful'] += 1
                        logger.info(
                            f"[{completed}/{len(tasks)}] Processed: {Path(task.path).name}"
                        )
                    else:
                        self.stats['failed'] += 1
                        logger.warning(
                            f"[{completed}/{len(tasks)}] Failed: {Path(task.path).name} "
                            f"- {result.error_message}"
                        )

                except Exception as e:
                    self.stats['failed'] += 1
                    results.append(ImageResult(
                        path=task.path,
                        index=task.index,
                        success=False,
                        error_message=str(e)
                    ))
                    logger.error(f"Error processing {task.path}: {e}")

                # Progress callback
                if progress_callback:
                    progress_callback(
                        completed,
                        len(tasks),
                        f"Processing {Path(task.path).name}"
                    )

                # Memory management
                if not self.memory_monitor.is_memory_available():
                    logger.warning("Memory limit approaching, forcing cleanup")
                    self.memory_monitor.cleanup()

        self.stats['total_processed'] += len(tasks)

        # Sort results by original index
        results.sort(key=lambda r: r.index)

        # Final cleanup
        self.memory_monitor.cleanup()

        return results

    def _process_single_image(self, task: ImageTask) -> ImageResult:
        """
        Process a single image through the full pipeline.

        Args:
            task: ImageTask with image path and parameters

        Returns:
            ImageResult with measurements
        """
        import time
        start_time = time.perf_counter()

        try:
            # 1. Load image
            image, scale_info = self.loader.load(task.path)

            # 2. Auto-level image
            leveled_image, level_angle = self.preprocessor.auto_level(image)

            # 3. Detect baseline
            baseline_info = self.baseline_detector.detect(
                leveled_image,
                method='auto',
                hint_y=task.baseline_hint_y
            )

            # 4. Generate preprocessing variants
            variants = self.preprocessor.generate_variants(
                leveled_image,
                methods=self.preprocessing_methods,
                rotation_angles=self.rotation_angles,
                scale_factors=[1.0]  # Keep scale at 1.0 to avoid confusion
            )

            # 5. Measure across all variants
            measurements = self.multi_measurer.measure_all_variants(
                image_variants=variants,
                baseline_y=baseline_info.y_position,
                depths_nm=task.depths_nm,
                scale_nm_per_pixel=scale_info.scale_nm_per_pixel
            )

            # Calculate processing time
            processing_time = (time.perf_counter() - start_time) * 1000

            return ImageResult(
                path=task.path,
                index=task.index,
                success=True,
                measurements=measurements,
                baseline_info=baseline_info,
                scale_info=scale_info,
                processing_time_ms=processing_time
            )

        except Exception as e:
            processing_time = (time.perf_counter() - start_time) * 1000
            logger.error(f"Error processing {task.path}: {e}")

            return ImageResult(
                path=task.path,
                index=task.index,
                success=False,
                error_message=str(e),
                processing_time_ms=processing_time
            )

        finally:
            # Clean up local variables
            gc.collect()

    def process_directory(
            self,
            directory: str,
            pattern: str = "*.tif*",
            depths_nm: List[float] = None,
            baseline_hint_y: Optional[int] = None,
            batch_size: int = 10
    ) -> List[ImageResult]:
        """
        Process all matching images in a directory.

        Automatically splits into working groups for memory efficiency.

        Args:
            directory: Directory path
            pattern: Glob pattern for image files
            depths_nm: Measurement depths
            baseline_hint_y: Optional baseline hint
            batch_size: Images per working group

        Returns:
            List of all ImageResults
        """
        directory = Path(directory)
        image_files = sorted(directory.glob(pattern))

        if not image_files:
            logger.warning(f"No images found matching {pattern} in {directory}")
            return []

        logger.info(f"Found {len(image_files)} images to process")

        # Split into working groups
        all_results = []
        depths = depths_nm or [5, 10, 15, 20]

        for i in range(0, len(image_files), batch_size):
            batch_files = image_files[i:i + batch_size]

            working_group = WorkingGroup(
                images=[str(f) for f in batch_files],
                name=f"batch_{i // batch_size + 1}",
                baseline_hint_y=baseline_hint_y,
                depths_nm=depths
            )

            results = self.process_working_group(working_group)
            all_results.extend(results)

            # Memory cleanup between batches
            self.memory_monitor.cleanup()

        return all_results

    def get_statistics(self) -> Dict[str, Any]:
        """Get processing statistics"""
        return {
            **self.stats,
            'memory_usage_mb': self.memory_monitor.get_memory_usage_mb(),
            'success_rate': (
                self.stats['successful'] / max(1, self.stats['total_processed'])
            ) * 100
        }


class StreamingBatchProcessor(BatchProcessor):
    """
    Streaming batch processor for very large datasets.

    Processes images one at a time and streams results,
    minimizing memory footprint.
    """

    def process_streaming(
            self,
            image_paths: List[str],
            depths_nm: List[float],
            baseline_hint_y: Optional[int] = None
    ):
        """
        Process images in streaming mode, yielding results as they complete.

        Args:
            image_paths: List of image paths
            depths_nm: Measurement depths
            baseline_hint_y: Optional baseline hint

        Yields:
            ImageResult objects as processing completes
        """
        results_queue = queue.Queue()

        def worker(task):
            result = self._process_single_image(task)
            results_queue.put(result)

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit tasks
            for i, path in enumerate(image_paths):
                task = ImageTask(
                    path=path,
                    index=i,
                    baseline_hint_y=baseline_hint_y,
                    depths_nm=depths_nm
                )
                executor.submit(worker, task)

            # Yield results as they complete
            for _ in range(len(image_paths)):
                result = results_queue.get()
                yield result

                # Memory management
                if not self.memory_monitor.is_memory_available():
                    self.memory_monitor.cleanup()
