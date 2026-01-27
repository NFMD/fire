#!/usr/bin/env python3
"""
HR-TEM Image Analysis CLI

Command-line interface for automated CD measurement of HR-TEM images.

Usage:
    # Single image
    python analyze.py single image.tiff -o results/ -d 5 10 15 20

    # Batch processing
    python analyze.py batch images/ -o results/ -w 4

    # Directory processing
    python analyze.py directory images/ -o results/ --pattern "*.tif"
"""
import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from loguru import logger
from pipeline.inference_pipeline import InferencePipeline, PipelineConfig, create_pipeline


def setup_logging(verbose: bool = False, log_file: str = None):
    """Configure logging"""
    logger.remove()

    level = "DEBUG" if verbose else "INFO"

    # Console output
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level=level,
        colorize=True
    )

    # File output
    if log_file:
        logger.add(
            log_file,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
            level="DEBUG",
            rotation="10 MB"
        )


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="HR-TEM Image Analysis - Automated CD Measurement",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze single image
  python analyze.py single sample.tiff -o results/ -d 5 10 15 20

  # Batch process multiple images
  python analyze.py batch img1.tiff img2.tiff img3.tiff -o results/

  # Process entire directory
  python analyze.py directory ./images/ -o results/ --pattern "*.tif*"

  # High precision mode with more workers
  python analyze.py directory ./images/ -o results/ --high-precision -w 8
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Processing mode')

    # Single image command
    single_parser = subparsers.add_parser('single', help='Process single image')
    single_parser.add_argument('image', type=str, help='Path to TIFF image')
    single_parser.add_argument('-o', '--output', type=str, required=True,
                               help='Output directory')
    single_parser.add_argument('-d', '--depths', type=float, nargs='+',
                               default=[5, 10, 15, 20],
                               help='Measurement depths in nm')
    single_parser.add_argument('-b', '--baseline', type=int, default=None,
                               help='Baseline Y position hint')

    # Batch command
    batch_parser = subparsers.add_parser('batch', help='Process multiple images')
    batch_parser.add_argument('images', type=str, nargs='+',
                              help='Paths to TIFF images')
    batch_parser.add_argument('-o', '--output', type=str, required=True,
                              help='Output directory')
    batch_parser.add_argument('-d', '--depths', type=float, nargs='+',
                              default=[5, 10, 15, 20],
                              help='Measurement depths in nm')
    batch_parser.add_argument('-b', '--baseline', type=int, default=None,
                              help='Baseline Y position hint')

    # Directory command
    dir_parser = subparsers.add_parser('directory', help='Process directory')
    dir_parser.add_argument('directory', type=str, help='Directory path')
    dir_parser.add_argument('-o', '--output', type=str, required=True,
                            help='Output directory')
    dir_parser.add_argument('-p', '--pattern', type=str, default='*.tif*',
                            help='File pattern (default: *.tif*)')
    dir_parser.add_argument('-d', '--depths', type=float, nargs='+',
                            default=[5, 10, 15, 20],
                            help='Measurement depths in nm')
    dir_parser.add_argument('-b', '--baseline', type=int, default=None,
                            help='Baseline Y position hint')

    # Common options
    for p in [single_parser, batch_parser, dir_parser]:
        p.add_argument('-w', '--workers', type=int, default=4,
                       help='Number of parallel workers (default: 4)')
        p.add_argument('--high-precision', action='store_true',
                       help='Use high precision mode')
        p.add_argument('-v', '--verbose', action='store_true',
                       help='Verbose output')
        p.add_argument('--log-file', type=str, default=None,
                       help='Log file path')
        p.add_argument('--no-json', action='store_true',
                       help='Skip JSON output')
        p.add_argument('--no-csv', action='store_true',
                       help='Skip CSV output')

    return parser.parse_args()


def main():
    """Main entry point"""
    args = parse_args()

    if not args.command:
        print("Please specify a command: single, batch, or directory")
        print("Use --help for more information")
        sys.exit(1)

    # Setup logging
    setup_logging(
        verbose=getattr(args, 'verbose', False),
        log_file=getattr(args, 'log_file', None)
    )

    logger.info("=" * 60)
    logger.info("HR-TEM Image Analysis - Automated CD Measurement")
    logger.info("=" * 60)

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create pipeline
    pipeline = create_pipeline(
        output_dir=str(output_dir),
        depths_nm=args.depths,
        max_workers=args.workers,
        high_precision=getattr(args, 'high_precision', False)
    )

    # Update config based on args
    if getattr(args, 'no_json', False):
        pipeline.config.save_json = False
    if getattr(args, 'no_csv', False):
        pipeline.config.save_csv = False

    # Process based on command
    try:
        if args.command == 'single':
            result = pipeline.process_single(
                image_path=args.image,
                baseline_hint_y=args.baseline,
                depths_nm=args.depths
            )
            _print_single_result(result)

        elif args.command == 'batch':
            results = pipeline.process_batch(
                image_paths=args.images,
                baseline_hint_y=args.baseline,
                depths_nm=args.depths
            )
            _print_batch_summary(results)

        elif args.command == 'directory':
            results = pipeline.process_directory(
                directory=args.directory,
                pattern=args.pattern,
                baseline_hint_y=args.baseline,
                depths_nm=args.depths
            )
            _print_batch_summary(results)

        logger.info("Processing complete!")
        logger.info(f"Results saved to: {output_dir}")

    except Exception as e:
        logger.error(f"Error during processing: {e}")
        if getattr(args, 'verbose', False):
            import traceback
            traceback.print_exc()
        sys.exit(1)


def _print_single_result(result: dict):
    """Print single image result"""
    if not result.get('success', False):
        logger.error(f"Processing failed: {result.get('error', 'Unknown error')}")
        return

    logger.info("\n" + "=" * 60)
    logger.info("MEASUREMENT SUMMARY")
    logger.info("=" * 60)

    measurements = result.get('measurements', {})
    for depth in sorted(measurements.keys()):
        m = measurements[depth]
        logger.info(
            f"  Depth {float(depth):5.1f} nm: "
            f"{m['thickness_nm']:7.2f} ± {m['thickness_std']:5.2f} nm"
        )

    logger.info("=" * 60)

    if result.get('output', {}).get('jpeg_path'):
        logger.info(f"Visualization: {result['output']['jpeg_path']}")


def _print_batch_summary(results: list):
    """Print batch processing summary"""
    successful = [r for r in results if r.get('success', False)]
    failed = [r for r in results if not r.get('success', False)]

    logger.info("\n" + "=" * 60)
    logger.info("BATCH PROCESSING SUMMARY")
    logger.info("=" * 60)
    logger.info(f"  Total images: {len(results)}")
    logger.info(f"  Successful: {len(successful)}")
    logger.info(f"  Failed: {len(failed)}")

    if failed:
        logger.info("\nFailed images:")
        for r in failed:
            logger.info(f"  - {Path(r['source_path']).name}: {r.get('error', 'Unknown')}")

    # Calculate average measurements
    if successful:
        all_depths = set()
        for r in successful:
            all_depths.update(r.get('measurements', {}).keys())

        logger.info("\nAverage measurements across all images:")
        for depth in sorted(all_depths):
            thicknesses = [
                r['measurements'][depth]['thickness_nm']
                for r in successful
                if depth in r.get('measurements', {})
            ]
            if thicknesses:
                import numpy as np
                mean_t = np.mean(thicknesses)
                std_t = np.std(thicknesses)
                logger.info(f"  Depth {float(depth):5.1f} nm: {mean_t:7.2f} ± {std_t:5.2f} nm")

    logger.info("=" * 60)


if __name__ == '__main__':
    main()
