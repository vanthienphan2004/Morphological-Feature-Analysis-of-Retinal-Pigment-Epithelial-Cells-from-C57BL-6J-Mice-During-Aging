"""Visualize red/green channels and masks for one example image.

Usage: python scripts/visualize_channels.py -c <config.json> [-o output_path] [-v]

This script reads the project's config.json to find the image directory, loads
the first TIFF it finds, computes green/red masks (Otsu or fixed thresholds),
and writes a composite PNG showing original, channels, and masks.
"""
from pathlib import Path
from typing import Dict, Optional, Tuple, Any
import argparse
import json
from PIL import Image
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu


def compute_masks(image_array: np.ndarray, config_params: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute red, green, gray channels and binary masks.

    Args:
        image_np: RGB image as numpy array.
        config_params: Configuration parameters.

    Returns:
        Tuple of red, green, gray, red_mask, green_mask.
    """
    red_channel = image_array[:, :, 0].astype(float)
    green_channel = image_array[:, :, 1].astype(float)
    grayscale_image = np.mean(image_array, axis=2).astype(np.uint8)

    red_normalized = red_channel / 255.0
    green_normalized = green_channel / 255.0

    if config_params.get('use_otsu', True):
        try:
            green_threshold = threshold_otsu(green_channel)
            green_binary = (green_channel > green_threshold).astype(np.uint8)
        except Exception:
            green_binary = (green_normalized > float(config_params.get('green_channel_threshold', 0.2))).astype(np.uint8)
        try:
            red_threshold = threshold_otsu(red_channel)
            red_binary = (red_channel > red_threshold).astype(np.uint8)
        except Exception:
            red_binary = (red_normalized > float(config_params.get('red_channel_threshold', 0.3))).astype(np.uint8)
    else:
        green_binary = (green_normalized > float(config_params.get('green_channel_threshold', 0.2))).astype(np.uint8)
        red_binary = (red_normalized > float(config_params.get('red_channel_threshold', 0.3))).astype(np.uint8)

    return red_channel, green_channel, grayscale_image, red_binary, green_binary


def find_first_tiff(image_dir: str) -> Optional[str]:
    """
    Find the first TIFF file in the directory tree.

    Args:
        image_dir: Directory to search.

    Returns:
        Path to the first TIFF file or None.
    """
    image_path = Path(image_dir)
    for file_path in image_path.rglob('*'):
        if file_path.suffix.lower() in ('.tif', '.tiff'):
            return str(file_path)
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description='Visualize red/green channels and masks for one example image')
    parser.add_argument('--config', '-c', default="config.json", help='Path to config.json')
    parser.add_argument('--output', '-o', default=None, help='Output image path (PNG). Defaults to <output_directory>/example_channels.png')
    parser.add_argument('--verbose', '-v', action='store_true')
    args = parser.parse_args()

    # normalize separators so forward-slash paths work on Windows
    config_path = Path(args.config).resolve()
    if not config_path.is_file():
        print(f"Config not found: {config_path}")
        return

    try:
        with open(config_path, 'r') as fh:
            config_data = json.load(fh)
    except Exception as e:
        print(f"Error loading config: {e}")
        return

    image_dir = config_data.get('paths', {}).get('image_directory')
    output_dir = config_data.get('paths', {}).get('output_directory', '.')
    if image_dir is None:
        print('image_directory not set in config')
        return

    image_path = find_first_tiff(image_dir)
    if image_path is None:
        print('No TIFF images found under', image_dir)
        return

    if args.verbose:
        print('Example image:', image_path)

    try:
        with Image.open(image_path).convert('RGB') as im:
            image_array = np.array(im, dtype=np.uint8)
    except Exception as e:
        print(f"Error loading image: {e}")
        return

    config_params = config_data.get('feature_params', {})
    red_channel, green_channel, grayscale_image, red_mask, green_mask = compute_masks(image_array, config_params)

    # Prepare figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()
    axes[0].imshow(image_array)
    axes[0].set_title('Original')
    axes[0].axis('off')

    axes[1].imshow(grayscale_image, cmap='gray')
    axes[1].set_title('Grayscale')
    axes[1].axis('off')

    axes[2].imshow(green_channel, cmap='Greens')
    axes[2].set_title('Green channel (raw)')
    axes[2].axis('off')

    axes[3].imshow(green_mask, cmap='gray')
    axes[3].set_title('Green mask (binary)')
    axes[3].axis('off')

    axes[4].imshow(red_channel, cmap='Reds')
    axes[4].set_title('Red channel (raw)')
    axes[4].axis('off')

    axes[5].imshow(red_mask, cmap='gray')
    axes[5].set_title('Red mask (binary)')
    axes[5].axis('off')

    plt.tight_layout()

    if args.output:
        output_path = args.output
    else:
        # prefer centralized helper
        from scripts.analysis import resolve_output_dirs
        base = Path(__file__).resolve().parents[1]
        _, _, _, plots_dir = resolve_output_dirs(base)
        output_path = str(plots_dir / 'example_channels.png')

    try:
        fig.savefig(output_path, dpi=150)
        print('Saved example channels image to', output_path)
    except Exception as e:
        print(f"Error saving image: {e}")


if __name__ == '__main__':
    main()
