"""Visualize red/green channels and masks for one example image.

Usage: python scripts/visualize_channels.py -c <config.json> [-o output_path] [-v]

This script reads the project's config.json to find the image directory, loads
the first TIFF it finds, computes green/red masks (Otsu or fixed thresholds),
and writes a composite PNG showing original, channels, and masks.
"""
import os
from pathlib import Path
import argparse
import json
from PIL import Image
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu


def compute_masks(image_np: np.ndarray, cfg_fp: dict):
    red = image_np[:, :, 0].astype(float)
    green = image_np[:, :, 1].astype(float)
    gray = np.mean(image_np, axis=2).astype(np.uint8)

    red_norm = red / 255.0
    green_norm = green / 255.0

    if cfg_fp.get('use_otsu', True):
        try:
            g_thresh = threshold_otsu(green)
            green_binary = (green > g_thresh).astype(np.uint8)
        except Exception:
            green_binary = (green_norm > float(cfg_fp.get('green_channel_threshold', 0.2))).astype(np.uint8)
        try:
            r_thresh = threshold_otsu(red)
            red_binary = (red > r_thresh).astype(np.uint8)
        except Exception:
            red_binary = (red_norm > float(cfg_fp.get('red_channel_threshold', 0.3))).astype(np.uint8)
    else:
        green_binary = (green_norm > float(cfg_fp.get('green_channel_threshold', 0.2))).astype(np.uint8)
        red_binary = (red_norm > float(cfg_fp.get('red_channel_threshold', 0.3))).astype(np.uint8)

    return red, green, gray, red_binary, green_binary


def find_first_tiff(image_dir: str):
    for root, dirs, files in os.walk(image_dir):
        for f in files:
            if f.lower().endswith(('.tif', '.tiff')):
                return os.path.join(root, f)
    return None


def main():
    parser = argparse.ArgumentParser(description='Visualize red/green channels and masks for one example image')
    parser.add_argument('--config', '-c', default=r"D:/GSU/CASA/Math Path 2/Code for RPE Crops/Final Code/config.json", help='Path to config.json')
    parser.add_argument('--output', '-o', default=None, help='Output image path (PNG). Defaults to <output_directory>/example_channels.png')
    parser.add_argument('--verbose', '-v', action='store_true')
    args = parser.parse_args()

    # normalize separators so forward-slash paths work on Windows
    config_path = os.path.normpath(args.config)
    if not os.path.isfile(config_path):
        print(f"Config not found: {config_path}")
        return

    with open(config_path, 'r') as fh:
        cfg = json.load(fh)

    image_dir = cfg.get('paths', {}).get('image_directory')
    out_dir = cfg.get('paths', {}).get('output_directory', '.')
    if image_dir is None:
        print('image_directory not set in config')
        return

    img_path = find_first_tiff(image_dir)
    if img_path is None:
        print('No TIFF images found under', image_dir)
        return

    if args.verbose:
        print('Example image:', img_path)

    with Image.open(img_path).convert('RGB') as im:
        img_np = np.array(im, dtype=np.uint8)

    cfg_fp = cfg.get('feature_params', {})
    red, green, gray, red_mask, green_mask = compute_masks(img_np, cfg_fp)

    # Prepare figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()
    axes[0].imshow(img_np)
    axes[0].set_title('Original')
    axes[0].axis('off')

    axes[1].imshow(gray, cmap='gray')
    axes[1].set_title('Grayscale')
    axes[1].axis('off')

    axes[2].imshow(green, cmap='Greens')
    axes[2].set_title('Green channel (raw)')
    axes[2].axis('off')

    axes[3].imshow(green_mask, cmap='gray')
    axes[3].set_title('Green mask (binary)')
    axes[3].axis('off')

    axes[4].imshow(red, cmap='Reds')
    axes[4].set_title('Red channel (raw)')
    axes[4].axis('off')

    axes[5].imshow(red_mask, cmap='gray')
    axes[5].set_title('Red mask (binary)')
    axes[5].axis('off')

    plt.tight_layout()

    if args.output:
        out_fp = args.output
    else:
        # prefer centralized helper
        from scripts.analysis import resolve_output_dirs
        base = Path(__file__).resolve().parents[1]
        _, _, _, plots_dir = resolve_output_dirs(base)
        out_fp = str(plots_dir / 'example_channels.png')

    fig.savefig(out_fp, dpi=150)
    print('Saved example channels image to', out_fp)


if __name__ == '__main__':
    main()
