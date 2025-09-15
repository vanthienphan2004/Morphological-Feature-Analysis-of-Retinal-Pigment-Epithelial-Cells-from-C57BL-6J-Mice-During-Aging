"""
Feature extraction module for RPE image analysis.

This module provides functions to extract morphological and texture features
from retinal pigment epithelial (RPE) cell images, including intensity statistics,
shape properties, texture features (LBP, GLCM, Gabor), and nuclear features.
"""

import json
from typing import Any, Dict, List
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from skimage import measure
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from skimage.filters import gabor, threshold_otsu
from scipy.stats import kurtosis, skew


def _compute_intensity_statistics(image: np.ndarray) -> List[float]:
    """
    Compute basic intensity statistics for a grayscale image.

    Args:
        image: Grayscale image as numpy array.

    Returns:
        List of [mean, std, skewness, kurtosis].
    """
    mean = np.mean(image)
    std = np.std(image)
    skewness = skew(image.flatten(), nan_policy='omit')
    kurt = kurtosis(image.flatten(), nan_policy='omit')
    return [mean, std, skewness if not np.isnan(skewness) else 0, kurt if not np.isnan(kurt) else 0]


def _compute_lbp_features(image: np.ndarray, P: int = 8, R: float = 1) -> List[float]:
    """
    Compute Local Binary Pattern (LBP) histogram features.

    Args:
        image: Grayscale image.
        P: Number of points for LBP.
        R: Radius for LBP.

    Returns:
        Normalized histogram of LBP values.
    """
    lbp = local_binary_pattern(image, P, R, method='uniform')
    n_bins = int(lbp.max() + 1)
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
    return (hist / (hist.sum() + 1e-6)).tolist()


def _flatten_features(features_dict: Dict[str, Any]) -> Dict[str, float]:
    """
    Flatten a nested features dictionary into a flat dict with string keys.

    Args:
        features_dict: Dictionary with potentially nested lists/arrays.

    Returns:
        Flattened dictionary with aggregated statistics for lists.
    """
    flat: Dict[str, float] = {}
    BIN_THRESHOLD = 20
    for key, val in features_dict.items():
        if isinstance(val, list):
            try:
                arr = np.array(val, dtype=float)
            except Exception:
                flat[f"{key}_count"] = float(len(val))
                continue

            if arr.size == 0:
                flat[f"{key}_count"] = 0.0
                flat[f"{key}_mean"] = 0.0
                flat[f"{key}_median"] = 0.0
                flat[f"{key}_std"] = 0.0
                flat[f"{key}_min"] = 0.0
                flat[f"{key}_max"] = 0.0
            elif arr.size <= BIN_THRESHOLD:
                for i, item in enumerate(arr.tolist()):
                    flat[f"{key}_{i}"] = float(item)
            else:
                flat[f"{key}_count"] = float(arr.size)
                flat[f"{key}_mean"] = float(np.mean(arr))
                flat[f"{key}_median"] = float(np.median(arr))
                flat[f"{key}_std"] = float(np.std(arr))
                flat[f"{key}_min"] = float(np.min(arr))
                flat[f"{key}_max"] = float(np.max(arr))
        else:
            if isinstance(val, (int, float, np.floating, np.integer)):
                flat[key] = float(val)
            else:
                flat[key] = val

    return flat


def agg(prefix: str, arr) -> Dict[str, Any]:
    """Aggregate a list/array of numeric values into summary statistics.

    Returns count, mean, median, std, min, max. Handles empty inputs safely.
    """
    arr = np.array(arr, dtype=float)
    if arr.size == 0:
        return {
            f"{prefix}_count": 0,
            f"{prefix}_mean": 0.0,
            f"{prefix}_median": 0.0,
            f"{prefix}_std": 0.0,
            f"{prefix}_min": 0.0,
            f"{prefix}_max": 0.0,
        }
    return {
        f"{prefix}_count": int(arr.size),
        f"{prefix}_mean": float(np.mean(arr)),
        f"{prefix}_median": float(np.median(arr)),
        f"{prefix}_std": float(np.std(arr)),
        f"{prefix}_min": float(np.min(arr)),
        f"{prefix}_max": float(np.max(arr)),
    }


def agg_nanaware(prefix: str, arr) -> Dict[str, Any]:
    """Aggregate numeric values while ignoring NaNs. Returns same keys as `agg`.

    If all entries are NaN or there are no valid values, returns zeros and count 0.
    """
    arr = np.array(arr, dtype=float)
    valid = arr[~np.isnan(arr)]
    if valid.size == 0:
        return {
            f"{prefix}_count": 0,
            f"{prefix}_mean": 0.0,
            f"{prefix}_median": 0.0,
            f"{prefix}_std": 0.0,
            f"{prefix}_min": 0.0,
            f"{prefix}_max": 0.0,
        }
    return {
        f"{prefix}_count": int(valid.size),
        f"{prefix}_mean": float(np.mean(valid)),
        f"{prefix}_median": float(np.median(valid)),
        f"{prefix}_std": float(np.std(valid)),
        f"{prefix}_min": float(np.min(valid)),
        f"{prefix}_max": float(np.max(valid)),
    }


def extract_rpe_features(image_path: str, config_params: dict, verbose: bool = True) -> Dict[str, Any]:
    """
    Extract features from a single RPE image.

    Args:
        image_path: Path to the image file.
        cfg_fp: Configuration parameters for feature extraction.
        verbose: Whether to print warnings.

    Returns:
        Dictionary of extracted features.
    """
    try:
        with Image.open(image_path).convert('RGB') as img:
            image_np = np.array(img, dtype=np.uint8)
    except IOError:
        if verbose:
            print(f"Warning: Could not open image file {image_path}. Skipping.")
        return {}

    red_channel = image_np[:, :, 0].astype(float)
    green_channel = image_np[:, :, 1].astype(float)
    gray = np.mean(image_np, axis=2).astype(np.uint8)

    lbp_points = int(config_params.get("lbp_points", 8))
    lbp_radius = float(config_params.get("lbp_radius", 1))
    green_thresh = float(config_params.get("green_channel_threshold", 0.2))
    red_thresh = float(config_params.get("red_channel_threshold", 0.3))

    red_norm = red_channel / 255.0
    green_norm = green_channel / 255.0

    try:
        if config_params.get("use_otsu", True):
            g_thresh = threshold_otsu(green_channel)
            green_binary = (green_channel > g_thresh).astype(np.uint8)
        else:
            green_binary = (green_norm > green_thresh).astype(np.uint8)
    except Exception:
        green_binary = (green_norm > green_thresh).astype(np.uint8)

    try:
        if config_params.get("use_otsu", True):
            r_thresh = threshold_otsu(red_channel)
            red_binary = (red_channel > r_thresh).astype(np.uint8)
        else:
            red_binary = (red_norm > red_thresh).astype(np.uint8)
    except Exception:
        red_binary = (red_norm > red_thresh).astype(np.uint8)

    features: Dict[str, Any] = {}

    mean, std, skewness, kurt = _compute_intensity_statistics(gray)
    features.update({
        "intensity_mean": mean,
        "intensity_std": std,
        "intensity_skew": skewness,
        "intensity_kurtosis": kurt,
    })

    labeled = measure.label(green_binary)
    props = measure.regionprops(labeled)
    n_cells = len(props)
    img_area = gray.shape[0] * gray.shape[1]
    features["cell_count"] = n_cells
    features["cell_density"] = n_cells / img_area if img_area > 0 else 0.0

    areas = [p.area for p in props] if props else [0]
    perimeters = [p.perimeter for p in props if p.perimeter is not None] if props else [0]
    major_axes = [p.major_axis_length for p in props if p.major_axis_length is not None] if props else [0]
    minor_axes = [p.minor_axis_length for p in props if p.minor_axis_length is not None] if props else [0]
    eccentricities = [p.eccentricity for p in props if p.eccentricity is not None] if props else [0]
    solidities = [p.solidity for p in props if p.solidity is not None] if props else [0]

    # Optional nuclear feature extraction (red channel) mapped to green-segmented cells
    # Controlled by config_params['extract_nuclear_features'] (defaults to False)
    if config_params.get("extract_nuclear_features", True):
        # Prepare lists to collect per-cell nuclear metrics
        nucleus_areas = []
        nucleus_perimeters = []
        nucleus_solidities = []
        nucleus_major_axes = []
        nucleus_minor_axes = []
        nucleus_circularity = []
        nc_ratios = []

        # Ensure red_binary is available and is binary (0/1)
        red_mask = (red_binary > 0).astype(np.uint8)

        # Iterate through each cell region and search for a nucleus within its bbox
        for p in props:
            # p.image is a boolean mask for the region within the bounding box
            minr, minc, maxr, maxc = p.bbox
            # Crop the red mask to the cell bounding box
            try:
                red_crop = red_mask[minr:maxr, minc:maxc]
            except Exception:
                # In case bbox is invalid, append zeros and continue
                nucleus_areas.append(0.0)
                nucleus_perimeters.append(0.0)
                nucleus_solidities.append(0.0)
                nucleus_major_axes.append(0.0)
                nucleus_minor_axes.append(0.0)
                nucleus_circularity.append(0.0)
                nc_ratios.append(0.0)
                continue

            # Restrict to inside the cell mask to avoid nuclei outside parent cell
            cell_mask = p.image.astype(np.uint8)
            # Align shapes: p.image has same shape as bbox region
            if cell_mask.shape != red_crop.shape:
                # If shapes don't match, try to pad or crop conservatively
                min_h = min(cell_mask.shape[0], red_crop.shape[0])
                min_w = min(cell_mask.shape[1], red_crop.shape[1])
                cell_mask = cell_mask[:min_h, :min_w]
                red_crop = red_crop[:min_h, :min_w]

            # Nucleus candidates inside the cell are where red_crop AND cell_mask are True
            nucleus_candidates = (red_crop & cell_mask).astype(np.uint8)

            # Label nucleus candidates within this cell bbox and pick the largest connected component (if any)
            n_lbl = measure.label(nucleus_candidates)
            n_props = measure.regionprops(n_lbl)

            if not n_props:
                # No nucleus found for this cell
                nucleus_areas.append(0.0)
                nucleus_perimeters.append(0.0)
                nucleus_solidities.append(0.0)
                nucleus_major_axes.append(0.0)
                nucleus_minor_axes.append(0.0)
                nucleus_circularity.append(0.0)
                nc_ratios.append(0.0)
                continue

            # Choose the largest nucleus by area within the cell
            n_props_sorted = sorted(n_props, key=lambda x: x.area, reverse=True)
            n = n_props_sorted[0]

            n_area = float(n.area)
            n_perim = float(n.perimeter) if getattr(n, 'perimeter', None) is not None else 0.0
            n_sol = float(n.solidity) if getattr(n, 'solidity', None) is not None else 0.0
            n_maj = float(n.major_axis_length) if getattr(n, 'major_axis_length', None) is not None else 0.0
            n_min = float(n.minor_axis_length) if getattr(n, 'minor_axis_length', None) is not None else 0.0

            # Circularity: 4*pi*area / perimeter^2 (guard against zero perimeter)
            if n_perim and n_perim > 0:
                n_circ = (4.0 * np.pi * n_area) / (n_perim * n_perim)
            else:
                n_circ = 0.0

            # N/C ratio: nucleus_area / (cell_area - nucleus_area) with division-by-zero guard
            cell_area = float(p.area)
            cytoplasm_area = max(cell_area - n_area, 0.0)
            if cytoplasm_area > 0:
                nc_ratio = n_area / cytoplasm_area
            else:
                # If cytoplasm area is zero or negative, fall back to NaN to indicate invalid
                nc_ratio = float('nan')

            nucleus_areas.append(n_area)
            nucleus_perimeters.append(n_perim)
            nucleus_solidities.append(n_sol)
            nucleus_major_axes.append(n_maj)
            nucleus_minor_axes.append(n_min)
            nucleus_circularity.append(n_circ)
            nc_ratios.append(nc_ratio)

        # Aggregate nuclear metrics using existing agg helper
        features.update(agg("nucleus_area", nucleus_areas))
        features.update(agg("nucleus_perimeter", nucleus_perimeters))
        features.update(agg("nucleus_major_axis", nucleus_major_axes))
        features.update(agg("nucleus_minor_axis", nucleus_minor_axes))
        features.update(agg("nucleus_circularity", nucleus_circularity))
        features.update(agg("nucleus_solidity", nucleus_solidities))
        # For N/C ratios, replace NaN with a large sentinel for aggregation guard, then restore NaN handling
        nc_arr = np.array([float(x) if not (isinstance(x, float) and np.isnan(x)) else np.nan for x in nc_ratios], dtype=float)
        # When computing statistics, np.nanmean / np.nanstd can be used; use the
        # module-level nan-aware aggregator for N/C ratios and the standard
        # aggregator for numeric lists.
        features.update(agg_nanaware("nc_ratio", nc_arr))

    features.update(agg("area", areas))
    features.update(agg("perimeter", perimeters))
    features.update(agg("major_axis", major_axes))
    features.update(agg("minor_axis", minor_axes))
    ar = [a / b if b > 0 else 0 for a, b in zip(major_axes, minor_axes)] if props else [0]
    features.update(agg("aspect_ratio", ar))
    features.update(agg("eccentricity", eccentricities))
    features.update(agg("solidity", solidities))

    circ = []
    for a, p in zip(areas, perimeters):
        if p and p > 0:
            circ.append((4 * np.pi * a) / (p * p))
        else:
            circ.append(0)
    features.update(agg("circularity", circ))

    centroids = [p.centroid for p in props] if props else []
    if len(centroids) > 1:
        pts = np.array(centroids)
        from scipy.spatial import distance
        dmat = distance.cdist(pts, pts)
        np.fill_diagonal(dmat, np.inf)
        nn = np.min(dmat, axis=1)
        features["nn_mean"] = float(np.mean(nn))
        features["nn_std"] = float(np.std(nn))
    else:
        features["nn_mean"] = 0.0
        features["nn_std"] = 0.0

    lbp_hist = _compute_lbp_features(gray, P=lbp_points, R=lbp_radius)
    for i, v in enumerate(lbp_hist):
        features[f"lbp_{i}"] = float(v)

    try:
        levels = int(config_params.get("glcm_levels", 64))
        gray_reduced = (gray / (256 // max(1, levels))).astype(np.uint8)
        distances = [1]
        angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
        glcm = graycomatrix(gray_reduced, distances=distances, angles=angles, levels=levels, symmetric=True, normed=True)
        props_names = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']
        for pn in props_names:
            try:
                vals = graycoprops(glcm, pn)
                features[f"glcm_{pn}_mean"] = float(np.mean(vals))
                features[f"glcm_{pn}_std"] = float(np.std(vals))
            except Exception:
                features[f"glcm_{pn}_mean"] = 0.0
                features[f"glcm_{pn}_std"] = 0.0
    except Exception:
        for pn in ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']:
            features[f"glcm_{pn}_mean"] = 0.0
            features[f"glcm_{pn}_std"] = 0.0

    try:
        gab_freqs = config_params.get("gabor_frequencies", [0.1, 0.2])
        gab_angles = config_params.get("gabor_angles", [0, np.pi/4, np.pi/2])
        for freq in gab_freqs:
            for ang in gab_angles:
                try:
                    real, imag = gabor(gray.astype(float), frequency=float(freq), theta=float(ang))
                    mag = np.hypot(real, imag)
                    mag = np.where(np.isfinite(mag), mag, np.nan)
                    mean_val = np.nanmean(mag)
                    std_val = np.nanstd(mag)
                    if not np.isfinite(mean_val):
                        mean_val = 0.0
                    if not np.isfinite(std_val):
                        std_val = 0.0
                    features[f"gabor_f{freq}_a{int(np.degrees(ang))}_mean"] = float(mean_val)
                    features[f"gabor_f{freq}_a{int(np.degrees(ang))}_std"] = float(std_val)
                except Exception:
                    features[f"gabor_f{freq}_a{int(np.degrees(ang))}_mean"] = 0.0
                    features[f"gabor_f{freq}_a{int(np.degrees(ang))}_std"] = 0.0
    except Exception:
        for freq in [0.1, 0.2]:
            for ang in [0, np.pi/4, np.pi/2]:
                features[f"gabor_f{freq}_a{int(np.degrees(ang))}_mean"] = 0.0
                features[f"gabor_f{freq}_a{int(np.degrees(ang))}_std"] = 0.0

    return features


def extract_features_from_directory(directory_path: str, config: dict, verbose: bool = False) -> pd.DataFrame:
    """
    Extract features from all images in a directory.

    Args:
        directory_path: Path to the directory containing image folders.
        config: Configuration dictionary.
        verbose: Whether to enable verbose output.

    Returns:
        DataFrame with extracted features and labels.
    """
    features_list = []
    labels = []
    from pathlib import Path
    config_params = config.get('feature_params', {})
    # collect all image paths first so we can show total progress
    image_paths = []
    dir_path = Path(directory_path)
    for foldername in dir_path.iterdir():
        if foldername.is_dir():
            for filename in foldername.iterdir():
                if filename.suffix.lower() in ('.tif', '.tiff'):
                    image_paths.append((foldername.name, str(filename)))

    disable_bar = not bool(verbose)
    for foldername, image_path in tqdm(image_paths, desc="Extracting features", unit="img", disable=disable_bar):
        features = extract_rpe_features(image_path, config_params, verbose=verbose)
        features_list.append(features)
        labels.append(foldername)

    flat_features = [ _flatten_features(f) for f in features_list ]
    df = pd.DataFrame(flat_features)
    df['label'] = labels
    df.fillna(0, inplace=True)
    return df
