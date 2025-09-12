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
    mean = np.mean(image)
    std = np.std(image)
    skewness = skew(image.flatten(), nan_policy='omit')
    kurt = kurtosis(image.flatten(), nan_policy='omit')
    return [mean, std, skewness if not np.isnan(skewness) else 0, kurt if not np.isnan(kurt) else 0]


def _compute_lbp_features(image: np.ndarray, P: int = 8, R: float = 1) -> List[float]:
    lbp = local_binary_pattern(image, P, R, method='uniform')
    n_bins = int(lbp.max() + 1)
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
    return (hist / (hist.sum() + 1e-6)).tolist()


def _flatten_features(features_dict: Dict[str, Any]) -> Dict[str, float]:
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


def extract_rpe_features(image_path: str, cfg_fp: dict, verbose: bool = True) -> Dict[str, Any]:
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

    lbp_points = int(cfg_fp.get("lbp_points", 8))
    lbp_radius = float(cfg_fp.get("lbp_radius", 1))
    green_thresh = float(cfg_fp.get("green_channel_threshold", 0.2))
    red_thresh = float(cfg_fp.get("red_channel_threshold", 0.3))

    red_norm = red_channel / 255.0
    green_norm = green_channel / 255.0

    try:
        if cfg_fp.get("use_otsu", True):
            g_thresh = threshold_otsu(green_channel)
            green_binary = (green_channel > g_thresh).astype(np.uint8)
        else:
            green_binary = (green_norm > green_thresh).astype(np.uint8)
    except Exception:
        green_binary = (green_norm > green_thresh).astype(np.uint8)

    try:
        if cfg_fp.get("use_otsu", True):
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

    def agg(prefix, arr):
        arr = np.array(arr, dtype=float)
        return {
            f"{prefix}_count": int(len(arr)),
            f"{prefix}_mean": float(np.mean(arr)),
            f"{prefix}_median": float(np.median(arr)),
            f"{prefix}_std": float(np.std(arr)),
            f"{prefix}_min": float(np.min(arr)),
            f"{prefix}_max": float(np.max(arr)),
        }

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
        levels = int(cfg_fp.get("glcm_levels", 64))
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
        gab_freqs = cfg_fp.get("gabor_frequencies", [0.1, 0.2])
        gab_angles = cfg_fp.get("gabor_angles", [0, np.pi/4, np.pi/2])
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
    features_list = []
    labels = []
    import os
    cfg_fp = config.get('feature_params', {})
    # collect all image paths first so we can show total progress
    image_paths = []
    for foldername in os.listdir(directory_path):
        folder_path = os.path.join(directory_path, foldername)
        if os.path.isdir(folder_path):
            for filename in os.listdir(folder_path):
                if filename.lower().endswith(('.tif', '.tiff')):
                    image_paths.append((foldername, os.path.join(folder_path, filename)))

    disable_bar = not bool(verbose)
    for foldername, image_path in tqdm(image_paths, desc="Extracting features", unit="img", disable=disable_bar):
        features = extract_rpe_features(image_path, cfg_fp, verbose=verbose)
        features_list.append(features)
        labels.append(foldername)

    flat_features = [ _flatten_features(f) for f in features_list ]
    df = pd.DataFrame(flat_features)
    df['label'] = labels
    df.fillna(0, inplace=True)
    return df
