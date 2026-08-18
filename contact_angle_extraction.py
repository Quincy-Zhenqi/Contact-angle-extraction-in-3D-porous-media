"""Physics-constrained contact-angle extraction from segmented 3D TIFF data.

The input volume is indexed as (z, y, x). By default, phase labels are
1 = solid, 2 = water, and 3 = gas. Exported point coordinates use (x, y, z).
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile
from scipy.ndimage import convolve, generate_binary_structure, label, uniform_filter


@dataclass(frozen=True)
class ContactAngleConfig:
    """Numerical labels and quality-control thresholds for the workflow."""

    solid_label: int = 1
    water_label: int = 2
    gas_label: int = 3
    min_water_component_voxels: int = 50
    min_gas_component_voxels: int = 50
    density_reject_neighbor_count: int = 4
    local_count_window_radius: int = 5
    min_local_phase_voxels: int = 20
    min_interface_component_voxels: int = 15
    sphere_radius: int = 15
    solid_ratio_min: float = 0.3
    solid_ratio_max: float = 0.7
    water_ratio_min: float = 0.1
    water_ratio_max: float = 0.9
    gas_ratio_min: float = 0.1
    gas_ratio_max: float = 0.9
    patch_window_radius: int = 4
    patch_max_points: int = 512
    min_patch_points: int = 15
    max_plane_rms: float = 1.0
    max_singular_value_ratio: float = 0.3
    normal_probe_steps: Tuple[int, ...] = (1, 2, 3)
    allow_third_phase_orientation_fallback: bool = True
    histogram_bins: int = 36
    low_angle_threshold: float = 10.0
    high_angle_threshold: float = 170.0
    progress_interval: int = 1000
    save_angle_map: bool = True
    save_npz: bool = True

    def validate(self) -> None:
        labels = (self.solid_label, self.water_label, self.gas_label)
        if len(set(labels)) != 3 or any(value < 0 for value in labels):
            raise ValueError("Solid, water, and gas labels must be distinct non-negative integers.")

        positive_integer_fields = {
            "min_water_component_voxels": self.min_water_component_voxels,
            "min_gas_component_voxels": self.min_gas_component_voxels,
            "density_reject_neighbor_count": self.density_reject_neighbor_count,
            "local_count_window_radius": self.local_count_window_radius,
            "min_local_phase_voxels": self.min_local_phase_voxels,
            "min_interface_component_voxels": self.min_interface_component_voxels,
            "sphere_radius": self.sphere_radius,
            "patch_window_radius": self.patch_window_radius,
            "patch_max_points": self.patch_max_points,
            "min_patch_points": self.min_patch_points,
            "histogram_bins": self.histogram_bins,
            "progress_interval": self.progress_interval,
        }
        for name, value in positive_integer_fields.items():
            if not isinstance(value, int) or value <= 0:
                raise ValueError("{} must be a positive integer.".format(name))

        if self.density_reject_neighbor_count > 7:
            raise ValueError("density_reject_neighbor_count must not exceed 7.")
        if self.patch_max_points < self.min_patch_points:
            raise ValueError("patch_max_points must be at least min_patch_points.")
        if self.max_plane_rms < 0 or self.max_singular_value_ratio < 0:
            raise ValueError("Plane-quality thresholds must be non-negative.")
        if not self.normal_probe_steps or any(step <= 0 for step in self.normal_probe_steps):
            raise ValueError("normal_probe_steps must contain positive integers.")
        if not (0 <= self.low_angle_threshold <= self.high_angle_threshold <= 180):
            raise ValueError("Angle thresholds must satisfy 0 <= low <= high <= 180.")

        ratio_ranges = {
            "solid": (self.solid_ratio_min, self.solid_ratio_max),
            "water": (self.water_ratio_min, self.water_ratio_max),
            "gas": (self.gas_ratio_min, self.gas_ratio_max),
        }
        for phase, (lower, upper) in ratio_ranges.items():
            if not (0 <= lower <= upper <= 1):
                raise ValueError("{} ratio bounds must satisfy 0 <= min <= max <= 1.".format(phase))


def load_config(path: Optional[Path]) -> ContactAngleConfig:
    """Load configuration overrides from JSON, or return defaults."""

    if path is None:
        config = ContactAngleConfig()
    else:
        with Path(path).open("r", encoding="utf-8") as handle:
            values = json.load(handle)
        valid_names = {field.name for field in fields(ContactAngleConfig)}
        unknown = sorted(set(values) - valid_names)
        if unknown:
            raise ValueError("Unknown configuration field(s): {}".format(", ".join(unknown)))
        if "normal_probe_steps" in values:
            values["normal_probe_steps"] = tuple(values["normal_probe_steps"])
        config = ContactAngleConfig(**values)
    config.validate()
    return config


def precompute_sphere_offsets(radius: int) -> np.ndarray:
    """Return integer (dz, dy, dx) offsets inside a voxel sphere."""

    coordinates = np.arange(-radius, radius + 1, dtype=np.int16)
    dz, dy, dx = np.meshgrid(coordinates, coordinates, coordinates, indexing="ij")
    mask = dz.astype(np.int32) ** 2 + dy.astype(np.int32) ** 2 + dx.astype(np.int32) ** 2 <= radius**2
    return np.column_stack((dz[mask], dy[mask], dx[mask])).astype(np.int16, copy=False)


def _retain_large_components(mask: np.ndarray, minimum_size: int) -> np.ndarray:
    structure = generate_binary_structure(3, 1)
    component_labels, component_count = label(mask, structure=structure)
    if component_count == 0:
        return np.zeros_like(mask, dtype=bool)
    component_sizes = np.bincount(component_labels.ravel())
    valid = component_labels != 0
    valid &= component_sizes[component_labels] >= minimum_size
    return valid


def filter_small_regions_3d(volume: np.ndarray, config: ContactAngleConfig) -> np.ndarray:
    """Replace small water regions with gas and small gas regions with water."""

    start = time.time()
    filtered = volume.copy()
    water_mask = volume == config.water_label
    gas_mask = volume == config.gas_label
    valid_water = _retain_large_components(water_mask, config.min_water_component_voxels)
    valid_gas = _retain_large_components(gas_mask, config.min_gas_component_voxels)
    filtered[water_mask & ~valid_water] = config.gas_label
    filtered[gas_mask & ~valid_gas] = config.water_label
    print("Small-region filtering completed in {:.2f} s".format(time.time() - start))
    return filtered


def extract_interface_6conn(volume: np.ndarray, primary_label: int, neighbour_label: int) -> np.ndarray:
    """Return primary-phase voxels sharing a face with the neighbour phase."""

    kernel = np.zeros((3, 3, 3), dtype=np.uint8)
    kernel[1, 1, 0] = kernel[1, 1, 2] = 1
    kernel[1, 0, 1] = kernel[1, 2, 1] = 1
    kernel[0, 1, 1] = kernel[2, 1, 1] = 1
    neighbour_count = convolve(
        (volume == neighbour_label).astype(np.uint8), kernel, mode="constant", cval=0
    )
    return (volume == primary_label) & (neighbour_count > 0)


def find_three_phase_points_3d(interface_water_gas: np.ndarray, interface_solid_water: np.ndarray) -> np.ndarray:
    """Return candidate contact voxels as integer (z, y, x) coordinates."""

    return np.argwhere(interface_water_gas & interface_solid_water)


def is_valid_triple_point_6neigh(
    volume: np.ndarray, z: int, y: int, x: int, config: ContactAngleConfig
) -> bool:
    """Require the six face neighbours to collectively contain all phases."""

    offsets = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))
    values = set()
    for dz, dy, dx in offsets:
        zz, yy, xx = z + dz, y + dy, x + dx
        if 0 <= zz < volume.shape[0] and 0 <= yy < volume.shape[1] and 0 <= xx < volume.shape[2]:
            values.add(int(volume[zz, yy, xx]))
    return {config.solid_label, config.water_label, config.gas_label}.issubset(values)


def compute_phase_ratios_in_sphere(
    volume: np.ndarray,
    z: int,
    y: int,
    x: int,
    offsets: np.ndarray,
    config: ContactAngleConfig,
) -> Tuple[float, float, float]:
    """Calculate solid, water, and gas fractions in a clipped voxel sphere."""

    zz = z + offsets[:, 0]
    yy = y + offsets[:, 1]
    xx = x + offsets[:, 2]
    inside = (
        (zz >= 0)
        & (zz < volume.shape[0])
        & (yy >= 0)
        & (yy < volume.shape[1])
        & (xx >= 0)
        & (xx < volume.shape[2])
    )
    if not np.any(inside):
        return 0.0, 0.0, 0.0
    voxels = volume[zz[inside], yy[inside], xx[inside]]
    total = float(voxels.size)
    return (
        float(np.count_nonzero(voxels == config.solid_label) / total),
        float(np.count_nonzero(voxels == config.water_label) / total),
        float(np.count_nonzero(voxels == config.gas_label) / total),
    )


def fit_plane(points_xyz: np.ndarray, minimum_points: int = 10) -> Tuple[Optional[np.ndarray], float, float]:
    """Fit a plane by SVD and return its normal, RMS residual, and s3/s2 ratio."""

    points = np.asarray(points_xyz, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] < minimum_points:
        return None, float("nan"), float("nan")
    centered = points - points.mean(axis=0)
    try:
        _, singular_values, right_vectors = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
        return None, float("nan"), float("nan")
    if singular_values.size < 3:
        return None, float("nan"), float("nan")
    normal = right_vectors[2]
    residual = float(np.linalg.norm(centered @ normal) / np.sqrt(points.shape[0]))
    ratio = float(singular_values[2] / singular_values[1]) if singular_values[1] > 1e-12 else float("inf")
    return normal, residual, ratio


def orient_normal(
    volume: np.ndarray,
    point_xyz: Tuple[int, int, int],
    normal: Optional[np.ndarray],
    positive_label: int,
    negative_label: int,
    probe_steps: Sequence[int],
    fallback_positive_priority: Sequence[int] = (),
) -> Optional[np.ndarray]:
    """Orient a normal from pure phase samples on its two probe sides.

    The expected pair is handled first. At a discrete three-phase line, a
    probe may cross the third phase instead. An optional priority list
    reproduces the original algorithm's fallback orientation convention.
    """

    if normal is None:
        return None
    x, y, z = point_xyz
    positive_values: List[int] = []
    negative_values: List[int] = []
    for step in probe_steps:
        plus = np.rint(np.array((x, y, z), dtype=float) + normal * step).astype(int)
        minus = np.rint(np.array((x, y, z), dtype=float) - normal * step).astype(int)
        px, py, pz = plus
        nx, ny, nz = minus
        if not (0 <= px < volume.shape[2] and 0 <= py < volume.shape[1] and 0 <= pz < volume.shape[0]):
            return None
        if not (0 <= nx < volume.shape[2] and 0 <= ny < volume.shape[1] and 0 <= nz < volume.shape[0]):
            return None
        positive_values.append(int(volume[pz, py, px]))
        negative_values.append(int(volume[nz, ny, nx]))

    if len(set(positive_values)) != 1 or len(set(negative_values)) != 1:
        return None
    plus_label = positive_values[0]
    minus_label = negative_values[0]
    if plus_label == minus_label:
        return None
    observed_pair = {plus_label, minus_label}
    if observed_pair == {positive_label, negative_label}:
        target_positive = positive_label
    else:
        target_positive = next(
            (label_value for label_value in fallback_positive_priority if label_value in observed_pair),
            None,
        )
        if target_positive is None:
            return None
    if plus_label == target_positive:
        return normal
    if minus_label == target_positive:
        return -normal
    return None


def compute_contact_angle(vector_a: Optional[np.ndarray], vector_b: Optional[np.ndarray]) -> Optional[float]:
    """Return the angle between two oriented normals in degrees."""

    if vector_a is None or vector_b is None:
        return None
    norm_a = float(np.linalg.norm(vector_a))
    norm_b = float(np.linalg.norm(vector_b))
    if norm_a <= 1e-12 or norm_b <= 1e-12:
        return None
    dot_product = float(np.clip(np.dot(vector_a / norm_a, vector_b / norm_b), -1.0, 1.0))
    return float(np.degrees(np.arccos(dot_product)))


def get_patch_points_from_global_labels(
    labelled_interface: np.ndarray,
    z: int,
    y: int,
    x: int,
    window_radius: int,
    maximum_points: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return local points from the same globally connected interface component."""

    component_id = labelled_interface[z, y, x]
    if component_id == 0:
        return np.empty((0, 3), dtype=int)
    z0, z1 = max(0, z - window_radius), min(labelled_interface.shape[0], z + window_radius + 1)
    y0, y1 = max(0, y - window_radius), min(labelled_interface.shape[1], y + window_radius + 1)
    x0, x1 = max(0, x - window_radius), min(labelled_interface.shape[2], x + window_radius + 1)
    local = labelled_interface[z0:z1, y0:y1, x0:x1]
    points = np.argwhere(local == component_id)
    if points.size == 0:
        return np.empty((0, 3), dtype=int)
    points += np.array((z0, y0, x0), dtype=int)
    if points.shape[0] > maximum_points:
        chosen = rng.choice(points.shape[0], size=maximum_points, replace=False)
        points = points[chosen]
    return points


def _coordinate_array(points: Sequence[Tuple[int, int, int]]) -> np.ndarray:
    return np.asarray(points, dtype=np.int32).reshape(-1, 3) if points else np.empty((0, 3), dtype=np.int32)


def _vector_array(points: Sequence[Tuple[float, ...]]) -> np.ndarray:
    return np.asarray(points, dtype=float).reshape(-1, 13) if points else np.empty((0, 13), dtype=float)


def _summary_value(values: Sequence[float], operation: str) -> float:
    if not values:
        return float("nan")
    array = np.asarray(values, dtype=float)
    functions = {"mean": np.mean, "median": np.median, "std": np.std, "min": np.min, "max": np.max}
    return float(functions[operation](array))


def _output_paths(output_directory: Path, image_stem: str, config: ContactAngleConfig) -> Dict[str, Path]:
    paths = {
        "histogram": output_directory / "{}_contact_angle_histogram.tiff".format(image_stem),
        "histogram_data": output_directory / "{}_contact_angle_histogram_data.csv".format(image_stem),
        "point_data": output_directory / "{}_contact_angle_all_data.csv".format(image_stem),
        "statistics": output_directory / "{}_contact_angle_statistics.csv".format(image_stem),
        "rejections": output_directory / "{}_contact_angle_rejection_summary.csv".format(image_stem),
        "config": output_directory / "{}_effective_config.json".format(image_stem),
    }
    if config.save_angle_map:
        paths["angle_map"] = output_directory / "{}_contact_angle_map.tif".format(image_stem)
    if config.save_npz:
        paths["npz"] = output_directory / "{}_results.npz".format(image_stem)
    return paths


def _check_outputs(paths: Dict[str, Path], overwrite: bool) -> None:
    existing = sorted(str(path) for path in paths.values() if path.exists())
    if existing and not overwrite:
        raise FileExistsError(
            "Output files already exist. Use --overwrite to replace them:\n{}".format("\n".join(existing))
        )


def _write_outputs(
    volume: np.ndarray,
    image_path: Path,
    paths: Dict[str, Path],
    config: ContactAngleConfig,
    contact_angles: Sequence[float],
    good_points: Sequence[Tuple[int, int, int]],
    bad_points: Sequence[Tuple[int, int, int]],
    vector_points: Sequence[Tuple[float, ...]],
    solid_ratios: Sequence[float],
    water_ratios: Sequence[float],
    gas_ratios: Sequence[float],
    diagnostic_points: Dict[str, Sequence[Tuple[int, int, int]]],
    rejection_counts: Counter,
    total_triple_points: int,
    processing_time: float,
) -> Dict[str, Any]:
    angles = np.asarray(contact_angles, dtype=float)
    point_coordinates = _coordinate_array(good_points)

    counts, bin_edges = np.histogram(angles, bins=config.histogram_bins, range=(0.0, 180.0))
    relative = counts / counts.sum() if counts.sum() else np.zeros_like(counts, dtype=float)
    pd.DataFrame(
        {
            "Bin_Start": bin_edges[:-1],
            "Bin_End": bin_edges[1:],
            "Bin_Center": 0.5 * (bin_edges[:-1] + bin_edges[1:]),
            "Frequency": counts,
            "Relative_Frequency": relative,
        }
    ).to_csv(paths["histogram_data"], index=False)

    figure, axis = plt.subplots(figsize=(10, 6))
    axis.hist(angles, bins=config.histogram_bins, range=(0, 180), color="skyblue", edgecolor="black")
    axis.set_xlabel("Contact angle (degrees)")
    axis.set_ylabel("Frequency")
    if angles.size:
        axis.set_title(
            "Contact Angle Distribution - {}\nMean: {:.2f}°, Std: {:.2f}°, Count: {}".format(
                image_path.stem, float(np.mean(angles)), float(np.std(angles)), angles.size
            )
        )
    else:
        axis.set_title("Contact Angle Distribution - {}\nNo valid contact angles".format(image_path.stem))
    figure.tight_layout()
    figure.savefig(paths["histogram"], dpi=300)
    plt.close(figure)

    pd.DataFrame(
        {
            "X": point_coordinates[:, 0] if point_coordinates.size else np.array([], dtype=int),
            "Y": point_coordinates[:, 1] if point_coordinates.size else np.array([], dtype=int),
            "Z": point_coordinates[:, 2] if point_coordinates.size else np.array([], dtype=int),
            "Contact_Angle_Deg": angles,
            "Solid_Ratio": np.asarray(solid_ratios, dtype=float),
            "Water_Ratio": np.asarray(water_ratios, dtype=float),
            "Gas_Ratio": np.asarray(gas_ratios, dtype=float),
        }
    ).to_csv(paths["point_data"], index=False)

    statistics = {
        "Image": image_path.name,
        "Dimensions_ZYX": str(tuple(int(value) for value in volume.shape)),
        "Total_Triple_Points": int(total_triple_points),
        "Valid_Contact_Angles": int(angles.size),
        "Mean_Contact_Angle_Deg": _summary_value(contact_angles, "mean"),
        "Median_Contact_Angle_Deg": _summary_value(contact_angles, "median"),
        "Std_Contact_Angle_Deg": _summary_value(contact_angles, "std"),
        "Min_Contact_Angle_Deg": _summary_value(contact_angles, "min"),
        "Max_Contact_Angle_Deg": _summary_value(contact_angles, "max"),
        "Mean_Solid_Ratio": _summary_value(solid_ratios, "mean"),
        "Mean_Water_Ratio": _summary_value(water_ratios, "mean"),
        "Mean_Gas_Ratio": _summary_value(gas_ratios, "mean"),
        "Processing_Time_sec": float(processing_time),
    }
    pd.DataFrame([statistics]).to_csv(paths["statistics"], index=False)
    pd.DataFrame(sorted(rejection_counts.items()), columns=("Rejection_Reason", "Count")).to_csv(
        paths["rejections"], index=False
    )
    with paths["config"].open("w", encoding="utf-8") as handle:
        json.dump(asdict(config), handle, indent=2)

    if config.save_angle_map:
        angle_volume = np.zeros(volume.shape, dtype=np.float32)
        for (x, y, z), angle in zip(good_points, contact_angles):
            angle_volume[z, y, x] = angle
        tifffile.imwrite(
            paths["angle_map"],
            angle_volume,
            photometric="minisblack",
            bigtiff=angle_volume.nbytes > (2**32 - 2**25),
        )

    if config.save_npz:
        archive_values: Dict[str, Any] = {
            "contact_angles": angles,
            "solid_ratios": np.asarray(solid_ratios, dtype=float),
            "water_ratios": np.asarray(water_ratios, dtype=float),
            "gas_ratios": np.asarray(gas_ratios, dtype=float),
            "good_points_xyz": point_coordinates,
            "bad_points_xyz": _coordinate_array(bad_points),
            "vector_points": _vector_array(vector_points),
            "rejection_reasons": np.asarray(sorted(rejection_counts), dtype=str),
            "rejection_counts": np.asarray([rejection_counts[key] for key in sorted(rejection_counts)], dtype=int),
        }
        for name, points in diagnostic_points.items():
            archive_values[name] = _coordinate_array(points)
        np.savez_compressed(paths["npz"], **archive_values)
    return statistics


def process_single_image(
    image_path: Path,
    output_directory: Optional[Path] = None,
    config: Optional[ContactAngleConfig] = None,
    seed: Optional[int] = 0,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """Run the complete workflow for one segmented 3D TIFF stack."""

    config = config or ContactAngleConfig()
    config.validate()
    image_path = Path(image_path).resolve()
    if not image_path.is_file():
        raise FileNotFoundError("Input TIFF not found: {}".format(image_path))
    if output_directory is None:
        output_directory = image_path.parent / "{}_contact_angle_results".format(image_path.stem)
    output_directory = Path(output_directory).resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    paths = _output_paths(output_directory, image_path.stem, config)
    _check_outputs(paths, overwrite)

    start_total = time.time()
    print("Reading volume: {}".format(image_path))
    volume = tifffile.imread(image_path)
    if volume.ndim != 3:
        raise ValueError("Input must be a single-channel 3D TIFF stack; received shape {}.".format(volume.shape))
    expected_labels = (config.solid_label, config.water_label, config.gas_label)
    missing_labels = [value for value in expected_labels if not np.any(volume == value)]
    if missing_labels:
        raise ValueError("Input volume is missing required phase label(s): {}".format(missing_labels))
    print("Volume shape (z, y, x): {}; dtype: {}".format(volume.shape, volume.dtype))

    filtered_volume = filter_small_regions_3d(volume, config)
    interface_water_gas = extract_interface_6conn(filtered_volume, config.water_label, config.gas_label)
    interface_solid_water = extract_interface_6conn(filtered_volume, config.water_label, config.solid_label)
    connectivity_18 = generate_binary_structure(3, 2)
    labelled_water_gas, _ = label(interface_water_gas, structure=connectivity_18)
    labelled_solid_water, _ = label(interface_solid_water, structure=connectivity_18)
    component_sizes_water_gas = np.bincount(labelled_water_gas.ravel())
    component_sizes_solid_water = np.bincount(labelled_solid_water.ravel())
    triple_points = find_three_phase_points_3d(interface_water_gas, interface_solid_water)
    print("Candidate three-phase contact voxels: {}".format(len(triple_points)))

    face_kernel = np.zeros((3, 3, 3), dtype=np.uint8)
    face_kernel[1, 1, 0] = face_kernel[1, 1, 2] = 1
    face_kernel[1, 0, 1] = face_kernel[1, 2, 1] = 1
    face_kernel[0, 1, 1] = face_kernel[2, 1, 1] = 1
    density_map = np.zeros(filtered_volume.shape, dtype=np.uint8)
    if len(triple_points):
        density_map[triple_points[:, 0], triple_points[:, 1], triple_points[:, 2]] = 1
    neighbour_counts = convolve(density_map, face_kernel, mode="constant", cval=0)

    window_width = 2 * config.local_count_window_radius + 1
    window_voxels = window_width**3
    solid_count_map = uniform_filter(
        (filtered_volume == config.solid_label).astype(np.float32), size=window_width, mode="constant", cval=0.0
    ) * window_voxels
    water_count_map = uniform_filter(
        (filtered_volume == config.water_label).astype(np.float32), size=window_width, mode="constant", cval=0.0
    ) * window_voxels
    gas_count_map = uniform_filter(
        (filtered_volume == config.gas_label).astype(np.float32), size=window_width, mode="constant", cval=0.0
    ) * window_voxels

    sphere_offsets = precompute_sphere_offsets(config.sphere_radius)
    rng = np.random.default_rng(seed)
    contact_angles: List[float] = []
    solid_ratios: List[float] = []
    water_ratios: List[float] = []
    gas_ratios: List[float] = []
    good_points: List[Tuple[int, int, int]] = []
    bad_points: List[Tuple[int, int, int]] = []
    vector_points: List[Tuple[float, ...]] = []
    rejection_counts: Counter = Counter()
    diagnostic_points: Dict[str, List[Tuple[int, int, int]]] = {
        "low_angle_points_xyz": [],
        "high_angle_points_xyz": [],
        "low_solid_ratio_points_xyz": [],
        "high_solid_ratio_points_xyz": [],
        "low_water_ratio_points_xyz": [],
        "high_water_ratio_points_xyz": [],
        "low_gas_ratio_points_xyz": [],
        "high_gas_ratio_points_xyz": [],
        "high_density_points_xyz": [],
    }

    def reject(reason: str, point: Tuple[int, int, int]) -> None:
        rejection_counts[reason] += 1
        bad_points.append(point)

    start_points = time.time()
    total_points = len(triple_points)
    for index, (z_value, y_value, x_value) in enumerate(triple_points):
        z, y, x = int(z_value), int(y_value), int(x_value)
        point = (x, y, z)
        if index % config.progress_interval == 0 or index == total_points - 1:
            if index == 0:
                print("Processed 1/{}".format(total_points))
            else:
                elapsed = time.time() - start_points
                speed = (index + 1) / elapsed
                remaining = (total_points - index - 1) / speed if speed else float("inf")
                print(
                    "Processed {}/{} ({:.1f} points/s; ETA {:.1f} s)".format(
                        index + 1, total_points, speed, remaining
                    )
                )

        if neighbour_counts[z, y, x] >= config.density_reject_neighbor_count:
            diagnostic_points["high_density_points_xyz"].append(point)
            reject("high_candidate_density", point)
            continue
        if not is_valid_triple_point_6neigh(filtered_volume, z, y, x, config):
            reject("invalid_six_neighbour_phase_set", point)
            continue
        if min(solid_count_map[z, y, x], water_count_map[z, y, x], gas_count_map[z, y, x]) < config.min_local_phase_voxels:
            reject("insufficient_local_phase_voxels", point)
            continue

        water_gas_component = labelled_water_gas[z, y, x]
        solid_water_component = labelled_solid_water[z, y, x]
        if (
            water_gas_component == 0
            or solid_water_component == 0
            or component_sizes_water_gas[water_gas_component] < config.min_interface_component_voxels
            or component_sizes_solid_water[solid_water_component] < config.min_interface_component_voxels
        ):
            reject("small_interface_component", point)
            continue

        solid_ratio, water_ratio, gas_ratio = compute_phase_ratios_in_sphere(
            filtered_volume, z, y, x, sphere_offsets, config
        )
        if solid_ratio < config.solid_ratio_min:
            diagnostic_points["low_solid_ratio_points_xyz"].append(point)
            reject("low_solid_ratio", point)
            continue
        if solid_ratio > config.solid_ratio_max:
            diagnostic_points["high_solid_ratio_points_xyz"].append(point)
            reject("high_solid_ratio", point)
            continue
        if water_ratio < config.water_ratio_min:
            diagnostic_points["low_water_ratio_points_xyz"].append(point)
            reject("low_water_ratio", point)
            continue
        if water_ratio > config.water_ratio_max:
            diagnostic_points["high_water_ratio_points_xyz"].append(point)
            reject("high_water_ratio", point)
            continue
        if gas_ratio < config.gas_ratio_min:
            diagnostic_points["low_gas_ratio_points_xyz"].append(point)
            reject("low_gas_ratio", point)
            continue
        if gas_ratio > config.gas_ratio_max:
            diagnostic_points["high_gas_ratio_points_xyz"].append(point)
            reject("high_gas_ratio", point)
            continue

        points_water_gas_zyx = get_patch_points_from_global_labels(
            labelled_water_gas, z, y, x, config.patch_window_radius, config.patch_max_points, rng
        )
        points_solid_water_zyx = get_patch_points_from_global_labels(
            labelled_solid_water, z, y, x, config.patch_window_radius, config.patch_max_points, rng
        )
        if (
            points_water_gas_zyx.shape[0] < config.min_patch_points
            or points_solid_water_zyx.shape[0] < config.min_patch_points
        ):
            reject("insufficient_interface_patch_points", point)
            continue

        normal_water_gas, residual_water_gas, ratio_water_gas = fit_plane(
            points_water_gas_zyx[:, (2, 1, 0)], config.min_patch_points
        )
        normal_solid_water, residual_solid_water, ratio_solid_water = fit_plane(
            points_solid_water_zyx[:, (2, 1, 0)], config.min_patch_points
        )
        if normal_water_gas is None or normal_solid_water is None:
            reject("plane_fit_failure", point)
            continue
        if (
            residual_water_gas > config.max_plane_rms
            or residual_solid_water > config.max_plane_rms
            or ratio_water_gas > config.max_singular_value_ratio
            or ratio_solid_water > config.max_singular_value_ratio
        ):
            reject("poor_plane_fit", point)
            continue

        oriented_water_gas = orient_normal(
            filtered_volume,
            point,
            normal_water_gas,
            positive_label=config.gas_label,
            negative_label=config.water_label,
            probe_steps=config.normal_probe_steps,
            fallback_positive_priority=(config.gas_label, config.solid_label, config.water_label)
            if config.allow_third_phase_orientation_fallback
            else (),
        )
        oriented_solid_water = orient_normal(
            filtered_volume,
            point,
            normal_solid_water,
            positive_label=config.water_label,
            negative_label=config.solid_label,
            probe_steps=config.normal_probe_steps,
            fallback_positive_priority=(config.water_label, config.gas_label, config.solid_label)
            if config.allow_third_phase_orientation_fallback
            else (),
        )
        angle = compute_contact_angle(oriented_water_gas, oriented_solid_water)
        if angle is None:
            reject("normal_orientation_failure", point)
            continue

        contact_angles.append(angle)
        solid_ratios.append(solid_ratio)
        water_ratios.append(water_ratio)
        gas_ratios.append(gas_ratio)
        good_points.append(point)
        vector_points.append(
            (
                x,
                y,
                z,
                oriented_water_gas[0],
                oriented_water_gas[1],
                oriented_water_gas[2],
                oriented_solid_water[0],
                oriented_solid_water[1],
                oriented_solid_water[2],
                angle,
                solid_ratio,
                water_ratio,
                gas_ratio,
            )
        )
        if angle < config.low_angle_threshold:
            diagnostic_points["low_angle_points_xyz"].append(point)
        if angle > config.high_angle_threshold:
            diagnostic_points["high_angle_points_xyz"].append(point)

    processing_time = time.time() - start_total
    statistics = _write_outputs(
        volume,
        image_path,
        paths,
        config,
        contact_angles,
        good_points,
        bad_points,
        vector_points,
        solid_ratios,
        water_ratios,
        gas_ratios,
        diagnostic_points,
        rejection_counts,
        total_points,
        processing_time,
    )
    statistics["Output_Directory"] = str(output_directory)
    print("Valid contact angles: {}".format(len(contact_angles)))
    print("Results saved to: {}".format(output_directory))
    print("Completed in {:.2f} s".format(processing_time))
    return statistics


def batch_process_images(
    image_directory: Path,
    output_directory: Path,
    config: ContactAngleConfig,
    seed: Optional[int] = 0,
    recursive: bool = False,
    overwrite: bool = False,
) -> pd.DataFrame:
    """Process all TIFF files in a directory and export a batch summary."""

    image_directory = Path(image_directory).resolve()
    output_directory = Path(output_directory).resolve()
    if not image_directory.is_dir():
        raise NotADirectoryError("Input directory not found: {}".format(image_directory))
    output_directory.mkdir(parents=True, exist_ok=True)
    summary_path = output_directory / "batch_processing_summary.csv"
    if summary_path.exists() and not overwrite:
        raise FileExistsError("Batch summary already exists: {}".format(summary_path))
    iterator = image_directory.rglob if recursive else image_directory.glob
    files = sorted(set(iterator("*.tif")) | set(iterator("*.tiff")))
    files = [path for path in files if output_directory not in path.resolve().parents]
    if not files:
        raise FileNotFoundError("No TIFF files found in {}".format(image_directory))

    summary_rows: List[Dict[str, Any]] = []
    for index, image_path in enumerate(files):
        relative_stem = image_path.relative_to(image_directory).with_suffix("")
        image_output = output_directory / relative_stem
        image_seed = None if seed is None else seed + index
        try:
            row = process_single_image(image_path, image_output, config, image_seed, overwrite)
        except Exception as error:
            row = {"Image": image_path.name, "Error": str(error)}
            print("Failed to process {}: {}".format(image_path, error))
        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(summary_path, index=False)
    print("Batch summary saved to: {}".format(summary_path))
    return summary


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract local 3D contact angles from a segmented three-phase TIFF volume."
    )
    parser.add_argument("input", type=Path, help="Input TIFF file or a directory of TIFF files.")
    parser.add_argument("--output-dir", type=Path, help="Output directory. A safe default is used if omitted.")
    parser.add_argument("--config", type=Path, help="JSON file overriding ContactAngleConfig defaults.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducible patch downsampling.")
    parser.add_argument("--recursive", action="store_true", help="Recursively search an input directory.")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing generated files.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_argument_parser().parse_args(argv)
    config = load_config(args.config)
    input_path = args.input.resolve()
    if input_path.is_dir():
        output_directory = args.output_dir or input_path / "contact_angle_results"
        batch_process_images(
            input_path, output_directory, config, args.seed, args.recursive, args.overwrite
        )
    else:
        process_single_image(input_path, args.output_dir, config, args.seed, args.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
