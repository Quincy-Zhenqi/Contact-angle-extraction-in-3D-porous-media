"""Contact-angle extraction from a three-phase labelled CT volume.

Install dependencies: pip install numpy scipy pandas tifffile
Labels: 1 = solid, 2 = liquid, 3 = gas; 4 may denote exterior background.
Input array order: (z, y, x). Output coordinates are original voxel indices.
G contains finite geometric angles; Q selects accepted rows without changing
their angles or the input labels. Set the two paths below and run this file.
"""
from __future__ import annotations
from pathlib import Path
from dataclasses import asdict, dataclass
from typing import Dict, Optional, Sequence, Tuple
import gc
import json
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy.ndimage import convolve, generate_binary_structure, label
from scipy.spatial import cKDTree
import tifffile

# Set these paths before running. The output folder must be new.
INPUT_PATH = Path("sample.tif")
OUTPUT_DIR = Path("contact_angle_results")


# Scientific settings. Counts and lengths are in voxels unless stated.
@dataclass
class Parameters:
    solid_label: int = 1
    liquid_label: int = 2
    gas_label: int = 3
    background_labels: Tuple[int, ...] = (4,)
    voxel_spacing_zyx: Tuple[float, float, float] = (9.14, 9.14, 9.14)
    voxel_spacing_unit: str = 'micrometre'
    phase_component_connectivity: int = 1
    interface_connectivity: int = 2
    contact_component_connectivity: int = 2
    min_liquid_component_voxels: int = 50
    min_gas_component_voxels: int = 50
    sphere_radius_voxels: int = 15
    boundary_margin_voxels: int = 15
    count_cube_half_width_voxels: int = 5
    min_phase_voxels_in_count_cube: int = 20
    max_candidate_face_neighbours: int = 3
    dense_candidate_min_face_neighbours: int = 4
    dense_candidate_core_min_face_neighbours: int = 4
    dense_candidate_guard_voxels: int = 2
    min_interface_component_voxels: int = 15
    solid_fraction_min: float = 0.3
    solid_fraction_max: float = 0.7
    liquid_fraction_min: float = 0.0
    liquid_fraction_max: Optional[float] = None
    gas_fraction_min: float = 0.0
    gas_fraction_max: Optional[float] = None
    plane_window_half_width_voxels: int = 4
    min_plane_points: int = 15
    max_plane_points: Optional[int] = None
    max_plane_rms_reference_voxels: float = 1.0
    max_singular_value_ratio: float = 0.3
    interface_representation: str = 'liquid_voxel_centers'
    orientation_method: str = 'centroid'
    min_orientation_alignment: float = 0.05
    low_tail_degrees: float = 10.0
    high_tail_degrees: float = 170.0

    def __post_init__(self) -> None:
        labels = self.scientific_labels + tuple(self.background_labels)
        if any((not isinstance(value, int) for value in labels)):
            raise ValueError('phase and background labels must be integers')
        if len(set(labels)) != len(labels):
            raise ValueError('phase and background labels must be distinct')
        if len(self.voxel_spacing_zyx) != 3 or any((v <= 0 for v in self.voxel_spacing_zyx)):
            raise ValueError('voxel_spacing_zyx must contain three positive values')
        if not isinstance(self.voxel_spacing_unit, str) or not self.voxel_spacing_unit.strip(
            ):
            raise ValueError('voxel_spacing_unit must be a nonempty string')
        if self.interface_representation not in {'liquid_voxel_centers'}:
            raise ValueError('interface_representation must be liquid_voxel_centers')
        if self.orientation_method not in {'centroid'}:
            raise ValueError('orientation_method must be centroid')
        for value, name in ((self.phase_component_connectivity,
            'phase_component_connectivity'), (self.interface_connectivity,
            'interface_connectivity'), (self.contact_component_connectivity,
            'contact_component_connectivity')):
            if value not in {1, 2, 3}:
                raise ValueError(f'{name} must be 1, 2, or 3')
        nonnegative_integer_fields = ((self.min_liquid_component_voxels,
            'min_liquid_component_voxels'), (self.min_gas_component_voxels,
            'min_gas_component_voxels'), (self.sphere_radius_voxels, 'sphere_radius_voxels'),
            (self.boundary_margin_voxels, 'boundary_margin_voxels'), (
            self.count_cube_half_width_voxels, 'count_cube_half_width_voxels'), (
            self.min_phase_voxels_in_count_cube, 'min_phase_voxels_in_count_cube'), (
            self.max_candidate_face_neighbours, 'max_candidate_face_neighbours'), (
            self.dense_candidate_min_face_neighbours, 'dense_candidate_min_face_neighbours'),
            (self.dense_candidate_core_min_face_neighbours,
            'dense_candidate_core_min_face_neighbours'), (self.dense_candidate_guard_voxels,
            'dense_candidate_guard_voxels'), (self.min_interface_component_voxels,
            'min_interface_component_voxels'), (self.plane_window_half_width_voxels,
            'plane_window_half_width_voxels'), (self.min_plane_points, 'min_plane_points'))
        for value, name in nonnegative_integer_fields:
            if not isinstance(value, int) or value < 0:
                raise ValueError(f'{name} must be a nonnegative integer')
        if self.sphere_radius_voxels < 1:
            raise ValueError('sphere_radius_voxels must be at least 1')
        if self.min_plane_points < 3:
            raise ValueError('min_plane_points must be at least 3')
        if self.max_candidate_face_neighbours > 6:
            raise ValueError('max_candidate_face_neighbours cannot exceed 6')
        if not 1 <= self.dense_candidate_min_face_neighbours <= 6:
            raise ValueError('dense_candidate_min_face_neighbours must be between 1 and 6')
        if not 1 <= self.dense_candidate_core_min_face_neighbours <= 6:
            raise ValueError(
                'dense_candidate_core_min_face_neighbours must be between 1 and 6')
        if self.max_plane_points is not None:
            if not isinstance(self.max_plane_points, int):
                raise ValueError('max_plane_points must be an integer or null')
            if self.max_plane_points < self.min_plane_points:
                raise ValueError('max_plane_points cannot be below min_plane_points')
        if not 0 <= self.min_orientation_alignment <= 1:
            raise ValueError('min_orientation_alignment must be in [0, 1]')
        if self.max_plane_rms_reference_voxels < 0:
            raise ValueError('max_plane_rms_reference_voxels must be nonnegative')
        if self.max_singular_value_ratio < 0:
            raise ValueError('max_singular_value_ratio must be nonnegative')
        required_margin = max(self.sphere_radius_voxels, self.count_cube_half_width_voxels,
            self.plane_window_half_width_voxels)
        if self.boundary_margin_voxels < required_margin:
            raise ValueError(
                'boundary_margin_voxels must cover the sphere, count cube, and plane window')
        if not 0 <= self.low_tail_degrees < self.high_tail_degrees <= 180:
            raise ValueError('tail thresholds must satisfy 0 <= low < high <= 180')
        for low, high, name in ((self.solid_fraction_min, self.solid_fraction_max, 'solid'), (
            self.liquid_fraction_min, self.liquid_fraction_max, 'liquid'), (
            self.gas_fraction_min, self.gas_fraction_max, 'gas')):
            if not 0 <= low <= 1:
                raise ValueError(f'{name} fraction minimum must be in [0, 1]')
            if high is not None and (not low <= high <= 1):
                raise ValueError(f'{name} fraction maximum must be in [minimum, 1]')
        if max(self.voxel_spacing_zyx) - min(self.voxel_spacing_zyx) > 1e-12:
            raise ValueError('This implementation requires isotropic voxel spacing')
        if self.plane_window_half_width_voxels < 1:
            raise ValueError('plane half width must be positive')

    @property
    def scientific_labels(self) -> Tuple[int, int, int]:
        return (self.solid_label, self.liquid_label, self.gas_label)

    @property
    def allowed_labels(self) -> Tuple[int, ...]:
        return self.scientific_labels + tuple(self.background_labels)

    @property
    def normalized_spacing_zyx(self) -> Tuple[float, float, float]:
        reference = min(self.voxel_spacing_zyx)
        return tuple((v / reference for v in self.voxel_spacing_zyx))


FACE_OFFSETS_ZYX = np.asarray([(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0,
    -1)], dtype=np.int8)


# Read labels only: no relabelling, smoothing or segmentation is applied.
def validate_volume(volume: np.ndarray, config: Parameters) -> Dict[int, int]:
    if volume.ndim != 3:
        raise ValueError(f'Expected a 3-D volume, received shape {volume.shape}')
    if any((length == 0 for length in volume.shape)):
        raise ValueError(f'Expected nonempty dimensions, received shape {volume.shape}')
    if not np.issubdtype(volume.dtype, np.integer):
        raise TypeError(f'Expected integer phase labels, received {volume.dtype}')
    counts_by_label: Dict[int, int] = {}
    target_chunk_voxels = 16 * 1024 * 1024
    slice_voxels = int(volume.shape[1] * volume.shape[2])
    chunk_depth = max(1, min(volume.shape[0], target_chunk_voxels // slice_voxels))
    for start in range(0, volume.shape[0], chunk_depth):
        labels, counts = np.unique(volume[start:start + chunk_depth], return_counts=True)
        for value, count in zip(labels, counts):
            label_value = int(value)
            counts_by_label[label_value] = counts_by_label.get(label_value, 0) + int(count)
    unexpected = sorted(set(counts_by_label) - set(config.allowed_labels))
    if unexpected:
        raise ValueError(f'Unexpected phase labels {unexpected}; allowed labels are {config.allowed_labels}')
    missing = sorted(set(config.scientific_labels) - set(counts_by_label))
    if missing:
        raise ValueError(f'Missing required scientific phase labels {missing}')
    return dict(sorted(counts_by_label.items()))


def face_neighbor_mask(mask: np.ndarray) -> np.ndarray:
    result = np.zeros_like(mask, dtype=bool)
    result[1:, :, :] |= mask[:-1, :, :]
    result[:-1, :, :] |= mask[1:, :, :]
    result[:, 1:, :] |= mask[:, :-1, :]
    result[:, :-1, :] |= mask[:, 1:, :]
    result[:, :, 1:] |= mask[:, :, :-1]
    result[:, :, :-1] |= mask[:, :, 1:]
    return result


# Both interfaces are represented by liquid voxel centres.
def liquid_side_interface(volume: np.ndarray, liquid_label: int,
    other_label: int) -> np.ndarray:
    liquid = volume == liquid_label
    other = volume == other_label
    return liquid & face_neighbor_mask(other)


# Connectivity 1 is 6-neighbour; connectivity 2 is 18-neighbour.
def component_labels_and_sizes(mask: np.ndarray, connectivity: int):
    structure = generate_binary_structure(3, connectivity)
    labels, count = label(mask, structure=structure)
    sizes = np.bincount(labels.ravel())
    if sizes.size:
        sizes[0] = 0
    return (labels, sizes, int(count))


def values_at_candidates(array: np.ndarray, candidates_zyx: np.ndarray) -> np.ndarray:
    return array[candidates_zyx[:, 0], candidates_zyx[:, 1], candidates_zyx[:, 2]]


# Use the largest face-adjacent gas component for the size check.
def largest_adjacent_component(component_labels: np.ndarray, component_sizes: np.ndarray,
    candidates_zyx: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n_candidates = len(candidates_zyx)
    neighbour_ids = np.zeros((n_candidates, len(FACE_OFFSETS_ZYX)), dtype=np.int32)
    shape = np.asarray(component_labels.shape)
    for column, offset in enumerate(FACE_OFFSETS_ZYX):
        points = candidates_zyx + offset
        in_bounds = np.all((points >= 0) & (points < shape), axis=1)
        valid_points = points[in_bounds]
        neighbour_ids[in_bounds, column] = component_labels[valid_points[:, 0], valid_points[
            :, 1], valid_points[:, 2]]
    neighbour_sizes = component_sizes[neighbour_ids]
    best_column = np.argmax(neighbour_sizes, axis=1)
    rows = np.arange(n_candidates)
    best_ids = neighbour_ids[rows, best_column]
    best_sizes = neighbour_sizes[rows, best_column]
    return (best_ids.astype(np.int32), best_sizes.astype(np.int64))


# Count other candidate voxels in the six face neighbours.
def candidate_face_neighbour_counts(candidate_mask: np.ndarray) -> np.ndarray:
    kernel = np.zeros((3, 3, 3), dtype=np.uint8)
    kernel[1, 1, 0] = kernel[1, 1, 2] = 1
    kernel[1, 0, 1] = kernel[1, 2, 1] = 1
    kernel[0, 1, 1] = kernel[2, 1, 1] = 1
    return convolve(candidate_mask.astype(np.uint8), kernel, mode='constant', cval=0)


# The centre is excluded from the six-neighbour topology check.
def candidate_neighbour_phase_presence(volume: np.ndarray, candidates_zyx: np.ndarray,
    phase_labels: Sequence[int]) -> Dict[int, np.ndarray]:
    shape = np.asarray(volume.shape)
    result = {int(phase): np.zeros(len(candidates_zyx), dtype=bool) for phase in phase_labels}
    for offset in FACE_OFFSETS_ZYX:
        points = candidates_zyx + offset
        in_bounds = np.all((points >= 0) & (points < shape), axis=1)
        valid_points = points[in_bounds]
        values = volume[valid_points[:, 0], valid_points[:, 1], valid_points[:, 2]]
        valid_rows = np.flatnonzero(in_bounds)
        for phase in phase_labels:
            result[int(phase)][valid_rows] |= values == phase
    return result


def array_boundary_distance(candidates_zyx: np.ndarray, shape: Sequence[int]) -> np.ndarray:
    shape_minus_one = np.asarray(shape, dtype=np.int64) - 1
    return np.minimum(candidates_zyx, shape_minus_one - candidates_zyx).min(axis=1)


# Radius-15 sphere: composition support, not plane-fitting support.
def sphere_composition(volume: np.ndarray, center_zyx: Sequence[int],
    sphere_offsets: np.ndarray, config: Parameters) -> Dict[str, float]:
    center = np.asarray(center_zyx, dtype=np.int64)
    points = center + sphere_offsets
    shape = np.asarray(volume.shape)
    in_bounds = np.all((points >= 0) & (points < shape), axis=1)
    if not np.all(in_bounds):
        return {'sphere_complete': False, 'sphere_invalid_fraction': 1.0 - float(np.mean(
            in_bounds)), 'solid_fraction': np.nan, 'liquid_fraction': np.nan,
            'gas_fraction': np.nan}
    values = volume[points[:, 0], points[:, 1], points[:, 2]]
    scientific = np.isin(values, config.scientific_labels)
    invalid_fraction = 1.0 - float(np.mean(scientific))
    if invalid_fraction > 0:
        return {'sphere_complete': False, 'sphere_invalid_fraction': invalid_fraction,
            'solid_fraction': np.nan, 'liquid_fraction': np.nan, 'gas_fraction': np.nan}
    total = float(len(values))
    return {'sphere_complete': bool(np.all(scientific)),
        'sphere_invalid_fraction': invalid_fraction, 'solid_fraction': float(np.count_nonzero(
        values == config.solid_label) / total), 'liquid_fraction': float(np.count_nonzero(
        values == config.liquid_label) / total), 'gas_fraction': float(np.count_nonzero(
        values == config.gas_label) / total)}


# Count each phase separately in the complete 11 x 11 x 11 cube.
def local_phase_counts(volume: np.ndarray, center_zyx: Sequence[int], half_width: int,
    config: Parameters) -> Dict[str, int]:
    z, y, x = map(int, center_zyx)
    z0, z1 = (z - half_width, z + half_width + 1)
    y0, y1 = (y - half_width, y + half_width + 1)
    x0, x1 = (x - half_width, x + half_width + 1)
    if z0 < 0 or y0 < 0 or x0 < 0 or (z1 > volume.shape[0]) or (y1 > volume.shape[1]) or (
        x1 > volume.shape[2]):
        return {'solid_count': -1, 'liquid_count': -1, 'gas_count': -1}
    patch = volume[z0:z1, y0:y1, x0:x1]
    return {'solid_count': int(np.count_nonzero(patch == config.solid_label)),
        'liquid_count': int(np.count_nonzero(patch == config.liquid_label)), 'gas_count': int(
        np.count_nonzero(patch == config.gas_label))}


# Array order is z,y,x; plane coordinates are x,y,z in reference voxels.
def _zyx_to_scaled_xyz(points_zyx: np.ndarray, config: Parameters) -> np.ndarray:
    spacing_z, spacing_y, spacing_x = config.normalized_spacing_zyx
    return np.column_stack((points_zyx[:, 2] * spacing_x, points_zyx[:, 1] * spacing_y,
        points_zyx[:, 0] * spacing_z))


# max_plane_points=None retains all local fitting points without subsampling.
def _deterministic_limit(points: np.ndarray, max_points: Optional[int]) -> np.ndarray:
    if max_points is None or len(points) <= max_points:
        return points
    indices = np.linspace(0, len(points) - 1, max_points, dtype=np.int64)
    return points[indices]


# Within the 9-cubed window, select the component containing the candidate.
def local_connected_interface_points(interface_mask: np.ndarray, center_zyx: Sequence[int],
    half_width: int, connectivity: int, max_points: Optional[int],
    config: Parameters) -> np.ndarray:
    z, y, x = map(int, center_zyx)
    z0, z1 = (max(0, z - half_width), min(interface_mask.shape[0], z + half_width + 1))
    y0, y1 = (max(0, y - half_width), min(interface_mask.shape[1], y + half_width + 1))
    x0, x1 = (max(0, x - half_width), min(interface_mask.shape[2], x + half_width + 1))
    patch = interface_mask[z0:z1, y0:y1, x0:x1]
    labels, _ = label(patch, structure=generate_binary_structure(3, connectivity))
    local_center = (z - z0, y - y0, x - x0)
    component_id = labels[local_center]
    if component_id == 0:
        return np.empty((0, 3), dtype=float)
    points = np.argwhere(labels == component_id).astype(float)
    points[:, 0] += z0
    points[:, 1] += y0
    points[:, 2] += x0
    return _deterministic_limit(_zyx_to_scaled_xyz(points, config), max_points)


# Centred SVD: smallest right singular vector is the fitted unit normal.
# Three non-collinear points suffice for G; the 15-point condition is for Q.
def fit_plane(points_xyz: np.ndarray):
    if len(points_xyz) < 3:
        return (None, np.nan, np.nan)
    centered = points_xyz - points_xyz.mean(axis=0)
    try:
        _, singular_values, vh = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
        return (None, np.nan, np.nan)
    if len(singular_values) < 3 or singular_values[1] <= 1e-12:
        return (None, np.nan, np.nan)
    normal = vh[2]
    rms = float(np.linalg.norm(centered @ normal) / np.sqrt(len(points_xyz)))
    singular_ratio = float(singular_values[2] / singular_values[1])
    return (normal, rms, singular_ratio)


# nLG points liquid to gas; nSL points solid to liquid. No 180-theta step.
def contact_angle_degrees(liquid_gas_normal: np.ndarray, solid_normal: np.ndarray) -> float:
    if liquid_gas_normal is None or solid_normal is None:
        return np.nan
    first = liquid_gas_normal / np.linalg.norm(liquid_gas_normal)
    second = solid_normal / np.linalg.norm(solid_normal)
    cosine = float(np.clip(np.dot(first, second), -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def _load_volume(path: Path) -> np.ndarray:
    try:
        return tifffile.memmap(str(path), mode='r')
    except Exception:
        volume = tifffile.imread(str(path))
        volume.setflags(write=False)
        return volume


# Component sizes are quality features; original phase labels stay unchanged.
def _phase_component_features(volume: np.ndarray, candidates_zyx: np.ndarray,
    config: Parameters):
    features: Dict[str, np.ndarray] = {}
    liquid_labels, liquid_sizes, liquid_count = component_labels_and_sizes(
        volume == config.liquid_label, config.phase_component_connectivity)
    liquid_ids = values_at_candidates(liquid_labels, candidates_zyx).astype(np.int32)
    features['liquid_component_id'] = liquid_ids
    features['liquid_component_size'] = liquid_sizes[liquid_ids].astype(np.int64)
    del liquid_labels, liquid_sizes
    gc.collect()
    gas_labels, gas_sizes, gas_count = component_labels_and_sizes(volume == config.gas_label,
        config.phase_component_connectivity)
    gas_ids, candidate_gas_sizes = largest_adjacent_component(gas_labels, gas_sizes,
        candidates_zyx)
    features['gas_component_id'] = gas_ids
    features['gas_component_size'] = candidate_gas_sizes
    del gas_labels, gas_sizes
    gc.collect()
    counts = {'liquid_component_count': liquid_count, 'gas_component_count': gas_count}
    return (features, counts)


def _component_size_at_candidates(mask: np.ndarray, candidates_zyx: np.ndarray,
    connectivity: int) -> Tuple[np.ndarray, np.ndarray, int]:
    labels, sizes, component_count = component_labels_and_sizes(mask, connectivity)
    component_ids = values_at_candidates(labels, candidates_zyx).astype(np.int32)
    component_sizes = sizes[component_ids].astype(np.int64)
    del labels, sizes
    gc.collect()
    return (component_ids, component_sizes, component_count)


def _candidate_base_features(volume: np.ndarray, liquid_gas_interface: np.ndarray,
    solid_liquid_interface: np.ndarray, candidate_mask: np.ndarray,
    candidates_zyx: np.ndarray, config: Parameters):
    base: Dict[str, np.ndarray] = {}
    base['candidate_id'] = np.arange(len(candidates_zyx), dtype=np.int64)
    base['z'] = candidates_zyx[:, 0].astype(np.int32)
    base['y'] = candidates_zyx[:, 1].astype(np.int32)
    base['x'] = candidates_zyx[:, 2].astype(np.int32)
    base['array_boundary_distance'] = array_boundary_distance(candidates_zyx,
        volume.shape).astype(np.int32)
    neighbour_count_map = candidate_face_neighbour_counts(candidate_mask)
    base['candidate_face_neighbour_count'] = values_at_candidates(neighbour_count_map,
        candidates_zyx).astype(np.uint8)
    del neighbour_count_map
    presence = candidate_neighbour_phase_presence(volume, candidates_zyx,
        config.scientific_labels)
    base['six_neighbour_has_solid'] = presence[config.solid_label]
    base['six_neighbour_has_liquid'] = presence[config.liquid_label]
    base['six_neighbour_has_gas'] = presence[config.gas_label]
    base['six_neighbour_has_all_three'] = presence[config.solid_label] & presence[
        config.liquid_label] & presence[config.gas_label]
    phase_features, phase_counts = _phase_component_features(volume, candidates_zyx, config)
    base.update(phase_features)
    contact_ids, contact_sizes, contact_count = _component_size_at_candidates(candidate_mask,
        candidates_zyx, config.contact_component_connectivity)
    base['contact_component_id'] = contact_ids
    base['contact_component_size'] = contact_sizes
    _, wg_sizes, wg_count = _component_size_at_candidates(liquid_gas_interface,
        candidates_zyx, config.interface_connectivity)
    base['wg_interface_component_size'] = wg_sizes
    _, sw_sizes, sw_count = _component_size_at_candidates(solid_liquid_interface,
        candidates_zyx, config.interface_connectivity)
    base['sw_interface_component_size'] = sw_sizes
    counts = dict(phase_counts)
    counts.update({'contact_component_count': contact_count,
        'wg_interface_component_count': wg_count, 'sw_interface_component_count': sw_count})
    return (base, counts)


def _blank_geometry_features() -> Dict[str, object]:
    return {'solid_count_11cube': -1, 'liquid_count_11cube': -1, 'gas_count_11cube': -1,
        'sphere_complete': False, 'sphere_invalid_fraction': np.nan, 'solid_fraction': np.nan,
        'liquid_fraction': np.nan, 'gas_fraction': np.nan, 'wg_plane_point_count': 0,
        'sw_plane_point_count': 0, 'wg_plane_rms': np.nan, 'sw_plane_rms': np.nan,
        'wg_singular_value_ratio': np.nan, 'sw_singular_value_ratio': np.nan,
        'wg_orientation_alignment': np.nan, 'sw_orientation_alignment': np.nan,
        'wg_orientation_status': 'not_evaluated', 'sw_orientation_status': 'not_evaluated',
        'n_lg_x': np.nan, 'n_lg_y': np.nan, 'n_lg_z': np.nan, 'n_solid_x': np.nan,
        'n_solid_y': np.nan, 'n_solid_z': np.nan, 'contact_angle_degrees': np.nan}


# Select the two liquid-side interface patches independently.
def _local_interface_points(volume: np.ndarray, liquid_gas_interface: np.ndarray,
    solid_liquid_interface: np.ndarray, center_zyx: Sequence[int], config: Parameters):
    wg_points = local_connected_interface_points(liquid_gas_interface, center_zyx,
        config.plane_window_half_width_voxels, config.interface_connectivity,
        config.max_plane_points, config)
    sw_points = local_connected_interface_points(solid_liquid_interface, center_zyx,
        config.plane_window_half_width_voxels, config.interface_connectivity,
        config.max_plane_points, config)
    return (wg_points, sw_points)


# Empirical statistics only. Standard deviation uses ddof=0.
def _angle_summary(values: np.ndarray, config: Parameters) -> Dict[str, object]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return {'n_contact_voxels': 0, 'mean_degrees': None, 'median_degrees': None,
            'population_std_degrees': None, 'q1_degrees': None, 'q3_degrees': None,
            'iqr_degrees': None, 'p05_degrees': None, 'p95_degrees': None,
            'low_tail_fraction': None, 'high_tail_fraction': None}
    q1, median, q3 = np.percentile(values, [25, 50, 75])
    return {'n_contact_voxels': int(len(values)), 'mean_degrees': float(np.mean(values)),
        'median_degrees': float(median), 'population_std_degrees': float(np.std(values,
        ddof=0)), 'q1_degrees': float(q1), 'q3_degrees': float(q3), 'iqr_degrees': float(
        q3 - q1), 'p05_degrees': float(np.percentile(values, 5)), 'p95_degrees': float(
        np.percentile(values, 95)), 'low_tail_fraction': float(np.mean(
        values < config.low_tail_degrees)), 'high_tail_fraction': float(np.mean(
        values > config.high_tail_degrees))}


def validate_parameters(cfg):
    values = asdict(cfg)
    defaults = asdict(Parameters())
    for key, value in values.items():
        example = defaults[key]
        if value is None:
            if key not in {'max_plane_points', 'liquid_fraction_max', 'gas_fraction_max'}:
                raise ValueError('null not allowed: ' + key)
            continue
        if isinstance(example, bool):
            if type(value) is not bool:
                raise ValueError('boolean required: ' + key)
        elif isinstance(example, int) or key == 'max_plane_points':
            if type(value) is not int:
                raise ValueError('integer required: ' + key)
        elif isinstance(example, float):
            if type(value) not in (float, int) or not np.isfinite(value):
                raise ValueError('finite numeric required: ' + key)
        elif isinstance(example, str):
            if not isinstance(value, str):
                raise ValueError('string required: ' + key)
    for key in ('voxel_spacing_zyx', 'background_labels'):
        if not isinstance(values[key], (list, tuple)):
            raise ValueError('sequence required: ' + key)
        for v in values[key]:
            if type(v) not in (int, float) or not np.isfinite(v):
                raise ValueError('invalid ' + key)
            if key != 'voxel_spacing_zyx' and type(v) is not int:
                raise ValueError('integer ' + key)
    return cfg


# Use int64 offsets for the voxel-space sphere.
def sphere_offsets(radius):
    values = np.arange(-radius, radius + 1, dtype=np.int64)
    z, y, x = np.meshgrid(values, values, values, indexing='ij')
    inside = z * z + y * y + x * x <= radius * radius
    return np.column_stack((z[inside], y[inside], x[inside]))


# Orient normals using local connected phase centroids in the fitting cube.
# If the centre is outside a phase, use its largest face-adjacent component.
def phase_support(volume, center, cfg, phase, global_gas=None):
    w = cfg.plane_window_half_width_voxels
    origin = np.asarray(center) - w
    slices = tuple((slice(int(c - w), int(c + w + 1)) for c in center))
    patch = volume[slices]
    labels, _ = ndi.label(patch == phase, structure=ndi.generate_binary_structure(3,
        cfg.phase_component_connectivity))
    local = np.array([w, w, w])
    chosen = int(labels[tuple(local)])
    adjacent = sorted(set((int(labels[tuple(local + o)]) for o in FACE_OFFSETS_ZYX)) - {0})
    sizes = np.bincount(labels.ravel())
    if not chosen:
        if not adjacent:
            return (None, {'local_id': 0, 'size': 0, 'global_id': 0,
                'adjacent_ids': adjacent})
        chosen = min(adjacent, key=lambda v: (-sizes[v], v))
    points = np.argwhere(labels == chosen) + origin
    centroid = _zyx_to_scaled_xyz(points.astype(float), cfg).mean(axis=0)
    global_id = int(global_gas[tuple(points[0])]) if global_gas is not None else 0
    return (centroid, {'local_id': chosen, 'size': int(sizes[chosen]), 'global_id': global_id,
        'adjacent_ids': adjacent})


# G requires a defined nonzero alignment; the 0.05 cutoff belongs to Q.
def orient_axis(normal, source, target):
    if normal is None or source is None or target is None:
        return (None, np.nan, 'missing_support')
    direction = target - source
    norm = np.linalg.norm(direction)
    if norm <= 1e-12:
        return (None, np.nan, 'centroid_degenerate')
    alignment = float(np.dot(normal, direction / norm))
    if alignment == 0:
        return (None, 0.0, 'zero_alignment')
    return (normal if alignment > 0 else -normal, abs(alignment), 'oriented_nonzero')


def geometry_schema():
    r = _blank_geometry_features()
    r.update(geometry_support_complete=False, geometry_status='not_evaluated',
        geometry_support_reason='not_evaluated', geometry_min_array_margin=0,
        orientation_min_alignment=np.nan, orientation_confidence_status='not_evaluated',
        count_support_complete=False, composition_array_complete=False,
        sphere_background_fraction=np.nan, sphere_observed_fraction=np.nan,
        composition_status='not_evaluated', count_status='not_evaluated',
        gas_centroid_global_id=0, gas_centroid_local_id=0, gas_centroid_local_size=0,
        liquid_centroid_local_id=0, liquid_centroid_local_size=0, solid_centroid_local_id=0,
        solid_centroid_local_size=0)
    for prefix in ('wg', 'sw'):
        for axis in 'xyz':
            r[prefix + '_axis_' + axis] = np.nan
        for k in ('s1', 's2', 's3', 'relative_smallest_gap'):
            r[prefix + '_' + k] = np.nan
        r[prefix + '_svd_status'] = 'not_evaluated'
    for phase in ('liquid', 'gas', 'solid'):
        for axis in 'xyz':
            r[phase + '_centroid_' + axis] = np.nan
    return r


# Compute geometry independently of composition and quality acceptance.
# A half-width-5 support envelope includes face neighbours beyond the fit cube.
def geometry_layer(volume, lg, sl, center, cfg, global_gas=None):
    r = geometry_schema()
    center = np.asarray(center, dtype=int)
    w = cfg.plane_window_half_width_voxels
    margin = w + 1
    r['geometry_min_array_margin'] = margin
    if np.minimum(center, np.asarray(volume.shape) - 1 - center).min() < margin:
        r.update(geometry_status='support_unavailable',
            geometry_support_reason='array_adjacency_support_truncated')
        return r
    support = volume[tuple((slice(int(c - margin), int(c + margin + 1)) for c in center))]
    if not np.isin(support, cfg.scientific_labels).all():
        r.update(geometry_status='support_unavailable',
            geometry_support_reason='background_in_geometry_support_envelope')
        return r
    r.update(geometry_support_complete=True, geometry_support_reason='complete')
    points = _local_interface_points(volume, lg, sl, center, cfg)
    normals = []
    for prefix, cloud in zip(('wg', 'sw'), points):
        r[prefix + '_plane_point_count'] = len(cloud)
        normal, rms, ratio = fit_plane(cloud)
        r[prefix + '_plane_rms'] = rms
        r[prefix + '_singular_value_ratio'] = ratio
        r[prefix + '_svd_status'] = 'defined' if normal is not None else 'insufficient_rank_or_svd_failure'
        if len(cloud) >= 3:
            try:
                sv = np.linalg.svd(cloud - cloud.mean(axis=0), full_matrices=False,
                    compute_uv=False)
                for key, val in zip(('s1', 's2', 's3'), sv):
                    r[prefix + '_' + key] = float(val)
                if sv[1] > 0:
                    r[prefix + '_relative_smallest_gap'] = float((sv[1] - sv[2]) / sv[1])
            except np.linalg.LinAlgError:
                pass
        if normal is not None:
            for axis, val in zip('xyz', normal):
                r[prefix + '_axis_' + axis] = float(val)
        normals.append(normal)
    centroids = {}
    for name, phase in [('liquid', cfg.liquid_label), ('gas', cfg.gas_label), ('solid',
        cfg.solid_label)]:
        c, info = phase_support(volume, center, cfg, phase,
            global_gas if name == 'gas' else None)
        centroids[name] = c
        r[name + '_centroid_local_id'] = info['local_id']
        r[name + '_centroid_local_size'] = info['size']
        if name == 'gas':
            r['gas_centroid_global_id'] = info['global_id']
        if c is not None:
            for axis, val in zip('xyz', c):
                r[name + '_centroid_' + axis] = float(val)
    oriented = []
    alignments = []
    for prefix, normal, source, target, out in [('wg', normals[0], 'liquid', 'gas', 'n_lg'), (
        'sw', normals[1], 'solid', 'liquid', 'n_solid')]:
        n, align, status = orient_axis(normal, centroids[source], centroids[target])
        r[prefix + '_orientation_alignment'] = align
        r[prefix + '_orientation_status'] = status
        if n is not None:
            for axis, val in zip('xyz', n):
                r[out + '_' + axis] = float(val)
        oriented.append(n)
        alignments.append(align)
    r['contact_angle_degrees'] = contact_angle_degrees(*oriented)
    available = bool(np.isfinite(r['contact_angle_degrees']))
    r['geometry_status'] = 'computed' if available else 'normal_or_orientation_unavailable'
    if available:
        r['orientation_min_alignment'] = float(min(alignments))
        r['orientation_confidence_status'] = 'nonzero_alignment_threshold_pending'
    return r


# Compute phase counts and sphere fractions without changing the angle.
def morphology_layer(volume, center, offsets, cfg):
    center = np.asarray(center)
    points = center + offsets
    shape = np.asarray(volume.shape)
    valid = np.all((points >= 0) & (points < shape), axis=1)
    observed = points[valid]
    values = volume[tuple(observed.T)]
    bg = ~np.isin(values, cfg.scientific_labels)
    r = sphere_composition(volume, center, offsets, cfg)
    r.update(composition_array_complete=bool(valid.all()), sphere_observed_fraction=float(
        valid.mean()), sphere_background_fraction=float(bg.sum() / len(offsets)),
        composition_status='computed' if valid.all() else 'truncated_partial_observation')
    for name, phase in [('solid', cfg.solid_label), ('liquid', cfg.liquid_label), ('gas',
        cfg.gas_label)]:
        r['sphere_observed_' + name + '_fraction'] = float(np.count_nonzero(
            values == phase) / len(offsets))
    counts = local_phase_counts(volume, center, cfg.count_cube_half_width_voxels, cfg)
    complete = counts['solid_count'] >= 0
    r.update(count_support_complete=complete,
        count_status='computed' if complete else 'array_truncated')
    for name in ('solid', 'liquid', 'gas'):
        r[name + '_count_11cube'] = counts[name + '_count'] if complete else np.nan
    return r


# Calculate every raw candidate before applying the acceptance criteria.
def extract(volume, cfg, progress_every=5000):
    validate_parameters(cfg)
    validate_volume(volume, cfg)
    lg = liquid_side_interface(volume, cfg.liquid_label, cfg.gas_label)
    sl = liquid_side_interface(volume, cfg.liquid_label, cfg.solid_label)
    mask = lg & sl
    coords = np.argwhere(mask)
    base, counts = _candidate_base_features(volume, lg, sl, mask, coords, cfg)
    neighbour = candidate_face_neighbour_counts(mask)
    dense = mask & (neighbour >= cfg.dense_candidate_min_face_neighbours)
    core = dense & (candidate_face_neighbour_counts(
        dense) >= cfg.dense_candidate_core_min_face_neighbours)
    core_coords = np.argwhere(core)
    distance = cKDTree(core_coords).query(coords, p=np.inf)[0] if len(core_coords) and len(
        coords) else np.full(len(coords), np.inf)
    base['dense_core_chebyshev_distance'] = distance
    del mask, neighbour, dense, core
    gc.collect()
    gas_labels, _, _ = component_labels_and_sizes(volume == cfg.gas_label,
        cfg.phase_component_connectivity)
    offsets = sphere_offsets(cfg.sphere_radius_voxels)
    rows = []
    for i, center in enumerate(coords):
        r = geometry_layer(volume, lg, sl, center, cfg, gas_labels)
        r.update(morphology_layer(volume, center, offsets, cfg))
        rows.append(r)
        if progress_every and (i + 1) % progress_every == 0:
            print('Candidates processed {}/{}'.format(i + 1, len(coords)), flush=True)
    del gas_labels, lg, sl
    gc.collect()
    empty = geometry_schema()
    empty.update(morphology_layer(volume, np.asarray(volume.shape) // 2, offsets, cfg))
    local_frame = pd.DataFrame(rows, columns=list(empty))
    for key, example in empty.items():
        if isinstance(example, (float, int, np.floating, np.integer)) and (not isinstance(
            example, (bool, np.bool_))):
            local_frame[key] = local_frame[key].astype(float)
        elif isinstance(example, (bool, np.bool_)):
            local_frame[key] = local_frame[key].astype(bool)
    frame = pd.concat([pd.DataFrame(base), local_frame], axis=1)
    frame['gas_global_local_component_mismatch'] = (frame.gas_centroid_global_id > 0) & (
        frame.gas_centroid_global_id != frame.gas_component_id)
    frame['geometry_available'] = np.isfinite(frame.contact_angle_degrees.to_numpy(float))
    return frame


# The 23 states are evaluated independently, not as early rejection steps.
# Unavailable features are not_evaluated; Q requires every state to be pass.
# wg denotes LG; sw denotes SL. Individual rejection reasons may overlap.
def policy(frame, cfg):
    n = len(frame)
    states = {}

    def add(name, test, known=None):
        test = np.asarray(test, dtype=bool)
        if test.ndim == 0:
            test = np.full(n, bool(test))
        known = np.ones(n, bool) if known is None else np.asarray(known, dtype=bool)
        states[name] = np.where(~known, 'not_evaluated', np.where(test, 'pass', 'fail'))
    add('numerical_geometry', frame.geometry_available)
    add('topology', frame.six_neighbour_has_all_three)
    add('legacy_margin', frame.array_boundary_distance >= cfg.boundary_margin_voxels)
    add('sphere', frame.sphere_complete)
    add('component_size', (frame.liquid_component_size >= cfg.min_liquid_component_voxels) & (
        frame.gas_component_size >= cfg.min_gas_component_voxels))
    add('candidate_density',
        frame.candidate_face_neighbour_count <= cfg.max_candidate_face_neighbours)
    add('dense_guard',
        True if cfg.dense_candidate_guard_voxels == 0 else frame.dense_core_chebyshev_distance > cfg.dense_candidate_guard_voxels)
    for name in ('solid', 'liquid', 'gas'):
        add(name + '_count', frame[
            name + '_count_11cube'] >= cfg.min_phase_voxels_in_count_cube,
            frame.count_support_complete)
        v = frame[name + '_fraction']
        lo = getattr(cfg, name + '_fraction_min')
        hi = getattr(cfg, name + '_fraction_max')
        add(name + '_fraction', (v >= lo) & (True if hi is None else v <= hi), np.isfinite(v))
    for prefix in ('wg', 'sw'):
        add(prefix + '_interface_size', frame[
            prefix + '_interface_component_size'] >= cfg.min_interface_component_voxels)
        add(prefix + '_plane_points', frame[
            prefix + '_plane_point_count'] >= cfg.min_plane_points,
            frame.geometry_support_complete)
        for key, threshold in [('plane_rms', cfg.max_plane_rms_reference_voxels), (
            'singular_value_ratio', cfg.max_singular_value_ratio), ('orientation_alignment',
            cfg.min_orientation_alignment)]:
            v = frame[prefix + '_' + key]
            add(prefix + '_' + key,
                v >= threshold if key == 'orientation_alignment' else v <= threshold,
                np.isfinite(v))
    table = pd.DataFrame(states, index=frame.index)
    accepted = (table == 'pass').all(axis=1).to_numpy()
    return (accepted, table)


def statistics(values, raw_n, cfg):
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    return dict(N=len(values), raw_denominator=int(raw_n), fraction_of_raw=len(
        values) / raw_n if raw_n else None, **_angle_summary(values, cfg))


# G contains finite angles. Q is a subset with exactly the same angle values.
def calculate(volume, cfg=None, progress_every=5000):
    if cfg is None:
        cfg = Parameters()
    frame = extract(volume, cfg, progress_every=progress_every)
    accepted, states = policy(frame, cfg)
    if (accepted & ~frame.geometry_available.to_numpy(bool)).any():
        raise RuntimeError('Q must be a subset of G')
    summary = {'G': statistics(frame.loc[frame.geometry_available, 'contact_angle_degrees'],
        len(frame), cfg), 'Q': statistics(frame.loc[accepted, 'contact_angle_degrees'], len(
        frame), cfg)}
    return (frame, accepted, states, summary)

def run_image(input_path, output):
    """Read the CT stack and save angles, features and acceptance states."""
    cfg = Parameters()
    input_path = Path(input_path)
    out = Path(output)
    if out.exists():
        raise FileExistsError("Use a new output directory: " + str(out))
    if not input_path.is_file():
        raise FileNotFoundError("Set INPUT_PATH to a labelled CT TIFF: " + str(input_path))
    volume = _load_volume(input_path)
    volume.setflags(write=False)
    out.mkdir(parents=True)
    frame, accepted, states, summary = calculate(volume, cfg)
    enriched = frame.copy()
    enriched['accepted_Q'] = accepted
    enriched['rejection_reasons'] = [
        ';'.join(states.columns[row == 'fail']) for _, row in states.iterrows()
    ]
    enriched['not_evaluated_criteria'] = [
        ';'.join(states.columns[row == 'not_evaluated']) for _, row in states.iterrows()
    ]

    def output_csv(table, name):
        table.to_csv(out / name, index=False, compression={'method': 'gzip', 'mtime': 0})

    output_csv(enriched, 'candidate_features.csv.gz')
    output_csv(states, 'criterion_states.csv.gz')
    output_csv(enriched.loc[frame.geometry_available], 'G.csv.gz')
    output_csv(enriched.loc[accepted], 'Q.csv.gz')
    with (out / 'summary.json').open('w', encoding='utf-8') as handle:
        json.dump(summary, handle, indent=2, allow_nan=False)
    print('Raw candidates: {}; G: {}; Q: {}'.format(
        len(frame), int(frame.geometry_available.sum()), int(accepted.sum())))
    print('Results saved to ' + str(out))
    return summary


if __name__ == '__main__':
    run_image(INPUT_PATH, OUTPUT_DIR)
