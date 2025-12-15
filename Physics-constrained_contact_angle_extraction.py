import numpy as np
import tifffile
import matplotlib.pyplot as plt
import os
from scipy.ndimage import (
    label, generate_binary_structure,
    uniform_filter, convolve
)
import time
import pandas as pd
import glob

# -----------------------------
# Utility: precompute sphere offsets (for fast indexing in spherical neighborhoods)
# -----------------------------
def precompute_sphere_offsets(radius):
    rr = range(-radius, radius + 1)
    dz, dy, dx = np.meshgrid(rr, rr, rr, indexing='ij')
    mask = (dz*dz + dy*dy + dx*dx) <= radius*radius
    dz = dz[mask].ravel().astype(np.int16)
    dy = dy[mask].ravel().astype(np.int16)
    dx = dx[mask].ravel().astype(np.int16)
    return np.stack([dz, dy, dx], axis=1)  # (N,3) in (z,y,x)

# -----------------------------
# 3D region filtering for water and gas
# (treat isolated small water clusters as gas, and isolated small gas clusters as water)
# -----------------------------
def filter_small_regions_3d(volume, min_water_voxels=100, min_air_voxels=100):
    """
    Filter out regions with voxel count smaller than the threshold (3D).
    """
    print("Filtering small water regions...")
    start_time = time.time()

    filtered_volume = np.copy(volume)

    print("Filtering small water regions...")
    water_mask = (volume == 2)
    structure = generate_binary_structure(3, 1)
    labeled_water, num_features = label(water_mask, structure=structure)
    component_sizes = np.bincount(labeled_water.ravel())
    valid_water_mask = np.zeros_like(water_mask, dtype=bool)
    for i in range(1, num_features + 1):
        if component_sizes[i] >= min_water_voxels:
            valid_water_mask |= (labeled_water == i)
    filtered_volume[(volume == 2) & ~valid_water_mask] = 3

    print("Filtering small gas regions...")
    air_mask = (volume == 3)
    labeled_air, num_features_air = label(air_mask, structure=structure)
    component_sizes_air = np.bincount(labeled_air.ravel())
    valid_air_mask = np.zeros_like(air_mask, dtype=bool)
    for i in range(1, num_features_air + 1):
        if component_sizes_air[i] >= min_air_voxels:
            valid_air_mask |= (labeled_air == i)
    filtered_volume[(volume == 3) & ~valid_air_mask] = 2

    print(f"Region filtering completed, elapsed: {time.time() - start_time:.2f} s")
    return filtered_volume

# -----------------------------
# 3D interface extraction - face connectivity (6-neighborhood)
# -----------------------------
def extract_interface_6conn(volume, val1, val2):
    """
    Extract interface using 6-neighborhood (3D).
    """
    print(f"Extracting {val1}-{val2} interface (6-neighborhood)...")
    start_time = time.time()

    mask1 = (volume == val1)
    mask2 = (volume == val2)
    neighbor_mask = np.zeros_like(mask2, dtype=bool)
    face_offsets = [
        (1, 0, 0), (-1, 0, 0),
        (0, 1, 0), (0, -1, 0),
        (0, 0, 1), (0, 0, -1)
    ]
    for dz, dy, dx in face_offsets:
        slices_in = [slice(None), slice(None), slice(None)]
        slices_out = [slice(None), slice(None), slice(None)]
        if dz > 0:
            slices_in[0] = slice(1, None)
            slices_out[0] = slice(0, -1)
        elif dz < 0:
            slices_in[0] = slice(0, -1)
            slices_out[0] = slice(1, None)
        if dy > 0:
            slices_in[1] = slice(1, None)
            slices_out[1] = slice(0, -1)
        elif dy < 0:
            slices_in[1] = slice(0, -1)
            slices_out[1] = slice(1, None)
        if dx > 0:
            slices_in[2] = slice(1, None)
            slices_out[2] = slice(0, -1)
        elif dx < 0:
            slices_in[2] = slice(0, -1)
            slices_out[2] = slice(1, None)
        try:
            neighbor_mask[tuple(slices_out)] |= mask2[tuple(slices_in)]
        except IndexError:
            pass

    interface = mask1 & neighbor_mask
    print(f"{val1}-{val2} interface extracted, voxels: {np.sum(interface)}, elapsed: {time.time() - start_time:.2f} s")
    return interface

# -----------------------------
# 3D triple-phase point detection
# -----------------------------
def find_three_phase_points_3d(interface1, interface2):
    # return (N,3) zyx integer array
    return np.argwhere(interface1 & interface2)

# -----------------------------
# 3D triple-phase point validation (6-neighborhood)
# -----------------------------
def is_valid_triple_point_6neigh(volume, x, y, z):
    """
    Check whether the 6-neighborhood contains all three phases.
    """
    neighbors = [
        (x + 1, y, z), (x - 1, y, z),
        (x, y + 1, z), (x, y - 1, z),
        (x, y, z + 1), (x, y, z - 1)
    ]
    unique_vals = set()
    for nx, ny, nz in neighbors:
        if (0 <= nx < volume.shape[2] and
                0 <= ny < volume.shape[1] and
                0 <= nz < volume.shape[0]):
            unique_vals.add(volume[nz, ny, nx])
    return {1, 2, 3}.issubset(unique_vals)

# -----------------------------
# 3D phase ratios in spherical neighborhood (fast: precomputed offsets + indexing)
# -----------------------------
# Precompute offsets inside sphere (radius=15)
SPHERE_OFFS = precompute_sphere_offsets(radius=15)

def compute_phase_ratios_in_sphere_fast(volume, z, y, x, offs=SPHERE_OFFS):
    zz = z + offs[:, 0]
    yy = y + offs[:, 1]
    xx = x + offs[:, 2]
    mask = (
        (zz >= 0) & (zz < volume.shape[0]) &
        (yy >= 0) & (yy < volume.shape[1]) &
        (xx >= 0) & (xx < volume.shape[2])
    )
    if not np.any(mask):
        return 0.0, 0.0, 0.0
    vox = volume[zz[mask], yy[mask], xx[mask]]
    total = vox.size
    s = np.count_nonzero(vox == 1) / total
    w = np.count_nonzero(vox == 2) / total
    g = np.count_nonzero(vox == 3) / total
    return s, w, g

# -----------------------------
# 3D plane fitting (PCA/SVD)
# -----------------------------
def fit_plane(points):
    """
    Input: points as (x,y,z)
    Output: normal vector, RMS residual, eigenvalue ratio
    """
    if len(points) < 10:
        return None, None, None
    points_arr = np.array(points, dtype=float)
    centered = points_arr - np.mean(points_arr, axis=0)
    try:
        U, s, Vh = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
        return None, None, None
    residuals = np.linalg.norm(centered @ Vh[2]) / np.sqrt(len(points))
    eigenvalue_ratio = s[2] / s[1] if s[1] > 1e-6 else float('inf')
    normal = Vh[2]
    return normal, residuals, eigenvalue_ratio

# -----------------------------
# 3D direction check (water-gas)
# -----------------------------
def check_wg_direction_3d(volume, x, y, z, normal, interface_type):
    """
    Check normal direction: enforce one side is water and the other side is gas.
    """
    if normal is None:
        return None
    steps = [1, 2, 3]
    side1, side2 = [], []
    for step in steps:
        px = int(round(x + normal[0] * step))
        py = int(round(y + normal[1] * step))
        pz = int(round(z + normal[2] * step))
        nx = int(round(x - normal[0] * step))
        ny = int(round(y - normal[1] * step))
        nz = int(round(z - normal[2] * step))
        if not (0 <= px < volume.shape[2] and 0 <= py < volume.shape[1] and 0 <= pz < volume.shape[0]):
            return None
        if not (0 <= nx < volume.shape[2] and 0 <= ny < volume.shape[1] and 0 <= nz < volume.shape[0]):
            return None
        side1.append(volume[pz, py, px])
        side2.append(volume[nz, ny, nx])
    if len(set(side1)) != 1 or len(set(side2)) != 1:
        return None
    t1, t2 = side1[0], side2[0]
    if t1 == t2:
        return None
    pair = set([t1, t2])
    if pair == set([2, 3]):  # water-gas
        return normal if t1 == 3 else -normal
    elif pair == set([1, 2]):  # solid-water
        return normal if t1 == 1 else -normal
    elif pair == set([1, 3]):  # solid-gas
        return normal if t1 == 3 else -normal
    else:
        return None

# -----------------------------
# 3D direction check (solid-water)
# -----------------------------
def check_sw_direction_3d(volume, x, y, z, normal, interface_type):
    """
    Check normal direction: enforce one side is solid and the other side is water.
    """
    if normal is None:
        return None
    steps = [1, 2, 3]
    side1, side2 = [], []
    for step in steps:
        px = int(round(x + normal[0] * step))
        py = int(round(y + normal[1] * step))
        pz = int(round(z + normal[2] * step))
        nx = int(round(x - normal[0] * step))
        ny = int(round(y - normal[1] * step))
        nz = int(round(z - normal[2] * step))
        if not (0 <= px < volume.shape[2] and 0 <= py < volume.shape[1] and 0 <= pz < volume.shape[0]):
            return None
        if not (0 <= nx < volume.shape[2] and 0 <= ny < volume.shape[1] and 0 <= nz < volume.shape[0]):
            return None
        side1.append(volume[pz, py, px])
        side2.append(volume[nz, ny, nx])
    if len(set(side1)) != 1 or len(set(side2)) != 1:
        return None
    t1, t2 = side1[0], side2[0]
    if t1 == t2:
        return None
    pair = set([t1, t2])
    if pair == set([1, 2]):  # solid-water
        return normal if t1 == 2 else -normal
    elif pair == set([1, 3]):  # solid-gas
        return normal if t1 == 3 else -normal
    elif pair == set([2, 3]):  # water-gas
        return normal if t1 == 2 else -normal
    else:
        return None

# -----------------------------
# Contact angle computation
# -----------------------------
def compute_contact_angle(vec1, vec2):
    if vec1 is None or vec2 is None:
        return None
    vec1_norm = vec1 / np.linalg.norm(vec1)
    vec2_norm = vec2 / np.linalg.norm(vec2)
    dot = np.clip(np.dot(vec1_norm, vec2_norm), -1.0, 1.0)
    angle = np.arccos(dot)
    return np.degrees(angle)

# -----------------------------
# Get local patch points from global labels (avoid per-point local labeling)
# -----------------------------
def get_patch_points_from_global_labels(labeled_global, z, y, x, window=4, max_points=512):
    """
    labeled_global: global label() results on interface masks
    Returns: (N,3) z,y,x coordinates in the local window belonging to the same label as the center point.
    Randomly downsamples to max_points if needed.
    """
    lab_id = labeled_global[z, y, x]
    if lab_id == 0:
        return np.empty((0, 3), dtype=int)

    Z0, Z1 = max(0, z - window), min(labeled_global.shape[0], z + window + 1)
    Y0, Y1 = max(0, y - window), min(labeled_global.shape[1], y + window + 1)
    X0, X1 = max(0, x - window), min(labeled_global.shape[2], x + window + 1)

    sub = labeled_global[Z0:Z1, Y0:Y1, X0:X1]
    pts = np.argwhere(sub == lab_id)
    if pts.size == 0:
        return np.empty((0, 3), dtype=int)

    pts[:, 0] += Z0
    pts[:, 1] += Y0
    pts[:, 2] += X0

    if pts.shape[0] > max_points:
        idx = np.random.choice(pts.shape[0], size=max_points, replace=False)
        pts = pts[idx]
    return pts

# -----------------------------
# Main workflow (single image)
# -----------------------------
def process_single_image(image_path):
    print(f"\n{'=' * 50}")
    print(f"Start processing: {image_path}")
    start_total = time.time()

    image_name = os.path.basename(image_path)
    image_base = os.path.splitext(image_name)[0]

    print("Reading volume...")
    volume = tifffile.imread(image_path)
    print(f"Original shape: {volume.shape}, dtype: {volume.dtype}")

    # 1) Small-region filtering (keep original logic)
    filtered_volume = filter_small_regions_3d(volume, min_water_voxels=50, min_air_voxels=50)

    # 2) Interface extraction (6-neighborhood)
    interface_wg = extract_interface_6conn(filtered_volume, 2, 3)  # water-gas
    interface_sw = extract_interface_6conn(filtered_volume, 2, 1)  # water-solid

    # 3) Global connected components once (for later patch extraction)
    print("Running connected-component labeling (global, once)...")
    NEIGH18 = generate_binary_structure(3, 2)
    labeled_wg, num_features_wg = label(interface_wg, structure=NEIGH18)
    component_sizes_wg = np.bincount(labeled_wg.ravel())
    labeled_sw, num_features_sw = label(interface_sw, structure=NEIGH18)
    component_sizes_sw = np.bincount(labeled_sw.ravel())

    # 4) Triple-phase candidates (ndarray in z,y,x order)
    print("Finding triple-phase points...")
    triple_points = find_three_phase_points_3d(interface_wg, interface_sw)  # (N,3) zyx
    print(f"Number of triple-phase points found: {len(triple_points)}")

    # --- Optimization: density map + one convolution for neighbor counts ---
    density_map = np.zeros_like(filtered_volume, dtype=np.uint8)
    if len(triple_points) > 0:
        density_map[triple_points[:, 0], triple_points[:, 1], triple_points[:, 2]] = 1
    # 6-neighborhood kernel (excluding center)
    K = np.zeros((3, 3, 3), dtype=np.uint8)
    K[1, 1, 0] = K[1, 1, 2] = 1
    K[1, 0, 1] = K[1, 2, 1] = 1
    K[0, 1, 1] = K[2, 1, 1] = 1
    neighbor_counts = convolve(density_map, K, mode='constant', cval=0)

    # --- Optimization: precompute local cube counts (window=5 -> 11^3 voxels) ---
    win = 2*5 + 1
    k3 = win**3
    mask_sand  = (filtered_volume == 1).astype(np.float32)
    mask_water = (filtered_volume == 2).astype(np.float32)
    mask_gas   = (filtered_volume == 3).astype(np.float32)
    cnt_sand_map  = uniform_filter(mask_sand,  size=win, mode='nearest') * k3
    cnt_water_map = uniform_filter(mask_water, size=win, mode='nearest') * k3
    cnt_gas_map   = uniform_filter(mask_gas,   size=win, mode='nearest') * k3

    # Per-point screening and angle computation
    contact_angles = []
    good_points = []
    vector_points = []
    bad_points = []
    low_angle_points = []
    large_angle_points = []
    sand_ratios = []
    water_ratios = []
    gas_ratios = []
    low_sand_ratio_points = []
    high_sand_ratio_points = []
    low_water_ratio_points = []
    high_water_ratio_points = []
    low_gas_ratio_points = []
    high_gas_ratio_points = []
    high_density_points = []

    total_points = len(triple_points)
    print(f"Processing {total_points} triple-phase points...")
    start_time = time.time()

    for i in range(total_points):
        z, y, x = triple_points[i]

        if i % 1000 == 0 or i == total_points - 1:
            elapsed = time.time() - start_time
            pts_per_sec = (i + 1) / elapsed if elapsed > 0 else 0
            remaining = (total_points - i - 1) / pts_per_sec if pts_per_sec > 0 else 0
            print(f"Processed {i + 1}/{total_points}, speed: {pts_per_sec:.1f} pts/s, ETA: {remaining:.1f} s")

        # ---- Remove overly dense points (using convolution result) ----
        if neighbor_counts[z, y, x] >= 4:
            high_density_points.append((x, y, z))
            bad_points.append((x, y, z))
            continue

        # ---- 6-neighborhood must contain all three phases ----
        if not is_valid_triple_point_6neigh(filtered_volume, x, y, z):
            bad_points.append((x, y, z))
            continue

        # ---- Local phase voxel counts (from precomputed maps) ----
        if cnt_water_map[z, y, x] < 20:
            bad_points.append((x, y, z))
            continue
        if cnt_gas_map[z, y, x] < 20:
            bad_points.append((x, y, z))
            continue
        if cnt_sand_map[z, y, x] < 20:
            bad_points.append((x, y, z))
            continue

        # ---- Interface patch size thresholds (from global labels) ----
        label_id_wg = labeled_wg[z, y, x]
        if component_sizes_wg[label_id_wg] < 15:
            bad_points.append((x, y, z))
            continue
        label_id_sw = labeled_sw[z, y, x]
        if component_sizes_sw[label_id_sw] < 15:
            bad_points.append((x, y, z))
            continue

        # ---- Phase ratios in a sphere neighborhood (fast) ----
        sand_ratio, water_ratio, gas_ratio = compute_phase_ratios_in_sphere_fast(filtered_volume, z, y, x)
        if sand_ratio < 0.3:
            low_sand_ratio_points.append((x, y, z)); bad_points.append((x, y, z)); continue
        elif sand_ratio > 0.7:
            high_sand_ratio_points.append((x, y, z)); bad_points.append((x, y, z)); continue
        if water_ratio < 0.1:
            low_water_ratio_points.append((x, y, z)); bad_points.append((x, y, z)); continue
        elif water_ratio > 0.9:
            high_water_ratio_points.append((x, y, z)); bad_points.append((x, y, z)); continue
        if gas_ratio < 0.1:
            low_gas_ratio_points.append((x, y, z)); bad_points.append((x, y, z)); continue
        elif gas_ratio > 0.9:
            high_gas_ratio_points.append((x, y, z)); bad_points.append((x, y, z)); continue

        # ---- Get interface points using global labels + local window ----
        pts_wg_zyx = get_patch_points_from_global_labels(labeled_wg, z, y, x, window=4, max_points=512)
        pts_sw_zyx = get_patch_points_from_global_labels(labeled_sw, z, y, x, window=4, max_points=512)
        if pts_wg_zyx.shape[0] < 15 or pts_sw_zyx.shape[0] < 15:
            bad_points.append((x, y, z))
            continue

        # Convert to (x,y,z) for plane fitting
        pts_wg_xyz = [(p[2], p[1], p[0]) for p in pts_wg_zyx]
        pts_sw_xyz = [(p[2], p[1], p[0]) for p in pts_sw_zyx]

        # Fit planes
        n_wg, res_wg, ratio_wg = fit_plane(pts_wg_xyz)
        n_sw, res_sw, ratio_sw = fit_plane(pts_sw_xyz)
        if n_wg is None or n_sw is None:
            bad_points.append((x, y, z))
            continue
        if (res_wg is None or res_sw is None or
                res_wg > 1.0 or res_sw > 1.0 or
                ratio_wg > 0.3 or ratio_sw > 0.3):
            bad_points.append((x, y, z))
            continue

        # Fix normal directions
        n_wg_fixed = check_wg_direction_3d(filtered_volume, x, y, z, n_wg, "wg")
        n_sw_fixed = check_sw_direction_3d(filtered_volume, x, y, z, n_sw, "sw")
        if n_wg_fixed is None or n_sw_fixed is None:
            bad_points.append((x, y, z))
            continue

        # Compute contact angle
        angle = compute_contact_angle(n_wg_fixed, n_sw_fixed)
        if angle is not None:
            contact_angles.append(angle)
            sand_ratios.append(sand_ratio)
            water_ratios.append(water_ratio)
            gas_ratios.append(gas_ratio)
            good_points.append((x, y, z))
            vector_points.append((
                x, y, z,
                n_wg_fixed[0], n_wg_fixed[1], n_wg_fixed[2],
                n_sw_fixed[0], n_sw_fixed[1], n_sw_fixed[2],
                angle, sand_ratio, water_ratio, gas_ratio
            ))
            if angle < 10:
                low_angle_points.append((x, y, z))
            if angle > 170:
                large_angle_points.append((x, y, z))
        else:
            bad_points.append((x, y, z))

    # Print summary statistics
    print("\n===== Results =====")
    print(f"Image: {image_name}")
    print("Volume shape:", volume.shape)
    print("Total triple-phase points:", len(triple_points))
    print("Valid contact angles:", len(contact_angles))

    if contact_angles:
        print("Mean contact angle (deg):", np.mean(contact_angles))
        print("Min contact angle:", np.min(contact_angles))
        print("Max contact angle:", np.max(contact_angles))
        print("Std contact angle:", np.std(contact_angles))
        print("Mean solid ratio:", np.mean(sand_ratios))
        print("Mean water ratio:", np.mean(water_ratios))
        print("Mean gas ratio:", np.mean(gas_ratios))
    else:
        print("No valid contact angles. Statistics not available.")

    # Save results (original + new feature)
    print("Saving results...")
    np.savez(f"{image_base}_results.npz",
             contact_angles=np.array(contact_angles),
             sand_ratios=np.array(sand_ratios),
             water_ratios=np.array(water_ratios),
             gas_ratios=np.array(gas_ratios),
             good_points=np.array(good_points),
             bad_points=np.array(bad_points),
             vector_points=np.array(vector_points),
             low_sand_ratio_points=np.array(low_sand_ratio_points),
             high_sand_ratio_points=np.array(high_sand_ratio_points),
             low_water_ratio_points=np.array(low_water_ratio_points),
             high_water_ratio_points=np.array(high_water_ratio_points),
             low_gas_ratio_points=np.array(low_gas_ratio_points),
             high_gas_ratio_points=np.array(high_gas_ratio_points),
             high_density_points=np.array(high_density_points))

    # Generate histogram + CSV + contact-angle voxel map
    if contact_angles:
        # ====== 1. Contact angle histogram ======
        plt.figure(figsize=(10, 6))
        n, bins, patches = plt.hist(contact_angles, bins=36, range=(0, 180),
                                    color='skyblue', edgecolor='black')
        plt.xlabel('Contact angle (degrees)', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        avg_angle = np.mean(contact_angles)
        std_angle = np.std(contact_angles)
        median_angle = np.median(contact_angles)
        min_angle = np.min(contact_angles)
        max_angle = np.max(contact_angles)
        count = len(contact_angles)
        title = (f'Contact Angle Distribution - {image_base}\n'
                 f'Mean: {avg_angle:.2f}°, Std: {std_angle:.2f}°, Median: {median_angle:.2f}°\n'
                 f'Min: {min_angle:.2f}°, Max: {max_angle:.2f}°, Count: {count}')
        plt.title(title)
        plt.tight_layout()
        plt.savefig(f"{image_base}_contact_angle_histogram.tiff", dpi=300)
        plt.close()

        # Export histogram data
        bin_centers = 0.5 * (bins[1:] + bins[:-1])
        hist_data = pd.DataFrame({
            'Bin_Start': bins[:-1],
            'Bin_End': bins[1:],
            'Bin_Center': bin_centers,
            'Frequency': n,
            'Relative_Frequency': n / np.sum(n)
        })
        hist_data.to_csv(f"{image_base}_contact_angle_histogram_data.csv", index=False)

        # ====== 2. Per-point table: angle + ratios + coordinates ======
        good_points_arr = np.array(good_points)  # (N, 3) -> (x, y, z)
        x_coords = good_points_arr[:, 0]
        y_coords = good_points_arr[:, 1]
        z_coords = good_points_arr[:, 2]

        all_angles_data = pd.DataFrame({
            'X': x_coords,
            'Y': y_coords,
            'Z': z_coords,
            'Contact_Angle': contact_angles,
            'Sand_Ratio': sand_ratios,
            'Water_Ratio': water_ratios,
            'Gas_Ratio': gas_ratios
        })
        all_angles_data.to_csv(f"{image_base}_contact_angle_all_data.csv", index=False)

        # ====== 3. Summary statistics ======
        stats_data = pd.DataFrame({
            'Image': [image_name],
            'Dimensions': [str(volume.shape)],
            'Total_Triple_Points': [len(triple_points)],
            'Valid_Contact_Angles': [len(contact_angles)],
            'Mean_Contact_Angle': [avg_angle],
            'Median_Contact_Angle': [median_angle],
            'Std_Contact_Angle': [std_angle],
            'Min_Contact_Angle': [min_angle],
            'Max_Contact_Angle': [max_angle],
            'Mean_Sand_Ratio': [np.mean(sand_ratios)],
            'Mean_Water_Ratio': [np.mean(water_ratios)],
            'Mean_Gas_Ratio': [np.mean(gas_ratios)],
            'Processing_Time_sec': [time.time() - start_total]
        })
        stats_data.to_csv(f"{image_base}_contact_angle_statistics.csv", index=False)

        # ====== 4. Contact-angle voxel map (same size as input TIFF) ======
        # Same size as input volume; 0 elsewhere; write contact angle (degrees) at valid triple points
        angle_volume = np.zeros_like(volume, dtype=np.float32)
        for (x, y, z), angle in zip(good_points, contact_angles):
            angle_volume[z, y, x] = angle  # note z,y,x indexing

        angle_tiff_path = f"{image_base}_contact_angle_map.tif"
        tifffile.imwrite(angle_tiff_path, angle_volume)
        print(f"Contact-angle voxel map saved to: {angle_tiff_path}")

        print(f"Outputs saved to {image_base}_* files")
    else:
        print("No contact-angle data. Skip histogram plotting and CSV export.")

    print(f"Finished! Elapsed: {time.time() - start_total:.2f} s")
    print('=' * 50)

    return {
        'image': image_name,
        'dimensions': volume.shape,
        'total_triple_points': len(triple_points),
        'valid_contact_angles': len(contact_angles),
        'mean_contact_angle': np.mean(contact_angles) if contact_angles else 0,
        'median_contact_angle': np.median(contact_angles) if contact_angles else 0,
        'std_contact_angle': np.std(contact_angles) if contact_angles else 0,
        'min_contact_angle': np.min(contact_angles) if contact_angles else 0,
        'max_contact_angle': np.max(contact_angles) if contact_angles else 0,
        'mean_sand_ratio': np.mean(sand_ratios) if sand_ratios else 0,
        'mean_water_ratio': np.mean(water_ratios) if water_ratios else 0,
        'mean_gas_ratio': np.mean(gas_ratios) if gas_ratios else 0,
        'processing_time': time.time() - start_total
    }

# -----------------------------
# Batch processing (kept, not called by default)
# -----------------------------
def batch_process_images(image_directory):
    """
    Batch-process all TIFF images in a directory (kept, not called by default).
    """
    image_directory = str(image_directory)
    if not os.path.isdir(image_directory):
        print(f"[Error] Directory does not exist: {image_directory}")
        print("Please check the path/drive/permissions. Use r'...' or '/' to avoid escape issues.")
        return

    patterns = ["*.tif", "*.tiff"]
    tiff_files = []
    for pat in patterns:
        tiff_files.extend(glob.glob(os.path.join(image_directory, "**", pat), recursive=True))

    if not tiff_files:
        print(f"[Info] No TIFF files found in the directory and subdirectories: {image_directory}")
        return

    print(f"Found {len(tiff_files)} TIFF files to process")
    summary_report = []

    for i, image_path in enumerate(tiff_files):
        filename = os.path.basename(image_path)
        try:
            print(f"\nProcessing file {i + 1}/{len(tiff_files)}: {image_path}")
            stats = process_single_image(image_path)
            summary_report.append(stats)
        except Exception as e:
            print(f"[Exception] Error processing file: {image_path}\nReason: {e}")
            summary_report.append({
                'image': filename,
                'error': str(e)
            })

    if summary_report:
        summary_df = pd.DataFrame(summary_report)
        summary_csv = os.path.join(image_directory, "batch_processing_summary.csv")
        summary_df.to_csv(summary_csv, index=False)
        print(f"\nBatch processing finished! Summary saved to: {summary_csv}")

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    # Process a single specified TIFF (your file)
    image_path = r"xxx.tif"

    print("Using image_path:", image_path)
    if not os.path.isfile(image_path):
        print(f"[Error] File not found: {image_path}")
    else:
        process_single_image(image_path)

    # If you need batch processing later, uncomment the following two lines,
    # and comment out the single-image part above.
    # image_dir = r"F:\2024-7-18_percolation\contact angle distribution"
    # batch_process_images(image_dir)
