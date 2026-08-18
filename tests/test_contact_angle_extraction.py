import shutil
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from contact_angle_extraction import (  # noqa: E402
    ContactAngleConfig,
    compute_contact_angle,
    fit_plane,
    load_config,
    orient_normal,
    precompute_sphere_offsets,
    process_single_image,
)


class ContactAngleExtractionTests(unittest.TestCase):
    def test_example_config_matches_program_defaults(self):
        loaded = load_config(REPOSITORY_ROOT / "example_config.json")
        self.assertEqual(loaded, ContactAngleConfig())

    def test_radius_one_sphere_has_seven_voxels(self):
        offsets = precompute_sphere_offsets(1)
        self.assertEqual(offsets.shape, (7, 3))

    def test_plane_fit_and_vector_angle(self):
        points = np.array([(x, y, 2.0) for x in range(5) for y in range(5)])
        normal, residual, ratio = fit_plane(points, minimum_points=5)
        self.assertIsNotNone(normal)
        self.assertAlmostEqual(residual, 0.0, places=12)
        self.assertAlmostEqual(ratio, 0.0, places=12)
        angle = compute_contact_angle(np.array((1.0, 0.0, 0.0)), np.array((0.0, 0.0, 1.0)))
        self.assertAlmostEqual(angle, 90.0, places=12)

    def test_third_phase_orientation_fallback_is_optional(self):
        volume = np.full((9, 9, 9), 2, dtype=np.uint8)
        volume[4, 4, 5:7] = 1
        volume[4, 4, 2:4] = 3
        normal = np.array((1.0, 0.0, 0.0))

        fallback = orient_normal(
            volume,
            point_xyz=(4, 4, 4),
            normal=normal,
            positive_label=3,
            negative_label=2,
            probe_steps=(1, 2),
            fallback_positive_priority=(3, 1, 2),
        )
        strict = orient_normal(
            volume,
            point_xyz=(4, 4, 4),
            normal=normal,
            positive_label=3,
            negative_label=2,
            probe_steps=(1, 2),
        )

        self.assertTrue(np.array_equal(fallback, -normal))
        self.assertIsNone(strict)

    def test_end_to_end_orthogonal_interfaces(self):
        artifact_root = REPOSITORY_ROOT / ".test-artifacts"
        shutil.rmtree(str(artifact_root), ignore_errors=True)
        artifact_root.mkdir(parents=True)
        self.addCleanup(lambda: shutil.rmtree(str(artifact_root), ignore_errors=True))

        volume = np.ones((25, 25, 25), dtype=np.uint8)
        volume[10:, :, :12] = 2
        volume[10:, :, 12:] = 3
        input_tiff = artifact_root / "orthogonal_interfaces.tif"
        output_directory = artifact_root / "results"
        tifffile.imwrite(input_tiff, volume, photometric="minisblack")

        config = ContactAngleConfig(
            min_water_component_voxels=1,
            min_gas_component_voxels=1,
            local_count_window_radius=3,
            min_local_phase_voxels=1,
            min_interface_component_voxels=5,
            sphere_radius=4,
            solid_ratio_min=0.0,
            solid_ratio_max=1.0,
            water_ratio_min=0.0,
            water_ratio_max=1.0,
            gas_ratio_min=0.0,
            gas_ratio_max=1.0,
            patch_window_radius=3,
            patch_max_points=128,
            min_patch_points=5,
            max_plane_rms=0.01,
            max_singular_value_ratio=0.01,
            normal_probe_steps=(1, 2),
            progress_interval=100,
        )
        statistics = process_single_image(
            input_tiff,
            output_directory=output_directory,
            config=config,
            seed=42,
            overwrite=True,
        )
        self.assertGreater(statistics["Valid_Contact_Angles"], 0)
        point_table = pd.read_csv(output_directory / "orthogonal_interfaces_contact_angle_all_data.csv")
        self.assertTrue(np.allclose(point_table["Contact_Angle_Deg"], 90.0, atol=1e-8))
        angle_map = tifffile.imread(output_directory / "orthogonal_interfaces_contact_angle_map.tif")
        self.assertEqual(angle_map.shape, volume.shape)
        self.assertAlmostEqual(float(angle_map.max()), 90.0, places=5)


if __name__ == "__main__":
    unittest.main()
