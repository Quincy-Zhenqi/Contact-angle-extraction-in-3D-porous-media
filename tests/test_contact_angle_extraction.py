import json
import sys
import unittest
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).absolute().parents[1]
sys.path.insert(0, str(ROOT))
import contact_angle_extraction as ca


def right_angle():
    volume = np.ones((43, 43, 43), dtype=np.uint8)
    volume[21:, :, :21] = 2
    volume[21:, :, 21:] = 3
    return volume


class ScientificContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = ca.Parameters()
        cls.volume = right_angle()
        original = cls.volume.copy()
        cls.volume.setflags(write=False)
        cls.frame, cls.q, cls.states, cls.summary = ca.calculate(cls.volume, progress_every=0)
        np.testing.assert_array_equal(cls.volume, original)

    def test_parameters(self):
        expected = json.loads((ROOT / 'example_config.json').read_text())
        self.assertEqual(expected, json.loads(json.dumps(asdict(self.cfg))))
        self.assertEqual(ca.validate_parameters(ca.Parameters(**expected)).scientific_labels, (1, 2, 3))
        self.assertIsNone(self.cfg.max_plane_points)
        with self.assertRaises(ValueError):
            ca.Parameters(voxel_spacing_zyx=(1, 1, 2))

    def test_labels(self):
        self.assertEqual(set(ca.validate_volume(self.volume, self.cfg)), {1, 2, 3})
        for bad in (np.ones((2, 2, 2), int), np.array([[[1, 2, 3, 5]]]), self.volume[0]):
            with self.assertRaises(ValueError):
                ca.validate_volume(bad, self.cfg)
        with self.assertRaises(TypeError):
            ca.validate_volume(self.volume.astype(float), self.cfg)

    def test_connectivity(self):
        for offset, expected in [((1, 0, 0), (1, 1)), ((1, 1, 0), (2, 1)),
                                 ((1, 1, 1), (2, 2))]:
            mask = np.zeros((3, 3, 3), bool)
            mask[0, 0, 0] = mask[offset] = True
            self.assertEqual(tuple(ca.component_labels_and_sizes(mask, c)[2]
                                   for c in (1, 2)), expected)
        lg = ca.liquid_side_interface(self.volume, 2, 3)
        sl = ca.liquid_side_interface(self.volume, 2, 1)
        self.assertTrue(np.all(self.volume[lg | sl] == 2))
        expected = np.zeros_like(lg)
        expected[21, :, 20] = True
        np.testing.assert_array_equal(lg & sl, expected)

    def test_rank_and_sign(self):
        cloud = np.array([(0, 0, 2), (1, 0, 2), (0, 1, 2)], float)
        n, rms, ratio = ca.fit_plane(cloud)
        np.testing.assert_allclose(abs(n), [0, 0, 1])
        self.assertEqual((rms, ratio), (0, 0))
        self.assertIsNone(ca.fit_plane(np.diag([1, 0, 0]))[0])
        n, alignment, _ = ca.orient_axis(np.array([-1., 0, 0]), np.zeros(3), np.array([1., 0, 0]))
        np.testing.assert_array_equal(n, [1, 0, 0])
        self.assertEqual(alignment, 1)
        self.assertIsNone(ca.orient_axis(n, np.zeros(3), np.array([0., 1, 0]))[0])
        self.assertAlmostEqual(ca.contact_angle_degrees(n, [-0.5, 3 ** .5 / 2, 0]), 120)

    def test_known_geometry_and_selection(self):
        f, q, s = self.frame, self.q, self.states
        self.assertEqual((len(f), self.summary['G']['N'], self.summary['Q']['N']), (43, 33, 13))
        np.testing.assert_array_equal(f.y[q], np.arange(15, 28))
        np.testing.assert_array_equal(f.y[f.geometry_available], np.arange(5, 38))
        for prefix, normal in [('n_lg', [1, 0, 0]), ('n_solid', [0, 0, 1])]:
            np.testing.assert_allclose(f.loc[q, [prefix + '_' + c for c in 'xyz']],
                                       np.tile(normal, (13, 1)), atol=1e-12)
        angles = f.set_index(['z', 'y', 'x']).contact_angle_degrees
        g_angles = angles[f.geometry_available.values]
        q_angles = angles[q]
        np.testing.assert_array_equal(g_angles.loc[q_angles.index], q_angles)
        np.testing.assert_allclose(g_angles, 90, atol=1e-12, rtol=0)
        self.assertEqual(s.shape, (43, 23))
        self.assertEqual(set(s.to_numpy().ravel()), {'pass', 'fail', 'not_evaluated'})
        np.testing.assert_array_equal(q, s.eq('pass').all(axis=1))
        edge = s.loc[f.y.eq(0)].iloc[0]
        self.assertEqual((edge.numerical_geometry, edge.wg_plane_points), ('fail', 'not_evaluated'))

    def test_q_thresholds(self):
        row = self.frame.loc[self.q].iloc[[0]]
        boundaries = [
            ('candidate_face_neighbour_count', 'candidate_density', 3, 4),
            ('dense_core_chebyshev_distance', 'dense_guard', 3, 2),
            ('array_boundary_distance', 'legacy_margin', 15, 14),
            ('solid_fraction', 'solid_fraction', .30, .2999),
            ('solid_fraction', 'solid_fraction', .70, .7001),
            ('liquid_component_size', 'component_size', 50, 49),
            ('gas_component_size', 'component_size', 50, 49),
        ]
        for phase in ('solid', 'liquid', 'gas'):
            boundaries.append((phase + '_count_11cube', phase + '_count', 20, 19))
        for prefix in ('wg_', 'sw_'):
            for field, state, good, bad in [('interface_component_size', 'interface_size', 15, 14),
                                           ('plane_point_count', 'plane_points', 15, 14),
                                           ('plane_rms', 'plane_rms', 1, 1.001),
                                           ('singular_value_ratio', 'singular_value_ratio', .30, .301),
                                           ('orientation_alignment', 'orientation_alignment', .05, .049)]:
                boundaries.append((prefix + field, prefix + state, good, bad))
        for field, state, good, bad in boundaries:
            for value, expected in [(good, 'pass'), (bad, 'fail')]:
                with self.subTest(field=field, value=value):
                    accepted, states = ca.policy(row.assign(**{field: value}), self.cfg)
                    self.assertEqual(states[state].iat[0], expected)
                    self.assertEqual(accepted[0], expected == 'pass')
                    self.assertEqual(states['numerical_geometry'].iat[0], 'pass')

    def test_dense_core(self):
        v = np.ones((3, 9, 9), np.uint8)
        v[1] = 2
        v[2] = 3
        f, _, s, _ = ca.calculate(v, progress_every=0)
        for i, row in f.iterrows():
            y, x = int(row.y), int(row.x)
            expected = max(2 - x, x - 6, 2 - y, y - 6, 0)
            self.assertEqual(row.dense_core_chebyshev_distance, expected)
            self.assertEqual(s.at[i, 'dense_guard'], 'fail')
            self.assertEqual(s.at[i, 'candidate_density'], 'fail' if 0 < x < 8 and 0 < y < 8 else 'pass')

    def test_background_and_empty(self):
        volume = right_angle()
        volume[21, 21, 17] = 4
        f, q, _, _ = ca.calculate(volume, progress_every=0)
        center = f.loc[f.y.eq(21)].iloc[0]
        self.assertFalse(center.geometry_available)
        self.assertEqual(center.geometry_support_reason, 'background_in_geometry_support_envelope')
        self.assertFalse(center.sphere_complete)
        self.assertFalse(q[f.y.eq(21)].any())
        separated = np.ones((20, 20, 20), np.uint8)
        separated[4:7] = 2
        separated[11:14] = 3
        f, q, states, stats = ca.calculate(separated, progress_every=0)
        self.assertEqual((len(f), len(q), states.shape), (0, 0, (0, 23)))
        self.assertIsNone(stats['Q']['mean_degrees'])


if __name__ == '__main__':
    unittest.main()
