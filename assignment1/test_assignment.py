"""Numerical checks: python -m unittest discover -s assignment1 -v."""
import unittest
import numpy as np
from part1 import newton_closest_point, golden_section_closest_point
from part2 import POINTS, analytical_fit, sequential_newton_fit


class AssignmentTests(unittest.TestCase):
    def test_required_points(self):
        for p in (0, -4, -8, 2, 6):
            n = newton_closest_point(lambda x:x*x+5, lambda x:2*x, lambda x:2, p, 0, 1)
            g = golden_section_closest_point(lambda x:x*x+5, p, 0, -2, 2)
            self.assertTrue(n['converged'] and g['converged'])
            self.assertAlmostEqual(n['x'], g['x'], places=6)
            self.assertLess(abs(4*n['x']**3+22*n['x']-2*p), 1e-9)
            if p == 0:
                self.assertAlmostEqual(n['distance'], 5)
            widths = [h['b']-h['a'] for h in g['history']]
            self.assertTrue(all(v < u for u, v in zip(widths, widths[1:])))

    def test_safeguards(self):
        n = newton_closest_point(lambda x:x*x, lambda x:2*x, lambda x:2, 0, .5, 0)
        self.assertEqual(n['status'], 'near_zero_curvature')
        n = newton_closest_point(lambda x:x*x, lambda x:2*x, lambda x:2, 0, 1, 0)
        self.assertEqual(n['status'], 'not_a_minimum')
        n = newton_closest_point(lambda x:x, lambda x:1, lambda x:0, 1, 1, 0, max_iter=0)
        self.assertFalse(n['converged'])
        g = golden_section_closest_point(lambda x:x, 0, 0, -1, 1, max_iter=0)
        self.assertEqual(g['status'], 'max_iter')
        with self.assertRaises(ValueError):
            golden_section_closest_point(lambda x:x, 0, 0, 2, -2)
        with self.assertRaises(ValueError):
            newton_closest_point(lambda x:float('nan'), lambda x:1, lambda x:0, 0, 0, 0)

    def test_fits_against_independent_least_squares(self):
        for power, expected in [(1, (2.3, -.2, .575)), (2, (75/98, 4/7, 5/392))]:
            exact = analytical_fit(power=power)
            fit = sequential_newton_fit(power=power)
            design = np.column_stack((POINTS[:, 0]**power, np.ones(4)))
            reference = np.linalg.lstsq(design, POINTS[:, 1], rcond=None)[0]
            np.testing.assert_allclose([fit['coefficient'], fit['b']], reference, atol=1e-9)
            np.testing.assert_allclose([exact[k] for k in ('coefficient', 'b', 'mse')], expected, atol=1e-12)
            self.assertTrue(fit['converged'])
            losses = [h['mse'] for h in fit['history']]
            self.assertTrue(all(v <= u+1e-14 for u, v in zip(losses, losses[1:])))
        early = sequential_newton_fit(power=2, max_iter=3)['history'][1:]
        np.testing.assert_allclose([[h['coefficient'], h['b']] for h in early],
                                   [[.84693877551, 2/7], [.80612244898, 3/7], [.78571428571, .5]], atol=1e-10)


if __name__ == '__main__':
    unittest.main()
