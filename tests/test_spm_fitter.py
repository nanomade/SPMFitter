import warnings
import unittest

from spm_fitter import *


class TestSPMFitterFitter(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Fix a deprecation warning from gwyfile
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # todo: There will be more than one test-image, we will get to that part...
        cls.fitter = SPMFitter('F1.002.gwy')

    def test_image_size(self):
        """
        Test that the size of the test images is parsed correctly
        (this really should not fail...)
        """
        x_size = self.fitter.size[0]
        y_size = self.fitter.size[1]
        self.assertAlmostEqual(x_size, 1.20702e-5, delta=1e-9)
        self.assertAlmostEqual(y_size, 1.20702e-5, delta=1e-9)

    def test_find_all_medians(self):
        medians = self.fitter._find_all_medians()
        self.assertAlmostEqual(medians[0], -6.407817731932293e-07, delta=1e-14)
        self.assertAlmostEqual(medians[127], -5.907531755957928e-07, delta=1e-14)
        self.assertAlmostEqual(medians[-1], -4.5861909865525434e-07, delta=1e-14)
        
    def test_find_modulated_lines(self):
        """
        Test that the auto-detection of modulated lines works.
        Expected result is to find all lines from 111 to 483
        """
        modulated_lines = self.fitter._find_modulated_lines()
        for line in range(111, 483):
            self.assertTrue(line in modulated_lines)

    def test_index_to_area(self):
        # Coordinates to test index-to-area. Happens to also be the
        # coordinates of the patterned area
        coords = {'x_l': 27, 'x_r': 469, 'y_t': 88, 'y_b': 507}
        area = self.fitter._index_to_area(**coords)
        expected_area = (
            (0.6365197265624999, 0.11787402343749999),
            (11.056583398437498, 9.995717187499999)
        )
        self.assertAlmostEqual(area[0][0], expected_area[0][0], delta=1e-9)
        self.assertAlmostEqual(area[0][1], expected_area[0][1], delta=1e-9)
        self.assertAlmostEqual(area[1][0], expected_area[1][0], delta=1e-9)
        self.assertAlmostEqual(area[1][1], expected_area[1][1], delta=1e-9)

    def test_hat_line_fit(self):
        # Test that we cannot fit a hat to line 111
        hat_fit = self.fitter._fit_hat_to_line(line_nr=111, plot=False)
        self.assertIsNone(hat_fit)

        # Test that we can fit a hat to line 112
        hat_fit = self.fitter._fit_hat_to_line(line_nr=112, plot=False)
        expected_fit = (86, 410, 112)
        self.assertEqual(hat_fit, expected_fit)

    def test_full_image_roughness(self):
        """
        Calculate the roughness of the entire image.
        """
        roughness = self.fitter.calculate_roughness()
        self.assertAlmostEqual(roughness, 6.053639690567813e-08, delta=1e-14)

    def test_plane_fit(self):
        """
        Test plane fit to entire area.
        TODO:
        Also need to test fit to sub-area as well as a fit to a masked area
        """
        z = self.fitter._plane_fit(plot=False)
        self.assertEqual(z.shape, (512, 512))
        self.assertAlmostEqual(sum(z[0][:]), -0.00033277827779927305, delta=1e-14)
        self.assertAlmostEqual(z[0][0], -6.419494146295475e-07, delta=1e-14)
        self.assertAlmostEqual(z[-1][0], -4.431543901977951e-07, delta=1e-14)
        self.assertAlmostEqual(z[125][125], -5.972383808630513e-07, delta=1e-14)


if __name__ == '__main__':
    unittest.main()
