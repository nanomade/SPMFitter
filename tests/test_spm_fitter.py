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
