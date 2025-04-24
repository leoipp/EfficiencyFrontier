import unittest

import numpy as np

from Markowitz import Markowitz
import os


class TestMarkowitz(unittest.TestCase):
    def setUp(self):
        self.valid_path = 'C:/Users/Leonardo/PycharmProjects/EfficiencyFrontier/Example/GPM_2019-09-0*.tif'
        self.invalid_path = 'C:/Invalid/Path/*.tif'
        self.markowitz = Markowitz(self.valid_path)
        self.markowitz_with_num_pixels = Markowitz(self.valid_path, num_pixels=10)

    def test_load_stack(self):
        self.markowitz.load_stack()
        self.assertIsNotNone(self.markowitz.stack)
        self.assertGreater(self.markowitz.stack.shape[0], 0)

    def test_load_stack_invalid_path(self):
        mk = Markowitz(self.invalid_path)
        with self.assertRaises(ValueError):
            mk.load_stack()

    def test_sample_pixels_without_stack(self):
        with self.assertRaises(ValueError):
            self.markowitz.sample_pixels()

    def test_sample_pixels_all_pixels(self):
        self.markowitz.load_stack()
        self.markowitz.sample_pixels(threshold=0.0, data_percent_tolerance=0.7)
        self.assertGreater(len(self.markowitz.coords), 0)
        self.assertEqual(self.markowitz.series.shape[0], len(self.markowitz.coords))

    def test_sample_pixels_with_num_pixels(self):
        self.markowitz_with_num_pixels.load_stack()
        self.markowitz_with_num_pixels.sample_pixels(threshold=0.0, data_percent_tolerance=0.7)
        self.assertEqual(len(self.markowitz_with_num_pixels.coords), 10)
        self.assertEqual(self.markowitz_with_num_pixels.series.shape[0], 10)

    def test_calculate_statistics_without_sampling(self):
        with self.assertRaises(ValueError):
            self.markowitz.calculate_statistics()

    def test_calculate_statistics(self):
        self.markowitz.load_stack()
        self.markowitz.sample_pixels(threshold=0.0, data_percent_tolerance=0.7)
        self.markowitz.calculate_statistics()
        self.assertIsNotNone(self.markowitz.mean_)
        self.assertIsNotNone(self.markowitz.std_)
        self.assertIsNotNone(self.markowitz.cov_matrix)

    def test_simulate_portfolios_without_statistics(self):
        with self.assertRaises(ValueError):
            self.markowitz.simulate_portfolios()

    def test_simulate_portfolios(self):
        self.markowitz.load_stack()
        self.markowitz.sample_pixels(threshold=0.0, data_percent_tolerance=0.7)
        self.markowitz.calculate_statistics()
        self.markowitz.simulate_portfolios(num_portfolios=100)
        self.assertIsNotNone(self.markowitz.results)
        self.assertEqual(self.markowitz.results.shape[1], 100)

    def test_simulate_portfolios_with_num_pixels(self):
        self.markowitz_with_num_pixels.load_stack()
        self.markowitz_with_num_pixels.sample_pixels(threshold=0.0, data_percent_tolerance=0.7)
        self.markowitz_with_num_pixels.calculate_statistics()
        self.markowitz_with_num_pixels.simulate_portfolios(num_portfolios=50)
        self.assertIsNotNone(self.markowitz_with_num_pixels.results)
        self.assertEqual(self.markowitz_with_num_pixels.results.shape[1], 50)

    def test_get_high_sharpe(self):
        self.markowitz.load_stack()
        self.markowitz.sample_pixels(threshold=0.0, data_percent_tolerance=0.7)
        self.markowitz.calculate_statistics()
        self.markowitz.simulate_portfolios(num_portfolios=100)
        high_sharpe_precips, binary_raster = self.markowitz.get_high_sharpe(threshold=1.0)
        self.assertIsInstance((high_sharpe_precips, binary_raster), tuple)
        self.assertIsInstance(high_sharpe_precips, list)
        self.assertIsInstance(binary_raster, np.ndarray)
        self.assertTrue(np.all(np.logical_or(binary_raster == 0, binary_raster == 1)))
        self.assertEqual(binary_raster.shape, self.markowitz.stack[0].shape)

    def test_dunder_repr(self):
        repr_output = repr(self.markowitz)
        self.assertIn("Markowitz(raster_path_pattern=", repr_output)
        self.assertIn("num_pixels=None", repr_output)

    def test_dunder_str(self):
        str_output = str(self.markowitz)
        self.assertIn("Markowitz Analysis:", str_output)
        self.assertIn("Raster Path Pattern:", str_output)
        self.assertIn("Number of Pixels: None", str_output)

    def test_dunder_len(self):
        self.markowitz.load_stack()
        self.markowitz.sample_pixels(threshold=0.0, data_percent_tolerance=0.7)
        self.assertEqual(len(self.markowitz), len(self.markowitz.coords))

    def test_dunder_getitem(self):
        self.markowitz.load_stack()
        self.markowitz.sample_pixels(threshold=0.0, data_percent_tolerance=0.7)
        first_pixel_series = self.markowitz[0]
        self.assertEqual(len(first_pixel_series), self.markowitz.stack.shape[0])
        self.assertTrue(isinstance(first_pixel_series, np.ndarray))

    def test_dunder_getitem_without_sampling(self):
        with self.assertRaises(ValueError):
            _ = self.markowitz[0]

    def tearDown(self):
        if os.path.exists(self.invalid_path):
            os.remove(self.invalid_path)


if __name__ == "__main__":
    unittest.main()
