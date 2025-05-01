import unittest

import numpy as np
import rasterio

from Markowitz.Markowitz import Markowitz
import os
import json
import gc  # Importar o coletor de lixo


class TestMarkowitz(unittest.TestCase):
    def setUp(self):
        self.valid_path = r'G:\PycharmProjects\EfficiencyFrontier\Example\GPM-2015-2024\*.tif'
        self.invalid_path = 'C:/Invalid/Path/*.tif'
        self.markowitz = Markowitz(self.valid_path)
        self.markowitz_with_seed = Markowitz(self.valid_path, seed=123)

    def test_load_stack(self):
        self.markowitz.load_stack(block_size=512, threshold=0.1, pixel_presence=0.8, dtype='float32',
                                  memmap_path='stack_test.dat', memmap_shape_path='stack_shape_test.json')
        self.assertIsNotNone(self.markowitz.stack)
        self.assertGreater(self.markowitz.stack.shape[0], 0)

    def test_load_stack_invalid_path(self):
        mk = Markowitz(self.invalid_path)
        with self.assertRaises(ValueError):
            mk.load_stack()

    def test_load_datstack(self):
        memmap_path = 'dummy_stack_test.dat'
        shape_path = 'dummy_stack_shape_test.json'
        dummy_data = np.random.rand(10, 100).astype('float32')

        # Criação do arquivo de memória mapeada
        memmap_file = np.memmap(memmap_path, dtype='float32', mode='w+', shape=dummy_data.shape)
        memmap_file[:] = dummy_data
        memmap_file.flush()
        del memmap_file
        gc.collect()

        # Criação do arquivo de metadados
        with open(shape_path, 'w') as f:
            json.dump({"shape": list(dummy_data.shape), "dtype": "float32"}, f)  # usa list para serializar corretamente

        # Carregar o arquivo de memória mapeada
        self.markowitz.load_datstack(memmap_path, shape_path, dtype='float32')
        self.assertIsNotNone(self.markowitz.stack)
        self.assertEqual(self.markowitz.stack.shape, dummy_data.shape)

        # Fechar o memmap antes de remover
        del self.markowitz.stack
        gc.collect()

        # Remover os arquivos após o teste
        try:
            os.remove(memmap_path)
            os.remove(shape_path)
        except PermissionError as e:
            self.fail(f"PermissionError ao remover arquivos: {e}")

    def test_sample_pixels(self):
        self.markowitz.load_stack(block_size=512, threshold=0.1, pixel_presence=0.8, dtype='float32',
                                  memmap_path='stack_test.dat', memmap_shape_path='stack_shape_test.json')
        self.markowitz.sample_pixels(num_pixels=50)
        self.assertEqual(len(self.markowitz.coords), 50)
        self.assertEqual(self.markowitz.series.shape[1], 50)

    def test_calculate_statistics(self):
        self.markowitz.load_stack(block_size=512, threshold=0.1, pixel_presence=0.8)
        self.markowitz.sample_pixels(num_pixels=50)
        self.markowitz.calculate_statistics(method="manual", shrinkage_intensity=0.2, normalize=True)
        self.assertIsNotNone(self.markowitz.mean_)
        self.assertIsNotNone(self.markowitz.std_)
        self.assertIsNotNone(self.markowitz.cov_matrix)

    def test_simulate_portfolios(self):
        self.markowitz.load_stack(block_size=512, threshold=0.1, pixel_presence=0.8, dtype='float32',
                                  memmap_path='stack_test.dat', memmap_shape_path='stack_shape_test.json')
        self.markowitz.sample_pixels(num_pixels=50)
        self.markowitz.calculate_statistics()
        self.markowitz.simulate_portfolios(num_portfolios=200)
        self.assertIsNotNone(self.markowitz.results)
        self.assertEqual(len(self.markowitz.results), 200)

    def test_plot_frontier(self):
        self.markowitz.load_stack(block_size=512, threshold=0.1, pixel_presence=0.8, dtype='float32',
                                  memmap_path='stack_test.dat', memmap_shape_path='stack_shape_test.json')
        self.markowitz.sample_pixels(num_pixels=50)
        self.markowitz.calculate_statistics()
        self.markowitz.simulate_portfolios(num_portfolios=200)
        try:
            self.markowitz.plot_frontier(optimize=True)
        except Exception as e:
            self.fail(f"plot_frontier raised an exception: {e}")

    def test_get_high_sharpe(self):
        self.markowitz.load_stack(block_size=512, threshold=0.1, pixel_presence=0.8)
        self.markowitz.sample_pixels(num_pixels=50)
        self.markowitz.calculate_statistics()
        self.markowitz.simulate_portfolios(num_portfolios=200)
        high_sharpe_precips, binary_raster = self.markowitz.get_high_sharpe(threshold=1.5)
        self.assertIsInstance(high_sharpe_precips, list)
        self.assertIsInstance(binary_raster, np.ndarray)

    def test_create_tif_from_array(self):
        self.markowitz.load_stack(block_size=512, threshold=0.1, pixel_presence=0.8)
        self.markowitz.sample_pixels(num_pixels=50)
        binary_raster = np.zeros_like(self.markowitz.stack[0], dtype=int)
        for idx in self.markowitz.coords:
            binary_raster[idx] = 1

        output_path = "test_output.tif"
        self.markowitz.create_tif_from_array(output_path, binary_raster)

        self.assertTrue(os.path.exists(output_path))
        with rasterio.open(output_path) as src:
            written_array = src.read(1)
            self.assertTrue(np.array_equal(written_array, binary_raster))

        os.remove(output_path)

    def tearDown(self):
        if os.path.exists(self.invalid_path):
            os.remove(self.invalid_path)


if __name__ == "__main__":
    unittest.main()
