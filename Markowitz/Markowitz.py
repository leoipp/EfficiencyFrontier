import rasterio
import numpy as np
import glob
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
import pandas as pd
import h5py

from typing import Optional, List

from utils import validate_array_dtype, normalize_weights, calculate_sharpe_ratio, convert_to_float16, \
    count_valid_pixels_blockwise
from checkpoints import check_consistent_crs, check_consistent_pixel_size

from logging_config import logger

class Markowitz:
    def __init__(self, raster_path_pattern: str,
                 seed: int = 42) -> None:
        """
        Initializes the Markowitz analysis on rasters.
            :param raster_path_pattern: Pattern for files, e.g., 'data/precip_2019-09-*.tif'
            :param seed: Seed for reproducibility
        """
        if not isinstance(raster_path_pattern, str) or not raster_path_pattern.strip():
            raise ValueError("The 'raster_path_pattern' parameter must be a valid string.")
        if not isinstance(seed, int) or seed < 0:
            raise ValueError("The 'seed' parameter must be a non-negative integer.")

        self.raster_path_pattern = raster_path_pattern
        self.seed = seed
        self.weights_list = []
        self.target_values = None
        self.stack = None
        self.series = None
        self.mean_ = None
        self.std_ = None
        self.cov_matrix = None
        self.results = None
        self.coords = None

        logger.info("Markowitz class successfully initialized.")

    def __repr__(self):
        return (f"Markowitz(raster_path_pattern={self.raster_path_pattern}")

    def __str__(self):
        return (f"Markowitz Analysis:\n"
                f" - Raster Path Pattern: {self.raster_path_pattern}\n"
                f" - Seed: {self.seed}")

    def __len__(self):
        return len(self.coords) if self.coords else 0

    def __getitem__(self, index):
        if self.series is None:
            raise ValueError("Pixels not sampled. Use .sample_pixels() first.")
        return self.series[index]

    def load_stack(
            self,
            block_size: int = 716,
            threshold: float = 0.0,
            pixel_presence: float = 0.95,
            save_as: Optional[str] = None,
            memmap_path: Optional[str] = "stack_float16.dat") -> None:
        """
        This method loads all rasters into a 3D array. This can be compared to collecting financial data from different assets over several days.
        Here, the assets are the daily precipitation values in various regions. Essentially, it samples N valid pixels and extracts time series.
            How it works:
                * The code reads all TIFF files matching the raster_path_pattern.
                * It stacks them into a 3D array: Each layer of the array will be a raster for a day, and the rows and columns represent the different pixels
                  (like different financial assets on different dates).
                * The resulting array will have the shape (n_days, n_rows, n_columns), where n_days is the number of days, and n_rows and n_columns are the raster dimensions.
        :return: None
        """
        files = sorted(glob.glob(self.raster_path_pattern))

        # Check if files exist
        if not files:
            logger.error(f"No files found for the pattern: {self.raster_path_pattern}")
            raise ValueError(f"No files found for the pattern: {self.raster_path_pattern}")

        # Check CRS and pixel size consistency
        if not check_consistent_crs(files, target=None, log=logger):
            logger.warning("Inconsistent CRS detected. Aborting stack loading.")
            return

        # Check if the target raster is valid
        if not check_consistent_pixel_size(files, target=None, log=logger):
            logger.warning("Inconsistent pixel sizes detected. Aborting stack loading.")
            return

        pixel_count, mask = count_valid_pixels_blockwise(files, block_size=block_size, threshold=threshold,
                                                   pixel_presence=pixel_presence, log=logger, save_as=save_as)

        # Check if pixel_count and mask are valid
        if pixel_count is None:
            logger.error("Error counting valid pixels. Aborting stack loading.")
            return
        if mask is None:
            logger.error("Error creating mask. Aborting stack loading.")
            return

        # Transform the mask to a 1D array
        mask_flat = mask.ravel()

        # First, determine n_days and n_valid_pixels
        n_days = len(files)
        n_valid_pixels = mask_flat.sum()

        # Create memory-mapped file
        self.stack = np.memmap(memmap_path, dtype='float16', mode='w+', shape=(n_days, n_valid_pixels))

        # Loop and write directly to the memmap
        for day_idx, file_path in enumerate(files):
            with rasterio.open(file_path) as src:
                data = src.read(1)
                flat_data = data.ravel()
                filtered_data = flat_data[mask_flat]
                self.stack[day_idx, :] = filtered_data.astype('float16')  # Write directly

        # Optionally flush to disk
        self.stack.flush()

        logger.info(f"Stack loaded: {self.stack.shape}")
        logger.info(f"Total NaNs in the stack: {np.isnan(self.stack).sum()}")
        logger.info(f"Total Valid pixels in the stack: {self.stack.size - np.isnan(self.stack).sum()}")

    def sample_pixels(self, num_pixels: Optional[int]=None) -> None:
        """
        The goal here is to select a set of random pixels from the assets stack for analysis.
        Here, we are essentially choosing some "assets" (or precipitation pixels) to observe how they vary over time.
            How it works:
                 * The code checks which pixels have valid values (precipitation greater than 0).
                 * Then, it samples a number of random pixels, as if we were choosing financial assets to build a portfolio.
                 * The time series of precipitation for these selected pixels will be our time series of returns (simulating the performance of assets over time).
        :param num_pixels: Number of pixels to sample. If None, all valid pixels are selected.
        :return: List of coordinates of sampled pixels
        """
        if self.stack is None:
            raise ValueError("Stack not loaded. Use .load_stack() first.")

        n_valid_pixels = self.stack.shape[1]

        if num_pixels is None:
            logger.info(f"All valid pixels selected: {n_valid_pixels}")
            self.series = np.array(self.stack, copy=True)  # Full copy
            self.coords = list(range(n_valid_pixels))  # Full list of flat indices
        else:
            if num_pixels > n_valid_pixels:
                logger.error(f"You requested {num_pixels} pixels, but only {n_valid_pixels} are valid.")
                raise ValueError(f"You requested {num_pixels} pixels, but only {n_valid_pixels} are valid.")

            np.random.seed(self.seed)
            sampled = np.random.choice(n_valid_pixels, num_pixels, replace=False)
            self.series = self.stack[:, sampled]
            self.coords = sampled.tolist()

            logger.info(f"{num_pixels} random pixels sampled.")

    def calculate_statistics(self):
        """
        Now, we will calculate means, standard deviations, and the covariance matrix of the precipitation time series
        from the sampled pixels. These statistics are essential to understand the historical performance and risk of
        each pixel.
            How it works:
                * Precipitation Mean: It is the average return for each pixel (asset).
                * Standard Deviation: We measure the volatility of each pixel over time, i.e., the uncertainty associated
                with its behavior (the risk).
                * Covariance Matrix: The covariance between the different pixels (or assets) shows how they behave
                together over time, helping to understand if they move together (correlation).
        :return: None
        """
        if self.series is None:
            raise ValueError("Pixels not sampled. Use .sample_pixels() first.")

        # Convert to float32 to avoid overflow
        series = self.series.astype(np.float32)

        self.mean_ = series.mean(axis=1)
        self.std_ = series.std(axis=1)
        self.cov_matrix = np.cov(series)
        logger.info("Statistics successfully calculated: mean, standard deviation, and covariance matrix.")

    def simulate_portfolios(self, num_portfolios: int=1000) -> None:
        """
        Now, the code will simulate the composition of various climate portfolios. Each portfolio will be a
        combination of different weights assigned to each pixel (asset). From this, we will calculate the average
        precipitation (return) and the risk (standard deviation) associated with each combination.
            How it works:
                * Random weights are assigned to each pixel.
                * For each weight combination, the expected return and risk of the portfolio are calculated, as well as the
                Sharpe Ratio.
                    * Return: Weighted average of the precipitations.
                    * Risk: Weighted standard deviation using the covariance matrix.
                    * Sharpe Ratio: Calculated by dividing the return by the risk, helping to determine which
                    combination has the best risk-adjusted return.
        :param num_portfolios: Number of portfolios to be simulated
        :return: None
        """
        if self.cov_matrix is None:
            raise ValueError("Statistics not calculated. Use .calculate_statistics() first.")

        n_assets = len(self.mean_)

        # Create a structured array: columns named 'risk', 'return', 'sharpe'
        dtype = [('risk', 'float32'), ('return', 'float32'), ('sharpe', 'float32')]
        self.results = np.zeros(num_portfolios, dtype=dtype)
        self.weights_list = []

        def simulate_portfolio():
            w = np.random.random(n_assets)
            w = normalize_weights(w)
            ret = np.dot(w, self.mean_)
            risk = np.sqrt(np.dot(w.T, np.dot(self.cov_matrix, w)))
            sharpe = calculate_sharpe_ratio(ret, risk)
            return risk, ret, sharpe, w

        for i in range(num_portfolios):
            risk, ret, sharpe, weights = simulate_portfolio()
            self.results[i] = (risk, ret, sharpe)
            self.weights_list.append(weights)

        logger.info(f"{num_portfolios} portfolios successfully simulated.")

    def plot_frontier(self):
        """
        Plots the Efficiency Frontier
        """
        if self.results is None:
            raise ValueError("Results not simulated. Use .simulate_portfolios() first.")

        risco = self.results['risk']
        retorno = self.results['return']
        sharpe = self.results['sharpe']

        plt.figure(figsize=(10, 6))
        plt.scatter(risco, retorno, c=sharpe, cmap='plasma', s=10)
        plt.xlabel('Risk (Volatility)')
        plt.ylabel('Return (Average)')
        plt.title('Efficiency Frontier')
        plt.colorbar(label='Sharpe Ratio')
        plt.grid(linestyle='--')
        plt.show()

        logger.info("Climate efficiency frontier plotted.")

    def get_high_sharpe(self, threshold: float=1.0) -> tuple[List[np.ndarray], np.ndarray]:
        """
        Returns the actual weighted variable of portfolios with Sharpe above the threshold.
        :param threshold: minimum Sharpe value
        :return: list of arrays above the weighted threshold and binary raster (xs, ys)
        """
        if self.results is None or self.weights_list is None:
            raise ValueError("Portfolios not simulated yet.")

        riscos, retornos, sharpes = self.results
        high_sharpe_indices = np.where(sharpes >= threshold)[0]

        if len(high_sharpe_indices) == 0:
            logger.warning("No portfolio with Sharpe above the threshold.")
            return [], np.zeros_like(self.stack[0], dtype=int)

        selected_precips = []
        binary_raster = np.zeros_like(self.stack[0], dtype=int)
        for idx in high_sharpe_indices:
            weights = self.weights_list[idx]
            # Combines the actual time series with the weights (weighted precipitation over time)
            combined = np.dot(weights, self.series)
            selected_precips.append(combined)

        # Marks the selected pixels in the binary raster
        for y, x in self.coords:
            binary_raster[y, x] = 1

        logger.info(f"{len(selected_precips)} portfolios selected with Sharpe >= {threshold}")
        return selected_precips, binary_raster

    def create_tif_from_array(self, output_path: str, array: np.ndarray) -> None:
        """
        Creates a GeoTIFF file from a numpy array using the first raster of the pattern as a reference.

        :param output_path: Path to save the generated GeoTIFF file.
        :param array: Numpy array representing the raster.
        """
        if not isinstance(output_path, str) or not output_path.strip():
            raise ValueError("The 'output_path' parameter must be a valid string.")
        if not isinstance(array, np.ndarray):
            raise ValueError("The 'array' parameter must be a numpy array.")

        array = validate_array_dtype(array, [np.uint8, np.int16, np.float32])

        # Gets the first raster of the pattern
        files = sorted(glob.glob(self.raster_path_pattern))
        if not files:
            logger.error(f"No files found for the pattern: {self.raster_path_pattern}")
            raise ValueError(f"No files found for the pattern: {self.raster_path_pattern}")

        reference_raster = files[0]

        with rasterio.open(reference_raster) as src:
            meta = src.meta.copy()
            meta.update({
                "driver": "GTiff",  # Explicitly defines the driver for GeoTIFF
                "dtype": array.dtype.name,
                "height": array.shape[0],
                "width": array.shape[1],
                "count": 1,
                "crs": src.crs,  # Copies the spatial reference system
                "transform": src.transform  # Copies the geospatial transformation
            })

            with rasterio.open(output_path, "w", **meta) as dst:
                dst.write(array, 1)

        logger.info(f"GeoTIFF file successfully created at: {output_path}")


mk = Markowitz(r'G:\PycharmProjects\EfficiencyFrontier\Example\Resample-test\*.tif')
mk.load_stack(
    block_size=716,
    threshold=0.0,
    pixel_presence=0.9,
    save_as=None,
    memmap_path='stack_float16.dat'
)
mk.sample_pixels()
mk.calculate_statistics()
mk.simulate_portfolios()
mk.plot_frontier()
sel, bin = mk.get_high_sharpe(.7)
mk.create_tif_from_array('output_mask.tif', bin)

