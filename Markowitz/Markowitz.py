import rasterio
import numpy as np
import glob
import matplotlib.pyplot as plt

from typing import Optional, List

from utils import validate_array_dtype, normalize_weights, calculate_sharpe_ratio
from checkpoints import check_consistent_crs, check_consistent_pixel_size

from .logging_config import logger

class Markowitz:
    def __init__(self, raster_path_pattern: str, target_raster: Optional[str] = None,
                 num_pixels: Optional[int] = None, seed: int = 42) -> None:
        """
        Initializes the Markowitz analysis on rasters.
            :param raster_path_pattern: Pattern for files, e.g., 'data/precip_2019-09-*.tif'
            :param target_raster: Return raster (production) as an optional argument
            :param num_pixels: Number of pixels to sample
            :param seed: Seed for reproducibility
        """
        if not isinstance(raster_path_pattern, str) or not raster_path_pattern.strip():
            raise ValueError("The 'raster_path_pattern' parameter must be a valid string.")
        if target_raster is not None and not isinstance(target_raster, str):
            raise ValueError("The 'target_raster' parameter must be a string or None.")
        if num_pixels is not None and (not isinstance(num_pixels, int) or num_pixels <= 0):
            raise ValueError("The 'num_pixels' parameter must be a positive integer or None.")
        if not isinstance(seed, int) or seed < 0:
            raise ValueError("The 'seed' parameter must be a non-negative integer.")

        self.raster_path_pattern = raster_path_pattern
        self.target_raster_path = target_raster
        self.num_pixels = num_pixels
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
        return (f"Markowitz(raster_path_pattern={self.raster_path_pattern}, "
                f"target_raster={self.target_raster_path}, num_pixels={self.num_pixels}, seed={self.seed})")

    def __str__(self):
        return (f"Markowitz Analysis:\n"
                f" - Raster Path Pattern: {self.raster_path_pattern}\n"
                f" - Target Raster: {self.target_raster_path}\n"
                f" - Number of Pixels: {self.num_pixels}\n"
                f" - Seed: {self.seed}")

    def __len__(self):
        return len(self.coords) if self.coords else 0

    def __getitem__(self, index):
        if self.series is None:
            raise ValueError("Pixels not sampled. Use .sample_pixels() first.")
        return self.series[index]

    def load_stack(self):
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
        if not files:
            self.logger.error(f"No files found for the pattern: {self.raster_path_pattern}")
            raise ValueError(f"No files found for the pattern: {self.raster_path_pattern}")

        # Check CRS consistency
        if not check_consistent_crs(files, self.logger):
            self.logger.warning("Inconsistent CRS detected. Aborting stack loading.")
            return

        # Check pixel size consistency
        if not check_consistent_pixel_size(files, self.logger):
            self.logger.warning("Inconsistent pixel sizes detected. Aborting stack loading.")
            return

        stack = [rasterio.open(f).read(1) for f in files]
        self.stack = np.array(stack)
        self.logger.info(f"Stack loaded: {self.stack.shape}")
        self.logger.info(f"Total NaNs in the stack: {np.isnan(self.stack).sum()}")

    def sample_pixels(self, threshold: float = 0.0, data_percent_tolerance: float = 0.7) -> None:
        """
        The goal here is to select a set of random pixels from the precipitation stack for analysis.
        Here, we are essentially choosing some "assets" (or precipitation pixels) to observe how they vary over time.
            How it works:
                 * The code checks which pixels have valid values (precipitation greater than 0).
                 * Then, it samples a number of random pixels, as if we were choosing financial assets to build a portfolio.
                 * The time series of precipitation for these selected pixels will be our time series of returns (simulating the performance of assets over time).
        :param threshold: Minimum precipitation value to consider a pixel valid
        :param data_percent_tolerance: Minimum percentage of valid data to consider a pixel valid
        :return: List of coordinates of sampled pixels
        """
        if self.stack is None:
            raise ValueError("Stack not loaded. Use .load_stack() first.")

        self.stack = np.nan_to_num(self.stack)  # removes NaNs by replacing them with 0
        self.stack[self.stack < threshold] = 0  # applies threshold

        # Mask for pixels valid in at least 70% of the dates
        valid_ratio = np.mean(self.stack > 0, axis=0)
        valid_mask = valid_ratio >= data_percent_tolerance

        self.logger.info(f"Valid pixels before masking: {np.sum(valid_mask)}")
        ys, xs = np.where(valid_mask)
        coords = list(zip(ys, xs))

        if self.num_pixels is None:
            self.coords = coords
            self.logger.info(f"All valid pixels sampled: {len(coords)}")
        else:
            if self.num_pixels > len(coords):
                self.logger.error(f"You requested {self.num_pixels} pixels, but only {len(coords)} are valid.")
                raise ValueError(f"You requested {self.num_pixels} pixels, but only {len(coords)} are valid.")
            np.random.seed(self.seed)
            sampled = np.random.choice(len(coords), self.num_pixels, replace=False)
            self.coords = [coords[i] for i in sampled]
            self.logger.info(f"{self.num_pixels} random pixels sampled.")

        self.series = np.array([self.stack[:, y, x] for y, x in self.coords])

        # Load actual return raster
        if self.target_raster_path:
            with rasterio.open(self.target_raster_path) as src:
                target_data = src.read(1)
                self.target_values = np.array([target_data[y, x] for y, x in self.coords])
                self.logger.info(f"Raster de retorno real carregado: {self.target_raster_path}")

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

        self.mean_ = self.series.mean(axis=1)
        self.std_ = self.series.std(axis=1)
        self.cov_matrix = np.cov(self.series)
        self.logger.info("Statistics successfully calculated: mean, standard deviation, and covariance matrix.")

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

        results = np.zeros((3, num_portfolios))
        n = len(self.mean_)

        def simulate_portfolio():
            weights = np.random.random(n)
            weights = normalize_weights(weights)
            retorno = np.dot(weights, self.mean_)
            risco = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
            sharpe = calculate_sharpe_ratio(retorno, risco)
            return risco, retorno, sharpe, weights

        for i in range(num_portfolios):
            risco, retorno, sharpe, weights = simulate_portfolio()
            results[0, i] = risco
            results[1, i] = retorno
            results[2, i] = sharpe
            self.weights_list.append(weights)

        self.results = results
        self.logger.info(f"{num_portfolios} portfolios successfully simulated.")

    def evaluate_against_target(self):
        """
        Calculates the actual return of the portfolios using the return raster
        :return: array of actual returns
        """
        if self.target_values is None or self.weights_list is None:
            raise ValueError("Actual return values or weights are not available.")

        real_returns = []
        for weights in self.weights_list:
            retorno_real = np.dot(weights, self.target_values)
            real_returns.append(retorno_real)

        self.logger.info("Evaluation against the return raster completed.")
        return np.array(real_returns)

    def plot_frontier(self):
        """
        Plots the Efficiency Frontier
        """
        if self.results is None:
            raise ValueError("Results not simulated. Use .simulate_portfolios() first.")

        risco, retorno, sharpe = self.results
        plt.figure(figsize=(10, 6))
        plt.scatter(risco, retorno, c=sharpe, cmap='plasma', s=10)
        plt.xlabel('Risk (Standard Deviation)')
        plt.ylabel('Return (Average)')
        plt.title('Efficiency Frontier')
        plt.colorbar(label='Sharpe Ratio')
        plt.grid(linestyle='--')
        plt.show()
        self.logger.info("Climate efficiency frontier plotted.")

    def plot_real_frontier(self):
        """
        Plots Risk vs. Actual Return based on the production raster
        """
        if self.results is None:
            raise ValueError("Results not simulated.")
        real_returns = self.evaluate_against_target()
        riscos = self.results[0]

        plt.figure(figsize=(10, 6))
        plt.scatter(riscos, real_returns, c=real_returns, cmap='viridis', s=10)
        plt.xlabel('Risk (Standard Deviation)')
        plt.ylabel('Actual Return (Production)')
        plt.title('Frontier Based on Production')
        plt.colorbar(label='Estimated Production')
        plt.grid(linestyle='--')
        plt.show()
        self.logger.info("Frontier based on production plotted.")

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
            self.logger.warning("No portfolio with Sharpe above the threshold.")
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

        self.logger.info(f"{len(selected_precips)} portfolios selected with Sharpe >= {threshold}")
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
            self.logger.error(f"No files found for the pattern: {self.raster_path_pattern}")
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

        self.logger.info(f"GeoTIFF file successfully created at: {output_path}")

mk = Markowitz('C:/Users/Leonardo/PycharmProjects/EfficiencyFrontier/Example/*.tif')
mk.load_stack()

"""
mk = Markowitz('C:/Users/Leonardo/PycharmProjects/EfficiencyFrontier/Example/GPM_2019-09-0*.tif')
mk = Markowitz('C:/Users/c0010261/Scripts/EfficiencyFrontier/Example/GPM_2019-09-012*.tif')
Variáveis climáticas com avaliação de retorno por pixel dentre as séries temporais:
    mk.load_stack()
    mk.sample_pixels()
    mk.calculate_statistics()
    mk.simulate_portfolios()
    mk.plot_frontier()
    sel, bin = mk.get_high_sharpe(.7)
    mk.create_tif_from_array('output_mask.tif', bin)
    
"""
"""
Variáveis climaticas com avaliação de retorno sobre outra variavel Ex. Produção volumétrica:
    mk.load_stack()
    mk.sample_pixels()
    mk.calculate_statistics()
    mk.simulate_portfolios()
    mk.plot_real_frontier()
"""

