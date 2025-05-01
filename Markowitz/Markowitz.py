import json

import rasterio
import numpy as np
import glob
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA

import cvxpy as cp

from typing import Optional, List

from utils import validate_array_dtype, normalize_weights, calculate_sharpe_ratio, count_valid_pixels_blockwise, \
    read_masked_stack_blockwise, normalize_stack
from checkpoints import check_consistent_crs, check_consistent_pixel_size

from logging_config import logger

class Markowitz:
    def __init__(
            self,
            raster_path_pattern: str,
            seed: int = 42
    ) -> None:
        """
        Initializes the Markowitz analysis on rasters.
            :param raster_path_pattern: Pattern for files, e.g., 'data/precip_2019-09-*.tif'
            :param seed: Seed for reproducibility
        """
        if not isinstance(raster_path_pattern, str) or not raster_path_pattern.strip():
            raise ValueError("The 'raster_path_pattern' parameter must be a valid string.")
        if not isinstance(seed, int) or seed < 0:
            raise ValueError("The 'seed' parameter must be a non-negative integer.")

        # --- Core parameters ---
        self.raster_path_pattern = raster_path_pattern
        self.seed = seed

        # --- Inputs and intermediate storage ---
        self.stack = None  # Raster stack (e.g., as xarray or np.array)
        self.series = None  # Time series or extracted values
        self.weights_list = []  # Storage for optimized weights
        self.target_values = None

        # --- Statistical properties ---
        self.mean_ = None
        self.std_ = None
        self.cov_matrix = None

        # --- Output/results ---
        self.results = None
        self.stats = None

        # --- Spatial properties ---
        self.coords = None
        self.num_pixels = None
        self.n_valid_pixels = None

        logger.info("Markowitz class successfully initialized.")

    def __repr__(self):
        return f"Markowitz(raster_path_pattern={self.raster_path_pattern}"

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
            pixel_presence: float = 0.99,
            dtype: str = 'float32',
            save_as: Optional[str] = None,
            memmap_path: Optional[str] = "stack_float32.dat",
            memap_shape_path: Optional[str] = "stack_shape.json"
    ) -> None:
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

        # --- Check if files exist ---
        if not files:
            logger.error(f"No files found for the pattern: {self.raster_path_pattern}")
            raise ValueError(f"No files found for the pattern: {self.raster_path_pattern}")

        # --- Check CRS and pixel size consistency ---
        if not check_consistent_crs(files, target=None, log=logger):
            logger.warning("Inconsistent CRS detected. Aborting stack loading.")
            return

        # --- Check if the target raster is valid ---
        if not check_consistent_pixel_size(files, target=None, log=logger):
            logger.warning("Inconsistent pixel sizes detected. Aborting stack loading.")
            return

        pixel_count, mask = count_valid_pixels_blockwise(files, block_size=block_size, threshold=threshold,
                                                   pixel_presence=pixel_presence, log=logger, save_as=save_as)

        # --- Check if pixel_count and mask are valid ---
        if pixel_count is None:
            logger.error("Error counting valid pixels. Aborting stack loading.")
            return
        if mask is None:
            logger.error("Error creating mask. Aborting stack loading.")
            return

        # --- Transform the mask to a 1D array - less power processing required ---
        mask_flat = mask.ravel()

        # --- First, determine n_days and n_valid_pixels ---
        n_days = len(files)
        n_valid_pixels = mask_flat.sum()

        # --- Create memory-mapped file ---
        self.stack = np.memmap(memmap_path, dtype=dtype, mode='w+', shape=(n_days, n_valid_pixels))

        # --- Loop and write directly to the memmap ---
        for day_idx, file_path in enumerate(files):
            day_data = read_masked_stack_blockwise(file_path, mask, block_size=block_size, dtype=dtype)
            self.stack[day_idx, :] = day_data
        self.stack.flush()

        # --- Save the shape to a .json file ---
        with open(memap_shape_path, 'w') as f:
            json.dump(
                {
                    "shape": [int(n_days), int(n_valid_pixels)],
                    "dtype": str(dtype)
                },
                f
            )

        self.n_valid_pixels = self.stack.shape[1]

        logger.info(f"Stack loaded: {self.stack.shape}")
        logger.debug(f"Total NaNs in the stack: {np.isnan(self.stack).sum()}")
        logger.debug(f"Total Valid pixels in the stack: {self.stack.size - np.isnan(self.stack).sum()}")

    def load_datstack(
            self,
            memmap_path: str,
            memap_shape_path: str,
            dtype: Optional[str] = None
    ) -> None:
        """
        Loads the memory-mapped raster stack from a .dat file using shape metadata saved as JSON.

        :param memmap_path: Path to the memory-mapped .dat file.
        :param shape_path: Path to the JSON file containing shape and dtype metadata.
        :param dtype: Optional override for dtype. If not provided, it will be read from metadata.
        """
        import json

        try:
            # Load shape and dtype from JSON
            with open(memap_shape_path, 'r') as f:
                meta = json.load(f)
                shape = tuple(meta["shape"])
                dtype = dtype or meta["dtype"]

            # Load memory-mapped array
            self.stack = np.memmap(memmap_path, dtype=dtype, mode='r', shape=shape)
            self.n_valid_pixels = self.stack.shape[1]

            logger.info(f"Stack loaded from {memmap_path} with shape {shape} and dtype {dtype}")
            logger.debug(f"Total NaNs in the stack: {np.isnan(self.stack).sum()}")
            logger.debug(f"Total Valid pixels in the stack: {self.stack.size - np.isnan(self.stack).sum()}")

        except FileNotFoundError:
            logger.error(f"File {memmap_path} or shape metadata {memap_shape_path} not found.")
        except Exception as e:
            logger.error(f"An error occurred while loading the stack: {str(e)}")

    def sample_pixels(
            self,
            normalize: bool=True,
            method: Optional[str]='standard',
            pca: bool=False,
            n_components: Optional[int]=None,
            num_pixels: Optional[int]=None
    ) -> None:
        """
        The goal here is to select a set of random pixels from the assets stack for analysis.
        Here, we are essentially choosing some "assets" (or precipitation pixels) to observe how they vary over time.
            How it works:
                 * The code checks which pixels have valid values (precipitation greater than 0).
                 * Then, it samples a number of random pixels, as if we were choosing financial assets to build a portfolio.
                 * The time series of precipitation for these selected pixels will be our time series of returns (simulating the performance of assets over time).
        :param num_pixels: Number of pixels to sample. If None, all valid pixels are selected.
        :param normalize: If True, normalizes the data.
        :param method: Normalization method. Options: 'standard', 'minmax', 'robust'.
        :return: List of coordinates of sampled pixels
        """
        if self.stack is None:
            raise ValueError("Stack not loaded. Use .load_stack() first.")

        if num_pixels is None:
            logger.info(f"All valid pixels selected: {self.n_valid_pixels}")
            self.series = np.array(self.stack, copy=True)  # Full copy
            self.coords = list(range(self.n_valid_pixels))  # Full list of flat indices
            self.num_pixels = self.n_valid_pixels
        else:
            if num_pixels > self.n_valid_pixels:
                logger.error(f"You requested {num_pixels} pixels, but only {self.n_valid_pixels} are valid.")
                raise ValueError(f"You requested {num_pixels} pixels, but only {self.n_valid_pixels} are valid.")

            np.random.seed(self.seed)
            sampled = np.random.choice(self.n_valid_pixels, num_pixels, replace=False)
            self.series = self.stack[:, sampled]
            self.coords = sampled.tolist()

            logger.info(f"{num_pixels} random pixels sampled.")
            self.num_pixels = num_pixels

        self.series = self.__calculate_returns__(self.series)

        if normalize:
            # Normalize the data
            self.series, self.stats = normalize_stack(self.series, method=method, axis=0)
            logger.info(f"Data normalized using method '{method}'.")

    def sample_pixels2(
            self,
            normalize: bool = True,
            method: Optional[str] = 'standard',
            use_pca: bool = True,  # Novo parâmetro para ativar/desativar PCA
            n_components: Optional[int] = None  # Número de componentes (None para autodetecção)
    ) -> None:
        # Aplica PCA se necessário
        if use_pca:

            # Define número de componentes para explicar 95% da variância
            if n_components is None:
                pca = PCA(n_components=0.95)
            else:
                pca = PCA(n_components=n_components)

            self.series = pca.fit_transform(self.series)
            logger.info(
                f"PCA aplicado: {self.series.shape[1]} componentes (explicam {100 * pca.explained_variance_ratio_.sum():.1f}% da variância)")

        # Normalização (opcional)
        if normalize:
            self.series, self.stats = normalize_stack(self.series, method=method, axis=0)
            logger.info(f"Dados normalizados com método '{method}'")

        self.coords = list(range(self.series.shape[1]))  # Coordenadas são os índices dos componentes

    @staticmethod
    def __calculate_returns__(data: np.ndarray) -> np.ndarray:
        """Calcula retornos percentuais com tratamento de divisão por zero."""
        with np.errstate(divide='ignore', invalid='ignore'):
            returns = (data[1:] - data[:-1]) / np.where(data[:-1] == 0, 1e-6, data[:-1])
            returns[~np.isfinite(returns)] = 0 # inf, -inf, NaN to 0
        return returns


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

        series = self.series.astype(np.float32)

        self.mean_ = series.mean(axis=0)
        self.std_ = series.std(axis=0)
        self.cov_matrix = np.cov(series, rowvar=False)

        if not np.isfinite(self.cov_matrix).all():
            raise ValueError("Covariance matrix contains NaNs or infinite values.")

        # Apply shrinkage
        # shrinkage_intensity = 0.1
        #self.cov_matrix = (1 - shrinkage_intensity) * self.cov_matrix + shrinkage_intensity * np.identity(
        # self.cov_matrix.shape[0], dtype=np.float32)

        # shrinkage_intensity = 0.5
        # self.cov_matrix = (1 - shrinkage_intensity) * self.cov_matrix + shrinkage_intensity * np.eye(self.cov_matrix.shape[0])

        # Matriz de covariância regularizada (Ledoit-Wolf shrinkage)

        cov_estimator = LedoitWolf().fit(self.series)
        self.cov_matrix = cov_estimator.covariance_

        logger.info(f"Statistics successfully calculated: mean, standard deviation, "
                    f"and covariance matrix.")

    def simulate_portfolios(
            self,
            num_portfolios: int=1000
    ) -> None:
        """
        Now, the code will simulate the composition of various climate portfolios. Each portfolio will be a
        combination of different weights assigned to each pixel (asset). From this, we will calculate the average
        precipitation (return) and the risk (standard deviation) associated with each combination.
            How it works:
                * Random weights are assigned to each pixel.
                * For each weight combination, the expected return and risk of the portfolio are calculated, as well as the
                Sharpe Ratio.
                    * Return: Weighted average.
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
            ret_ = np.dot(w, self.mean_)
            risk_ = np.sqrt(np.dot(w.T, np.dot(self.cov_matrix, w)))
            sharpe_ = calculate_sharpe_ratio(ret_, risk_)
            return risk_, ret_, sharpe_, w

        for i in range(num_portfolios):
            risk, ret, sharpe, weights = simulate_portfolio()
            self.results[i] = (risk, ret, sharpe)
            self.weights_list.append(weights)
        logger.info(f"{num_portfolios} portfolios successfully simulated.")

    def __optimize__(self):
        """
        Optimizes the efficient frontier by minimizing portfolio variance for a range of target returns.
        """
        y_ = np.linspace(self.results['return'].min(), self.results['return'].max(), 10)

        # Predefine functions
        mean_ = np.array(self.mean_)
        cov = self.cov_matrix

        def portfolio_return(w):
            return np.dot(w, mean_)

        def portfolio_variance(w):
            return np.dot(w, np.dot(cov, w))

        def weight_sum(w):
            return np.sum(w) - 1

        # Initial equal weights
        n_assets = len(mean_)
        initial_w = np.ones(n_assets) / n_assets
        bounds = [(0, 1) for _ in range(n_assets)]

        x_ = []
        for target_return in y_:
            constraints = [
                {'type': 'eq', 'fun': weight_sum},
                {'type': 'eq', 'fun': lambda w, target=target_return: portfolio_return(w) - target}
            ]
            result = minimize(portfolio_variance, initial_w, method='SLSQP', bounds=bounds, constraints=constraints)

            if result.success:
                x_.append(np.sqrt(result.fun))  # Standard deviation (risk)
                initial_w = result.x  # Warm start for the next optimization
            else:
                x_.append(np.nan)

        # Ensure the frontier aligns with the simulated data
        x_ = np.array(x_)
        y_ = np.array(y_)
        valid_indices = ~np.isnan(x_)
        return x_[valid_indices], y_[valid_indices]

    def __optimizeCVPX__(self):
        """Optimizes the efficient frontier using convex optimization with CVXPY"""
        n_assets = len(self.mean_)
        returns = self.mean_
        cov = self.cov_matrix

        target_returns = np.linspace(returns.min(), returns.max(), 20)
        x_opt, y_opt = [], []

        for target in target_returns:
            w = cp.Variable(n_assets)
            risk = cp.quad_form(w, cov)
            ret = returns @ w

            constraints = [
                cp.sum(w) == 1,
                ret >= target,
                w >= -0.2,  # Permite 20% de venda a descoberto
                w <= 1.2  # Limite superior relaxado
            ]

            prob = cp.Problem(cp.Minimize(risk), constraints)
            prob.solve(solver=cp.ECOS)

            if prob.status == 'optimal':
                x_opt.append(np.sqrt(risk.value))
                y_opt.append(target)

        return np.array(x_opt), np.array(y_opt)

    def plot_frontier(
            self,
            optimize: bool = False
    ) -> None:
        """
        Plots the Efficiency Frontier, adjusting for large numbers of pixels if necessary.
        :param optimize: Whether to optimize the efficient frontier.
        """
        plt.figure(figsize=(10, 6))
        risk_ = self.results['risk']
        return_ = self.results['return']
        sharpe = self.results['sharpe']

        plt.scatter(risk_, return_, c=sharpe, cmap='plasma', s=10, zorder=4)
        # if optimize:
        #     x, y = self.__optimize__()
        #     plt.plot(x, y, color='red', linewidth=2, label='Efficient Frontier (Optimized)', zorder=3)
        #     plt.legend()

        if optimize:
            x, y = self.__optimize__()
            # Clip to reasonable bounds
            x = np.clip(x, risk_.min(), risk_.max())
            plt.plot(x, y, color='red', linewidth=2,
                     label='Efficient Frontier', zorder=5)

        plt.xlabel('Risk (Volatility)')
        plt.ylabel('Return (Average)')
        plt.title(f'Optimized Efficiency Frontier - Sampled Pixels={self.num_pixels}')
        plt.colorbar(label='Sharpe Ratio')
        plt.grid(linestyle='--', zorder=1)

        plt.show()

        logger.info("Optimized climate efficiency frontier plotted.")

    def get_high_sharpe(
            self,
            threshold: float=1.0
    ) -> tuple[List[np.ndarray], np.ndarray]:
        """
        Returns the actual weighted variable of portfolios with Sharpe above the threshold.
        :param threshold: minimum Sharpe value
        :return: list of arrays above the weighted threshold and binary raster (xs, ys)
        """
        if self.results is None or self.weights_list is None:
            raise ValueError("Portfolios not simulated yet.")

        rk_, re_, sh_ = self.results
        high_sharpe_indices = np.where(sh_ >= threshold)[0]

        if len(high_sharpe_indices) == 0:
            logger.warning("No portfolio with Sharpe above the threshold.")
            return [], np.zeros_like(self.stack[0], dtype=int)

        selected_vars = []
        binary_raster = np.zeros_like(self.stack[0], dtype=int)
        for idx in high_sharpe_indices:
            weights = self.weights_list[idx]
            # Combines the actual time series with the weights (weighted precipitation over time)
            combined = np.dot(weights, self.series)
            selected_vars.append(combined)

        # Marks the selected pixels in the binary raster
        for y, x in self.coords:
            binary_raster[y, x] = 1

        logger.info(f"{len(selected_vars)} portfolios selected with Sharpe >= {threshold}")
        return selected_vars, binary_raster

    def create_tif_from_array(
            self,
            output_path: str,
            array: np.ndarray
    ) -> None:
        """
        Creates a GeoTIFF file from a numpy array using the first raster of the pattern as a reference.

        :param output_path: Path to save the generated GeoTIFF file.
        :param array: Numpy array representing the raster.
        """
        if not isinstance(output_path, str) or not output_path.strip():
            raise ValueError("The 'output_path' parameter must be a valid string.")
        if not isinstance(array, np.ndarray):
            raise ValueError("The 'array' parameter must be a numpy array.")

        array = validate_array_dtype(array, [np.uint8, np.int16, np.float16, np.float32, np.float64])

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


mk = Markowitz(r'G:\PycharmProjects\EfficiencyFrontier\Example\GPM-2015-2024\*.tif')
# mk.load_stack(
#     block_size=716,
#     threshold=0.0,
#     pixel_presence=0.99,
#     save_as=None,
#     memmap_path='stack_float32.dat',
#     dtype='float32'
# )
mk.load_datstack(memmap_path='stack_float32.dat', memap_shape_path='stack_shape.json', dtype='float32')
# mk.sample_pixels(num_pixels=150, normalize=True, method='standard')
# mk.sample_pixels2(normalize=True, method='standard', use_pca=False, n_components=100)
# mk.calculate_statistics()
# mk.simulate_portfolios(num_portfolios=1000)
# mk.plot_frontier(optimize=True)
# sel, bn = mk.get_high_sharpe(.7)
# mk.create_tif_from_array('output_mask.tif', bn)

