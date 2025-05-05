# -*- coding: utf-8 -*-
import json

import rasterio
import numpy as np
import glob
import matplotlib.pyplot as plt
from scipy.optimize import minimize

import cvxpy as cp

from typing import Optional, List, Literal

from .utils import (validate_array_dtype, normalize_weights, calculate_sharpe_ratio, count_valid_pixels_blockwise,
                    read_masked_stack_blockwise, normalize_stack, cov_shrinkage)
from .checkpoints import check_consistent_crs, check_consistent_pixel_size

from .logging_config import logger

from .PixelSampler import PixelSampler

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
        self.sampling_method = None

        # --- Output/results ---
        self.results = None
        self.stats = None
        self.pca_components = None
        self.sampled_indexes = None

        # --- Spatial properties ---
        self.coords = None
        self.num_pixels = None
        self.n_valid_pixels = None
        self.dimensions = None
        self.xy_coords = None
        self.transform = None
        self.crs = None

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
            dtype: Literal['float16', 'float32', 'float64'] = 'float32',
            save_as: Optional[str] = None,
            memmap_path: str = "stack_data.dat",
            memmap_shape_path: str = "stack_shape.json",
            stack_metadata: str = "stack_metadata.npz"
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
            logger.error("Inconsistent CRS detected. Aborting stack loading.")
            return

        # --- Check if the target raster is valid ---
        if not check_consistent_pixel_size(files, target=None, log=logger):
            logger.error("Inconsistent pixel sizes detected. Aborting stack loading.")
            return

        # --- Check mmemmap path and mmemmap shape path ---
        if not memmap_path or not memmap_shape_path:
            logger.error("Memory-mapped path and shape path must be provided.")
            raise ValueError("Memory-mapped path and shape path must be provided.")

        # --- Validate file extensions ---
        if not memmap_path.endswith('.dat'):
            logger.error("memmap_path must end with '.dat'")
            raise ValueError("memmap_path must end with '.dat'")

        if not memmap_shape_path.endswith('.json'):
            logger.error("memmap_shape_path must end with '.json'")
            raise ValueError("memmap_shape_path must end with '.json'")

        if not stack_metadata.endswith('.npz'):
            logger.error("stack_metadata must end with '.npz'")
            raise ValueError("stack_metadata must end with '.npz'")

        pixel_count, mask, self.dimensions, self.xy_coords, self.transform, self.crs = count_valid_pixels_blockwise(
            files, block_size=block_size, threshold=threshold, pixel_presence=pixel_presence, log=logger,
            save_as=save_as
        )

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
        with open(memmap_shape_path, 'w') as f:
            json.dump(
                {
                    "shape": [int(n_days), int(n_valid_pixels)],
                    "dtype": str(dtype)
                },
                f
            )

        # --- Save extra metadata to .npz ---
        np.savez_compressed(
            stack_metadata,
            dimensions=self.dimensions,
            xy_coords=self.xy_coords,
            transform=np.array(self.transform),  # Affine is not serializable directly
            crs_wkt=self.crs.to_wkt()  # Serialize CRS as WKT string
        )

        self.n_valid_pixels = self.stack.shape[1]

        logger.info(f"Stack loaded: {self.stack.shape}")
        logger.debug(f"Total NaNs in the stack: {np.isnan(self.stack).sum()}")
        logger.debug(f"Total Valid pixels in the stack: {self.stack.size - np.isnan(self.stack).sum()}")

    def load_datstack(
            self,
            memmap_path: str,
            memmap_shape_path: str,
            stack_metadata: str = str,
            dtype: Optional[str] = None
    ) -> None:
        """
        Loads the memory-mapped raster stack from a .dat file using shape metadata saved as JSON.

        :param memmap_path: Path to the memory-mapped .dat file.
        :param memmap_shape_path: Path to the JSON file containing shape and dtype metadata.
        :param stack_metadata: Path to the .npz file containing additional metadata.
        :param dtype: Optional override for dtype. If not provided, it will be read from metadata.
        """
        try:
            # Load shape and dtype from JSON
            with open(memmap_shape_path, 'r') as f:
                meta = json.load(f)
                shape = tuple(meta["shape"])
                dtype = dtype or meta["dtype"]

            # Load memory-mapped array
            self.stack = np.memmap(memmap_path, dtype=dtype, mode='r', shape=shape)
            self.n_valid_pixels = self.stack.shape[1]

            data = np.load(stack_metadata, allow_pickle=True)
            self.dimensions = tuple(data["dimensions"])
            self.xy_coords = data["xy_coords"]
            self.transform = rasterio.Affine(*data["transform"])
            self.crs = rasterio.crs.CRS.from_wkt(str(data["crs_wkt"]))

            logger.info(f"Stack loaded from {memmap_path} with shape {shape} and dtype {dtype}")
            logger.debug(f"Total NaNs in the stack: {np.isnan(self.stack).sum()}")
            logger.debug(f"Total Valid pixels in the stack: {self.stack.size - np.isnan(self.stack).sum()}")

        except FileNotFoundError:
            logger.error(f"File {memmap_path} or shape metadata {memmap_shape_path} or stack metadata "
                         f"{stack_metadata} not found.")
        except Exception as e:
            logger.error(f"An error occurred while loading the stack: {str(e)}")

    @staticmethod
    def __calculate_returns__(data: np.ndarray) -> np.ndarray:
        """Calcula retornos percentuais com tratamento de divisão por zero."""
        with np.errstate(divide='ignore', invalid='ignore'):
            returns = (data[1:] - data[:-1]) / np.where(data[:-1] == 0, 1e-6, data[:-1])
            returns[~np.isfinite(returns)] = 0 # inf, -inf, NaN to 0
        return returns

    def calculate_statistics(
            self,
            num_pixels: Optional[int] = None,
            method: Literal["manual", "ledoitwolf"] = "ledoitwolf",
            norm_axis: Optional[int] = 0,
            shrinkage_intensity: float = 0.1,
            normalize: bool = True,
            norm_method: Literal["standard", "minmax"] = 'standard',
            sampling_method: Optional[Literal['bayesian', 'kriging', 'pca', 'kmeans_only', 'pca_kmeans']] = None,
            bayesian_grouping: Optional[int] = 5,
            n_components: Optional[int] = 2
    ) -> None:
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
        # Primeiro, fazemos a amostragem dos pixels
        # Instanciando o PixelSampler e realizando a amostragem
        pixel_sampler = PixelSampler(self.stack, self.xy_coords, self.sampled_indexes, self.n_valid_pixels, self.seed,
                                     self.__calculate_returns__)
        # Verificando se sampling_method não é None antes de usar o operador 'in'
        if sampling_method is not None:
            self.sampled_indexes = pixel_sampler.sample_pixels(
                num_pixels=num_pixels,
                bayesian=sampling_method == "bayesian",
                bayesian_grouping=bayesian_grouping,
                kriging=sampling_method == "kriging",
                pca=sampling_method == "pca",
                kmeans_only=sampling_method == "kmeans_only",
                pca_kmeans=sampling_method == "pca_kmeans",
                n_components=n_components,
                normalize=normalize,
                normalize_method=norm_method,
                normalize_axis=norm_axis
            )
        else:
            # Caso sampling_method seja None, use o metodo de amostragem padrão (ex: aleatório)
            self.sampled_indexes = pixel_sampler.sample_pixels(num_pixels=num_pixels)

        self.num_pixels = num_pixels if num_pixels is not None else self.n_valid_pixels
        self.sampling_method = sampling_method if sampling_method is not None else "random"

        # Atualizando a série com os pixels amostrados
        self.series = pixel_sampler.series

        if self.series is None:
            raise ValueError("Pixels not sampled. Use .sample_pixels() first.")

        # Convert to float32 for compatibility with covariance matrix calculation
        series = self.series.astype(np.float32)

        self.mean_ = series.mean(axis=0)
        self.std_ = series.std(axis=0)
        self.cov_matrix = cov_shrinkage(series, method=method, shrinkage_intensity=shrinkage_intensity)

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

    def plot_sampled_space(
            self,
            use_hexbin: bool = False,
            gridsize: int = 60
    ) -> None:
        """
        Plota a visualização espacial de uma única amostragem de pixels.

        Parâmetros:
        - use_hexbin: se True, usa hexbin para fundo; senão, usa scatter simples.
        - gridsize: tamanho da grade para hexbin (quanto maior, menor o hexágono).
        """
        coords = np.array(self.xy_coords)
        sampled_coords = coords[self.sampled_indexes]

        plt.figure(figsize=(10, 8))

        if use_hexbin:
            # Hexbin dos dados totais (todos os pixels)
            hb = plt.hexbin(coords[:, 0], coords[:, 1], gridsize=gridsize, cmap='Greys', bins='log', mincnt=1,
                            alpha=0.5)
            plt.colorbar(hb, label='Densidade (log)')
        else:
            # Scatter de todos os pixels
            plt.scatter(coords[:, 0], coords[:, 1], alpha=0.1, color='gray', label='All Pixels', zorder=2)

        # Scatter dos pixels amostrados
        plt.scatter(sampled_coords[:, 0], sampled_coords[:, 1], color='green',
                    label=f'Sampled Pixels - {self.num_pixels / self.n_valid_pixels:.2%}',
                    s=15, zorder=3)

        plt.title(f"Spatial Sampling - Método {self.sampling_method}")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.legend()
        plt.grid(linestyle='--', zorder=1)
        plt.tight_layout()
        plt.show()

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
