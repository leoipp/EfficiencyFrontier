from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import numpy as np
from typing import Optional, Literal

from .logging_config import logger
from .utils import normalize_stack


class PixelSampler:
    def __init__(self, stack, xy_coords, sampled_indexes, n_valid_pixels, seed, __calculate_returns__):
        self.stack = stack
        self.xy_coords = xy_coords
        self.sampled_indexes = sampled_indexes
        self.n_valid_pixels = n_valid_pixels
        self.seed = seed
        self.__calculate_returns__ = __calculate_returns__

    def sample_pixels(
            self,
            num_pixels: Optional[int] = None,
            bayesian: bool = False,
            bayesian_grouping: int = 5,
            kriging: bool = False,
            pca: bool = False,
            n_components: int = 2,
            kmeans_only: bool = False,
            pca_kmeans: bool = False,
            normalize: bool = False,
            normalize_method: Literal["standard", "minmax"] = "standard",
            normalize_axis: Optional[int] = 0,
    ) -> np.ndarray:
        """
        Main function that chooses the appropriate sampling method based on the parameters.
        """
        if self.stack is None:
            raise ValueError("Stack not loaded. Use .load_stack() first.")

        if sum([bayesian, kriging, pca, kmeans_only, pca_kmeans]) > 1:
            raise ValueError("Choose only one sampling method: bayesian, kriging, pca, kmeans_only, or pca_kmeans.")

        if num_pixels is None:
            self._select_all_pixels()
            self.sampling_method = "all"
            self.num_pixels = self.n_valid_pixels
            return np.arange(self.n_valid_pixels)

        if num_pixels > self.n_valid_pixels:
            logger.error(f"Requested {num_pixels}, but only {self.n_valid_pixels} are valid.")
            raise ValueError(f"You requested {num_pixels} pixels, but only {self.n_valid_pixels} are valid.")

        np.random.seed(self.seed)
        self.num_pixels = num_pixels  # salva para visualização

        # Métodos que requerem normalização
        needs_normalization = any([pca, pca_kmeans, kmeans_only])

        if normalize:
            if not needs_normalization:
                logger.warning("Normalization requested but selected method does not require it.")
            self.stack, _ = normalize_stack(data=self.stack, method=normalize_method, axis=normalize_axis)

        # Escolha do metodo
        if bayesian:
            self.sampling_method = "bayesian"
            self.sampled_indexes = self._sample_bayesian(num_pixels, bayesian_grouping)
        elif kriging:
            self.sampling_method = "kriging"
            self.sampled_indexes = self._sample_kriging(num_pixels)
        elif pca_kmeans:
            self.sampling_method = "pca_kmeans"
            self.sampled_indexes = self._sample_pca_kmeans(num_pixels)
        elif pca:
            self.sampling_method = "pca"
            self.sampled_indexes = self._sample_pca(num_pixels, n_components)
        elif kmeans_only:
            self.sampling_method = "kmeans"
            self.sampled_indexes = self._sample_kmeans(num_pixels)
        else:
            self.sampling_method = "random"
            self.sampled_indexes = self._sample_random(num_pixels)

        logger.info(f"Sampling method: {self.sampling_method} | Pixels sampled: {num_pixels}")

        self._apply_sampled_pixels(self.sampled_indexes)
        return self.sampled_indexes

    def _select_all_pixels(self) -> None:
        """
        Select all valid pixels for sampling.
        """
        logger.info(f"All valid pixels selected: {self.n_valid_pixels}")
        self.series = np.array(self.stack, copy=True)
        self.coords = list(range(self.n_valid_pixels))
        self.num_pixels = self.n_valid_pixels
        self.series = self.__calculate_returns__(self.series)

    def _sample_bayesian(self, num_pixels: int, bayesian_grouping: int) -> np.ndarray:
        """
        Perform sampling using Bayesian-style regional grouping.
        """
        logger.info("Sampling using Bayesian-style regional grouping (simulated).")
        region_ids = np.random.randint(0, bayesian_grouping, size=self.n_valid_pixels)
        sampled_pixels = []
        for region in np.unique(region_ids):
            region_idx = np.where(region_ids == region)[0]
            if len(region_idx) >= num_pixels // bayesian_grouping:
                sampled_pixels.extend(np.random.choice(region_idx, size=num_pixels // bayesian_grouping, replace=False))
        return np.array(sampled_pixels)

    def _sample_kriging(self, num_pixels: int) -> np.ndarray:
        """
        Perform sampling using a spatial farthest-point strategy (approximate Kriging-style).
        """
        logger.info("Sampling using farthest point strategy (Kriging-style).")
        coords = np.array(self.xy_coords)
        selected = [np.random.randint(len(coords))]  # Começa com ponto aleatório
        distances = np.full(len(coords), np.inf)

        for _ in range(1, num_pixels):
            last = coords[selected[-1]]
            # Atualiza a menor distância de cada ponto aos pontos já escolhidos
            dist_to_last = np.linalg.norm(coords - last, axis=1)
            distances = np.minimum(distances, dist_to_last)
            next_index = np.argmax(distances)
            selected.append(next_index)

        return np.array(selected)

    def _sample_pca_kmeans(self, num_pixels: int) -> np.ndarray:
        """
        Perform sampling using PCA followed by KMeans clustering.
        """
        logger.info("Sampling using PCA for dimensionality reduction followed by KMeans clustering.")
        pixel_matrix = self.stack.T
        pca_result = PCA(n_components=2).fit_transform(pixel_matrix)
        clusters = KMeans(n_clusters=5, random_state=self.seed).fit_predict(pca_result)

        sampled = []
        for c in range(5):
            idxs = np.where(clusters == c)[0]
            n_sample = min(num_pixels // 5, len(idxs))
            sampled.extend(np.random.choice(idxs, size=n_sample, replace=False))
        return np.array(sampled)

    def _sample_pca(self, num_pixels: int, n_components: int = 2) -> np.ndarray:
        """
        Perform sampling using PCA for dimensionality reduction without clustering.
        Selects pixels that are most extreme in the PCA space.
        """
        logger.info("Sampling using PCA for dimensionality reduction without KMeans clustering.")
        pixel_matrix = self.stack.T
        pca_result = PCA(n_components=n_components).fit_transform(pixel_matrix)

        # Calcula a distância de cada ponto até a origem do espaço PCA
        distances = np.linalg.norm(pca_result, axis=1)

        # Seleciona os 'num_pixels' mais distantes (mais extremos no espaço PCA)
        sampled_idxs = np.argsort(distances)[-num_pixels:]

        return sampled_idxs

    def _sample_kmeans(self, num_pixels: int) -> np.ndarray:
        """
        Perform sampling using KMeans directly on the data.
        """
        logger.info("Sampling using KMeans directly on the data.")
        pixel_matrix = self.stack.T
        clusters = KMeans(n_clusters=5, random_state=self.seed).fit_predict(pixel_matrix)

        sampled = []
        for c in range(5):
            idxs = np.where(clusters == c)[0]
            n_sample = min(num_pixels // 5, len(idxs))
            sampled.extend(np.random.choice(idxs, size=n_sample, replace=False))
        return np.array(sampled)

    def _sample_random(self, num_pixels: int) -> np.ndarray:
        """
        Perform random sampling.
        """
        logger.info("Sampling using random method.")
        return np.random.choice(self.n_valid_pixels, num_pixels, replace=False)

    def _apply_sampled_pixels(self, sampled: np.ndarray) -> None:
        """
        Apply the sampled pixels to the data and update internal state.
        """
        self.series = self.stack[:, sampled]
        self.coords = sampled.tolist()
        self.num_pixels = len(self.coords)
        self.series = self.__calculate_returns__(self.series)
        self.series = np.nan_to_num(self.series, nan=0.0, posinf=0.0, neginf=0.0)

        logger.info(f"{self.num_pixels} pixels sampled.")
