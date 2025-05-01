from typing import Optional, List, Tuple

import numpy as np
import rasterio
import rasterio.warp

import logging
from logging_config import logger

from rasterio.windows import Window
from tqdm import tqdm


def validate_array_dtype(
        array: np.ndarray,
        valid_dtypes: Optional[list] = None,
        log: Optional[logging.Logger] = None
) -> np.ndarray:
    """
    Validates and converts the data type of array to a compatible type.

    :param array: Numpy array to be validated.
    :param valid_dtypes: List of valid data types. Defaults to [np.float16, np.float32, np.float64].
    :param log: Optional logger to record messages.
    :return: Array converted to a valid data type, if necessary.
    """
    if valid_dtypes is None:
        valid_dtypes = [np.float16, np.float32, np.float64]

    if array.dtype not in valid_dtypes:
        if log:
            log.warning(f"Converting array from dtype {array.dtype} to float32.")
        return array.astype(np.float32)
    if log:
        log.info(f"Array validated with dtype {array.dtype}.")
    return array


def normalize_weights(
        weights: np.ndarray,
        log: Optional[logging.Logger] = None
) -> np.ndarray:
    """
    Normalizes weights so that their sum equals 1.

    :param weights: Array of weights.
    :param log: Optional logger to record messages.
    :return: Array of normalized weights.
    """
    normalized_weights = weights / np.sum(weights)
    if log:
        logger.info(f"Normalized weights: sum = {np.sum(normalized_weights)}.")
    return normalized_weights


def calculate_sharpe_ratio(
        return_mean: float,
        risk: float,
        log: Optional[logging.Logger] = None
) -> float:
    """
    Calculates the Sharpe ratio given the mean return and risk.

    :param return_mean: Portfolio mean return.
    :param risk: Portfolio risk (standard deviation).
    :param log: Optional logger to record messages.
    :return: Sharpe ratio.
    """
    if risk == 0:
        if log:
            logger.warning("Risk is 0. Returning Sharpe ratio as 0.")
        return 0
    sharpe_ratio = return_mean / risk
    if log:
        logger.info(f"Calculated Sharpe ratio: {sharpe_ratio}.")
    return sharpe_ratio


def resample_raster(
        input_path: str,
        output_path: str,
        target_pixel_size: tuple,
        log: Optional[logging.Logger] = None
) -> None:
    """
    Resamples a raster to a specified pixel size.

    :param input_path: Path to the input raster file.
    :param output_path: Path to save the resampled raster.
    :param target_pixel_size: Target pixel size as a tuple (x_res, y_res).
    :param log: Optional logger to log messages.
    """
    try:
        with rasterio.open(input_path) as src:
            # Check for invalid or extreme values in the raster
            data = src.read(1)  # Read the first band
            if log:
                log.info(f"Original raster statistics: min={data.min()}, max={data.max()}, mean={data.mean()}.")

            # Replace extreme values with NaN (or another placeholder)
            invalid_mask = (data > 1e10) | (data < -1e10)  # Example threshold for invalid values
            if invalid_mask.any():
                if log:
                    log.warning(f"Raster contains extreme values. Replacing {invalid_mask.sum()} invalid pixels with NaN.")
                data[invalid_mask] = np.nan

            # Normalize data if necessary
            if np.nanmax(data) - np.nanmin(data) == 0:
                if log:
                    log.error("Raster has no variation (all values are the same). Resampling cannot proceed.")
                raise ValueError("Raster has no variation (all values are the same).")

            # Update the raster with cleaned data
            transform = src.transform
            new_transform = rasterio.Affine(
                target_pixel_size[0], transform.b, transform.c,
                transform.d, -target_pixel_size[1], transform.f
            )
            new_width = int((src.bounds.right - src.bounds.left) / target_pixel_size[0])
            new_height = int((src.bounds.top - src.bounds.bottom) / target_pixel_size[1])

            # Ensure dimensions are valid
            if new_width <= 0 or new_height <= 0:
                if log:
                    log.error(f"Invalid dimensions for resampling raster '{input_path}': width={new_width}, height={new_height}.")
                raise ValueError(f"Invalid dimensions for resampling raster '{input_path}': width={new_width}, height={new_height}.")

            meta = src.meta.copy()
            meta.update({
                "transform": new_transform,
                "width": new_width,
                "height": new_height
            })

            with rasterio.open(output_path, "w", **meta) as dst:
                rasterio.warp.reproject(
                    source=data,
                    destination=rasterio.band(dst, 1),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=new_transform,
                    dst_crs=src.crs,
                    resampling=rasterio.warp.Resampling.bilinear
                )

        if log:
            log.info(f"Raster resampled and saved to {output_path} with pixel size {target_pixel_size}.")
    except Exception as e:
        if log:
            log.error(f"Error resampling raster '{input_path}': {e}")
        raise


def count_valid_pixels_blockwise(
    files: List[str],
    block_size: int = 1024,
    threshold: Optional[float] = 0,
    pixel_presence: Optional[float] = 0.7,
    log: Optional[logging.Logger] = None,
    save_as: Optional[str] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Counts how many times each pixel was valid (> threshold) across all rasters, using blockwise reading.
    Also calculates the final mask of minimum valid pixel presence.

    :param files: List of paths to rasters.
    :param block_size: Block size for window reading.
    :param threshold: Minimum threshold to consider a pixel as valid.
    :param pixel_presence: Minimum proportion of valid data presence to consider the pixel as valid.
    :param log: Optional logger.
    :param save_as: Base path to save the count matrix and mask (without extension).
    :return: Tuple with (absolute count matrix, final binary mask).
    """
    with rasterio.open(files[0]) as src:
        height, width = src.height, src.width
        valid_counts = np.zeros((height, width), dtype=np.uint16)
        if log:
            log.info(f"Raster dimensions: height={height}, width={width}.")

    for path in tqdm(files, desc="Window - validating pixels", ncols=100):
        with rasterio.open(path, 'r', sharing=True) as src:
            for i in range(0, height, block_size):
                for j in range(0, width, block_size):
                    h = min(block_size, height - i)
                    w = min(block_size, width - j)
                    window = Window(j, i, w, h)
                    arr = src.read(1, window=window, masked=False)
                    arr = validate_array_dtype(arr, log=None)
                    valid_counts[i:i+h, j:j+w] += arr > threshold

    # Calculate proportion and binary mask
    valid_ratio = valid_counts / len(files)
    final_mask = valid_ratio >= pixel_presence

    if log:
        total_valid = np.sum(final_mask)
        log.info(f"Valid pixels with presence >= {pixel_presence:.0%}: {total_valid}")

    # Save results, if requested
    if save_as:
        np.save(f"{save_as}_counts.npy", valid_counts)
        np.save(f"{save_as}_mask.npy", final_mask.astype(np.uint8))
        if log:
            log.info(f"Files saved: {save_as}_counts.npy and {save_as}_mask.npy")

    return valid_counts, final_mask


def read_masked_stack_blockwise(
    file_path: str,
    mask: np.ndarray,
    block_size: int = 1024,
    dtype: str = 'float32'
) -> np.ndarray:
    """
    Lê um raster aplicando uma máscara, usando leitura por blocos para emitter estouro de memória.

    :param file_path: Caminho para o arquivo raster.
    :param mask: Máscara booleana 2D (True para pixels válidos).
    :param block_size: Tamanho do bloco para leitura (em pixels).
    :param dtype: Tipo de dado final (ex: 'float16', 'float32').
    :return: 1D array com apenas os pixels válidos aplicados da máscara.
    """
    with rasterio.open(file_path) as src:
        height, width = src.height, src.width
        mask_flat = mask.ravel()
        n_pixels = mask_flat.sum()
        day_data = np.zeros((n_pixels,), dtype=dtype)

        cursor = 0

        for i in range(0, height, block_size):
            for j in range(0, width, block_size):
                h = min(block_size, height - i)
                w = min(block_size, width - j)
                window = Window(j, i, w, h)

                block = src.read(1, window=window)
                block_flat = block.ravel()

                mask_block = mask[i:i+h, j:j+w].ravel()
                valid_block_values = block_flat[mask_block]

                n_valid = len(valid_block_values)
                day_data[cursor:cursor + n_valid] = valid_block_values.astype(dtype)
                cursor += n_valid

        assert cursor == n_pixels, f"Cursor ({cursor}) doesn't match expected pixels ({n_pixels})"

    return day_data


def normalize_stack(
        data,
        method='standard',
        mean: Optional[np.ndarray]=None,
        std: Optional[np.ndarray]=None,
        min_val: Optional[float]=None,
        max_val: Optional[float]=None,
        axis: Optional[int]=0
):
    """
    Normaliza dados stackados, podendo usar estatísticas já fornecidas.

    Parâmetros:
    - data: np.ndarray (2D) [n_amostras, n_variaveis]
    - method: str, 'standard' (z-score) ou 'minmax' (0 a 1)
    - mean, std: para metodo 'standard' (podem ser None, calcula se não fornecido)
    - min_val, max_val: para metodo 'minmax' (idem)

    Retorna:
    - data_normalized: np.ndarray normalizado
    - stats: dicionário com parâmetros usados
    """
    if not isinstance(data, np.ndarray):
        raise ValueError("Os dados precisam ser um np.ndarray")

    if method == 'standard':
        if mean is None:
            mean = np.mean(data, axis=axis, keepdims=True)
        if std is None:
            std = np.std(data, axis=axis, keepdims=True)

        data_normalized = (data - mean) / std
        stats = {'mean': mean, 'std': std}

    elif method == 'minmax':
        if min_val is None:
            min_val = np.min(data, axis=axis, keepdims=True)
        if max_val is None:
            max_val = np.max(data, axis=axis, keepdims=True)

        data_normalized = (data - min_val) / (max_val - min_val)
        stats = {'min': min_val, 'max': max_val}

    else:
        raise ValueError("Método de normalização não reconhecido. Use 'standard' ou 'minmax'.")

    return data_normalized, stats
