from typing import Optional, List, Tuple

import numpy as np
import rasterio
import rasterio.warp

import logging
from logging_config import logger

from rasterio.windows import Window
from tqdm import tqdm


def validate_array_dtype(array: np.ndarray, valid_dtypes: list, log: Optional[logging.Logger] = None) -> np.ndarray:
    """
    Valida e converte o tipo de dado de um array para um tipo compatível.

    :param array: Array numpy a ser validado.
    :param valid_dtypes: Lista de tipos de dados válidos.
    :param log: Logger opcional para registrar mensagens.
    :return: Array convertido para um tipo de dado válido, se necessário.
    """
    if array.dtype not in valid_dtypes:
        if log:
            logger.warning(f"Convertendo array de dtype {array.dtype} para float32.")
        return array.astype(np.float32)
    if log:
        logger.info(f"Array validado com dtype {array.dtype}.")
    return array


def normalize_weights(weights: np.ndarray, log: Optional[logging.Logger] = None) -> np.ndarray:
    """
    Normaliza os pesos para que a soma seja igual a 1.

    :param weights: Array de pesos.
    :param log: Logger opcional para registrar mensagens.
    :return: Array de pesos normalizados.
    """
    normalized_weights = weights / np.sum(weights)
    if log:
        logger.info(f"Pesos normalizados: soma = {np.sum(normalized_weights)}.")
    return normalized_weights


def calculate_sharpe_ratio(return_mean: float, risk: float, log: Optional[logging.Logger] = None) -> float:
    """
    Calcula o índice de Sharpe dado o retorno médio e o risco.

    :param return_mean: Retorno médio do portfólio.
    :param risk: Risco (desvio padrão) do portfólio.
    :param log: Logger opcional para registrar mensagens.
    :return: Índice de Sharpe.
    """
    if risk == 0:
        if log:
            logger.warning("Risco é 0. Retornando índice de Sharpe como 0.")
        return 0
    sharpe_ratio = return_mean / risk
    if log:
        logger.info(f"Índice de Sharpe calculado: {sharpe_ratio}.")
    return sharpe_ratio


def resample_raster(input_path: str, output_path: str, target_pixel_size: tuple, log: Optional[logging.Logger] = None) -> None:
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


def convert_to_float16(array: np.ndarray, log: Optional[logging.Logger] = None) -> np.ndarray:
    """
    Converte um array numpy para o tipo float32.

    :param array: Array numpy a ser convertido.
    :param log: Logger opcional para registrar mensagens.
    :return: Array convertido para float16.
    """
    if array.dtype != np.float16:
        if log:
            log.info(f"Convertendo array de dtype {array.dtype} para float16.")
        return array.astype(np.float16)
    if log:
        log.info("Array já está no tipo float16.")
    return array


def count_valid_pixels_blockwise(
    files: List[str],
    block_size: int = 1024,
    threshold: Optional[float] = 0,
    pixel_presence: Optional[float] = 0.7,
    log: Optional[logging.Logger] = None,
    save_as: Optional[str] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Conta quantas vezes cada pixel foi válido (> threshold) em todos os rasters, usando leitura em blocos.
    Também calcula a máscara final de presença mínima de pixels válidos.

    :param files: Lista de caminhos para rasters.
    :param block_size: Tamanho do bloco para leitura por janela.
    :param threshold: Limite mínimo para considerar um pixel como válido.
    :param pixel_presence: Proporção mínima de presença de dados válidos para considerar o pixel como válido.
    :param log: Logger opcional.
    :param save_as: Caminho base para salvar a matriz de contagem e a máscara (sem extensão).
    :return: Tuple com (matriz de contagem absoluta, máscara final binária).
    """
    with rasterio.open(files[0]) as src:
        height, width = src.height, src.width
        valid_counts = np.zeros((height, width), dtype=np.uint16)
        if log:
            log.info(f"Dimensões do raster: altura={height}, largura={width}.")

    for path in tqdm(files, desc="Contando pixels válidos", ncols=100):
        with rasterio.open(path, 'r', sharing=True) as src:
            for i in range(0, height, block_size):
                for j in range(0, width, block_size):
                    h = min(block_size, height - i)
                    w = min(block_size, width - j)
                    window = Window(j, i, w, h)
                    arr = src.read(1, window=window, masked=False)
                    arr = convert_to_float16(arr, log=None)
                    valid_counts[i:i+h, j:j+w] += arr > threshold

    # Calcula proporção e máscara binária
    valid_ratio = valid_counts / len(files)
    final_mask = valid_ratio >= pixel_presence

    if log:
        total_valid = np.sum(final_mask)
        log.info(f"Pixels válidos com presença >= {pixel_presence:.0%}: {total_valid}")

    # Salvar resultados, se solicitado
    if save_as:
        np.save(f"{save_as}_counts.npy", valid_counts)
        np.save(f"{save_as}_mask.npy", final_mask.astype(np.uint8))
        if log:
            log.info(f"Arquivos salvos: {save_as}_counts.npy e {save_as}_mask.npy")

    return valid_counts, final_mask
