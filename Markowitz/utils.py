from typing import Optional

import numpy as np
import rasterio
import rasterio.warp

import logging
from logging_config import logger

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


import glob
import os

output_path = r"C:\Users\Leonardo\PycharmProjects\EfficiencyFrontier\Example\Resample-teste"
files = glob.glob(r"C:\Users\Leonardo\PycharmProjects\EfficiencyFrontier\Example\*.tif")
for file in files:
    print(file)
    otp = os.path.join(output_path, os.path.basename(file))
    resample_raster(
        input_path=file,
        output_path=otp,
        target_pixel_size=(0.0002, 0.0002),
        log=logger
    )