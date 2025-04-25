import os.path
from typing import Optional

import numpy as np
import rasterio
import rasterio.warp  # Certifique-se de importar o submódulo warp corretamente

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
            # Check if CRS is in degrees
            if src.crs.is_geographic:
                if log:
                    log.warning(f"The raster '{input_path}' is in a geographic CRS (degrees). "
                                f"Consider reprojecting to a projected CRS for accurate resampling.")

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
                    source=rasterio.band(src, 1),
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

import glob
import os

# Define o diretório de saída
output_dir = 'C:/Users/Leonardo/PycharmProjects/EfficiencyFrontier/Example/Resample-teste'

# Garante que o diretório de saída exista
os.makedirs(output_dir, exist_ok=True)

# Obtém os arquivos .tif
files = sorted(glob.glob('C:/Users/Leonardo/PycharmProjects/EfficiencyFrontier/Example/*.tif'))

for file in files:
    print(os.path.normpath(file))
    # Define o caminho de saída para o arquivo resampleado
    output_path = os.path.join(output_dir, os.path.basename(file).replace('.tif', '_resampled.tif'))

    # Chama a função de resample
    resample_raster(file, output_path, (0.0005, 0.0005), logger)

