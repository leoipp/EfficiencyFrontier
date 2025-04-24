from typing import Optional

import numpy as np
import logging
import rasterio

# Configuração básica do logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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


def check_raster_crs(raster_path: str, log: Optional[logging.Logger] = None) -> bool:
    """
    Checks if a raster file has a valid CRS (Coordinate Reference System).

    :param raster_path: Path to the raster file.
    :param log: Optional logger to log messages.
    :return: True if the raster has a CRS, False otherwise.
    """
    import rasterio

    try:
        with rasterio.open(raster_path) as src:
            if src.crs is None:
                if log:
                    log.warning(f"The raster file '{raster_path}' has no CRS defined.")
                return False
            if log:
                log.info(f"The raster file '{raster_path}' has a valid CRS: {src.crs}.")
            return True
    except Exception as e:
        if log:
            log.error(f"Error checking CRS for raster file '{raster_path}': {e}")
        return False


def check_consistent_crs(files: list, log: Optional[logging.Logger] = None) -> bool:
    """
    Checks if all raster files matching the given path pattern have the same CRS.

    :param files: list of raster files.
    :param log: Optional logger to log messages.
    :return: True if all rasters have the same CRS, False otherwise.
    """
    if not files:
        if log:
            log.error(f"No files found for the pattern provided")
        return False

    crs_set = set()
    for file in files:
        try:
            with rasterio.open(file) as src:
                if src.crs is None:
                    if log:
                        log.warning(f"The raster file '{file}' has no CRS defined.")
                    return False
                crs_set.add(src.crs)
        except Exception as e:
            if log:
                log.error(f"Error reading CRS for raster file '{file}': {e}")
            return False

    if len(crs_set) > 1:
        if log:
            log.warning(f"Inconsistent CRS found among rasters: {crs_set}")
        return False

    if log:
        log.info(f"All rasters have a consistent CRS: {crs_set.pop()}")
    return True


def check_consistent_pixel_size(files: list, log: Optional[logging.Logger] = None) -> bool:
    """
    Checks if all raster files in the list have the same pixel size.

    :param files: List of raster file paths.
    :param log: Optional logger to log messages.
    :return: True if all rasters have the same pixel size, False otherwise.
    """
    if not files:
        if log:
            log.error("No files provided to check pixel size.")
        return False

    pixel_sizes = set()
    for file in files:
        try:
            with rasterio.open(file) as src:
                pixel_size = src.res  # Resolution (pixel size) as a tuple (x_res, y_res)
                pixel_sizes.add(pixel_size)
        except Exception as e:
            if log:
                log.error(f"Error reading pixel size for raster file '{file}': {e}")
            return False

    if len(pixel_sizes) > 1:
        if log:
            log.warning(f"Inconsistent pixel sizes found among rasters: {pixel_sizes}")
        return False

    if log:
        log.info(f"All rasters have a consistent pixel size: {pixel_sizes.pop()}")
    return True

