from typing import Optional

import rasterio

import logging
from logging_config import logger


def check_consistent_crs(files: list, log: Optional[logging.Logger] = None) -> bool:
    """
    Checks if all raster files matching the given path pattern have the same CRS.

    :param files: list of raster files.
    :param log: Optional logger to log messages.
    :return: True if all rasters have the same CRS, False otherwise.
    """
    if not files:
        if log:
            logger.error(f"No files found for the pattern provided")
        return False

    crs_set = set()
    for file in files:
        try:
            with rasterio.open(file) as src:
                if src.crs is None:
                    if log:
                        logger.warning(f"The raster file '{file}' has no CRS defined.")
                    return False
                crs_set.add(src.crs)
        except Exception as e:
            if log:
                logger.error(f"Error reading CRS for raster file '{file}': {e}")
            return False

    if len(crs_set) > 1:
        if log:
            logger.warning(f"Inconsistent CRS found among rasters: {crs_set}")
        return False

    if log:
        logger.info(f"All rasters have a consistent CRS: {crs_set.pop()}")
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
            logger.error("No files provided to check pixel size.")
        return False

    pixel_sizes = set()
    for file in files:
        try:
            with rasterio.open(file) as src:
                pixel_size = src.res  # Resolution (pixel size) as a tuple (x_res, y_res)
                pixel_sizes.add(pixel_size)
        except Exception as e:
            if log:
                logger.error(f"Error reading pixel size for raster file '{file}': {e}")
            return False

    if len(pixel_sizes) > 1:
        if log:
            logger.warning(f"Inconsistent pixel sizes found among rasters: {pixel_sizes}")
        return False

    if log:
        logger.info(f"All rasters have a consistent pixel size: {pixel_sizes.pop()}")
    return True

