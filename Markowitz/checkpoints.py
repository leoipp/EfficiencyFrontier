from typing import Optional

import rasterio

import logging
from logging_config import logger


def check_consistency(files: list, target: Optional[str] = None, check_type: str = "crs", log: Optional[logging.Logger] = None) -> bool:
    """
    Checks if all raster files (and optionally a target raster) have consistent CRS or pixel size.

    :param files: List of raster file paths.
    :param target: Optional path to the target raster file.
    :param check_type: Type of consistency to check: "crs" or "pixel_size".
    :param log: Optional logger to log messages.
    :return: True if all rasters (and the target, if provided) have consistent CRS or pixel size, False otherwise.
    """
    if not files:
        if log:
            logger.error("No files provided to check consistency.")
        return False

    values_set = set()

    # Check target file if provided
    if target:
        try:
            with rasterio.open(target) as src:
                value = src.crs if check_type == "crs" else src.res
                if value is None and check_type == "crs":
                    if log:
                        logger.warning(f"The target raster '{target}' has no CRS defined.")
                    return False
                values_set.add(value)
        except Exception as e:
            if log:
                logger.error(f"Error reading {check_type} for target raster '{target}': {e}")
            return False

    # Check all files
    for file in files:
        try:
            with rasterio.open(file) as src:
                value = src.crs if check_type == "crs" else src.res
                if value is None and check_type == "crs":
                    if log:
                        logger.warning(f"The raster file '{file}' has no CRS defined.")
                    return False
                values_set.add(value)
        except Exception as e:
            if log:
                logger.error(f"Error reading {check_type} for raster file '{file}': {e}")
            return False

    # Check consistency
    if len(values_set) > 1:
        if log:
            logger.warning(f"Inconsistent {check_type} found among rasters: {values_set}")
        return False

    if log:
        logger.info(f"All rasters have a consistent {check_type}: {values_set.pop()}")
    return True


def check_consistent_crs(files: list, target: Optional[str] = None, log: Optional[logging.Logger] = None) -> bool:
    """
    Wrapper for checking CRS consistency.
    """
    return check_consistency(files, target, check_type="crs", log=log)


def check_consistent_pixel_size(files: list, target: Optional[str] = None, log: Optional[logging.Logger] = None) -> bool:
    """
    Wrapper for checking pixel size consistency.
    """
    return check_consistency(files, target, check_type="pixel_size", log=log)

