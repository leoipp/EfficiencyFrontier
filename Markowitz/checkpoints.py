from typing import Optional
import rasterio
import logging


def check_consistency(
        files: list,
        target: Optional[str] = None,
        check_type: str = "crs",
        log: Optional[logging.Logger] = None
) -> bool:
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
            log.error("No files provided to check consistency.")
        return False

    values_set = set()
    all_paths = [target] + files if target else files

    for path in all_paths:
        try:
            with rasterio.open(path) as src:
                value = src.crs if check_type == "crs" else src.res
                if value is None and check_type == "crs":
                    if log:
                        log.warning(f"The raster '{path}' has no CRS defined.")
                    return False
                values_set.add(value)
        except Exception as e:
            if log:
                log.error(f"Error reading {check_type} for raster '{path}': {e}")
            return False

    if len(values_set) > 1:
        if log:
            log.warning(f"Inconsistent {check_type} found among rasters: {values_set}")
        return False

    if log:
        log.info(f"All rasters have a consistent {check_type}: {values_set.pop()}")
    return True


def check_consistent_crs(
        files: list,
        target: Optional[str] = None,
        log: Optional[logging.Logger] = None
) -> bool:
    """
    Wrapper for checking CRS consistency.
    """
    return check_consistency(files, target, check_type="crs", log=log)


def check_consistent_pixel_size(
        files: list,
        target: Optional[str] = None,
        log: Optional[logging.Logger] = None
) -> bool:
    """
    Wrapper for checking pixel size consistency.
    """
    return check_consistency(files, target, check_type="pixel_size", log=log)
