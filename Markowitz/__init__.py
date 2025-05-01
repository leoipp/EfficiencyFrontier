"""
Markowitz - A package for simulating the efficiency frontier for environmental modeling.

This package provides tools for analyzing raster data using concepts inspired by
Markowitz portfolio theory, enabling the simulation of climate efficiency frontiers.

Modules:
- Markowitz: Core class for performing the analysis.
- utils: Utility functions for raster validation, resampling, and other operations.

Author: Leonardo Ippolito Rodrigues
License: MIT
"""

# Package metadata
__title__ = "Markowitz"
__version__ = "0.1-beta"
__author__ = "Leonardo Ippolito Rodrigues"
__email__ = "leoippef@gmail.com"
__license__ = "MIT"
__status__ = "Development"
__date__ = "2025-04-23"
__description__ = (
    "The Markowitz class simulates the climate efficiency frontier based on rasters, "
    "inspired by financial portfolio optimization theory."
)

# Import main components
from .Markowitz import Markowitz
from . import utils
from . import checkpoints
from . import logging_config

# Define what is exported when using `from Markowitz import *`
__all__ = ["Markowitz", "utils", "checkpoints", "logging_config"]
