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
__version__ = "0.1b"
__author__ = "<Leonardo Ippolito Rodrigues>"
__email__ = "<leoippef@gmail.com>"
__license__ = "MIT"
__status__ = "Development"
__date__ = "2025-04-23"
__description__ = """The Markowitz class is the core of the code. Its purpose is to simulate the climate efficiency frontier based on precipitation rasters. 
The analogy would be something like an investment analyst who wants to analyze climate data as if they were financial assets."""


# Import main components
from .Markowitz import Markowitz
from . import utils

# Define what is exported when using `from Markowitz import *`
__all__ = ["Markowitz", "utils"]
