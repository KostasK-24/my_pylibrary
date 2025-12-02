"""
Stand Brary Library
A collection of utility functions for various calculations and data processing.
"""

from .core import (
    K_BOLTZMANN,
    Q_ELEMENTARY,
    calculate_thermal_voltage,
    check_file_access,
    calculate_centered_derivative,
    find_abs_min_or_max,
    load_text_file_by_column
)

__version__ = "1.0.0"
__author__ = "Kostas"
__email__ = "gkaralis@tuc.gr"

__all__ = [
    "K_BOLTZMANN",
    "Q_ELEMENTARY", 
    "calculate_thermal_voltage",
    "check_file_access",
    "calculate_centered_derivative",
    "find_abs_min_or_max",
    "load_text_file_by_column"
]
