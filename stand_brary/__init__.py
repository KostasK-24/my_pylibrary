"""
Stand Brary Library
A collection of utility functions for various calculations, semiconductor 
parameter extraction, and data processing.
"""

from .core import (
    # Constants
    K_BOLTZMANN,
    Q_ELEMENTARY,
    EPSILON_OX,
    EPSILON_SI,
    NI_300K,
    
    # Utilities
    calculate_thermal_voltage,
    check_file_access,
    parse_simulation_file,
    export_vectors_and_scalars,
    calculate_centered_derivative,
    find_abs_min_or_max,
    load_text_file_by_column,
    load_scalar_map,
    load_vector_map,
    calculate_linear_interpolation,
    get_temp_key,
    
    # Process & Physics Parameters
    calculate_cox_prime,
    calculate_gamma,
    calculate_fermi_potential,
    calculate_pinch_off_voltage,
    calculate_slope_factor,
    calculate_beta_eff,
    calculate_mobility,
    
    # Current & Inversion
    calculate_ispec,
    calculate_theoretical_ispec,
    calculate_inversion_coefficient,
    calculate_surface_potential_approx,
    calculate_drain_current_strong,
    calculate_drain_current_weak,
    calculate_vds_sat,
    
    # AC & Capacitance
    calculate_cgs_ekv,
    calculate_cgd_ekv,
    calculate_cgb_ekv,
    calculate_tau_0,
    calculate_ft_saturation,
    
    # Noise & Mismatch
    calculate_flicker_noise,
    calculate_thermal_noise,
    calculate_current_mismatch,
    
    # Plotting
    plot_four_styles
)

__version__ = "1.5.0"
__author__ = "Kostas"
__email__ = "gkaralis@tuc.gr"

__all__ = [
    "K_BOLTZMANN",
    "Q_ELEMENTARY",
    "EPSILON_OX",
    "EPSILON_SI",
    "NI_300K",
    "calculate_thermal_voltage",
    "check_file_access",
    "parse_simulation_file",
    "export_vectors_and_scalars",
    "calculate_centered_derivative",
    "find_abs_min_or_max",
    "load_text_file_by_column",
    "calculate_linear_interpolation",
    "calculate_cox_prime",
    "load_scalar_map",
    "load_vector_map",
    "calculate_gamma",
    "calculate_fermi_potential",
    "calculate_pinch_off_voltage",
    "calculate_slope_factor",
    "calculate_beta_eff",
    "calculate_mobility",
    "calculate_ispec",
    "calculate_theoretical_ispec",
    "calculate_inversion_coefficient",
    "calculate_surface_potential_approx",
    "calculate_drain_current_strong",
    "calculate_drain_current_weak",
    "calculate_vds_sat",
    "calculate_cgs_ekv",
    "calculate_cgd_ekv",
    "calculate_cgb_ekv",
    "calculate_tau_0",
    "calculate_ft_saturation",
    "calculate_flicker_noise",
    "calculate_thermal_noise",
    "calculate_current_mismatch",
    "get_temp_key",
    "plot_four_styles"
]