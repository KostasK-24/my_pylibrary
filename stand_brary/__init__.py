"""
Stand Brary Library
A collection of utility functions for various calculations, semiconductor 
parameter extraction, and data processing.
"""

from .core import (
    # Constants
    K_BOLTZMANN, Q_ELEMENTARY, EPSILON_OX, EPSILON_SI, NI_300K,
    
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
    load_scalar_data_from_dir,
    load_vector_data_from_dir,
    calculate_linear_interpolation,
    get_temp_key,
    get_temp_from_filename,
    find_col_index,
    
    # Process & Physics Parameters
    calculate_cox_prime, calculate_gamma, calculate_fermi_potential,
    calculate_vto, calculate_n0, 
    calculate_pinch_off_voltage, calculate_slope_factor,
    calculate_beta_eff, calculate_mobility,
    
    # Current & Inversion
    calculate_normalization_charge_q0,
    calculate_ispec, calculate_theoretical_ispec,
    calculate_inversion_coefficient, 
    calculate_normalized_charge_ekv, calculate_normalized_current_ekv,
    calculate_surface_potential_approx, # Alias for backward compatibility
    calculate_drain_current_ekv_all_regions,
    calculate_drain_current_strong, calculate_drain_current_weak,
    calculate_vds_sat, calculate_early_voltage,
    
    # Transconductances
    calculate_gms_ekv, calculate_gmg_ekv,
    calculate_gmd_ekv, calculate_gmb_ekv,
    
    # AC & Capacitance
    calculate_cgs_ekv, calculate_cgd_ekv, calculate_cgb_ekv,
    calculate_cbs_ekv, calculate_cbd_ekv,
    calculate_tau_0, calculate_tau_qs, 
    calculate_ft_saturation, calculate_ft_general,
    
    # Noise & Mismatch
    calculate_flicker_noise, calculate_thermal_noise,
    calculate_current_mismatch, calculate_voltage_mismatch_variance,
    
    # Extended Extraction Helpers
    calculate_gms_over_id, calculate_gmg_over_id,
    
    # Plotting & Reporting
    plot_four_styles,
    plot_family_of_curves,
    export_current_plot_to_tex,
    inject_plots_into_tex
)

__version__ = "1.18.0"
__author__ = "Kostas"
__email__ = "gkaralis@tuc.gr"

__all__ = [
    # Constants
    "K_BOLTZMANN", "Q_ELEMENTARY", "EPSILON_OX", "EPSILON_SI", "NI_300K",
    
    # Utilities
    "calculate_thermal_voltage", "check_file_access", "parse_simulation_file",
    "export_vectors_and_scalars", "load_scalar_map", "load_vector_map",
    "load_scalar_data_from_dir", "load_vector_data_from_dir",
    "calculate_centered_derivative", "find_abs_min_or_max", "load_text_file_by_column",
    "calculate_linear_interpolation", "get_temp_key", "get_temp_from_filename", "find_col_index",
    
    # Physics - Basic
    "calculate_cox_prime", "calculate_gamma", "calculate_fermi_potential",
    "calculate_vto", "calculate_n0", 
    "calculate_pinch_off_voltage", "calculate_slope_factor", 
    "calculate_beta_eff", "calculate_mobility", 
    
    # Physics - Current/Inversion
    "calculate_normalization_charge_q0",
    "calculate_ispec", "calculate_theoretical_ispec",
    "calculate_inversion_coefficient", 
    "calculate_normalized_charge_ekv", "calculate_normalized_current_ekv",
    "calculate_surface_potential_approx",
    "calculate_drain_current_ekv_all_regions",
    "calculate_drain_current_strong", "calculate_drain_current_weak", 
    "calculate_vds_sat", "calculate_early_voltage",
    
    # Physics - Transconductances
    "calculate_gms_ekv", "calculate_gmg_ekv",
    "calculate_gmd_ekv", "calculate_gmb_ekv",
    
    # Physics - Capacitance/AC
    "calculate_cgs_ekv", "calculate_cgd_ekv", "calculate_cgb_ekv",
    "calculate_cbs_ekv", "calculate_cbd_ekv",
    "calculate_tau_0", "calculate_tau_qs", 
    "calculate_ft_saturation", "calculate_ft_general", 
    
    # Physics - Noise/Mismatch
    "calculate_flicker_noise", "calculate_thermal_noise", 
    "calculate_current_mismatch", "calculate_voltage_mismatch_variance",
    
    # Helpers
    "calculate_gms_over_id", "calculate_gmg_over_id",
    
    # Plotting
    "plot_four_styles", "plot_family_of_curves", 
    "export_current_plot_to_tex", "inject_plots_into_tex"
]