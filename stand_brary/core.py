"""
Stand Brary Library - Core Physics & Utility Functions

This module provides essential tools for semiconductor parameter extraction, 
numerical analysis, file handling, and plotting.

CONTEXT OF EQUATIONS & FUNCTIONS:

1. FUNDAMENTAL CONSTANTS:
   - K_BOLTZMANN (k): 1.38e-23 J/K [Source: 15]
   - Q_ELEMENTARY (q): 1.602e-19 C [Source: 15]
   - EPSILON_OX: 3.45e-11 F/m [Source: 9]
   - EPSILON_SI: 1.04e-10 F/m [Source: 9]

2. PHYSICS CALCULATIONS:
   - Thermal Voltage (Ut): Ut = kT/q. [Source: 20]
   - Specific Current (Ispec): The normalization current in the EKV model.
     Formula: Ispec = 2 * n * Ut^2 * (W/L) * Beta. [Source: 19]
   - Inversion Coefficient (IC): Measures the level of inversion.
     Formula: IC = Id / Ispec. [Source: 19]
   - Surface Potential (qs): Normalized surface potential approximation.
     Formula: qs = sqrt(0.25 + IC) - 0.5. [Source: 35]
   - Effective Gain Factor (Beta_eff): Extracted from Ispec or Source Current.
   - Mobility (Mu): Carrier mobility extracted from Beta_eff.
     Formula: Mu = (Beta_eff * L) / (C'ox * W). [Source: 19]

3. CAPACITANCE MODELING (EKV):
   - Cgs: Normalized Gate-Source Capacitance. [Source: 85]
   - Cgd: Normalized Gate-Drain Capacitance. [Source: 77]
   - Cgb: Normalized Gate-Bulk Capacitance. [Source: 80]

4. NUMERICAL & PLOTTING TOOLS:
   - Centered Derivative, Linear Interpolation, Smart Loader.
   - Plotting tools for IEEE/Scientific visualization.
"""

import os
import math
# New imports for plotting
try:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import ScalarFormatter
    import numpy as np
except ImportError:
    print("Warning: Matplotlib or Numpy not found. Plotting functions will fail.")

# --- Fundamental Physical Constants ---

# Boltzmann's constant (J/K) [Source: 15]
K_BOLTZMANN = 1.380649e-23

# Elementary charge (C) [Source: 15]
Q_ELEMENTARY = 1.602176634e-19

# Permittivity of Oxide (F/m) [Source: 9]
EPSILON_OX = 3.45e-11

# Permittivity of Silicon (F/m) [Source: 9]
EPSILON_SI = 1.04e-10

# Intrinsic Carrier Concentration (cm^-3 at 300K) [Source: 13]
NI_300K = 1.19e10

# --- Reusable Utility Functions ---

def calculate_thermal_voltage(temperature_celsius):
    """
    Calculates the thermal voltage (Ut) for a given temperature in Celsius.
    Ut = (k * T_Kelvin) / q [Source: 20]
    
    Args:
        temperature_celsius (float): Temperature in degrees Celsius.
        
    Returns:
        tuple: (float: Thermal voltage (Ut) in Volts, float: T_kelvin)
    """
    T_kelvin = temperature_celsius + 273.15
    Ut = (K_BOLTZMANN * T_kelvin) / Q_ELEMENTARY
    return Ut, T_kelvin

def check_file_access(filepath, mode='r'):
    """Checks if a file exists and if the user has the required permissions."""
    if not os.path.exists(filepath):
        print(f"File Check Error: File not found at {filepath}")
        return False
        
    if mode == 'r' and not os.access(filepath, os.R_OK):
        print(f"File Check Error: No permission to read file at {filepath}")
        return False
    elif mode == 'w' and not os.access(filepath, os.W_OK):
        print(f"File Check Error: No permission to write to file at {filepath}")
        return False
        
    return True

def calculate_centered_derivative(y_data, x_data):
    """
    Calculates the centered-difference derivative of y with respect to x.
    dY/dX = (y[i+1] - y[i-1]) / (x[i+1] - x[i-1])
    """
    n = len(y_data)
    if n != len(x_data):
        print("Derivative Error: Y and X data lists must have the same length.")
        return [None] * n
        
    derivative_list = [None] * n
    
    for i in range(1, n - 1):
        delta_y = y_data[i+1] - y_data[i-1]
        delta_x = x_data[i+1] - x_data[i-1]
        
        if delta_x != 0:
            derivative_list[i] = delta_y / delta_x
        else:
            derivative_list[i] = 0.0
            
    return derivative_list

def find_abs_min_or_max(data_list, find_min=True):
    """Finds the absolute value of the minimum or maximum value in a list."""
    valid_data = [d for d in data_list if isinstance(d, (int, float))]
    
    if not valid_data:
        return 0.0
        
    if find_min:
        return abs(min(valid_data))
    else:
        return abs(max(valid_data))

def load_text_file_by_column(filepath):
    """
    Opens a space-delimited .txt file, and loads all data into a 
    dictionary of lists (vectors).
    """
    if not check_file_access(filepath, mode='r'):
        return None
        
    data_vectors = {}
    current_column_names = []
    
    try:
        with open(filepath, 'r') as f:
            print(f"Starting parse of: {os.path.basename(filepath)}")
            
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue

                try:
                    float(parts[0])
                    is_data_row = True
                except ValueError:
                    is_data_row = False

                if is_data_row:
                    if not current_column_names:
                        print(f"Skipping data line (no header seen yet): {line.strip()}")
                        continue
                        
                    if len(parts) == len(current_column_names):
                        try:
                            for i, name in enumerate(current_column_names):
                                data_vectors[name].append(float(parts[i]))
                        except ValueError:
                            print(f"Skipping malformed data line: {line.strip()}")
                    else:
                        print(f"Skipping line (column count mismatch): {line.strip()}")
                else:
                    print(f"\nFound new header row:")
                    print(f"  -> {line.strip()}")
                    current_column_names = parts
                    for name in current_column_names:
                        if name not in data_vectors:
                            data_vectors[name] = []
                            print(f"    - Initializing new vector: '{name}'")

            print("\nFile parsing complete.")
            return data_vectors

    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def calculate_linear_interpolation(x0, y0, x1, y1, target_y):
    """Finds x corresponding to target_y using linear interpolation."""
    if y1 == y0:
        return None  
    return x0 + (target_y - y0) * (x1 - x0) / (y1 - y0)

def get_temp_key(val):
    """Returns rounded integer temperature."""
    return int(round(val))


# --- PHYSICS & MODELING FUNCTIONS ---

def calculate_cox_prime(t_ox):
    """
    Calculates Oxide Capacitance per unit area (Cox').
    Formula: C'ox = E_ox / T_ox [Source: 8]
    """
    if t_ox <= 0: return 0.0
    return EPSILON_OX / t_ox

def calculate_gamma(n_sub, cox_prime):
    """
    Calculates Body Effect Parameter (Gamma).
    Formula: Gamma = sqrt(2 * q * E_si * N_sub) / C'ox [Source: 8]
    """
    if cox_prime == 0: return 0.0
    numerator = math.sqrt(2 * Q_ELEMENTARY * EPSILON_SI * n_sub)
    return numerator / cox_prime

def calculate_fermi_potential(ut, n_sub, ni=NI_300K):
    """
    Calculates Fermi Potential (Phi).
    Formula: Phi = 2 * Ut * ln(N_sub / ni) [Source: 8]
    """
    if ni == 0 or n_sub <= 0: return 0.0
    return 2 * ut * math.log(n_sub / ni)

def calculate_pinch_off_voltage(vg, vto, n):
    """
    Calculates Pinch-off Voltage (Vp).
    Formula: Vp ~= (Vg - Vto) / n [Source: 11]
    """
    if n == 0: return 0.0
    return (vg - vto) / n

def calculate_slope_factor(gamma, vp, phi):
    """
    Calculates Slope Factor (n).
    Formula: n = 1 + gamma / (2 * sqrt(Vp + Phi)) [Source: 17]
    """
    denom_inner = vp + phi
    if denom_inner <= 0: return 1.0 # Safety fallback
    return 1 + (gamma / (2 * math.sqrt(denom_inner)))

def calculate_ispec(max_derivative, thermal_voltage):
    """
    Calculates extracted Ispec from transconductance data.
    Formula: Ispec = (2 * max_derivative * Ut)^2
    """
    return (2 * max_derivative * thermal_voltage) ** 2

def calculate_theoretical_ispec(n, ut, mobility, cox_prime, w, l):
    """
    Calculates Theoretical Specific Current (Ispec).
    Formula: Ispec = 2 * n * Ut^2 * Beta_tech * (W/L) [Source: 19]
    """
    if l == 0: return 0.0
    beta_tech = mobility * cox_prime
    return 2 * n * (ut**2) * beta_tech * (w / l)

def calculate_inversion_coefficient(id_abs, ispec):
    """
    Calculates Inversion Coefficient (IC = Id / Ispec).
    [Source: 19]
    """
    if ispec == 0:
        return 0.0
    return id_abs / ispec

def calculate_surface_potential_approx(ic):
    """
    Calculates normalized surface potential (qs) from IC.
    Formula: qs = sqrt(0.25 + IC) - 0.5 [Source: 35]
    """
    if ic < -0.25: return 0.0
    return math.sqrt(0.25 + ic) - 0.5

# --- Capacitance Functions ---

def calculate_cgs_ekv(qs):
    """
    Calculates normalized Gate-Source Capacitance (cgs).
    Formula: cgs = (qs/3) * (2qs + 3) / (qs + 1)^2 [Source: 85]
    (Simplified Saturation/Weak Inv form)
    """
    if qs <= -1: 
        return 0.0
    term1 = qs / 3.0
    term2 = (2.0 * qs + 3.0)
    term3 = (qs + 1.0) ** 2
    return term1 * (term2 / term3)

def calculate_cgd_ekv(qs, qd):
    """
    Calculates normalized Gate-Drain Capacitance (cgd).
    Formula: cgd = (qd/3) * (2qd + 4qs + 3) / (qs + qd + 1)^2 [Source: 77]
    """
    denom = (qs + qd + 1.0)**2
    if denom == 0: return 0.0
    term = (2.0 * qd + 4.0 * qs + 3.0)
    return (qd / 3.0) * (term / denom)

def calculate_cgb_ekv(n, cgs, cgd=0.0):
    """
    Calculates normalized Gate-Bulk Capacitance (cgb).
    Formula: cgb = ((n-1)/n) * (1 - cgs - cgd) [Source: 80]
    """
    if n == 0: 
        return 0.0 
    term_n = (n - 1.0) / n
    return term_n * (1.0 - cgs - cgd)

# --- Process Extraction Functions ---

def calculate_beta_eff(isource, n0, ut):
    """
    Calculates Beta_eff (Total Device Beta).
    Formula: Beta_eff = Is / (n0 * Ut^2) [Source: 19 derived]
    """
    if n0 == 0 or ut == 0:
        return 0.0
    return isource / (n0 * (ut ** 2))

def calculate_mobility(beta_eff, cox_prime, w, l):
    """
    Calculates Mobility (Mu) from effective Beta.
    
    Formula: Mu = (Beta_eff * L) / (C'ox * W) [Source: 19 rearranged]
    
    *Altered*: Added 'w' and 'l' arguments to ensure dimensional correctness.
    """
    if cox_prime == 0 or w == 0:
        return 0.0
    return (beta_eff * l) / (cox_prime * w)

# --- Current & Time Constants ---

def calculate_drain_current_strong(n, beta, vp, vs):
    """
    Calculates Drain Current in Strong Inversion (Saturation).
    Formula: Id ~= (n * Beta / 2) * (Vp - Vs)^2 [Source: 51]
    Note: Beta here is total device beta (Beta_eff).
    """
    if vp < vs: return 0.0
    return (n * beta / 2.0) * ((vp - vs) ** 2)

def calculate_drain_current_weak(id0, vg, vs, n, ut):
    """
    Calculates Drain Current in Weak Inversion.
    Formula: Id = Id0 * exp((Vg - n*Vs) / (n*Ut)) [Source: 64]
    """
    if n == 0 or ut == 0: return 0.0
    exponent = (vg - n * vs) / (n * ut)
    return id0 * math.exp(exponent)

def calculate_vds_sat(ut, ic):
    """
    Calculates Drain-Source Saturation Voltage.
    Formula: Vds_sat = 2*Ut * sqrt(IC + 0.25) + 3*Ut [Source: 106]
    """
    if ic < -0.25: return 3*ut
    return 2 * ut * math.sqrt(ic + 0.25) + 3 * ut

def calculate_tau_0(l, mobility, ut):
    """
    Calculates intrinsic time constant (Tau_0).
    Formula: Tau_0 = L^2 / (Mu * Ut) [Source: 23]
    """
    if mobility == 0 or ut == 0: return 0.0
    return (l**2) / (mobility * ut)

def calculate_ft_saturation(mobility, ut, l_eff, ic):
    """
    Calculates Unity Gain Transit Frequency (fT) in Saturation.
    Formula: fT = (Mu * Ut) / (2 * pi * L^2) * (sqrt(1 + 4*IC) - 1) [Source: 121]
    """
    if l_eff == 0: return 0.0
    prefactor = (mobility * ut) / (2 * math.pi * (l_eff**2))
    term = math.sqrt(1 + 4 * ic) - 1
    return prefactor * term

# --- Noise & Mismatch ---

def calculate_flicker_noise(kf, cox_prime, w, l, freq, af, gm):
    """
    Calculates Flicker Noise Spectral Density (S_ID).
    Formula: S_ID = (gm^2 * KF) / (C'ox * W * L * f^AF) [Source: 129]
    """
    if cox_prime == 0 or w == 0 or l == 0 or freq == 0: return 0.0
    denom = cox_prime * w * l * (freq ** af)
    return ((gm**2) * kf) / denom

def calculate_thermal_noise(k, temp_kelvin, gamma_noise, gms):
    """
    Calculates Thermal Noise Spectral Density (S_ID).
    Formula: S_ID = 4 * k * T * gamma * gms [Source: 130]
    """
    return 4 * k * temp_kelvin * gamma_noise * gms

def calculate_current_mismatch(sigma_vt, gm, id_val, a_beta, w, l):
    """
    Calculates Drain Current Mismatch (Sigma_dId/Id).
    Formula: sqrt( (A_beta/sqrt(WL))^2 + (gm/Id * Sigma_vt)^2 ) [Source: 133]
    Note: A_beta/sqrt(WL) represents Sigma_beta/beta.
    """
    if w == 0 or l == 0 or id_val == 0: return 0.0
    
    sigma_beta_norm = a_beta / math.sqrt(w * l)
    term2 = (gm / id_val) * sigma_vt
    
    return math.sqrt(sigma_beta_norm**2 + term2**2)


# --- PLOTTING FUNCTIONS ---

def plot_four_styles(x_data, y_data, x_label="X-Axis", y_label="Y-Axis", title_base="Plot"):
    """
    Generates a single window with 4 subplots showing the data in:
    1. Linear Scale
    2. Scientific Axis
    3. Log Scale (Semilogy)
    4. IEEE-like Style (Serif font, compact)
    
    Args:
        x_data (list/array): The X-axis data.
        y_data (list/array): The Y-axis data.
        x_label (str): Label for X-axis.
        y_label (str): Label for Y-axis.
        title_base (str): Main title for the window.
    """
    # Create a 2x2 grid of subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle(f"{title_base} - 4 Views", fontsize=14)

    # 1. Linear Scale
    ax1.plot(x_data, y_data, 'o-', linewidth=2, label="Data")
    ax1.set_xlabel(x_label)
    ax1.set_ylabel(y_label)
    ax1.set_title("1. Linear Scale")
    ax1.grid(True)
    ax1.legend()

    # 2. Scientific Axis
    ax2.plot(x_data, y_data, 'o-', linewidth=2, label="Data", color='orange')
    ax2.set_xlabel(x_label)
    ax2.set_ylabel(y_label)
    ax2.set_title("2. Scientific Axis")
    ax2.grid(True)
    
    # Apply scientific formatting to Y axis
    formatter = ScalarFormatter()
    formatter.set_powerlimits((0, 0))
    ax2.yaxis.set_major_formatter(formatter)

    # 3. Log Scale
    # Handle negative values safely for log plot by taking absolute value
    # Check if input is a list or numpy array
    if hasattr(y_data, 'tolist'):
        y_abs = [abs(y) for y in y_data]
    else:
        y_abs = [abs(y) for y in y_data]
        
    ax3.semilogy(x_data, y_abs, 'o-', linewidth=2, label="|Data|", color='green')
    ax3.set_xlabel(x_label)
    ax3.set_ylabel(f"|{y_label}|")
    ax3.set_title("3. Log Scale")
    ax3.grid(True, which="both")

    # 4. IEEE-like Style
    # We mimic the style locally
    ax4.plot(x_data, y_data, 'o-', linewidth=1.5, color='black', markersize=4, label="Data")
    
    # Use serif font family for this subplot's labels
    font_ieee = {'family': 'serif', 'size': 10}
    
    ax4.set_xlabel(x_label, fontdict=font_ieee)
    ax4.set_ylabel(y_label, fontdict=font_ieee)
    ax4.set_title("4. IEEE Style", fontdict=font_ieee)
    
    # Thinner grid lines
    ax4.grid(True, linewidth=0.5, linestyle='--')
    ax4.tick_params(axis='both', which='major', labelsize=9)
    
    # Use scientific notation for IEEE as well usually
    ax4.yaxis.set_major_formatter(formatter)

    plt.tight_layout()
    # plt.show() must be called by the user script to keep the window open
    pass