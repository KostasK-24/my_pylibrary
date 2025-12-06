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
   - NI_300K: 1.19e10 cm^-3 [Source: 13]

2. UTILITY & FILE HANDLING:
   - calculate_thermal_voltage(temperature_celsius)
   - check_file_access(filepath, mode)
   - parse_simulation_file(input_path, output_dir, output_name, col_mapping)
   - export_vectors_and_scalars(filepath, vectors_dict, scalars_dict, delimiter)
   - load_scalar_map(directory, target_column)
   - load_vector_map(directory, target_column)
   - load_text_file_by_column(filepath)
   - get_temp_key(val)

3. NUMERICAL ANALYSIS:
   - calculate_centered_derivative(y_data, x_data)
   - find_abs_min_or_max(data_list, find_min)
   - calculate_linear_interpolation(x0, y0, x1, y1, target_y)

4. PHYSICS - MOS FUNDAMENTALS:
   - calculate_cox_prime(t_ox)
   - calculate_gamma(n_sub, cox_prime)
   - calculate_fermi_potential(ut, n_sub, ni)
   - calculate_pinch_off_voltage(vg, vto, n)
   - calculate_slope_factor(gamma, vp, phi)

5. PHYSICS - EKV SPECIFIC:
   - calculate_ispec(max_derivative, thermal_voltage)
   - calculate_theoretical_ispec(n, ut, mobility, cox_prime, w, l)
   - calculate_inversion_coefficient(id_abs, ispec)
   - calculate_surface_potential_approx(ic)

6. CAPACITANCE MODELING (EKV):
   - calculate_cgs_ekv(qs)
   - calculate_cgd_ekv(qs, qd)
   - calculate_cgb_ekv(n, cgs, cgd)

7. PROCESS EXTRACTION:
   - calculate_beta_eff(isource, n0, ut)
   - calculate_mobility(beta_eff, cox_prime, w, l)

8. CURRENT & TIME CONSTANTS:
   - calculate_drain_current_strong(n, beta, vp, vs)
   - calculate_drain_current_weak(id0, vg, vs, n, ut)
   - calculate_vds_sat(ut, ic)
   - calculate_tau_0(l, mobility, ut)
   - calculate_ft_saturation(mobility, ut, l_eff, ic)

9. NOISE & MISMATCH:
   - calculate_flicker_noise(kf, cox_prime, w, l, freq, af, gm)
   - calculate_thermal_noise(k, temp_kelvin, gamma_noise, gms)
   - calculate_current_mismatch(sigma_vt, gm, id_val, a_beta, w, l)

10. PLOTTING:
    - plot_four_styles(x_data, y_data, x_label, y_label, title_base)
"""

import os
import math
import sys
import glob  # Added for directory scanning

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
        # Check if directory is writable if file doesn't exist yet
        directory = os.path.dirname(filepath)
        if not os.path.exists(filepath) and not os.access(directory, os.W_OK):
             print(f"File Check Error: No permission to write to directory {directory}")
             return False
        # If file exists, check if it is writable
        if os.path.exists(filepath) and not os.access(filepath, os.W_OK):
            print(f"File Check Error: No permission to write to file at {filepath}")
            return False
        
    return True

def parse_simulation_file(input_file_path, output_directory, output_filename, column_mapping):
    """
    Parses a simulation file with "Forward Fill" logic for sparse data.
    If a column is missing in the current line (e.g., Temperature), 
    it reuses the value from the last valid row.
    
    Args:
        input_file_path (str): Absolute path to source file.
        output_directory (str): The directory where the result file will be saved.
        output_filename (str): Name of the file to save (e.g., 'results.tsv').
        column_mapping (list): List of columns to keep (use "-" to skip).
    """
    output_full_path = os.path.join(output_directory, output_filename)

    # Use existing library function to check access
    if not check_file_access(input_file_path, 'r'):
        sys.exit(1)
        
    # Ensure output directory exists
    if not os.path.exists(output_directory):
        print(f"Error: Output directory does not exist: {output_directory}")
        sys.exit(1)

    print(f"Processing: {input_file_path}")
    print(f"Saving to:  {output_full_path}")

    # --- Determine Format ---
    is_tsv = output_filename.lower().endswith('.tsv')
    is_csv = output_filename.lower().endswith('.csv')
    
    if is_tsv:
        separator = "\t"
        use_pretty_align = False
    elif is_csv:
        separator = ","
        use_pretty_align = False
    else:
        separator = "" 
        use_pretty_align = True
        col_width = 25

    # --- Prepare Columns ---
    indices_to_keep = [i for i, col in enumerate(column_mapping) if col != "-"]
    new_header_list = [col for col in column_mapping if col != "-"]
    
    # --- Construct Header ---
    if use_pretty_align:
        formatted_header_list = [f"{col:^{col_width}}" for col in new_header_list]
        new_header_str = "".join(formatted_header_list)
    else:
        new_header_str = separator.join(new_header_list)

    # --- Cache for Forward Fill ---
    last_seen_values = {} 

    try:
        with open(input_file_path, 'r') as infile, open(output_full_path, 'w') as outfile:
            
            # Write the new header at the very top of the file
            outfile.write(new_header_str + "\n")
            
            for line_num, line in enumerate(infile):
                parts = line.split()
                if not parts: continue

                # Detect Header vs Data
                is_header = False
                try:
                    float(parts[0])
                except ValueError:
                    is_header = True

                if is_header:
                    # Skip Headers (we only wrote the first custom one)
                    continue
                
                else:
                    # It is Data
                    selected_data = []
                    try:
                        for i in indices_to_keep:
                            # 1. Check if the column exists in this line
                            if i < len(parts):
                                val = parts[i]
                                # Store it in cache
                                last_seen_values[i] = val
                            else:
                                # 2. If missing, retrieve from cache (Forward Fill)
                                if i in last_seen_values:
                                    val = last_seen_values[i]
                                else:
                                    # If not in cache, we have a real problem
                                    print(f"Error at line {line_num + 1}: Column {i} missing and no previous value found.")
                                    sys.exit(1)
                            
                            selected_data.append(val)
                        
                        # Write the row
                        if use_pretty_align:
                            formatted_data = [f"{val:^{col_width}}" for val in selected_data]
                            line_to_write = "".join(formatted_data)
                        else:
                            line_to_write = separator.join(selected_data)
                        
                        outfile.write(line_to_write + "\n")

                    except Exception as e:
                        print(f"Error processing line {line_num + 1}: {e}")
                        sys.exit(1)

        print(f"Success! Clean file saved to: {output_full_path}")

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        sys.exit(1)

def export_vectors_and_scalars(filepath, vectors_dict, scalars_dict, delimiter='\t'):
    """
    Exports data to a file with columns (vectors) and single-row constants (scalars).
    
    Args:
        filepath (str): Full path to save the file.
        vectors_dict (dict): Dictionary of lists {'Header': [values]}. 
                             All lists must be the same length.
        scalars_dict (dict): Dictionary of constants {'Header': value}. 
                             These are only written in the first row.
        delimiter (str): Separator (default is tab for TSV).
    """
    vector_keys = list(vectors_dict.keys())
    scalar_keys = list(scalars_dict.keys())
    all_headers = vector_keys + scalar_keys
    
    if not vector_keys:
        print("Export Error: No vector data provided.")
        return
    n_rows = len(vectors_dict[vector_keys[0]])

    try:
        with open(filepath, 'w', newline='') as f:
            # Write Header
            f.write(delimiter.join(all_headers) + "\n")
            
            for i in range(n_rows):
                row_items = []
                # Vectors
                for key in vector_keys:
                    val = vectors_dict[key][i]
                    if val is None: row_items.append("")
                    elif isinstance(val, (int, float)): row_items.append(f"{val:.8e}")
                    else: row_items.append(str(val))
                
                # Scalars (First row only)
                if i == 0:
                    for key in scalar_keys:
                        val = scalars_dict[key]
                        if isinstance(val, (int, float)): row_items.append(f"{val:.8e}")
                        else: row_items.append(str(val))
                else:
                    row_items.extend([""] * len(scalar_keys))
                
                f.write(delimiter.join(row_items) + "\n")
                
    except IOError as e:
        print(f"Error writing file {filepath}: {e}")

def load_scalar_map(directory, target_column):
    """
    Scans TSV files in the directory and extracts a single CONSTANT value 
    (from the first data row) for a specific column.
    
    Args:
        directory (str): Path to the folder containing TSV files.
        target_column (str): The column header name (or substring) to find (e.g., "Ispec", "Vth").
        
    Returns:
        dict: { temperature_int: scalar_value_float }
    """
    scalar_map = {}
    
    if not os.path.exists(directory):
        print(f"Warning: Directory not found: {directory}")
        return scalar_map

    files = glob.glob(os.path.join(directory, "*.tsv"))
    
    for filepath in files:
        try:
            with open(filepath, 'r') as f:
                lines = f.readlines()
                # Need header + at least one data row
                if len(lines) < 2: continue
                
                # Parse Header (Line 1) and Data (Line 2)
                headers = lines[0].strip().split('\t')
                data = lines[1].strip().split('\t')
                
                # Generic Column Search
                target_idx = -1
                for i, h in enumerate(headers):
                    if target_column.lower() in h.lower():
                        target_idx = i
                        break
                
                # If column found and data exists at that index
                if target_idx != -1 and target_idx < len(data):
                    try:
                        val = float(data[target_idx])
                        
                        # Extract Temperature from filename
                        fname = os.path.basename(filepath)
                        if "C.tsv" in fname:
                            parts = fname.split('_')
                            for p in parts:
                                if "C.tsv" in p:
                                    temp_str = p.replace("C.tsv", "")
                                    scalar_map[int(temp_str)] = val
                                    break
                    except ValueError:
                        pass
        except Exception: 
            continue
            
    return scalar_map

def load_vector_map(directory, target_column):
    """
    Scans TSV files and extracts an ENTIRE COLUMN vector for each temperature.
    
    Args:
        directory (str): Path to the folder containing TSV files.
        target_column (str): The column header name (or substring) to find (e.g., "n", "gm").
        
    Returns:
        dict: { temperature_int: [list_of_values] }
    """
    vector_map = {}
    
    if not os.path.exists(directory):
        # print(f"Warning: Directory not found: {directory}")
        return vector_map

    files = glob.glob(os.path.join(directory, "*.tsv"))
    
    for filepath in files:
        try:
            vector_data = []
            temp_key = None
            
            # 1. Extract Temp Key from filename first
            fname = os.path.basename(filepath)
            if "C.tsv" in fname:
                try:
                    parts = fname.split('_')
                    for p in parts:
                        if "C.tsv" in p:
                            temp_key = int(p.replace("C.tsv", ""))
                            break
                except: continue

            if temp_key is None: continue

            # 2. Open file and read column
            with open(filepath, 'r') as f:
                header_line = f.readline()
                headers = header_line.strip().split('\t')
                
                # Generic Column Search
                target_idx = -1
                for i, h in enumerate(headers):
                    if target_column.lower() in h.lower():
                        target_idx = i
                        break
                
                # If not found, skip this file
                if target_idx == -1: continue

                # Read remaining data rows
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) > target_idx:
                        try:
                            val_str = parts[target_idx].strip()
                            if val_str and val_str != "---" and val_str != "NaN":
                                vector_data.append(float(val_str))
                            else:
                                # Default for missing vector data
                                if "n" == target_column.lower():
                                    vector_data.append(1.0)
                                else:
                                    vector_data.append(0.0)
                        except ValueError:
                            vector_data.append(0.0)
            
            vector_map[temp_key] = vector_data

        except Exception: continue
            
    return vector_map

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