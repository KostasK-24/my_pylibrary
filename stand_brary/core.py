"""
Stand Brary Library - Core Physics & Utility Functions

This module provides essential tools for semiconductor parameter extraction, 
numerical analysis, and file handling.

CONTEXT OF EQUATIONS & FUNCTIONS:

1. FUNDAMENTAL CONSTANTS:
   - K_BOLTZMANN (k): 1.38e-23 J/K
   - Q_ELEMENTARY (q): 1.602e-19 C

2. PHYSICS CALCULATIONS:
   - Thermal Voltage (Ut): Ut = kT/q. Essential for normalization in EKV model.
   - Specific Current (Ispec): The normalization current in the EKV model.
     Formula: Ispec = (2 * n * Ut)^2 * (W/L) * Beta. 
     (Simplified extraction uses derivative method).
   - Inversion Coefficient (IC): Measures the level of inversion (Weak/Moderate/Strong).
     Formula: IC = Id / Ispec.
   - Surface Potential (qs): Normalized surface potential approximation.
     Formula: qs = sqrt(0.25 + IC) - 0.5.
   - Effective Gain Factor (Beta_eff): Extracted from Ispec or Source Current.
     Formula: Beta_eff = Is / (n0 * Ut^2).
   - Mobility (Mu): Carrier mobility extracted from Beta_eff.
     Formula: Mu = Beta_eff / C'ox.

3. CAPACITANCE MODELING (EKV):
   - Cgs: Normalized Gate-Source Capacitance.
     Formula: cgs = (qs/3) * (2*qs + 3) / (qs + 1)^2.
   - Cgb: Normalized Gate-Bulk Capacitance.
     Formula: cgb = ((n-1)/n) * (1 - cgs - cgd).

4. NUMERICAL TOOLS:
   - Centered Derivative: Calculates dy/dx using (y_next - y_prev) / (x_next - x_prev).
     Crucial for extracting slope (n) and transconductance (gm).
   - Linear Interpolation: Finds exact crossing points (e.g., Vth where Vs=0).
     Formula: x = x0 + (target_y - y0) * (x1 - x0) / (y1 - y0).
   - Find Min/Max: Robust finder for max derivative or min value ignoring noise.

5. UTILITIES:
   - File Access: Safely checks file existence and permissions.
   - Smart Loader: Parses complex text files with multiple data blocks.
   - Temp Key: Standardizes temperature keys (float -> int) for consistent mapping.
"""

import os
import math

# --- Fundamental Physical Constants ---

# Boltzmann's constant (J/K)
K_BOLTZMANN = 1.380649e-23

# Elementary charge (C)
Q_ELEMENTARY = 1.602176634e-19

# --- Reusable Utility Functions ---

def calculate_thermal_voltage(temperature_celsius):
    """
    Calculates the thermal voltage (Ut) for a given temperature in Celsius.
    
    Ut = (k * T_Kelvin) / q
    
    Args:
        temperature_celsius (float): Temperature in degrees Celsius.
        
    Returns:
        tuple: (float: Thermal voltage (Ut) in Volts, float: T_kelvin)
    """
    T_kelvin = temperature_celsius + 273.15
    Ut = (K_BOLTZMANN * T_kelvin) / Q_ELEMENTARY
    return Ut, T_kelvin

def check_file_access(filepath, mode='r'):
    """
    Checks if a file exists and if the user has the required permissions.
    
    Args:
        filepath (str): The full path to the file.
        mode (str): 'r' for read access, 'w' for write access.
        
    Returns:
        bool: True if file exists and access is granted, False otherwise.
    """
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
    
    The first and last points are set to None as they cannot be calculated.
    
    Args:
        y_data (list): A list of 'y' numerical values.
        x_data (list): A list of 'x' numerical values (must be same length as y).
        
    Returns:
        list: A list of derivative values. Edges are marked with None.
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
    """
    Finds the absolute value of the minimum or maximum value in a list,
    ignoring non-numerical entries (like None or "").
    
    Args:
        data_list (list): A list that can contain numbers, Nones, or strings.
        find_min (bool): If True, finds abs(min(data)). 
                         If False, finds abs(max(data)).
                         
    Returns:
        float: The absolute min or max value. Returns 0.0 if no valid data.
    """
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
    
    This function is designed to be "smart":
    - It reads the file line by line.
    - If a line's first element is text, it's treated as a NEW HEADER.
    - All subsequent data lines are assumed to belong to that header.
    - This handles files with multiple data blocks and changing headers.
    
    Args:
        filepath (str): The full path to the .txt file.
        
    Returns:
        dict: A dictionary where keys are column names (from all headers)
              and values are lists of data (as floats).
              Returns None if the file is not found or is empty.
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
    """
    Finds x corresponding to target_y using linear interpolation
    between two points (x0, y0) and (x1, y1).
    Formula: x = x0 + (target_y - y0) * (x1 - x0) / (y1 - y0)
    
    Args:
        x0, y0 (float): Coordinates of the first point.
        x1, y1 (float): Coordinates of the second point.
        target_y (float): The known y-value we want to find x for.
    
    Returns:
        float: The interpolated x value. Returns None if y1 == y0.
    """
    if y1 == y0:
        return None  # Avoid division by zero
    
    return x0 + (target_y - y0) * (x1 - x0) / (y1 - y0)


def calculate_ispec(max_derivative, thermal_voltage):
    """
    Calculates the Specific Current (Ispec) based on the EKV model extraction method.
    
    Formula: Ispec = (2 * n * Ut)^2 * (W/L) * Beta 
             Here simplified extraction assumes: Ispec = (2 * Max_Derivative * Ut)^2
             Where Max_Derivative = max(d(sqrt(Id))/dVg) which approximates sqrt(Beta/2)
    
    Args:
        max_derivative (float): The maximum value of the derivative d(sqrt(Id))/dVg.
        thermal_voltage (float): Thermal voltage (Ut) in Volts.
        
    Returns:
        float: The calculated Specific Current (Ispec) in Amps.
    """
    return (2 * max_derivative) * (thermal_voltage ** 2)

def calculate_inversion_coefficient(id_abs, ispec):
    """
    Calculates the Inversion Coefficient (IC).
    
    Formula: IC = Id / Ispec
    
    Args:
        id_abs (float): Absolute value of Drain Current (Id).
        ispec (float): Specific Current (Ispec).
        
    Returns:
        float: The Inversion Coefficient (unitless). Returns 0 if Ispec is 0.
    """
    if ispec == 0:
        return 0.0
    return id_abs / ispec

def calculate_surface_potential_approx(ic):
    """
    Calculates the normalized surface potential (qs) approximation from Inversion Coefficient.
    
    Formula: qs = sqrt(0.25 + IC) - 0.5
    
    Args:
        ic (float): Inversion Coefficient.
        
    Returns:
        float: Normalized surface potential (qs).
    """
    return math.sqrt(0.25 + ic) - 0.5

def calculate_cgs_ekv(qs):
    """
    Calculates the normalized Gate-Source Capacitance (cgs) using EKV charge model approximation.
    
    Formula: cgs = (qs/3) * (2*qs + 3) / (qs + 1)^2
    
    Args:
        qs (float): Normalized surface potential.
        
    Returns:
        float: Normalized cgs.
    """
    # Avoid division by zero if qs approaches -1 (though physically qs >= 0)
    if qs <= -1: 
        return 0.0
        
    term1 = qs / 3.0
    term2 = (2.0 * qs + 3.0)
    term3 = (qs + 1.0) ** 2
    return term1 * (term2 / term3)

def calculate_cgb_ekv(n, cgs, cgd=0.0):
    """
    Calculates the normalized Gate-Bulk Capacitance (cgb) using EKV model.
    
    Formula: cgb = ((n-1)/n) * (1 - cgs - cgd)
    
    Args:
        n (float): Slope factor (n).
        cgs (float): Normalized cgs.
        cgd (float): Normalized cgd (default 0.0).
        
    Returns:
        float: Normalized cgb.
    """
    if n == 0: 
        return 0.0 # Safety check
        
    term_n = (n - 1.0) / n
    return term_n * (1.0 - cgs - cgd)

def calculate_beta_eff(isource, n0, ut):
    """
    Calculates the Effective Gain Factor (Beta_eff).
    
    Formula: Beta_eff = Is / (n0 * Ut^2)
             Where Is = 2 * Isource (Specific Current)
             Adjusted to user logic: Beta_eff derived from source current at Vth.
    
    Args:
        isource (float): Source current at Vth (or Ispec/2).
        n0 (float): Slope factor at Vth.
        ut (float): Thermal voltage in Volts.
        
    Returns:
        float: Beta_eff (A/V^2).
    """
    if n0 == 0 or ut == 0:
        return 0.0
    return isource / (n0 * (ut ** 2))

def calculate_mobility(beta_eff, cox_prime):
    """
    Calculates Mobility (Mu).
    
    Formula: Mu = Beta_eff / C'ox
    
    Args:
        beta_eff (float): Effective Gain Factor (A/V^2).
        cox_prime (float): Oxide Capacitance per unit area (F/m^2 or F).
        
    Returns:
        float: Mobility (m^2/V*s).
    """
    if cox_prime == 0:
        return 0.0
    return beta_eff / cox_prime

def get_temp_key(val):
    """
    Standardize temperature keys to integers to avoid float mismatch issues
    when mapping data between files (e.g. 27.0 vs 27).
    
    Args:
        val (float): Temperature value.
        
    Returns:
        int: Rounded integer temperature.
    """
    return int(round(val))