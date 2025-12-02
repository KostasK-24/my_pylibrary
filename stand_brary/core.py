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
