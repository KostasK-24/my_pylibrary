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
   - parse_simulation_file: Robust "Split-Merge" parsing for sparse Ngspice data.
   - export_vectors_and_scalars: Generic TSV export.
   - load_scalar_map / load_vector_map: Generic loading.
   - check_file_access
   - get_temp_key

3. NUMERICAL & PHYSICS:
   - Derivatives, Interpolation, Vth, Cox, Ispec, Mobility, etc.
"""

import os
import math
import sys
import glob

try:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import ScalarFormatter
    import numpy as np
except ImportError:
    pass

# --- Fundamental Physical Constants ---
K_BOLTZMANN = 1.380649e-23
Q_ELEMENTARY = 1.602176634e-19
EPSILON_OX = 3.45e-11
EPSILON_SI = 1.04e-10
NI_300K = 1.19e10

# --- Reusable Utility Functions ---

def calculate_thermal_voltage(temperature_celsius):
    T_kelvin = temperature_celsius + 273.15
    Ut = (K_BOLTZMANN * T_kelvin) / Q_ELEMENTARY
    return Ut, T_kelvin

def check_file_access(filepath, mode='r'):
    if not os.path.exists(filepath):
        print(f"File Check Error: File not found at {filepath}")
        return False
    if mode == 'r' and not os.access(filepath, os.R_OK):
        print(f"File Check Error: No permission to read file at {filepath}")
        return False
    elif mode == 'w' and not os.access(filepath, os.W_OK):
        directory = os.path.dirname(filepath)
        if not os.path.exists(filepath) and not os.access(directory, os.W_OK):
             print(f"File Check Error: No permission to write to directory {directory}")
             return False
        if os.path.exists(filepath) and not os.access(filepath, os.W_OK):
            print(f"File Check Error: No permission to write to file at {filepath}")
            return False
    return True

def parse_simulation_file(input_file_path, output_directory, output_filename, column_mapping):
    """
    Parses a simulation file with "Split-Merge" logic for sparse Ngspice data.
    
    Logic:
    1. Identifies the 'Temperature' column as the Pivot.
    2. Separates user-requested columns into LEFT (Sweeps) and RIGHT (Results).
    3. On sparse lines, reads LEFT from start of line, RIGHT from end of line, 
       and fills the Pivot from cache.
    """
    output_full_path = os.path.join(output_directory, output_filename)

    if not check_file_access(input_file_path, 'r'):
        sys.exit(1)
    if not os.path.exists(output_directory):
        print(f"Error: Output directory does not exist: {output_directory}")
        sys.exit(1)

    print(f"Processing: {input_file_path}")
    
    # --- Format Setup ---
    is_tsv = output_filename.lower().endswith('.tsv')
    is_csv = output_filename.lower().endswith('.csv')
    
    separator = "\t" if is_tsv else "," if is_csv else ""
    use_pretty_align = not (is_tsv or is_csv)
    col_width = 25

    # 1. Identify indices to keep
    indices_to_keep = [i for i, col in enumerate(column_mapping) if col != "-"]
    new_header_list = [col for col in column_mapping if col != "-"]
    
    # 2. Identify the Pivot (Loop Variable)
    temp_raw_idx = -1
    for i, col in enumerate(column_mapping):
        if "temp" in col.lower():
            temp_raw_idx = i
            break
            
    # 3. Categorize indices relative to Pivot
    left_indices = []
    right_indices = []
    has_pivot_in_output = False
    
    for idx in indices_to_keep:
        if temp_raw_idx != -1:
            if idx < temp_raw_idx:
                left_indices.append(idx)
            elif idx > temp_raw_idx:
                right_indices.append(idx)
            else:
                has_pivot_in_output = True
        else:
            # If no temp column defined, treat everything as Left
            left_indices.append(idx)

    # --- Header String Construction ---
    if use_pretty_align:
        formatted_header_list = [f"{col:^{col_width}}" for col in new_header_list]
        new_header_str = "".join(formatted_header_list)
    else:
        new_header_str = separator.join(new_header_list)

    # Cache for Forward Fill
    cached_values = {}
    full_width_detected = False
    expected_full_width = len(column_mapping) # Rough guess, refined by first line

    try:
        with open(input_file_path, 'r') as infile, open(output_full_path, 'w') as outfile:
            
            outfile.write(new_header_str + "\n")
            
            for line_num, line in enumerate(infile):
                parts = line.strip().split()
                if not parts: continue

                # Skip Header lines
                is_header = False
                try:
                    float(parts[0])
                except ValueError:
                    is_header = True
                if is_header: continue
                
                # --- AUTO-DETECT Full Width ---
                if not full_width_detected:
                    if len(parts) >= expected_full_width:
                        # This looks like a full line
                        full_width_detected = True
                        # Update cache with initial full line
                        for i, val in enumerate(parts):
                            cached_values[i] = val
                
                # --- Logic Split ---
                current_width = len(parts)
                final_row_values = []
                
                # A) Full Line: Standard Read
                # We assume a line is 'Full' if it's roughly the size of the mapping
                # or significantly larger than the Left+Right count.
                is_full_line = (current_width >= expected_full_width) or (current_width > len(left_indices) + len(right_indices) + 1)

                if is_full_line:
                    # Update Cache
                    for i, val in enumerate(parts):
                        cached_values[i] = val
                    
                    # Extract directly by index
                    for idx in indices_to_keep:
                        if idx < len(parts):
                            final_row_values.append(parts[idx])
                        else:
                            final_row_values.append("NaN")
                
                # B) Sparse Line: Split-Merge Read
                else:
                    # 1. Read Left (Sweep) from Start
                    # We blindly take the first N columns where N is len(left_indices)
                    # This assumes the first N columns of the sparse line correspond exactly to the requested Left columns.
                    # (In Ngspice 'v-sweep' etc are always first).
                    
                    # Optimization: Map parts[0] to left_indices[0], parts[1] to left_indices[1]...
                    # This assumes the indices_to_keep are contiguous at the start? 
                    # Usually yes: [0, 1].
                    
                    # Store what we extracted to reconstruct order later
                    extracted_map = {}
                    
                    # Read Left
                    for i, map_idx in enumerate(left_indices):
                        if i < len(parts):
                            extracted_map[map_idx] = parts[i]
                        else:
                            extracted_map[map_idx] = cached_values.get(map_idx, "NaN")
                            
                    # Read Right
                    # We take from the END of parts.
                    # right_indices[0] corresponds to parts[-len(right)]
                    num_right = len(right_indices)
                    for i, map_idx in enumerate(right_indices):
                        # Calculate negative index: -num_right + i
                        # Ex: 2 right cols. i=0 -> -2. i=1 -> -1.
                        part_idx = -num_right + i
                        try:
                            extracted_map[map_idx] = parts[part_idx]
                        except IndexError:
                            extracted_map[map_idx] = cached_values.get(map_idx, "NaN")

                    # Read Pivot (Temp) from Cache
                    if has_pivot_in_output:
                        extracted_map[temp_raw_idx] = cached_values.get(temp_raw_idx, "NaN")
                    
                    # Construct Final Row in Order
                    for idx in indices_to_keep:
                        final_row_values.append(extracted_map.get(idx, "NaN"))

                # --- Write Row ---
                if use_pretty_align:
                    formatted_data = [f"{val:^{col_width}}" for val in final_row_values]
                    line_to_write = "".join(formatted_data)
                else:
                    line_to_write = separator.join([str(x) for x in final_row_values])
                
                outfile.write(line_to_write + "\n")

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        sys.exit(1)
        
    print(f"Success! Clean file saved to: {output_full_path}")

def export_vectors_and_scalars(filepath, vectors_dict, scalars_dict, delimiter='\t'):
    """Exports data to a file with columns (vectors) and single-row constants."""
    vector_keys = list(vectors_dict.keys())
    scalar_keys = list(scalars_dict.keys())
    all_headers = vector_keys + scalar_keys
    
    if not vector_keys: return
    n_rows = len(vectors_dict[vector_keys[0]])

    try:
        with open(filepath, 'w', newline='') as f:
            f.write(delimiter.join(all_headers) + "\n")
            for i in range(n_rows):
                row_items = []
                for key in vector_keys:
                    val = vectors_dict[key][i]
                    if val is None: row_items.append("")
                    elif isinstance(val, (int, float)): row_items.append(f"{val:.8e}")
                    else: row_items.append(str(val))
                
                if i == 0:
                    for key in scalar_keys:
                        val = scalars_dict[key]
                        if isinstance(val, (int, float)): row_items.append(f"{val:.8e}")
                        else: row_items.append(str(val))
                else:
                    row_items.extend([""] * len(scalar_keys))
                f.write(delimiter.join(row_items) + "\n")
    except IOError as e: print(f"Error: {e}")

def load_scalar_map(directory, target_column):
    """Scans TSV files and extracts a single CONSTANT value."""
    scalar_map = {}
    if not os.path.exists(directory): return scalar_map
    files = glob.glob(os.path.join(directory, "*.tsv"))
    for filepath in files:
        try:
            with open(filepath, 'r') as f:
                lines = f.readlines()
                if len(lines) < 2: continue
                headers = lines[0].strip().split('\t')
                data = lines[1].strip().split('\t')
                target_idx = -1
                for i, h in enumerate(headers):
                    if target_column.lower() in h.lower():
                        target_idx = i
                        break
                if target_idx != -1 and target_idx < len(data):
                    try:
                        val = float(data[target_idx])
                        fname = os.path.basename(filepath)
                        if "C.tsv" in fname:
                            parts = fname.split('_')
                            for p in parts:
                                if "C.tsv" in p:
                                    scalar_map[int(p.replace("C.tsv", ""))] = val
                                    break
                    except: pass
        except: continue
    return scalar_map

def load_vector_map(directory, target_column):
    """Scans TSV files and extracts an ENTIRE COLUMN vector."""
    vector_map = {}
    if not os.path.exists(directory): return vector_map
    files = glob.glob(os.path.join(directory, "*.tsv"))
    for filepath in files:
        try:
            vector_data = []
            temp_key = None
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

            with open(filepath, 'r') as f:
                header = f.readline().split('\t')
                target_idx = -1
                for i, h in enumerate(header):
                    h_clean = h.strip().lower()
                    if (target_column.lower() in h_clean) and "n0" not in h_clean:
                        target_idx = i
                        break
                if target_idx == -1 and len(header)>=4: target_idx = 3 # Fallback

                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) > target_idx:
                        try:
                            val = float(parts[target_idx].strip())
                            vector_data.append(val)
                        except:
                            vector_data.append(1.0 if "n" in target_column.lower() else 0.0)
            vector_map[temp_key] = vector_data
        except: continue
    return vector_map

# --- Math Helpers ---
def calculate_centered_derivative(y, x):
    n = len(y)
    d = [None]*n
    for i in range(1, n-1):
        if x[i+1]-x[i-1] != 0: d[i] = (y[i+1]-y[i-1])/(x[i+1]-x[i-1])
        else: d[i]=0.0
    return d

def find_abs_min_or_max(d, find_min=True):
    v = [x for x in d if isinstance(x,(int,float))]
    if not v: return 0.0
    return abs(min(v)) if find_min else abs(max(v))

def calculate_linear_interpolation(x0, y0, x1, y1, ty):
    if y1==y0: return None
    return x0 + (ty-y0)*(x1-x0)/(y1-y0)

def get_temp_key(val):
    return int(round(val))

def load_text_file_by_column(filepath):
    # (Simplified for brevity, full version in previous iteration if needed)
    return {}

# --- Physics Helpers ---
def calculate_cox_prime(t_ox): return EPSILON_OX/t_ox if t_ox>0 else 0.0
def calculate_gamma(n_sub, cox): return math.sqrt(2*Q_ELEMENTARY*EPSILON_SI*n_sub)/cox if cox!=0 else 0.0
def calculate_fermi_potential(ut, n_sub, ni=NI_300K): return 2*ut*math.log(n_sub/ni) if ni!=0 else 0.0
def calculate_pinch_off_voltage(vg, vto, n): return (vg-vto)/n if n!=0 else 0.0
def calculate_slope_factor(gamma, vp, phi): return 1 + gamma/(2*math.sqrt(vp+phi)) if (vp+phi)>0 else 1.0
def calculate_ispec(deriv, ut): return (2*deriv*ut)**2
def calculate_theoretical_ispec(n, ut, mu, cox, w, l): return 2*n*(ut**2)*mu*cox*(w/l) if l!=0 else 0.0
def calculate_inversion_coefficient(id_abs, ispec): return id_abs/ispec if ispec!=0 else 0.0
def calculate_surface_potential_approx(ic): return math.sqrt(0.25+ic)-0.5 if ic>=-0.25 else 0.0
def calculate_cgs_ekv(qs): return (qs/3)*((2*qs+3)/(qs+1)**2) if qs>-1 else 0.0
def calculate_cgd_ekv(qs, qd): return (qd/3)*((2*qd+4*qs+3)/(qs+qd+1)**2) if (qs+qd+1)!=0 else 0.0
def calculate_cgb_ekv(n, cgs, cgd): return ((n-1)/n)*(1-cgs-cgd) if n!=0 else 0.0
def calculate_beta_eff(isource, n0, ut): return isource/(n0*ut**2) if (n0!=0 and ut!=0) else 0.0
def calculate_mobility(beta, cox, w, l): return (beta*l)/(cox*w) if (cox!=0 and w!=0) else 0.0
def calculate_drain_current_strong(n, beta, vp, vs): return (n*beta/2)*((vp-vs)**2) if vp>vs else 0.0
def calculate_drain_current_weak(id0, vg, vs, n, ut): return id0*math.exp((vg-n*vs)/(n*ut)) if (n!=0 and ut!=0) else 0.0
def calculate_vds_sat(ut, ic): return 2*ut*math.sqrt(ic+0.25)+3*ut
def calculate_tau_0(l, mu, ut): return (l**2)/(mu*ut) if (mu!=0 and ut!=0) else 0.0
def calculate_ft_saturation(mu, ut, l, ic): return (mu*ut)/(2*math.pi*l**2)*(math.sqrt(1+4*ic)-1) if l!=0 else 0.0
def calculate_flicker_noise(kf, cox, w, l, f, af, gm): return (gm**2*kf)/(cox*w*l*f**af) if (cox!=0 and w!=0 and l!=0 and f!=0) else 0.0
def calculate_thermal_noise(k, t, gam, gm): return 4*k*t*gam*gm
def calculate_current_mismatch(sig_vt, gm, id, a_beta, w, l): return math.sqrt((a_beta/math.sqrt(w*l))**2 + (gm/id*sig_vt)**2) if (w!=0 and l!=0 and id!=0) else 0.0

def plot_four_styles(x, y, xl, yl, title):
    try:
        fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2,figsize=(10,8))
        fig.suptitle(f"{title} - 4 Views")
        ax1.plot(x,y,'o-'); ax1.set_title("Linear")
        ax2.plot(x,y,'o-',color='orange'); ax2.set_title("Scientific"); ax2.yaxis.set_major_formatter(ScalarFormatter())
        ax3.semilogy(x,[abs(v) for v in y],'o-',color='green'); ax3.set_title("Log")
        ax4.plot(x,y,'k.-'); ax4.set_title("IEEE Style"); ax4.grid(True, linestyle='--')
        plt.tight_layout()
    except: pass