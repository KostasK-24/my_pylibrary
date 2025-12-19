"""
Stand Brary Library - Core Physics & Utility Functions (v1.18.0)

This module provides a comprehensive suite of tools for semiconductor parameter 
extraction (EKV Model), numerical analysis, file handling, plotting, and LaTeX reporting.

===============================================================================
                                 CONTENT INDEX
===============================================================================

1. FUNDAMENTAL PHYSICAL CONSTANTS
   - K_BOLTZMANN, Q_ELEMENTARY, EPSILON_OX, EPSILON_SI, NI_300K

2. UTILITY & FILE HANDLING
   - calculate_thermal_voltage, check_file_access, get_temp_key
   - get_temp_from_filename, find_col_index, parse_simulation_file
   - export_vectors_and_scalars, load_scalar_map, load_vector_map
   - load_scalar_data_from_dir, load_vector_data_from_dir

3. MATH HELPERS
   - calculate_centered_derivative, find_abs_min_or_max
   - calculate_linear_interpolation

4. PHYSICS: EKV MODEL & PARAMETERS (Based on Bucher et al.)
   - calculate_cox_prime, calculate_gamma, calculate_fermi_potential
   - calculate_vto, calculate_n0, calculate_slope_factor
   - calculate_pinch_off_voltage, calculate_normalization_charge_q0
   - calculate_ispec, calculate_theoretical_ispec
   - calculate_beta_eff, calculate_mobility
   - calculate_inversion_coefficient, calculate_normalized_charge_ekv
   - calculate_normalized_current_ekv
   - calculate_drain_current_strong, calculate_drain_current_weak
   - calculate_drain_current_ekv_all_regions

5. PHYSICS: TRANSCONDUCTANCES & CAPACITANCES
   - calculate_gms_ekv, calculate_gmg_ekv, calculate_gmd_ekv, calculate_gmb_ekv
   - calculate_cgs_ekv, calculate_cgd_ekv, calculate_cgb_ekv
   - calculate_cbs_ekv, calculate_cbd_ekv

6. PHYSICS: AC, NOISE & MISMATCH
   - calculate_vds_sat, calculate_early_voltage, calculate_tau_0, calculate_tau_qs
   - calculate_ft_saturation, calculate_ft_general
   - calculate_flicker_noise, calculate_thermal_noise
   - calculate_current_mismatch, calculate_voltage_mismatch_variance

7. EXTENDED EXTRACTION HELPERS
   - calculate_gms_over_id, calculate_gmg_over_id

8. PLOTTING & REPORTING
   - plot_four_styles, plot_family_of_curves
   - export_current_plot_to_tex
   - inject_plots_into_tex (NEW: Targeted insertion)
===============================================================================
"""

import os
import math
import sys
import glob

try:
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    from matplotlib.ticker import ScalarFormatter, FuncFormatter
    import numpy as np
except ImportError:
    pass

# --- 1. Fundamental Physical Constants ---
K_BOLTZMANN = 1.380649e-23
Q_ELEMENTARY = 1.602176634e-19
EPSILON_OX = 3.45e-11
EPSILON_SI = 1.04e-10
NI_300K = 1.19e10

# --- 2. Utility & File Handling ---

def calculate_thermal_voltage(temperature_celsius):
    """Calculates Ut = kT/q."""
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
             return False
        if os.path.exists(filepath) and not os.access(filepath, os.W_OK):
            print(f"File Check Error: No permission to write to file at {filepath}")
            return False
    return True

def get_temp_key(val): return int(round(val))

def get_temp_from_filename(filename):
    try:
        parts = filename.split('_')
        for p in parts:
            if "C.tsv" in p:
                return float(p.replace("C.tsv", ""))
    except: pass
    return None

def find_col_index(headers, keywords, exclude=None):
    for i, h in enumerate(headers):
        h_low = h.lower()
        if any(k in h_low for k in keywords):
            if exclude and any(ex in h_low for ex in exclude): continue
            return i
    return -1

def parse_simulation_file(input_file_path, output_directory, output_filename, column_mapping):
    output_full_path = os.path.join(output_directory, output_filename)
    if not check_file_access(input_file_path, 'r'): sys.exit(1)
    if not os.path.exists(output_directory):
        print(f"Error: Output directory does not exist: {output_directory}")
        sys.exit(1)

    print(f"Processing: {input_file_path}")
    is_tsv = output_filename.lower().endswith('.tsv')
    is_csv = output_filename.lower().endswith('.csv')
    separator = "\t" if is_tsv else "," if is_csv else ""
    use_pretty_align = not (is_tsv or is_csv)
    col_width = 25

    indices_to_keep = [i for i, col in enumerate(column_mapping) if col != "-"]
    new_header_list = [col for col in column_mapping if col != "-"]
    
    temp_raw_idx = -1
    for i, col in enumerate(column_mapping):
        if "temp" in col.lower():
            temp_raw_idx = i
            break
            
    left_indices = []
    right_indices = []
    has_pivot_in_output = False
    for idx in indices_to_keep:
        if temp_raw_idx != -1:
            if idx < temp_raw_idx: left_indices.append(idx)
            elif idx > temp_raw_idx: right_indices.append(idx)
            else: has_pivot_in_output = True
        else: left_indices.append(idx)

    if use_pretty_align:
        formatted_header_list = [f"{col:^{col_width}}" for col in new_header_list]
        new_header_str = "".join(formatted_header_list)
    else:
        new_header_str = separator.join(new_header_list)

    cached_values = {}
    full_width_detected = False
    expected_full_width = len(column_mapping)

    try:
        with open(input_file_path, 'r') as infile, open(output_full_path, 'w') as outfile:
            outfile.write(new_header_str + "\n")
            for line_num, line in enumerate(infile):
                parts = line.strip().split()
                if not parts: continue
                try: float(parts[0])
                except ValueError: continue 
                
                if not full_width_detected:
                    if len(parts) >= expected_full_width:
                        full_width_detected = True
                        for i, val in enumerate(parts): cached_values[i] = val
                
                current_width = len(parts)
                final_row_values = []
                is_full_line = (current_width >= expected_full_width) or (current_width > len(left_indices) + len(right_indices) + 1)

                if is_full_line:
                    for i, val in enumerate(parts): cached_values[i] = val
                    for idx in indices_to_keep:
                        if idx < len(parts): final_row_values.append(parts[idx])
                        else: final_row_values.append("NaN")
                else:
                    extracted_map = {}
                    for i, map_idx in enumerate(left_indices):
                        if i < len(parts): extracted_map[map_idx] = parts[i]
                        else: extracted_map[map_idx] = cached_values.get(map_idx, "NaN")
                    num_right = len(right_indices)
                    for i, map_idx in enumerate(right_indices):
                        part_idx = -num_right + i
                        try: extracted_map[map_idx] = parts[part_idx]
                        except IndexError: extracted_map[map_idx] = cached_values.get(map_idx, "NaN")
                    if has_pivot_in_output:
                        extracted_map[temp_raw_idx] = cached_values.get(temp_raw_idx, "NaN")
                    for idx in indices_to_keep:
                        final_row_values.append(extracted_map.get(idx, "NaN"))

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
                target_idx = find_col_index(headers, [target_column.lower()])
                if target_idx != -1 and target_idx < len(data):
                    try:
                        val = float(data[target_idx])
                        t = get_temp_from_filename(os.path.basename(filepath))
                        if t is not None: scalar_map[int(t)] = val
                    except: pass
        except: continue
    return scalar_map

def load_vector_map(directory, target_column):
    vector_map = {}
    if not os.path.exists(directory): return vector_map
    files = glob.glob(os.path.join(directory, "*.tsv"))
    for filepath in files:
        try:
            t = get_temp_from_filename(os.path.basename(filepath))
            if t is None: continue
            vector_data = []
            with open(filepath, 'r') as f:
                header = f.readline().split('\t')
                exclude_list = ["n0"] if "n" == target_column.lower() else None
                target_idx = find_col_index(header, [target_column.lower()], exclude=exclude_list)
                if target_idx == -1 and len(header)>=4: target_idx = 3
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) > target_idx:
                        try:
                            val = float(parts[target_idx].strip())
                            vector_data.append(val)
                        except: vector_data.append(np.nan)
            vector_map[int(t)] = vector_data
        except: continue
    return vector_map

def load_scalar_data_from_dir(directory, columns_to_load):
    data_map = {}
    if not os.path.exists(directory): return {}
    for fpath in glob.glob(os.path.join(directory, "*.tsv")):
        t = get_temp_from_filename(os.path.basename(fpath))
        if t is None: continue
        try:
            with open(fpath, 'r') as f:
                lines = f.readlines()
                if len(lines) < 2: continue
                head = lines[0].strip().split('\t')
                row = lines[1].strip().split('\t')
                if t not in data_map: data_map[t] = {}
                for col in columns_to_load:
                    exclude = ["n0", "n at"] if col.lower() == "n" and "vector" in col.lower() else None
                    if col.lower() == 'n': keywords = ["n0", "n at", "slope factor"]
                    else: keywords = [col.lower()]
                    idx = find_col_index(head, keywords, exclude=exclude)
                    if idx != -1 and idx < len(row):
                        data_map[t][col] = float(row[idx])
        except: pass
    sorted_keys = sorted(data_map.keys())
    result = {'Temp': np.array(sorted_keys)}
    for col in columns_to_load:
        arr = []
        for k in sorted_keys: arr.append(data_map[k].get(col, np.nan))
        result[col] = np.array(arr)
    return result

def load_vector_data_from_dir(directory, vector_keys):
    data_map = {}
    if not os.path.exists(directory): return data_map
    for fpath in glob.glob(os.path.join(directory, "*.tsv")):
        t = get_temp_from_filename(os.path.basename(fpath))
        if t is None: continue
        try:
            with open(fpath, 'r') as f:
                header = f.readline().strip().split('\t')
                indices = {}
                for key in vector_keys:
                    exclude = ["n0"] if key.lower() == "n" else None
                    idx = find_col_index(header, [key.lower()], exclude=exclude)
                    if idx != -1: indices[key] = idx
                if not indices: continue
                temp_vecs = {k: [] for k in indices}
                for line in f:
                    parts = line.strip().split('\t')
                    for key, idx in indices.items():
                        if len(parts) > idx:
                            try: temp_vecs[key].append(float(parts[idx]))
                            except: temp_vecs[key].append(np.nan)
                if any(temp_vecs.values()):
                    data_map[t] = {k: np.array(v) for k, v in temp_vecs.items()}
        except: pass
    return data_map

# --- 3. Math Helpers ---

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

def load_text_file_by_column(filepath): return {}

# --- 4. Physics: Basic Parameters & Threshold ---

def calculate_cox_prime(t_ox):
    return EPSILON_OX/t_ox if t_ox>0 else 0.0

def calculate_gamma(n_sub, cox):
    return math.sqrt(2*Q_ELEMENTARY*EPSILON_SI*n_sub)/cox if cox!=0 else 0.0

def calculate_fermi_potential(ut, n_sub, ni=NI_300K):
    return 2*ut*math.log(n_sub/ni) if ni!=0 else 0.0

def calculate_vto(vfb, phi, gamma):
    if phi < 0: return vfb + phi
    return vfb + phi + gamma * math.sqrt(phi)

def calculate_n0(gamma, phi):
    if phi <= 0: return 1.0
    return 1 + gamma / (2 * math.sqrt(phi))

def calculate_slope_factor(gamma, vp, phi):
    return 1 + gamma/(2*math.sqrt(vp+phi)) if (vp+phi)>0 else 1.0

def calculate_pinch_off_voltage(vg, vto, n):
    return (vg-vto)/n if n!=0 else 0.0

# --- 5. Physics: Normalization & EKV Basics ---

def calculate_normalization_charge_q0(ut, cox):
    return 2 * ut * cox

def calculate_ispec(deriv, ut):
    return (2*deriv*ut)**2

def calculate_theoretical_ispec(n, ut, mu, cox, w, l):
    return 2*n*(ut**2)*mu*cox*(w/l) if l!=0 else 0.0

def calculate_beta_eff(isource, n0, ut):
    return isource/(n0*ut**2) if (n0!=0 and ut!=0) else 0.0

def calculate_mobility(beta, cox, w, l):
    return (beta*l)/(cox*w) if (cox!=0 and w!=0) else 0.0

# --- 6. Physics: Currents & Inversion Charges ---

def calculate_inversion_coefficient(id_abs, ispec):
    return id_abs/ispec if ispec!=0 else 0.0

def calculate_normalized_charge_ekv(ic):
    return math.sqrt(0.25+ic)-0.5 if ic>=-0.25 else 0.0

calculate_surface_potential_approx = calculate_normalized_charge_ekv # Alias

def calculate_normalized_current_ekv(q):
    return q**2 + q

def calculate_drain_current_ekv_all_regions(ispec, ifwd, irev):
    return ispec * (ifwd - irev)

def calculate_drain_current_strong(n, beta, vp, vs):
    return (n*beta/2)*((vp-vs)**2) if vp>vs else 0.0

def calculate_drain_current_weak(id0, vg, vs, n, ut):
    return id0*math.exp((vg-n*vs)/(n*ut)) if (n!=0 and ut!=0) else 0.0

# --- 7. Physics: Transconductances ---

def calculate_gms_ekv(ispec, ut, qs):
    if ut == 0: return 0.0
    return (ispec / ut) * qs

def calculate_gmg_ekv(gms, n):
    return gms / n if n != 0 else 0.0

def calculate_gmd_ekv(ispec, ut, qd):
    if ut == 0: return 0.0
    return (ispec / ut) * qd

def calculate_gmb_ekv(n, gms, gmd):
    if n == 0: return 0.0
    return ((n - 1) / n) * (gms - gmd)

# --- 8. Physics: Capacitances ---

def calculate_cgs_ekv(qs, qd=0.0):
    denom = (qs + qd + 1)**2
    if denom == 0: return 0.0
    return (qs/3.0) * (2*qs + 4*qd + 3) / denom

def calculate_cgd_ekv(qs, qd):
    denom = (qs + qd + 1)**2
    if denom == 0: return 0.0
    return (qd/3.0) * (2*qd + 4*qs + 3) / denom

def calculate_cgb_ekv(n, cgs, cgd):
    return ((n-1)/n)*(1-cgs-cgd) if n!=0 else 0.0

def calculate_cbs_ekv(n, cgs):
    return (n - 1) * cgs

def calculate_cbd_ekv(n, cgd):
    return (n - 1) * cgd

# --- 9. Physics: Small Signal & AC ---

def calculate_vds_sat(ut, ic):
    return 2*ut*math.sqrt(ic+0.25)+3*ut

def calculate_early_voltage(id_vector, vds_vector):
    gds = calculate_centered_derivative(id_vector, vds_vector)
    va_list = []
    for i in range(len(id_vector)):
        val_id = id_vector[i]
        val_gds = gds[i]
        if val_gds is not None and val_gds != 0 and val_id != 0:
            va_list.append(val_id / val_gds)
        else:
            va_list.append(None)
    return va_list

def calculate_tau_0(l, mu, ut):
    return (l**2)/(mu*ut) if (mu!=0 and ut!=0) else 0.0

def calculate_tau_qs(tau0, qs, qd=0.0):
    denom = (qs + qd + 1)**3
    if denom == 0: return 0.0
    num = 4*(qs**2) + 10*qs + 12*qs*qd + 4*(qd**2) + 5
    return tau0 * (1.0/30.0) * (num / denom)

def calculate_ft_saturation(mu, ut, l, ic):
    if l==0: return 0.0
    return (mu*ut)/(2*math.pi*l**2)*(math.sqrt(1+4*ic)-1)

def calculate_ft_general(gm, c_total):
    if c_total == 0: return 0.0
    return gm / (2 * math.pi * c_total)

# --- 10. Physics: Noise & Mismatch ---

def calculate_flicker_noise(kf, cox, w, l, f, af, gm):
    return (gm**2*kf)/(cox*w*l*f**af) if (cox!=0 and w!=0 and l!=0 and f!=0) else 0.0

def calculate_thermal_noise(k, t, gamma_noise, gms):
    return 4*k*t*gamma_noise*gms

def calculate_current_mismatch(sig_vt, gm, id, a_beta, w, l):
    term1 = (a_beta / math.sqrt(w*l))**2
    term2 = (gm/id * sig_vt)**2
    return math.sqrt(term1 + term2) if (w!=0 and l!=0 and id!=0) else 0.0

def calculate_voltage_mismatch_variance(sigma_vt, gm, ib, sigma_beta):
    if gm == 0: return 0.0
    return (sigma_vt**2) + ((ib/gm)**2)*(sigma_beta**2)

# --- 11. Extended Extraction Helpers ---

def calculate_gms_over_id(v_source_vector, id_vector, temp_c, normalize=False):
    Ut, _ = calculate_thermal_voltage(temp_c)
    ln_id = []
    for val in id_vector:
        if val > 1e-15: ln_id.append(math.log(abs(val)))
        else: ln_id.append(None)
    deriv = calculate_centered_derivative(ln_id, v_source_vector)
    result = []
    for d in deriv:
        if d is None: result.append(None)
        else:
            if normalize: result.append(d * Ut)
            else: result.append(d)
    return result

def calculate_gmg_over_id(v_gate_vector, id_vector, temp_c, normalize=False):
    Ut, _ = calculate_thermal_voltage(temp_c)
    ln_id = []
    for val in id_vector:
        if val > 1e-15: ln_id.append(math.log(abs(val)))
        else: ln_id.append(None)
    deriv = calculate_centered_derivative(ln_id, v_gate_vector)
    result = []
    for d in deriv:
        if d is None: result.append(None)
        else:
            if normalize: result.append(d * Ut)
            else: result.append(d)
    return result

# --- 12. Plotting Functions ---

def smart_formatter(x, pos):
    if x == 0: return "0"
    if abs(x) < 1e-4 or abs(x) > 1e5: 
        return f"{x:.2e}"
    s = f"{x:.3f}"
    return s.rstrip('0').rstrip('.') if '.' in s else s

def format_axis(ax):
    try:
        formatter = FuncFormatter(smart_formatter)
        if ax.get_xscale() == 'linear': ax.xaxis.set_major_formatter(formatter)
        if ax.get_yscale() == 'linear': ax.yaxis.set_major_formatter(formatter)
    except: pass

def plot_four_styles(x_data, y_data, x_label="X-Axis", y_label="Y-Axis", title_base="Plot"):
    try:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 8))
        fig.suptitle(f"{title_base} - 4 Views", fontsize=14)

        ax1.plot(x_data, y_data, 'o-', linewidth=2, label="Data")
        ax1.set_xlabel(x_label); ax1.set_ylabel(y_label); ax1.set_title("1. Linear"); ax1.grid(True)
        format_axis(ax1)

        ax2.plot(x_data, y_data, 'o-', linewidth=2, label="Data", color='orange')
        ax2.set_xlabel(x_label); ax2.set_ylabel(y_label); ax2.set_title("2. Scientific"); ax2.grid(True)
        ax2.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        ax2.ticklabel_format(axis='y', style='sci', scilimits=(0,0))

        if hasattr(y_data, 'tolist'): y_abs = [abs(y) for y in y_data]
        else: y_abs = [abs(y) for y in y_data]
        ax3.semilogy(x_data, y_abs, 'o-', linewidth=2, label="|Data|", color='green')
        ax3.set_xlabel(x_label); ax3.set_ylabel(f"|{y_label}|"); ax3.set_title("3. Log Scale"); ax3.grid(True, which="both")

        ax4.plot(x_data, y_data, 'o-', linewidth=1.5, color='black', markersize=4, label="Data")
        font_ieee = {'family': 'serif', 'size': 10}
        ax4.set_xlabel(x_label, fontdict=font_ieee); ax4.set_ylabel(y_label, fontdict=font_ieee); ax4.set_title("4. IEEE Style", fontdict=font_ieee)
        ax4.grid(True, linewidth=0.5, linestyle='--'); ax4.tick_params(axis='both', which='major', labelsize=9)
        format_axis(ax4)

        plt.tight_layout()
    except Exception as e: print(f"Plot Error: {e}")

def plot_family_of_curves(data_map, x_key, y_key, x_label, y_label, title_base, log_x=False):
    try:
        sorted_keys = sorted(data_map.keys())
        if not sorted_keys: return
        plot_keys = sorted_keys[::max(1, len(sorted_keys)//20)]
        colors = cm.jet(np.linspace(0, 1, len(plot_keys)))

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f"{title_base}", fontsize=16)

        def draw_lines(ax, mode='linear'):
            for i, k in enumerate(plot_keys):
                d = data_map[k]
                if x_key not in d or y_key not in d: continue
                x_raw = d[x_key]
                y_raw = d[y_key]
                min_len = min(len(x_raw), len(y_raw))
                if min_len == 0: continue
                try:
                    xa = np.array(x_raw[:min_len], dtype=float)
                    ya = np.array(y_raw[:min_len], dtype=float)
                except: continue
                mask = np.isfinite(xa) & np.isfinite(ya)
                if not np.any(mask): continue
                x_plot = xa[mask]
                y_plot = ya[mask]
                
                if mode == 'log':
                    y_safe = np.abs(y_plot)
                    y_safe[y_safe==0] = 1e-15
                    if log_x:
                        x_safe = np.abs(x_plot)
                        x_safe[x_safe==0] = 1e-15
                        ax.loglog(x_safe, y_safe, linewidth=1.2, color=colors[i])
                    else:
                        ax.semilogy(x_plot, y_safe, linewidth=1.2, color=colors[i])
                else:
                    if log_x:
                        x_safe = np.abs(x_plot)
                        x_safe[x_safe==0] = 1e-15
                        ax.semilogx(x_safe, y_plot, linewidth=1.2, color=colors[i])
                    else:
                        ax.plot(x_plot, y_plot, linewidth=1.2, color=colors[i])

        ax1.set_title("1. Linear"); ax1.grid(True); draw_lines(ax1, 'linear'); ax1.set_xlabel(x_label); ax1.set_ylabel(y_label)
        if not log_x: format_axis(ax1)

        ax2.set_title("2. Scientific"); ax2.grid(True); draw_lines(ax2, 'linear'); ax2.set_xlabel(x_label); ax2.set_ylabel(y_label)
        try: ax2.yaxis.set_major_formatter(ScalarFormatter(useMathText=True)); ax2.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
        except: pass

        ax3.set_title("3. Log Scale"); ax3.grid(True, which="both"); draw_lines(ax3, 'log'); ax3.set_xlabel(x_label); ax3.set_ylabel(f"|{y_label}|")

        font_ieee = {'family': 'serif', 'size': 10}
        ax4.set_title("4. IEEE Style", fontdict=font_ieee); ax4.grid(True, linewidth=0.5, linestyle='--')
        draw_lines(ax4, 'linear'); ax4.set_xlabel(x_label, fontdict=font_ieee); ax4.set_ylabel(y_label, fontdict=font_ieee)
        if not log_x: format_axis(ax4)

        sm = plt.cm.ScalarMappable(cmap=cm.jet, norm=plt.Normalize(vmin=min(sorted_keys), vmax=max(sorted_keys)))
        sm.set_array([])
        fig.colorbar(sm, ax=[ax1, ax2, ax3, ax4], shrink=0.9).set_label('Temperature (C)')
        
    except Exception as e: print(f"Plotting Error: {e}")

def export_current_plot_to_tex(title, tex_file_path, img_dir_path, tex_relative_dir, write_to_file=True):
    """
    Saves the current matplotlib figure to the given directory.
    If write_to_file=True, it appends to the .tex file immediately.
    If False, it returns the latex string for later use.

    Args:
        title (str): Plot title (used for filename and caption).
        tex_file_path (str): Path to the target .tex file.
        img_dir_path (str): Absolute path where the image file will be saved.
        tex_relative_dir (str): The directory prefix for the LaTeX \includegraphics command.
        write_to_file (bool): Whether to write to file immediately or return string.
    """
    if not os.path.exists(img_dir_path):
        os.makedirs(img_dir_path)

    # 1. Safe Filename
    safe_filename_str = title.replace(" ", "_").replace("(", "").replace(")", "").replace("*", "x").replace("/", "div")
    img_filename = f"{safe_filename_str}.png"
    abs_img_path = os.path.join(img_dir_path, img_filename)
    
    # 2. Save Figure
    try:
        plt.gcf().savefig(abs_img_path, dpi=300, bbox_inches='tight')
    except Exception as e:
        print(f"Error saving image {img_filename}: {e}")
        return ""

    # 3. Safe Caption
    safe_caption = title.replace("_", r"\_") 
    
    # 4. Generate LaTeX
    # Join the user-provided relative directory with the filename
    # We use os.path.join then replace backslashes to forward slashes for LaTeX compatibility
    rel_img_path = os.path.join(tex_relative_dir, img_filename).replace(os.sep, '/')
    
    latex_content = f"""
% Auto-generated plot for {title}
\\begin{{figure}}[H]
    \\centering
    \\includegraphics[width=0.85\\textwidth]{{{rel_img_path}}}
    \\caption{{{safe_caption}}}
    \\label{{fig:{safe_filename_str}}}
\\end{{figure}}
"""
    
    if write_to_file:
        try:
            with open(tex_file_path, "a") as f:
                f.write(latex_content)
            print(f" -> Exported to TeX: '{title}'")
        except Exception as e:
            print(f"Error writing to .tex file: {e}")
        return ""
    else:
        print(f" -> Generated LaTeX for: '{title}'")
        return latex_content

def inject_plots_into_tex(tex_path, target_section_name, new_subsection_name, content_to_inject):
    """
    Reads the .tex file, finds \section{target_section_name},
    navigates to the end of that section (before the next section or end doc),
    and inserts the new subsection and plots there.
    """
    if not os.path.exists(tex_path):
        print(f"Error: LaTeX file not found at {tex_path}")
        return

    with open(tex_path, 'r') as f:
        lines = f.readlines()

    # 1. Find the Target Section
    start_index = -1
    target_str = f"\\section{{{target_section_name}}}"
    
    for i, line in enumerate(lines):
        if target_str in line:
            start_index = i
            break
            
    if start_index == -1:
        print(f"Warning: Section '{target_section_name}' not found! Appending to end of file instead.")
        with open(tex_path, 'a') as f:
            f.write(f"\n\\section{{{target_section_name}}}\n")
            if new_subsection_name:
                f.write(f"\\subsection{{{new_subsection_name}}}\n")
            f.write(content_to_inject)
        return

    # 2. Find the insertion point (End of that section)
    # The end is either the next \section, \chapter, \end{document}, or EOF
    insertion_index = len(lines) # Default to end
    
    for i in range(start_index + 1, len(lines)):
        line = lines[i].strip()
        if line.startswith(r"\section") or line.startswith(r"\chapter") or line.startswith(r"\end{document}"):
            insertion_index = i
            break
            
    # 3. Construct the new content block
    injection_block = []
    if new_subsection_name:
        injection_block.append(f"\n\\subsection{{{new_subsection_name}}}\n")
    injection_block.append(content_to_inject)
    
    # 4. Insert and Write Back
    new_lines = lines[:insertion_index] + injection_block + lines[insertion_index:]
    
    try:
        with open(tex_path, 'w') as f:
            f.writelines(new_lines)
        print(f" -> Successfully inserted plots into Section '{target_section_name}'")
    except Exception as e:
        print(f"Error writing to .tex file: {e}")