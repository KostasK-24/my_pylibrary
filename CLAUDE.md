# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Stand Brary** (`stand_brary`, v1.24.0) is a Python library for semiconductor parameter extraction, EKV model calculations, file parsing, plotting, LaTeX reporting, and MOS dimensioning. Target audience: electronics researchers and engineers.

Dependencies: `numpy`, `matplotlib` (both soft-imported with `try/except` in `core.py` to allow the library to load without them in constrained environments).

## Installation

```bash
# Development (editable install)
pip install -e .

# User install from GitHub
pip install git+https://github.com/KostasK-24/stand_brary.git
```

No build step is required — this is a pure Python package.

## Code Architecture

All logic lives in a single module: `stand_brary/core.py`. The `__init__.py` re-exports everything from `core.py` via explicit `from .core import (...)` — there are no sub-packages.

`core.py` is divided into numbered sections (see its content index at the top):

| Section | Contents |
|---------|----------|
| 1 | Physical constants (`K_BOLTZMANN`, `Q_ELEMENTARY`, `EPSILON_OX`, `EPSILON_SI`, `NI_300K`) |
| 2 | Utility & file handling: `parse_simulation_file`, `parse_simulation_file_tsv`, `load_scalar_map`, `load_vector_map`, `load_scalar_data_from_dir`, `load_vector_data_from_dir` |
| 3 | Math helpers: centered derivative, linear interpolation, abs min/max |
| 4 | EKV model basics: Cox, gamma, Fermi potential, Vt0, slope factor, pinch-off |
| 5 | EKV normalization & currents: Ispec, inversion coefficient, normalized charges, drain current in all regions |
| 6–7 | Transconductances (`gms`, `gmg`, `gmd`, `gmb`) and capacitances (`cgs`–`cbd`) |
| 8–9 | AC/small-signal: Vds_sat, Early voltage, transit frequency; Noise & mismatch |
| 10 | Extended extraction helpers: `gms_over_id`, `gmg_over_id` |
| 11 | Plotting: `plot_four_styles` (2×2 panel: linear/scientific/log/IEEE), `plot_family_of_curves` (temperature-swept family with jet colorbar) |
| 12 | LaTeX reporting: `export_current_plot_to_tex`, `inject_plots_into_tex` |
| 13 | Dimensioning tools: `Widths_calculation`, `Id_Temperature_Modeling` |

## Key Conventions

**File naming for temperature data**: TSV files must follow the pattern `*_{temp}C.tsv` (e.g., `data_25C.tsv`). `get_temp_from_filename` parses temperature by splitting on `_` and looking for the token ending in `C.tsv`.

**`parse_simulation_file` vs `parse_simulation_file_tsv`**: Both parse space-delimited simulation output files. The non-TSV variant handles "short rows" by latching the temperature column value across incomplete lines (a format emitted by some SPICE simulators where the temperature column appears only on the first row of a sweep group).

**`load_scalar_map` / `load_vector_map`**: Scan a directory for `*.tsv` files, use `find_col_index` for fuzzy keyword matching on headers, and return `{int(temp): value}` dicts keyed by temperature.

**IC groups in `Widths_calculation`**: Inversion coefficient sweep is split into WI (0.001–0.1), MI (0.1–10), SI (10–20). Widths below 0.13 µm are rejected as `None`. Output TSV filenames encode L, Io, MOS type, and Id.

**`Id_Temperature_Modeling`**: Reads `Widths_calculated_*.tsv` files from `input_path`, pairs each (W, IC) operating point with a `(Io_list, T_list)` sweep, and writes `Id_Model_Temp_*.tsv` files. `Io_list` and `T_list` must have equal length.

## Adding New Functions

1. Implement in `core.py` under the relevant numbered section.
2. Add to the explicit import list in `stand_brary/__init__.py` (both the `from .core import (...)` block and `__all__`).
3. Update the content index comment at the top of `core.py` if adding a new section.
4. Bump `__version__` in both `__init__.py` and `setup.py`.
