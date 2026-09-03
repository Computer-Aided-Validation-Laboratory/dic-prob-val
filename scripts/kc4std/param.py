"""
================================================================================
KC4 Standard Workflow: Analysis Parameters & Grid Configuration Constants
================================================================================
Centralized configuration controlling all stages of the KC4 standard pipeline.
"""

from pathlib import Path

# Execution & Workspace
OUT_DIR = Path("./out/kc4std")
SIM_RUNS_DIR = OUT_DIR / "sim_runs"
NUM_WORKERS = 4
NUM_THREADS_PER_SIM = 1
RESTART = True
VERBOSE = True

# Stage 1: DOE Sampling
NUM_DOE_SAMPLES = 8
DOE_SEED = 42

# Stage 2: Physical Model & Output Quantities
SCALAR_OUTPUTS = (
    "react_y_top",
    "stress_vm_max",
    "plastic_strain_eq_max",
)
FIELD_OUTPUTS = (
    "vonmises_stress",
    "effective_plastic_strain_out",
)

# Stage 3: Common-Grid Field Standardisation (ROI & Spatial Resolution)
GRID_SHAPE = (50, 25)  # (Ny, Nx) regular validation grid
GRID_BOUNDS = ((-12.5, 12.5), (0.0, 50.0))  # ((x_min, x_max), (y_min, y_max))

# Stage 4: Modal Field Dimensionality Reduction
ENERGY_THRESHOLD = 0.9999
MAX_MODES = 6

# Stage 5: Gaussian Process Regression
GP_RESTARTS = 3
GP_SEED = 42

# Stage 6: Epistemic & Aleatory Uncertainty Propagation
NUM_EPISTEMIC = 40
NUM_ALEATORY = 300
QUANTILES = (0.05, 0.50, 0.95)
UQ_SEED = 42
