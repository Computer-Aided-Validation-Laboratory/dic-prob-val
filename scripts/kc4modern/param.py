"""
================================================================================
KC4 Modern Workflow: Analysis Parameters & Grid Configuration Constants
================================================================================
Centralized configuration controlling all stages of the KC4 modern pipeline.
"""

from pathlib import Path

# Execution & Workspace
OUT_DIR = Path("./out/kc4modern")
SIM_RUNS_DIR = OUT_DIR / "sim_runs"
NUM_WORKERS = 4
NUM_THREADS_PER_SIM = 1
RESTART = True
VERBOSE = True

# Stage 1: Modern Mixed Uncertainty & Initial DOE Sampling
NUM_DOE_SAMPLES = 8
DOE_SEED = 42

# Stage 2: Sensitivity Screening
SENSITIVITY_SAMPLES = 1024
SENSITIVITY_SEED = 42
SCREENING_THRESHOLD = 0.05

# Stage 3: Physical Model & Output Quantities
SCALAR_OUTPUTS = (
    "react_y_top",
    "stress_vm_max",
    "plastic_strain_eq_max",
)
FIELD_OUTPUTS = (
    "vonmises_stress",
    "effective_plastic_strain_out",
)

# Stage 4: Common-Grid Field Standardisation (ROI & Spatial Resolution)
GRID_SHAPE = (50, 25)  # (Ny, Nx) regular validation grid
GRID_BOUNDS = ((-12.5, 12.5), (0.0, 50.0))  # ((x_min, x_max), (y_min, y_max))
POD_ENERGY_THRESHOLD = 0.999
MAX_POD_MODES = 6

# Stage 5: Modal Field Gaussian Process Surrogate
GP_RESTARTS = 3
GP_SEED = 42

# Stage 6: Convergence-Monitored Spatial Field UQ
NUM_EPISTEMIC = 30
NUM_ALEATORY = 200
QUANTILES = (0.05, 0.50, 0.95)
UQ_SEED = 42
