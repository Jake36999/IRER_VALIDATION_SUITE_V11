"""
settings.py
CLASSIFICATION: Central Configuration (IRER V11.0)
GOAL: Consolidates all file paths, script names, and metric keys
      for use by the entire V11.0 suite.
"""
import os

# --- Directory layout ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_DIR = os.path.join(BASE_DIR, "input_configs")
DATA_DIR = os.path.join(BASE_DIR, "simulation_data")
PROVENANCE_DIR = os.path.join(BASE_DIR, "provenance_reports")
LOG_DIR = os.path.join(BASE_DIR, "logs")
LEDGER_FILE = os.path.join(LOG_DIR, "aste_hunt_ledger.csv")

# --- Script entry points (placeholders for HPC jobs) ---
WORKER_SCRIPT = os.path.join(BASE_DIR, "worker_sncgl_sdg.py")
VALIDATOR_SCRIPT = os.path.join(BASE_DIR, "validation_pipeline.py")

# --- Execution parameters (defaults) ---
NUM_GENERATIONS = 10
POPULATION_SIZE = 10

# --- Metric keys ---
# This is the "Unified Hashing Mandate" key
HASH_KEY = "job_uuid"
# This is the "Fidelity" metric
SSE_METRIC_KEY = "log_prime_sse"
# This is the "Stability" metric
STABILITY_METRIC_KEY = "sdg_h_norm_l2"
