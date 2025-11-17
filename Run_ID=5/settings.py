# settings.py
# CLASSIFICATION: Central Configuration (IRER V11.0)
# GOAL: Consolidates all file paths, script names, and metric keys
#       for use by the entire V11.0 suite.


import os


# --- Directory layout ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_DIR = os.path.join(BASE_DIR, "input_configs")
DATA_DIR = os.path.join(BASE_DIR, "simulation_data")
PROVENANCE_DIR = os.path.join(BASE_DIR, "provenance_reports")


# --- Ledger File ---
# Central record for the evolutionary algorithm (Hunter)
LEDGER_FILE = os.path.join(BASE_DIR, "simulation_ledger.csv")


# --- Script Names ---
# Defines the executable scripts for the orchestrator
WORKER_SCRIPT = "worker_sncgl_sdg.py"
VALIDATOR_SCRIPT = "validation_pipeline.py"


# --- Data Contract Keys ---
# These keys ensure the worker, validator, and hunter all refer to
# metrics using the same canonical names.
SSE_METRIC_KEY = "log_prime_sse"
STABILITY_METRIC_KEY = "H_Norm_L2"
HASH_KEY = "job_uuid"
