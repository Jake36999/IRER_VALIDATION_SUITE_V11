import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_DIR = os.path.join(BASE_DIR, "input_configs")
DATA_DIR = os.path.join(BASE_DIR, "simulation_data")
PROVENANCE_DIR = os.path.join(BASE_DIR, "provenance_reports")
LOG_DIR = os.path.join(BASE_DIR, "logs")
LEDGER_FILE = os.path.join(LOG_DIR, "aste_hunt_ledger.csv")

WORKER_SCRIPT = os.path.join(BASE_DIR, "worker_sncgl_sdg.py")
VALIDATOR_SCRIPT = os.path.join(BASE_DIR, "validation_pipeline.py")

NUM_GENERATIONS = 10
POPULATION_SIZE = 10

HASH_KEY = "job_uuid"
SSE_METRIC_KEY = "log_prime_sse"
STABILITY_METRIC_KEY = "sdg_h_norm_l2"
