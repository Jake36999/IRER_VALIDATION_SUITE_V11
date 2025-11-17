"""Centralized configuration for the FMIA adaptive hunt."""
import os

# Core directories
ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(ROOT_DIR, "data")
PROVENANCE_DIR = os.path.join(ROOT_DIR, "provenance")
CONFIG_DIR = os.path.join(ROOT_DIR, "configs")

# Ledger file for the hunter
LEDGER_FILE = os.path.join(ROOT_DIR, "simulation_ledger.csv")

# Script names for the hybrid build (worker/validator)
WORKER_SCRIPT = "worker_sncgl_sdg.py"
VALIDATOR_SCRIPT = "validation_pipeline_v11.py"

# --- EVOLUTIONARY HUNT PARAMETERS ---
NUM_GENERATIONS = 10
POPULATION_SIZE = 10
LAMBDA_FALSIFIABILITY = 0.1
MUTATION_RATE = 0.3
MUTATION_STRENGTH = 0.1

# --- DATA CONTRACT KEYS ---
# These keys MUST be used consistently across all components
HASH_KEY = "config_hash"
SSE_METRIC_KEY = "log_prime_sse"
STABILITY_METRIC_KEY = "sdg_h_norm_l2"
METRIC_BLOCK_SPECTRAL = "spectral_fidelity"

# --- AI ASSISTANT CONFIGURATION ---
AI_ASSISTANT_MODE = "MOCK"  # 'MOCK' or 'GEMINI_PRO'

# --- RESOURCE MANAGEMENT ---
JOB_TIMEOUT_SECONDS = 600  # 10 minutes
