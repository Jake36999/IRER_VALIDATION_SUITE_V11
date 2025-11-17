"""
core_engine.py
V11.0: Refactored Adaptive Hunt Core Module.
This is the V11.0 orchestrator logic, converted into an importable module
to be called by the Flask meta-orchestrator in a background thread.
"""
import os
import json
import subprocess
import logging
import time
import random
import hashlib
import settings
from typing import List, Dict, Any

# Destructure settings for clarity
CONFIG_DIR = settings.CONFIG_DIR
DATA_DIR = settings.DATA_DIR
PROVENANCE_DIR = settings.PROVENANCE_DIR
WORKER_SCRIPT = settings.WORKER_SCRIPT
VALIDATOR_SCRIPT = settings.VALIDATOR_SCRIPT
NUM_GENERATIONS = settings.NUM_GENERATIONS
POPULATION_SIZE = settings.POPULATION_SIZE
JOB_TIMEOUT_SECONDS = settings.JOB_TIMEOUT_SECONDS

# Simplified Hunter logic for demonstration
class Hunter:
    def get_next_parameters(self, generation: int) -> Dict[str, Any]:
        """Generates new parameters. A real implementation would use evolutionary logic."""
        return {
            "sncgl_epsilon": random.uniform(0.1, 0.5),
            "sncgl_lambda": random.uniform(0.01, 0.1),
            "sncgl_g_nonlocal": random.uniform(0.0005, 0.005),
            "sdg_alpha": random.uniform(1.0, 2.0),
            "sdg_rho_vac": 1.0,
            "sdg_kappa": 1.0,
            "sdg_eta": 0.5
        }
    def process_generation_results(self, job_hash: str, generation: int):
        """Placeholder for hunter to learn from results."""
        logging.info(f"[Hunter] Processing result for {job_hash[:8]} from generation {generation}")
        pass

def generate_deterministic_hash(params: dict) -> str:
    """Generates the content-based hash for a configuration."""
    payload = json.dumps(params, sort_keys=True).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()[:12]

def run_simulation_job(job_uuid: str, config_path: str) -> bool:
    """Executes the full worker->validator pipeline for a single job."""
    try:
        # 1. Execute Worker
        worker_cmd = ["python", WORKER_SCRIPT, "--config_hash", job_uuid, "--config_path", config_path]
        subprocess.run(worker_cmd, check=True, capture_output=True, text=True, timeout=JOB_TIMEOUT_SECONDS)

        # 2. Execute Validator
        validator_cmd = ["python", VALIDATOR_SCRIPT, "--config_hash", job_uuid]
        subprocess.run(validator_cmd, check=True, capture_output=True, text=True, timeout=JOB_TIMEOUT_SECONDS)

    except subprocess.CalledProcessError as e:
        logging.error(f"[CoreEngine] JOB FAILED for {job_uuid[:8]}. Exit code {e.returncode}")
        logging.error(f"  STDOUT: {e.stdout}")
        logging.error(f"  STDERR: {e.stderr}")
        return False
    except subprocess.TimeoutExpired:
        logging.error(f"[CoreEngine] JOB TIMED OUT for {job_uuid[:8]}.")
        return False

    logging.info(f"--- [CoreEngine] JOB SUCCEEDED: {job_uuid[:8]} ---")
    return True

def execute_hunt():
    """
    This is the refactored main() function. It is now called by app.py
    in a background thread to run the full evolutionary hunt.
    """
    logging.info("[CoreEngine] V11.0 HUNT EXECUTION STARTED.")

    os.makedirs(CONFIG_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(PROVENANCE_DIR, exist_ok=True)

    hunter = Hunter()

    logging.info(f"[CoreEngine] Starting Hunt: {NUM_GENERATIONS} generations...")

    for generation in range(NUM_GENERATIONS):
        logging.info(f"--- [CoreEngine] STARTING GENERATION {generation} ---")

        params_batch = [hunter.get_next_parameters(generation) for _ in range(POPULATION_SIZE)]

        jobs_to_run = []
        for params in params_batch:
            full_params = {
                "time_steps": 100,
                "spatial_resolution": 64,
                **params
            }
            job_uuid = generate_deterministic_hash(full_params)

            config_path = os.path.join(CONFIG_DIR, f"config_{job_uuid}.json")
            with open(config_path, 'w') as f:
                json.dump(full_params, f, indent=2)

            jobs_to_run.append({"uuid": job_uuid, "path": config_path})

        completed_job_hashes = []
        for job in jobs_to_run:
            if run_simulation_job(job["uuid"], job["path"]):
                completed_job_hashes.append(job["uuid"])
            else:
                logging.warning(f"Job {job['uuid']} failed. See logs for details.")

        logging.info(f"[CoreEngine] GENERATION {generation} COMPLETE. Processing results...")
        for job_hash in completed_job_hashes:
            hunter.process_generation_results(job_hash, generation)

    logging.info("[CoreEngine] --- ALL GENERATIONS COMPLETE. HUNT FINISHED. ---")
