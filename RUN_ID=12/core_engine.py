"""
core_engine.py
CLASSIFICATION: V11.0 Data Plane Orchestrator
GOAL: Encapsulates the blocking, long-running evolutionary hunt logic.
      This script is a module, not an executable. It is designed to be
      imported by the Control Plane (app.py) and run in a background
      thread, which is the core fix for the V10.x "Blocking Server"
      failure.
"""

import os
import sys
import json
import subprocess
import uuid
import logging
import time
from typing import Dict, Any, List, Optional

import settings
from aste_hunter import Hunter

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [CoreEngine] - %(message)s')

def _generate_config_file(job_uuid: str, params: Dict, gen: int, i: int) -> str:
    """Generates a unique JSON config file for a specific job."""
    config = {
        settings.HASH_KEY: job_uuid,
        "generation": gen,
        "params": params,
        "N_grid": 64, # Default simulation parameters
        "T_steps": 200,
        "seed": (gen * 100) + i
    }

    config_path = os.path.join(settings.CONFIG_DIR, f"config_{job_uuid}.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    return config_path

def _run_simulation_job(job_uuid: str, config_path: str) -> bool:
    """Runs a single Worker + Validator job as a subprocess."""

    # --- 1. Run Worker (Data Plane) ---
    # Call the script defined in the central settings file.
    worker_cmd = [sys.executable, settings.WORKER_SCRIPT, "--job_uuid", job_uuid, "--params", config_path]
    try:
        logging.info(f"Job {job_uuid[:8]}: Starting Worker...")
        subprocess.run(worker_cmd, check=True, capture_output=True, text=True, timeout=600)
    except subprocess.CalledProcessError as e:
        logging.error(f"Job {job_uuid[:8]}: WORKER FAILED.\nSTDOUT: {e.stdout}\nSTDERR: {e.stderr}")
        return False
    except subprocess.TimeoutExpired:
        logging.error(f"Job {job_uuid[:8]}: WORKER TIMED OUT.")
        return False

    # --- 2. Run Validator (Analysis Plane) ---
    # Call the script defined in the central settings file.
    validator_cmd = [sys.executable, settings.VALIDATOR_SCRIPT, "--job_uuid", job_uuid]
    try:
        logging.info(f"Job {job_uuid[:8]}: Starting Validator...")
        subprocess.run(validator_cmd, check=True, capture_output=True, text=True, timeout=300)
    except subprocess.CalledProcessError as e:
        logging.error(f"Job {job_uuid[:8]}: VALIDATOR FAILED.\nSTDOUT: {e.stdout}\nSTDERR: {e.stderr}")
        return False
    except subprocess.TimeoutExpired:
        logging.error(f"Job {job_uuid[:8]}: VALIDATOR TIMED OUT.")
        return False

    logging.info(f"Job {job_uuid[:8]}: Run SUCCEEDED.")
    return True

def execute_hunt(num_generations: int, population_size: int) -> Dict:
    """
    The main evolutionary hunt loop. This function is designed to
    be called by app.py in a background thread.
    """
    logging.info(f"--- V11.0 HUNT STARTING ---")
    logging.info(f"Gens: {num_generations}, Pop: {population_size}")

    # Ensure all state directories exist
    for d in [settings.CONFIG_DIR, settings.PROVENANCE_DIR, settings.DATA_DIR]:
        os.makedirs(d, exist_ok=True)

    hunter = Hunter(population_size=population_size)
    final_best_run: Optional[Dict[str, Any]] = None

    for gen in range(num_generations):
        logging.info(f"--- GENERATION {gen}/{num_generations-1} ---")

        # 1. Get new parameter batch from the "Brain"
        param_batch = hunter.get_next_generation_parameters()

        # 2. Execute all jobs for this generation
        job_contexts = []
        for i, params in enumerate(param_batch):

            # --- UNIFIED HASHING MANDATE (Generation) ---
            # Generate the single, authoritative UUID for this job.
            job_uuid = str(uuid.uuid4())
            # ----------------------------------------------

            config_path = _generate_config_file(job_uuid, params, gen, i)
            job_contexts.append({"uuid": job_uuid, "params": params, "config": config_path})

            logging.info(f"Gen {gen}, Job {i}: Spawning run {job_uuid[:8]}...")
            # This is a simple, synchronous loop for execution.
            # A V12.0 implementation would use a parallel job queue.
            _run_simulation_job(job_uuid, config_path)

        # 3. Process results and update Hunter's population
        logging.info(f"--- Gen {gen} Complete. Processing results... ---")
        for job in job_contexts:
            # The Hunter reads the provenance.json file generated
            # by the validator to calculate the fitness.
            run_data = hunter.process_generation_results(job["uuid"], job["params"])
            if run_data["fitness"] > 0:
                logging.info(f"Result {job['uuid'][:8]}: Fitness={run_data['fitness']:.4f}, SSE={run_data['sse']:.4f}, H_Norm={run_data['h_norm']:.4f}")

        final_best_run = hunter.get_best_run()
        if final_best_run:
            logging.info(f"Current Best: {final_best_run['job_uuid'][:8]} (Fitness: {final_best_run['fitness']:.4f})")

    logging.info(f"--- V11.0 HUNT COMPLETE ---")
    return final_best_run if final_best_run else {}
