"""
core_engine.py
CLASSIFICATION: Core Engine (IRER V11.0)
GOAL: Refactored orchestrator, now a callable module.
      This is the 'locked' HPC core.
"""

import os
import json
import subprocess
import sys
import uuid
import time
import logging
import random # Added for seed generation
import settings
import aste_hunter # This is the REAL hunter module

# --- THIS IS THE KEY REFACTOR ---
# The old `main()` function is renamed `execute_hunt()`
def execute_hunt(num_generations, population_size):
    """
    This is the refactored main() function.
    It's now called by app.py in a background thread.
    It returns the final "best run" dictionary on completion.
    """

    # --- Centralized Logging ---
    # Get the root logger configured by app.py
    log = logging.getLogger() 
    log.info("--- [CoreEngine] V11.0 HUNT EXECUTION STARTED ---")

    # --- 1. Setup ---
    log.info("[CoreEngine] Ensuring I/O directories exist...")
    os.makedirs(settings.CONFIG_DIR, exist_ok=True)
    os.makedirs(settings.DATA_DIR, exist_ok=True)
    os.makedirs(settings.PROVENANCE_DIR, exist_ok=True)

    hunter = aste_hunter.Hunter(ledger_file=settings.LEDGER_FILE)

    start_gen = hunter.get_current_generation()
    end_gen = start_gen + num_generations
    log.info(f"[CoreEngine] Starting Hunt: {num_generations} generations (from {start_gen} to {end_gen-1})")

    # --- 2. Main Evolutionary Loop ---
    for gen in range(start_gen, end_gen):
        log.info(f"--- [CoreEngine] STARTING GENERATION {gen} ---")

        parameter_batch = hunter.get_next_generation(population_size)

        jobs_to_run = []
        jobs_to_register = []

        for phys_params in parameter_batch:
            # --- UNIFIED HASHING MANDATE ---
            job_uuid = str(uuid.uuid4())

            full_params = {
                settings.HASH_KEY: job_uuid, # Use UUID as the single hash source
                "global_seed": random.randint(0, 2**32 - 1),
                "simulation": {"N_grid": 32, "T_steps": 200}, # Example params
                "sncgl_params": phys_params
            }

            params_filepath = os.path.join(settings.CONFIG_DIR, f"config_{job_uuid}.json")
            with open(params_filepath, 'w') as f:
                json.dump(full_params, f, indent=2)

            jobs_to_run.append({"job_uuid": job_uuid, "params_filepath": params_filepath})

            ledger_entry = {
                settings.HASH_KEY: job_uuid,
                "generation": gen,
                **phys_params
            }
            jobs_to_register.append(ledger_entry)

        hunter.register_new_jobs(jobs_to_register)

        # --- 3. Execute Batch Loop (Worker + Validator) ---
        job_hashes_completed = []
        for job in jobs_to_run:
            # This is the "Layer 1" JAX/HPC loop.
            # It calls the *real* scripts.
            if run_simulation_job(job["job_uuid"], job["params_filepath"]):
                job_hashes_completed.append(job["job_uuid"])

        # --- 4. Ledger Step (Cycle Completion) ---
        log.info(f"[CoreEngine] GENERATION {gen} COMPLETE. Processing {len(job_hashes_completed)} results...")
        hunter.process_generation_results(settings.PROVENANCE_DIR, job_hashes_completed)

        best_run = hunter.get_best_run()
        if best_run:
            log.info(f"[CoreEngine] Best Run So Far: {best_run.get(settings.HASH_KEY, 'N/A')[:8]}... (Fitness: {best_run.get('fitness', 0):.4f})")

    log.info("--- [CoreEngine] ALL GENERATIONS COMPLETE ---")

    final_best_run = hunter.get_best_run()
    if final_best_run:
        log.info(f"Final Best Run: {final_best_run[settings.HASH_KEY]}")
        return final_best_run
    else:
        log.info("No successful runs completed.")
        return {"error": "No successful runs completed."}


def run_simulation_job(job_uuid: str, params_filepath: str) -> bool:
    """
    This is the "Layer 1" HPC loop.
    It runs the REAL worker and validator scripts as subprocesses.
    """
    log = logging.getLogger() # Get the root logger
    log.info(f"--- [CoreEngine] STARTING JOB {job_uuid[:10]}... ---")

    # --- 1. Execute Worker (worker_sncgl_sdg.py) ---
    worker_cmd = [
        sys.executable, settings.WORKER_SCRIPT,
        "--params", params_filepath,
        "--job_uuid", job_uuid
    ]
    try:
        # Note: We set a timeout (e.g., 10 minutes)
        worker_result = subprocess.run(worker_cmd, capture_output=True, text=True, check=True, timeout=600)
        log.info(f"  [CoreEngine] <- Worker OK for {job_uuid[:10]}")
    except subprocess.CalledProcessError as e:
        log.error(f"  [CoreEngine] WORKER FAILED: {job_uuid[:10]}. STDERR: {e.stderr}")
        return False
    except subprocess.TimeoutExpired:
        log.error(f"  [CoreEngine] WORKER TIMED OUT: {job_uuid[:10]}")
        return False
    except FileNotFoundError:
        log.error(f"  [CoreEngine] Worker script not found: {settings.WORKER_SCRIPT}")
        return False

    # --- 2. Execute Validator (validation_pipeline.py) ---
    validator_cmd = [
        sys.executable, settings.VALIDATOR_SCRIPT,
        "--job_uuid", job_uuid, # This is the "Unified Hashing Mandate"
    ]
    try:
        # Validator should be fast (e.g., 5 min timeout)
        validator_result = subprocess.run(validator_cmd, capture_output=True, text=True, check=True, timeout=300)
        log.info(f"  [CoreEngine] <- Validator OK for {job_uuid[:10]}")
    except subprocess.CalledProcessError as e:
        log.error(f"  [CoreEngine] VALIDATOR FAILED: {job_uuid[:10]}. STDERR: {e.stderr}")
        return False
    except subprocess.TimeoutExpired:
        log.error(f"  [CoreEngine] VALIDATOR TIMED OUT: {job_uuid[:10]}")
        return False
    except FileNotFoundError:
        log.error(f"  [CoreEngine] Validator script not found: {settings.VALIDATOR_SCRIPT}")
        return False
        
    log.info(f"--- [CoreEngine] JOB SUCCEEDED {job_uuid[:10]} ---")
    return True
