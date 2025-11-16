import os
import json
import subprocess
import sys
import uuid
import time
import logging
import random
import settings
import aste_hunter

def execute_hunt(num_generations, population_size):
    log = logging.getLogger()
    log.info("--- [CoreEngine] V11.0 HUNT EXECUTION STARTED ---")

    os.makedirs(settings.CONFIG_DIR, exist_ok=True)
    os.makedirs(settings.DATA_DIR, exist_ok=True)
    os.makedirs(settings.PROVENANCE_DIR, exist_ok=True)

    hunter = aste_hunter.Hunter(ledger_file=settings.LEDGER_FILE)

    start_gen = hunter.get_current_generation()
    end_gen = start_gen + num_generations
    log.info(f"[CoreEngine] Starting Hunt: {num_generations} generations (from {start_gen} to {end_gen-1})")

    for gen in range(start_gen, end_gen):
        log.info(f"--- [CoreEngine] STARTING GENERATION {gen} ---")

        parameter_batch = hunter.get_next_generation(population_size)

        jobs_to_run = []
        jobs_to_register = []

        for phys_params in parameter_batch:
            job_uuid = str(uuid.uuid4())

            full_params = {
                settings.HASH_KEY: job_uuid,
                "global_seed": random.randint(0, 2**32 - 1),
                "simulation": {"N_grid": 32, "T_steps": 200},
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

        job_hashes_completed = []
        for job in jobs_to_run:
            if run_simulation_job(job["job_uuid"], job["params_filepath"]):
                job_hashes_completed.append(job["job_uuid"])

        log.info(f"[CoreEngine] GENERATION {gen} COMPLETE. Processing {len(job_hashes_completed)} results...")
        hunter.process_generation_results(settings.PROVENANCE_DIR, job_hashes_completed)

        best_run = hunter.get_best_run()
        if best_run:
            log.info(f"[CoreEngine] Best Run So Far: {best_run.get(settings.HASH_KEY)[:8]}... (Fitness: {best_run.get('fitness', 0):.4f})")

    log.info("--- [CoreEngine] ALL GENERATIONS COMPLETE ---")

    final_best_run = hunter.get_best_run()
    if final_best_run:
        log.info(f"Final Best Run: {final_best_run[settings.HASH_KEY]}")
        return final_best_run
    else:
        log.info("No successful runs completed.")
        return {"error": "No successful runs completed."}


def run_simulation_job(job_uuid: str, params_filepath: str) -> bool:
    log = logging.getLogger()
    log.info(f"--- [CoreEngine] STARTING JOB {job_uuid[:10]}... ---")

    worker_cmd = [
        sys.executable, settings.WORKER_SCRIPT,
        "--params", params_filepath,
        "--job_uuid", job_uuid
    ]
    try:
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

    validator_cmd = [
        sys.executable, settings.VALIDATOR_SCRIPT,
        "--job_uuid", job_uuid,
    ]
    try:
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
