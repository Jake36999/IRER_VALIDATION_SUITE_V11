"""
core_engine.py
CLASSIFICATION: Data Plane (V11.0 Control Hub)
GOAL: Encapsulates the blocking, long-running hunt logic.
      Called by the Flask app in a background thread.
"""
import os
import sys
import json
import subprocess
import hashlib
import logging
from typing import Dict, Any, List


try:
    import settings
    from aste_hunter import Hunter
except ImportError:
    print("FATAL: core_engine requires settings.py and aste_hunter.py", file=sys.stderr)
    sys.exit(1)


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def _run_subprocess(cmd: List[str], job_hash: str) -> bool:
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=settings.JOB_TIMEOUT_SECONDS)
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"[Job {job_hash[:8]}] FAILED (Exit Code {e.returncode}).\nSTDOUT: {e.stdout}\nSTDERR: {e.stderr}")
        return False
    except subprocess.TimeoutExpired:
        logging.error(f"[Job {job_hash[:8]}] TIMED OUT after {settings.JOB_TIMEOUT_SECONDS}s.")
        return False
    except Exception as e:
        logging.error(f"[Job {job_hash[:8]}] UNHANDLED EXCEPTION: {e}")
        return False


def execute_hunt(num_generations: int, population_size: int) -> Dict:
    logging.info(f"Core Engine: Starting hunt with {num_generations} generations, {population_size} population.")
    
    for d in [settings.CONFIG_DIR, settings.DATA_DIR, settings.PROVENANCE_DIR]:
        os.makedirs(d, exist_ok=True)


    hunter = Hunter()


    for gen in range(num_generations):
        logging.info(f"--- Starting Generation {gen}/{num_generations-1} ---")
        
        param_batch = hunter.breed_next_generation(population_size)
        
        jobs_to_run = []
        for i, params in enumerate(param_batch):
            param_str = json.dumps(params, sort_keys=True).encode('utf-8')
            config_hash = hashlib.sha256(param_str).hexdigest()
            
            config = {
                "config_hash": config_hash,
                "params": params,
                "grid_size": 32,
                "T_steps": 500,
                "global_seed": i + gen * population_size
            }
            config_path = os.path.join(settings.CONFIG_DIR, f"config_{config_hash}.json")
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=4)
            
            run_data = {"generation": gen, settings.HASH_KEY: config_hash, **params}
            jobs_to_run.append((run_data, config_path, config_hash))


        hunter.population.extend([job[0] for job in jobs_to_run])
        hunter._save_ledger()
        
        for run_data, config_path, config_hash in jobs_to_run:
            logging.info(f"Running job for hash: {config_hash[:10]}...")
            
            worker_cmd = [sys.executable, settings.WORKER_SCRIPT, "--params", config_path, "--output_dir", settings.DATA_DIR]
            if not _run_subprocess(worker_cmd, config_hash):
                continue # Skip validation if worker failed


            validator_cmd = [sys.executable, settings.VALIDATOR_SCRIPT, "--config_hash", config_hash]
            _run_subprocess(validator_cmd, config_hash)
            
        hunter.process_generation_results()


    best_run = hunter.get_best_run()
    logging.info("Core Engine: Hunt complete.")
    return best_run if best_run else {}
