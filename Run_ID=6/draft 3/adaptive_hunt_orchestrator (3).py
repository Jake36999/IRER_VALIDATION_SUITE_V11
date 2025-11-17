# V11.0: Orchestrator - Unified Hashing Mandate (Phase 1 Hotfix)
# Mandate: Serve as the SOLE source of the deterministic UUID/config_hash.

import json
import hashlib
import subprocess
import os
import sys
import settings

# --- CONFIGURATION (Example Placeholder Params) ---
HPC_PARAMS = {
    "simulation_name": "HPC_SDG_V11_TestRun",
    "time_steps": 100,
    "spatial_resolution": 64,
    "sncgl_epsilon": 0.15,
    "sncgl_lambda": 0.05,
    "sncgl_g_nonlocal": 0.001,
    "sdg_alpha": 1.5,
    "sdg_rho_vac": 1.0,
    "sdg_kappa": 1.0,
    "sdg_eta": 0.5,
}

def generate_deterministic_hash(params: dict) -> str:
    """
    Generates a deterministic configuration hash (serving as the run UUID).
    MANDATE: The non-deterministic time.time() salt MUST be removed.
    """
    payload = json.dumps(params, sort_keys=True).encode("utf-8")
    config_hash = hashlib.sha1(payload).hexdigest()[:12]
    return config_hash

def launch_pipeline_step(uuid: str, config_path: str):
    """
    Launches the worker and validator subprocesses, passing the UUID.
    """
    print(f"Starting run with UUID: {uuid}")

    # 1. Launch Worker (S-NCGL/SDG Co-evolution)
    print(f"Dispatching {settings.WORKER_SCRIPT}...")
    worker_cmd = [
        sys.executable, settings.WORKER_SCRIPT,
        "--config_hash", uuid,
        "--config_path", config_path
    ]
    subprocess.run(worker_cmd, check=True, timeout=settings.JOB_TIMEOUT_SECONDS)
    print("Worker completed successfully.")

    # 2. Launch Validator (Core Metrics Check)
    print(f"Dispatching {settings.VALIDATOR_SCRIPT}...")
    validator_cmd = [
        sys.executable, settings.VALIDATOR_SCRIPT,
        "--config_hash", uuid
    ]
    subprocess.run(validator_cmd, check=True, timeout=settings.JOB_TIMEOUT_SECONDS)
    print("Validator completed successfully. Pipeline UNBLOCKED.")

if __name__ == "__main__":
    os.makedirs(settings.DATA_DIR, exist_ok=True)
    os.makedirs(settings.PROVENANCE_DIR, exist_ok=True)
    os.makedirs(settings.CONFIG_DIR, exist_ok=True)

    run_uuid = generate_deterministic_hash(HPC_PARAMS)

    config_file_path = os.path.join(settings.CONFIG_DIR, f"config_{run_uuid}.json")
    with open(config_file_path, 'w') as f:
        json.dump(HPC_PARAMS, f, indent=4)

    try:
        launch_pipeline_step(run_uuid, config_file_path)
        print(f"\nV11.0 Pipeline Hotfix Confirmed for Run {run_uuid}.")
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        print(f"\nPipeline failed during execution. Error: {e}")
        sys.exit(1)
