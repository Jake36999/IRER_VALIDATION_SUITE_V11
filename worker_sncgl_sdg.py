"""
worker_sncgl_sdg.py (STUB)
CLASSIFICATION: HPC Core (Layer 1)
GOAL: Runs the S-NCGL + SDG coupled system.
      This stub simulates the work by generating a mock HDF5 file.
"""
import argparse
import time
import os
import json
import logging
import random
import sys
import h5py # Import h5py for HDF5 operations
import numpy as np # Import numpy for numerical computations
import settings # Import settings to get DATA_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger()

def run_sncgl_sdg_simulation(params: dict, job_uuid: str):
    """
    Placeholder for the S-NCGL + SDG coupled system simulation.
    Generates mock simulation data and metrics, and saves them to HDF5.
    """
    log.info(f"[WorkerStub {job_uuid[:8]}] Simulating S-NCGL + SDG with params: {params.get('sncgl_params')}")

    # Simulate JAX/HPC work duration
    simulation_duration = random.uniform(1.0, 3.0)
    time.sleep(simulation_duration)

    # --- Generate Mock Simulation Data ---
    # In a real scenario, this would be the output of the JAX simulation.
    grid_size = params['simulation']['N_grid']
    time_steps = params['simulation']['T_steps']

    # Create a simple mock data array (e.g., a dynamic field)
    mock_field_data = np.random.rand(time_steps, grid_size, grid_size).astype(np.float32)
    mock_field_data += np.sin(np.linspace(0, 10, time_steps))[:, np.newaxis, np.newaxis]

    # --- Generate Mock Metrics ---
    # These would be derived from the simulation output.
    param_D = params['sncgl_params'].get('param_D', 0.5)
    param_eta = params['sncgl_params'].get('param_eta', 0.1)

    # Fake metrics, possibly influenced by input parameters
    mock_sse = random.uniform(0.001, 0.5) * (1 + param_D / 2) # Example influence
    mock_h_norm = random.uniform(0.001, 0.1) * (1 + param_eta / 2) # Example influence

    metrics_data = {
        settings.SSE_METRIC_KEY: mock_sse,
        settings.STABILITY_METRIC_KEY: mock_h_norm,
        "simulation_duration_s": simulation_duration
    }

    # --- Save to HDF5 File ---
    output_filename = f"simulation_data_{job_uuid}.h5"
    output_path = os.path.join(settings.DATA_DIR, output_filename)

    os.makedirs(settings.DATA_DIR, exist_ok=True)

    try:
        with h5py.File(output_path, 'w') as f:
            # Save simulation parameters
            f.attrs['job_uuid'] = job_uuid
            f.attrs['global_seed'] = params['global_seed']
            for key, value in params['simulation'].items():
                f.attrs[f'sim_{key}'] = value
            for key, value in params['sncgl_params'].items():
                f.attrs[f'sncgl_{key}'] = value

            # Save mock field data
            f.create_dataset('field_data', data=mock_field_data)

            # Save mock metrics as attributes or a separate group
            metrics_group = f.create_group('metrics')
            for key, value in metrics_data.items():
                metrics_group.attrs[key] = value
        log.info(f"[WorkerStub {job_uuid[:8]}] HDF5 data saved to: {output_path}")
        return True
    except Exception as e:
        log.error(f"[WorkerStub {job_uuid[:8]}] FAILED to write HDF5: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="S-NCGL+SDG Worker Stub")
    parser.add_argument("--params", required=True, help="Path to the config_{job_uuid}.json file")
    parser.add_argument("--job_uuid", required=True, help="The unified job_uuid")
    args = parser.parse_args()

    log.info(f"[WorkerStub {args.job_uuid[:8]}] Starting...")

    try:
        with open(args.params, 'r') as f:
            params = json.load(f)
        log.info(f"[WorkerStub {args.job_uuid[:8]}] Loaded params (Seed: {params.get('global_seed')}) from {args.params}")
    except Exception as e:
        log.error(f"[WorkerStub {args.job_uuid[:8]}] Failed to load params file: {e}")
        sys.exit(1)

    # Call the actual (mock) simulation logic
    if not run_sncgl_sdg_simulation(params, args.job_uuid):
        log.error(f"[WorkerStub {args.job_uuid[:8]}] Simulation failed.")
        sys.exit(1)

    log.info(f"[WorkerStub {args.job_uuid[:8]}] Work complete.")

if __name__ == "__main__":
    main()
