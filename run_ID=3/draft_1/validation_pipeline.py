"""
validation_pipeline.py
CLASSIFICATION: HPC Core (Layer 1)
GOAL: Calculates metrics from the worker's output and writes the
      critical provenance.json file.
"""
import argparse
import time
import os
import json
import random # Keep for potential future use or if some metrics are still random
import logging
import settings # Need this to find the PROVENANCE_DIR and metric keys
import h5py # Import h5py to read HDF5 files
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger()

def main():
    parser = argparse.ArgumentParser(description="Validator Stub")
    parser.add_argument("--job_uuid", required=True, help="The unified job_uuid")
    args = parser.parse_args()

    log.info(f"[Validator {args.job_uuid[:8]}] Starting...")

    # --- Construct path to the worker's HDF5 output ---
    h5_filename = f"simulation_data_{args.job_uuid}.h5"
    h5_filepath = os.path.join(settings.DATA_DIR, h5_filename)

    # Initialize metrics with default (error) values
    sse_metric = 999.0
    h_norm_metric = 999.0
    simulation_duration = 0.0

    try:
        # --- Read metrics from HDF5 file ---
        if not os.path.exists(h5_filepath):
            raise FileNotFoundError(f"HDF5 file not found: {h5_filepath}")

        with h5py.File(h5_filepath, 'r') as f:
            # Access metrics from the 'metrics' group attributes
            if 'metrics' in f and f['metrics'].attrs:
                sse_metric = f['metrics'].attrs.get(settings.SSE_METRIC_KEY, sse_metric)
                h_norm_metric = f['metrics'].attrs.get(settings.STABILITY_METRIC_KEY, h_norm_metric)
                simulation_duration = f['metrics'].attrs.get('simulation_duration_s', simulation_duration)
            else:
                log.warning(f"[Validator {args.job_uuid[:8]}] Metrics group or attributes not found in {h5_filename}.")

        log.info(f"[Validator {args.job_uuid[:8]}] Extracted metrics: SSE={sse_metric:.4f}, H_Norm={h_norm_metric:.4f}")

    except FileNotFoundError as e:
        log.error(f"[Validator {args.job_uuid[:8]}] Failed to find HDF5 file: {e}. Cannot calculate metrics.")
    except Exception as e:
        log.error(f"[Validator {args.job_uuid[:8]}] Error reading HDF5 file {h5_filepath}: {e}. Metrics will be default.")

    # Simulate analysis work (if any, separate from file I/O)
    time.sleep(random.uniform(0.1, 0.5))

    # --- Use extracted metrics for provenance file creation ---
    metrics = {
        settings.SSE_METRIC_KEY: sse_metric,
        settings.STABILITY_METRIC_KEY: h_norm_metric,
        "simulation_duration_s": simulation_duration
    }

    # --- PROVENANCE FILE CREATION ---
    payload = {
        settings.HASH_KEY: args.job_uuid,
        "metrics": metrics,
        "timestamp": time.time()
    }

    output_filename = f"provenance_{args.job_uuid}.json"
    output_path = os.path.join(settings.PROVENANCE_DIR, output_filename)

    try:
        os.makedirs(settings.PROVENANCE_DIR, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(payload, f, indent=2)
        log.info(f"[Validator {args.job_uuid[:8]}] Provenance file saved: {output_path}")
    except Exception as e:
        log.error(f"[Validator {args.job_uuid[:8]}] FAILED to write provenance: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
