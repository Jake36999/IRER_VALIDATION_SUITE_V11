# validation_pipeline.py
# CLASSIFICATION: Decoupled Validation Suite (IRER V11.0)
# GOAL: Receive UUID, deterministically locate the HDF5 artifact,
#       extract core metrics, and generate a final provenance report.


import argparse
import os
import sys
import json
import h5py
import logging

# Import centralized configuration
try:
    import settings
except ImportError:
    print("FATAL: 'settings.py' not found. Ensure all modules are in place.", file=sys.stderr)
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger()

def main():
    parser = argparse.ArgumentParser(description="V11.0 Validation and Provenance Pipeline")
    parser.add_argument("--job_uuid", required=True, help="Unique identifier for the completed run")
    parser.add_argument("--params", required=True, help="Path to the original parameter config JSON file")
    args = parser.parse_args()

    log.info(f"[Validator {args.job_uuid[:8]}] Starting validation...")

    # --- 1. Artifact Retrieval ---
    hdf5_path = os.path.join(settings.DATA_DIR, f"simulation_data_{args.job_uuid}.h5")

    if not os.path.exists(hdf5_path):
        log.error(f"[Validator {args.job_uuid[:8]}] DEADLOCK FAILURE: Worker artifact not found at {hdf5_path}")
        sys.exit(1)

    # --- 2. Metric Extraction ---
    # This direct HDF5 attribute access is the mandated fix for the V10.0 data contract
    # failures, which were caused by inconsistent identifiers and data formats.
    try:
        with h5py.File(hdf5_path, 'r') as f:
            metrics_group = f['metrics']
            sse = metrics_group.attrs[settings.SSE_METRIC_KEY]
            h_norm = metrics_group.attrs[settings.STABILITY_METRIC_KEY]
        log.info(f"[Validator {args.job_uuid[:8]}] Metrics extracted successfully from HDF5.")
    except Exception as e:
        log.error(f"[Validator {args.job_uuid[:8]}] FAILED to read metrics from HDF5: {e}")
        sys.exit(1)
        
    # --- 3. Provenance Artifact Generation ---
    try:
        with open(args.params, 'r') as f:
            params_data = json.load(f)
    except Exception as e:
        log.warning(f"[Validator {args.job_uuid[:8]}] Could not load params file {args.params}: {e}")
        params_data = {}

    payload = {
        "job_uuid": args.job_uuid,
        "params": params_data,
        "metrics": {
            settings.SSE_METRIC_KEY: sse,
            settings.STABILITY_METRIC_KEY: h_norm,
        }
    }
    
    # --- 4. Save Final Report ---
    output_path = os.path.join(settings.PROVENANCE_DIR, f"provenance_{args.job_uuid}.json")
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
