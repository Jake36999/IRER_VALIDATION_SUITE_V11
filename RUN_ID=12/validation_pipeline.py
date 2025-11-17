"""
validation_pipeline.py
CLASSIFICATION: V11.0 Validation Service
GOAL: Acts as the streamlined validator for the V11.0 suite.
      Its sole purpose is to load a completed simulation's RAW data
      artifact, CALCULATE the core scientific and stability metrics,
      and generate a canonical provenance report.

This script strictly adheres to the "Unified Hashing Mandate"  and
the "Trust but Verify" audit protocol. By performing analysis
independently on the raw simulation output, it correctly decouples the
CPU-bound validation task from the HPC core.
"""

import os
import argparse
import json
import h5py
import numpy as np

import settings # Import data contract keys

def calculate_log_prime_sse(rho_data: np.ndarray) -> float:
    """
    Placeholder for scientific fidelity metric calculation.
    In a real implementation, this would perform a 2D FFT, identify
    spectral peaks, and calculate the Sum of Squared Errors against
    the log-prime targets.
    """
    # Mock calculation based on the variance of the final field.
    # Lower variance might imply a more ordered, crystalline state.
    variance = np.var(rho_data)
    mock_sse = 0.001 + variance / 10.0
    return float(mock_sse)

def calculate_sdg_h_norm_l2(metric_data: np.ndarray) -> float:
    """
    Placeholder for geometric stability metric calculation.
    In a real implementation, this would calculate the L2 norm of the
    Hamiltonian constraint violation from the SDG solver's output.
    """
    # Mock calculation based on the deviation of the metric from flat space.
    # A value of -1.0 is the flat-space target for g_tt.
    deviation = np.mean(np.abs(metric_data - (-1.0)))
    mock_h_norm = deviation * 0.5
    return float(mock_h_norm)


def validate_run(job_uuid: str):
    """
    Loads a raw HDF5 artifact, calculates key metrics, and saves
    a JSON provenance report.
    """
    print(f"[Validator {job_uuid[:8]}] Starting validation...")

    # --- 1. Artifact Retrieval (V11 Hashing Mandate) ---
    # Deterministically locate the artifact using the passed job_uuid.
    artifact_path = os.path.join(settings.DATA_DIR, f"rho_history_{job_uuid}.h5")
    if not os.path.exists(artifact_path):
        print(f"[Validator {job_uuid[:8]}] CRITICAL FAILURE: Artifact not found at {artifact_path}")
        # Write a failure provenance so the hunter is not blocked
        provenance = {
            settings.HASH_KEY: job_uuid,
            settings.SSE_METRIC_KEY: 999.0,
            settings.STABILITY_METRIC_KEY: 999.0,
            "error": "FileNotFoundError"
        }
    else:
        # --- 2. Independent Metric Calculation (V11 Audit Mandate) ---
        # Load RAW data from the artifact, per "Trust but Verify".
        try:
            with h5py.File(artifact_path, 'r') as f:
                raw_rho = f['final_rho'][()]
                raw_g_tt = f['final_g_tt'][()]

            # Independently calculate all metrics from the raw data.
            sse = calculate_log_prime_sse(raw_rho)
            h_norm = calculate_sdg_h_norm_l2(raw_g_tt)

            print(f"[Validator {job_uuid[:8]}] Metrics calculated: SSE={sse:.4f}, H_Norm={h_norm:.4f}")

            provenance = {
                settings.HASH_KEY: job_uuid,
                settings.SSE_METRIC_KEY: sse,
                settings.STABILITY_METRIC_KEY: h_norm
            }

        except Exception as e:
            print(f"[Validator {job_uuid[:8]}] CRITICAL FAILURE: Failed to read HDF5 artifact: {e}")
            provenance = {
                settings.HASH_KEY: job_uuid,
                settings.SSE_METRIC_KEY: 998.0,
                settings.STABILITY_METRIC_KEY: 998.0,
                "error": str(e)
            }

    # --- 3. Save Provenance Report (V11 Data Contract) ---
    # The output filename MUST use the job_uuid.
    # The content keys MUST use the constants from settings.py.
    output_path = os.path.join(settings.PROVENANCE_DIR, f"provenance_{job_uuid}.json")
    try:
        os.makedirs(settings.PROVENANCE_DIR, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(provenance, f, indent=2)
        print(f"[Validator {job_uuid[:8]}] Provenance report saved to {output_path}")
    except Exception as e:
        print(f"[Validator {job_uuid[:8]}] CRITICAL FAILURE: Failed to write provenance JSON: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="V11.0 Validation & Provenance Service")

    # MANDATE (Unified Hashing): Validator MUST receive the job_uuid
    # from the orchestrator.
    parser.add_argument("--job_uuid", required=True, help="Unique identifier for the completed run.")
    args = parser.parse_args()

    validate_run(args.job_uuid)
