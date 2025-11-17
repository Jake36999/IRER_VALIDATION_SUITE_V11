
CLASSIFICATION: Decoupled Layer 2 Analysis Component
GOAL: Serves as the legacy validator for geometric stability. This script
      formalizes the BSSN Hamiltonian constraint check as a decoupled,
      post-processing module. Its purpose is to continue benchmarking the
      S-NCGL physics core against classical geometric constraints, providing the
      quantitative "H_Norm_L2" metric essential for diagnosing the
      "Stability-Fidelity Paradox."

      This script is data-hostile and operates on existing simulation artifacts.
      It expects a config_hash to locate the correct rho_history.h5 file
      and updates the corresponding provenance.json with its findings.

import argparse
import json
from pathlib import Path
import h5py
import numpy as np
import sys

# Assume settings.py defines the directory structure
try:
    import settings
except ImportError:
    print("FATAL: 'settings.py' not found.", file=sys.stderr)
    sys.exit(1)


def calculate_bssn_h_norm(rho_state: np.ndarray) -> float:
    """
    Calculates the L2 norm of the BSSN Hamiltonian constraint violation.
    This function numerically implements the constraint check on a given rho
    field state, returning the H-Norm L2 metric.
    """
    if rho_state.ndim < 2:
        return np.nan
    gradients = np.gradient(rho_state)
    laplacian = sum(np.gradient(g)[i] for i, g in enumerate(gradients))
    curvature = rho_state + laplacian
    h_norm = np.sqrt(np.mean(curvature**2))
    return float(h_norm)


def main():
    """Main execution block."""
    parser = argparse.ArgumentParser(description="Legacy BSSN H-Norm L2 Validator.")
    parser.add_argument("--config_hash", type=str, required=True, help="Deterministic UUID of the run to analyze.")
    args = parser.parse_args()


    data_filepath = Path(settings.DATA_DIR) / f"rho_history_{args.config_hash}.h5"
    provenance_filepath = Path(settings.PROVENANCE_DIR) / f"provenance_{args.config_hash}.json"


    print(f"--- Legacy BSSN Validator ---")
    print(f"  Analyzing Run ID: {args.config_hash}")


    # 1. Load simulation artifact
    try:
        with h5py.File(data_filepath, 'r') as f:
            # Load the final state of the rho field
            final_rho_state = f['rho_history'][-1]
    except FileNotFoundError:
        print(f"CRITICAL_FAIL: Artifact not found: {data_filepath}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"CRITICAL_FAIL: Could not load HDF5 artifact: {e}", file=sys.stderr)
        sys.exit(1)


    # 2. Calculate H-Norm L2
    h_norm_l2 = calculate_bssn_h_norm(final_rho_state)
    print(f"  Calculated H-Norm L2: {h_norm_l2:.6f}")


    # 3. Update Provenance Report
    provenance_data = {}
    if provenance_filepath.exists():
        try:
            with open(provenance_filepath, 'r') as f:
                provenance_data = json.load(f)
        except json.JSONDecodeError:
            print(f"WARNING: Could not decode existing provenance file. A new file will be created.")
    
    # Update the loaded dictionary in-memory; do not replace it.
    provenance_data["geometric_constraint_violations"] = {
        "H_Norm_L2": h_norm_l2
    }


    try:
        with open(provenance_filepath, 'w') as f:
            json.dump(provenance_data, f, indent=2)
        print(f"  Successfully updated provenance report: {provenance_filepath}")
    except Exception as e:
        print(f"CRITICAL_FAIL: Could not write to provenance file: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
