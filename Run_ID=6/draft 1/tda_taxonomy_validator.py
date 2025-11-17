"""
tda_taxonomy_validator.py
CLASSIFICATION: Structural Validation Module (ASTE V10.0)
GOAL: Performs Topological Data Analysis (TDA) to validate the
      structural integrity of emergent phenomena ("Quantules") by
      computing and visualizing their persistent homology.
"""


import os
import sys
import argparse
import pandas as pd
import numpy as np


# --- Dependency Check for TDA Libraries ---
try:
    from ripser import ripser
    from persim import plot_diagrams
    import matplotlib.pyplot as plt
    TDA_LIBS_AVAILABLE = True
except ImportError:
    TDA_LIBS_AVAILABLE = False


def load_collapse_data(filepath: str) -> np.ndarray:
    """Loads the (x, y, z) coordinates from a quantule_events.csv file."""
    print(f"[TDA] Loading collapse data from: {filepath}...")
    if not os.path.exists(filepath):
        print(f"ERROR: File not found: {filepath}", file=sys.stderr)
        return None
    try:
        df = pd.read_csv(filepath)
        if 'x' not in df.columns or 'y' not in df.columns or 'z' not in df.columns:
            print("ERROR: CSV must contain 'x', 'y', and 'z' columns.", file=sys.stderr)
            return None


        point_cloud = df[['x', 'y', 'z']].values
        if point_cloud.shape[0] == 0:
            print("WARNING: CSV contains no data points.", file=sys.stderr)
            return None


        print(f"[TDA] Loaded {len(point_cloud)} collapse events.")
        return point_cloud
    except Exception as e:
        print(f"ERROR: Could not load data. {e}", file=sys.stderr)
        return None


def compute_persistence(data: np.ndarray, max_dim: int = 2) -> dict:
    """Computes persistent homology up to max_dim (H0, H1, H2)."""
    print(f"[TDA] Computing persistent homology (max_dim={max_dim})...")
    result = ripser(data, maxdim=max_dim)
    dgms = result['dgms']
    print("[TDA] Computation complete.")
    return dgms


def plot_taxonomy(dgms: list, run_id: str, output_dir: str):
    """Generates and saves a persistence diagram plot with subplots."""
    print(f"[TDA] Generating persistence diagram plot for {run_id}...")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Persistence Diagrams for {run_id[:10]}", fontsize=16)


    # Plot H0
    plot_diagrams(dgms[0], ax=axes[0], show=False)
    axes[0].set_title("H0 (Connected Components)")


    # Plot H1
    if len(dgms) > 1 and dgms[1].size > 0:
        plot_diagrams(dgms[1], ax=axes[1], show=False)
        axes[1].set_title("H1 (Loops/Tunnels)")
    else:
        axes[1].set_title("H1 (No Features Found)")
        axes[1].text(0.5, 0.5, "No H1 features detected.", ha='center', va='center')


    output_path = os.path.join(output_dir, f"tda_persistence_{run_id}.png")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_path)
    plt.close()
    print(f"[TDA] Plot saved to {output_path}")


def main():
    if not TDA_LIBS_AVAILABLE:
        print("FATAL: TDA Module is BLOCKED.", file=sys.stderr)
        print("Please install dependencies: pip install ripser persim matplotlib", file=sys.stderr)
        sys.exit(1)


    parser = argparse.ArgumentParser(description="TDA Taxonomy Validator")
    parser.add_argument("--hash", required=True, help="The config_hash of the run to analyze.")
    parser.add_argument("--datadir", default="./simulation_data", help="Directory containing event CSVs.")
    parser.add_argument("--outdir", default="./provenance_reports", help="Directory to save plots.")
    args = parser.parse_args()


    print(f"--- TDA Taxonomy Validator for Run: {args.hash[:10]} ---")


    # 1. Load Data
    csv_filename = f"{args.hash}_quantule_events.csv"
    csv_filepath = os.path.join(args.datadir, csv_filename)
    point_cloud = load_collapse_data(csv_filepath)


    if point_cloud is None:
        print("[TDA] Aborting due to data loading failure.")
        sys.exit(1)


    # 2. Compute Persistence
    diagrams = compute_persistence(point_cloud)


    # 3. Generate Plot
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
    plot_taxonomy(diagrams, args.hash, args.outdir)


    print("--- TDA Validation Complete ---")


if __name__ == "__main__":
    main()
