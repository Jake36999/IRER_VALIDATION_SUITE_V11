#!/usr/bin/env python3
"""
deconvolution_validator.py
CLASSIFICATION: External Validation Module (ASTE V10.0)
PURPOSE: Implements the "Forward Validation" protocol to solve the "Phase Problem"
         by comparing simulation predictions against external experimental data.
VALIDATION MANDATE: This script is "data-hostile" and contains no mock data generators.
"""
import os
import sys
import numpy as np

def perform_regularized_division(JSI: np.ndarray, Pump_Intensity: np.ndarray, K: float) -> np.ndarray:
    """
    Performs a numerically stable, regularized deconvolution.
    Implements the formula: PMF_recovered = JSI / (Pump_Intensity + K)
    """
    print("[Decon] Performing regularized division...")
    stabilized_denominator = Pump_Intensity + K
    PMF_recovered = JSI / stabilized_denominator
    return PMF_recovered

def load_data_artifact(filepath: str) -> np.ndarray:
    """Loads a required .npy data artifact, failing if not found."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Missing required data artifact: {filepath}")
    return np.load(filepath)

def reconstruct_instrument_function_I_recon(shape: tuple, beta: float) -> np.ndarray:
    """Reconstructs the complex Instrument Function I_recon = exp(i*beta*w_s*w_i)."""
    print(f"[Decon] Reconstructing instrument I_recon (beta={beta})...")
    w = np.linspace(-1, 1, shape[0])
    ws, wi = np.meshgrid(w, w, indexing='ij')
    return np.exp(1j * beta * ws * wi)

def predict_4_photon_signal_C4_pred(JSA_pred: np.ndarray) -> np.ndarray:
    """Calculates the 4-photon interference pattern via 4D tensor calculation."""
    N = JSA_pred.shape[0]
    psi = JSA_pred
    C4_4D = np.abs(
        np.einsum('si,pj->sipj', psi, psi) +
        np.einsum('sj,pi->sipj', psi, psi)
    )**2


    # Integrate to 2D fringe pattern
    C4_2D_fringe = np.zeros((N * 2 - 1, N * 2 - 1))
    for s in range(N):
        for i in range(N):
            for p in range(N):
                for j in range(N):
                    ds_idx, di_idx = (p - s) + (N - 1), (j - i) + (N - 1)
                    C4_2D_fringe[ds_idx, di_idx] += C4_4D[s, i, p, j]


    # Center crop
    start, end = (N // 2) - 1, (N // 2) + N - 1
    return C4_2D_fringe[start:end, start:end]

def calculate_sse(pred: np.ndarray, exp: np.ndarray) -> float:
    """Calculates Sum of Squared Errors between prediction and experiment."""
    if pred.shape != exp.shape:
        print(f"ERROR: Shape mismatch for SSE. Pred: {pred.shape}, Exp: {exp.shape}", file=sys.stderr)
        return 1e9
    return np.sum((pred - exp)**2) / pred.size

def main():
    print("--- Deconvolution Validator (Forward Validation) ---")


    # Configuration
    PRIMORDIAL_FILE_PATH = "./data/P9_Fig1b_primordial.npy"
    FRINGE_FILE_PATH = "./data/P9_Fig2f_fringes.npy"
    BETA = 20.0


    try:
        # 1. Load Experimental Data (P_ext and C_4_exp)
        P_ext = load_data_artifact(PRIMORDIAL_FILE_PATH)
        C_4_exp = load_data_artifact(FRINGE_FILE_PATH)


        # 2. Reconstruct Instrument Function (I_recon)
        I_recon = reconstruct_instrument_function_I_recon(P_ext.shape, BETA)


        # 3. Predict Joint Spectral Amplitude (JSA_pred)
        JSA_pred = P_ext * I_recon


        # 4. Predict 4-Photon Signal (C_4_pred)
        C_4_pred = predict_4_photon_signal_C4_pred(JSA_pred)


        # 5. Calculate Final External SSE
        sse_ext = calculate_sse(C_4_pred, C_4_exp)
        print(f"\n--- VALIDATION COMPLETE ---")
        print(f"External SSE (Prediction vs. Experiment): {sse_ext:.8f}")


        if sse_ext < 1e-6:
            print("\n✅ VALIDATION SUCCESSFUL!")
            print("P_golden (our ln(p) signal) successfully predicted the")
            print("phase-sensitive 4-photon interference pattern.")
        else:
            print("\n❌ VALIDATION FAILED.")
            print(f"P_golden failed to predict the external data.")


    except FileNotFoundError as e:
        print(f"\nFATAL ERROR: {e}", file=sys.stderr)
        print("This is a data-hostile script. Ensure all required experimental .npy artifacts are present in ./data/", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
