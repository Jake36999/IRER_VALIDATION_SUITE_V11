#!/usr/bin/env python3
"""
run_invariance_test_p11.py
CLASSIFICATION: Advanced Validation Module (ASTE V10.0)
PURPOSE: Validates that the deconvolution process is invariant to the
         instrument function, recovering the same primordial signal
         from multiple measurements. Confirms the physical reality of the signal.
"""
import os
import sys
import numpy as np
from typing import Dict, List


# Import the mandated deconvolution function
try:
    from deconvolution_validator import perform_regularized_division, calculate_sse
except ImportError:
    print("FATAL: 'deconvolution_validator.py' not found.", file=sys.stderr)
    sys.exit(1)


def load_convolved_signal_P11(filepath: str) -> np.ndarray:
    """Loads a convolved signal artifact, failing if not found."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Missing P11 data artifact: {filepath}")
    return np.load(filepath)


def _reconstruct_pump_intensity_alpha_sq(shape: tuple, bandwidth_nm: float) -> np.ndarray:
    """Reconstructs the Gaussian Pump Intensity |alpha|^2."""
    w_range = np.linspace(-3, 3, shape[0])
    w_s, w_i = np.meshgrid(w_range, w_range, indexing='ij')
    sigma_w = 1.0 / (bandwidth_nm * 0.5)
    pump_amplitude = np.exp(- (w_s + w_i)**2 / (2 * sigma_w**2))
    pump_intensity = np.abs(pump_amplitude)**2
    return pump_intensity / np.max(pump_intensity)


def _reconstruct_pmf_intensity_phi_sq(shape: tuple, L_mm: float = 20.0) -> np.ndarray:
    """Reconstructs the Phase-Matching Function Intensity |phi|^2 for a 20mm ppKTP crystal."""
    w_range = np.linspace(-3, 3, shape[0])
    w_s, w_i = np.meshgrid(w_range, w_range, indexing='ij')
    sinc_arg = L_mm * 0.1 * (w_s - w_i)
    pmf_amplitude = np.sinc(sinc_arg / np.pi)
    return np.abs(pmf_amplitude)**2


def reconstruct_instrument_function_P11(shape: tuple, bandwidth_nm: float) -> np.ndarray:
    """Constructs the full instrument intensity from pump and PMF components."""
    Pump_Intensity = _reconstruct_pump_intensity_alpha_sq(shape, bandwidth_nm)
    PMF_Intensity = _reconstruct_pmf_intensity_phi_sq(shape)
    return Pump_Intensity * PMF_Intensity


def main():
    print("--- Invariance Test (Candidate P11) ---")
    DATA_DIR = "./data"


    if not os.path.isdir(DATA_DIR):
        print(f"FATAL: Data directory '{DATA_DIR}' not found.", file=sys.stderr)
        sys.exit(1)


    P11_RUNS = {
        "C1": {"bandwidth_nm": 4.1, "path": os.path.join(DATA_DIR, "P11_C1_4.1nm.npy")},
        "C2": {"bandwidth_nm": 2.1, "path": os.path.join(DATA_DIR, "P11_C2_2.1nm.npy")},
        "C3": {"bandwidth_nm": 1.0, "path": os.path.join(DATA_DIR, "P11_C3_1.0nm.npy")},
    }


    DECON_K = 1e-3
    all_recovered_signals = []


    try:
        print(f"[P11 Test] Starting Invariance Test on {len(P11_RUNS)} datasets...")
        for run_name, config in P11_RUNS.items():
            print(f"\n--- Processing Run: {run_name} (BW: {config['bandwidth_nm']}nm) ---")


            # 1. LOAD the convolved signal (JSI_n)
            JSI = load_convolved_signal_P11(config['path'])


            # 2. RECONSTRUCT the instrument function (I_n)
            Instrument_Func = reconstruct_instrument_function_P11(JSI.shape, config['bandwidth_nm'])


            # 3. DECONVOLVE to recover the primordial signal (P_recovered_n)
            P_recovered = perform_regularized_division(JSI, Instrument_Func, DECON_K)
            all_recovered_signals.append(P_recovered)
            print(f"[P11 Test] Deconvolution for {run_name} complete.")


        # 4. VALIDATE INVARIANCE by comparing the recovered signals
        if len(all_recovered_signals) < 2:
            print("\nWARNING: Need at least two signals to test invariance.")
            return


        reference_signal = all_recovered_signals[0]
        all_sses = []
        for i, signal in enumerate(all_recovered_signals[1:], 1):
            sse = calculate_sse(signal, reference_signal)
            all_sses.append(sse)
            print(f"[P11 Test] SSE between Run 0 and Run {i}: {sse:.6f}")


        mean_sse = np.mean(all_sses)
        std_dev = np.std(all_sses)
        rel_std_dev = (std_dev / mean_sse) * 100 if mean_sse > 1e-9 else 0.0


        print("\n--- Invariance Analysis ---")
        print(f"Mean SSE: {mean_sse:.6f}")
        print(f"Std Deviation: {std_dev:.6f}")
        print(f"Relative Std Dev: {rel_std_dev:.2f}%")


        if rel_std_dev < 15.0:
            print("\n✅ INVARIANCE TEST SUCCESSFUL!")
            print("The recovered primordial signal is stable across all instrument functions.")
        else:
            print("\n❌ INVARIANCE TEST FAILED.")
            print("The recovered signal is not invariant, suggesting a model or data error.")


    except FileNotFoundError as e:
        print(f"\nFATAL ERROR: {e}", file=sys.stderr)
        print("This script requires P11 data artifacts. Ensure they are present in ./data/", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\nAn unexpected error occurred during the test: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
