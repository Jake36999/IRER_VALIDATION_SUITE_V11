"""
quantulemapper_real.py
CLASSIFICATION: Spectral Analysis Service (CEPP Profiler V2.0)
GOAL: Perform rigorous, quantitative spectral analysis on simulation artifacts
      to calculate the Sum of Squared Errors (SSE) against the
      Log-Prime Spectral Attractor (k ~ ln(p)). Includes mandatory
      falsifiability null tests.
"""


import math
import random
from typing import List, Tuple, Dict, Any, Optional


# --- Dependency Shim ---
try:
    import numpy as np
    from numpy.fft import fftn, ifftn, rfft
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    print("WARNING: 'numpy' not found. Falling back to 'lite-core' mode.")

try:
    import scipy.signal
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("WARNING: 'scipy' not found. Falling back to 'lite-core' mode.")


# --- Constants ---
LOG_PRIME_TARGETS = [math.log(p) for p in [2, 3, 5, 7, 11, 13, 17, 19]]


# --- Falsifiability Null Tests ---
def _null_a_phase_scramble(rho: np.ndarray) -> Optional[np.ndarray]:
    """Null A: Scramble phases while preserving amplitude."""
    if not HAS_NUMPY:
        print("Skipping Null A (Phase Scramble): NumPy not available.")
        return None
    F = fftn(rho)
    amps = np.abs(F)
    phases = np.random.uniform(0, 2 * np.pi, F.shape)
    F_scr = amps * np.exp(1j * phases)
    scrambled_field = ifftn(F_scr).real
    return scrambled_field

def _null_b_target_shuffle(targets: list) -> list:
    """Null B: Shuffle the log-prime targets."""
    shuffled_targets = list(targets)
    random.shuffle(shuffled_targets)
    return shuffled_targets


# --- Core Spectral Analysis Functions ---
def _quadratic_interpolation(data: list, peak_index: int) -> float:
    """Finds the sub-bin accurate peak location."""
    if peak_index < 1 or peak_index >= len(data) - 1:
        return float(peak_index)
    y0, y1, y2 = data[peak_index - 1 : peak_index + 2]
    denominator = (y0 - 2 * y1 + y2)
    if abs(denominator) < 1e-9:
        return float(peak_index)
    p = 0.5 * (y0 - y2) / denominator
    return float(peak_index) + p if math.isfinite(p) else float(peak_index)

def _get_multi_ray_spectrum(rho: np.ndarray, num_rays: int = 128) -> Tuple[np.ndarray, np.ndarray]:
    """Implements the 'Multi-Ray Directional Sampling' protocol."""
    grid_size = rho.shape[0]
    aggregated_spectrum = np.zeros(grid_size // 2 + 1)
    
    for _ in range(num_rays):
        axis = np.random.randint(3)
        x_idx, y_idx = np.random.randint(grid_size, size=2)
        
        if axis == 0: ray_data = rho[:, x_idx, y_idx]
        elif axis == 1: ray_data = rho[x_idx, :, y_idx]
        else: ray_data = rho[x_idx, y_idx, :]
            
        if len(ray_data) < 4: continue
        
        # Apply mandatory Hann window
        windowed_ray = ray_data * scipy.signal.hann(len(ray_data))
        spectrum = np.abs(rfft(windowed_ray))**2
        
        if np.max(spectrum) > 1e-9:
            aggregated_spectrum += spectrum / np.max(spectrum)
            
    freq_bins = np.fft.rfftfreq(grid_size, d=1.0 / grid_size)
    return freq_bins, aggregated_spectrum / num_rays

def _find_spectral_peaks(freqs: np.ndarray, spectrum: np.ndarray) -> np.ndarray:
    """Finds and interpolates spectral peaks."""
    peaks_indices, _ = scipy.signal.find_peaks(spectrum, height=np.max(spectrum) * 0.1, distance=5)
    if len(peaks_indices) == 0:
        return np.array([])
    
    accurate_peak_bins = np.array([_quadratic_interpolation(spectrum, p) for p in peaks_indices])
    observed_peak_freqs = np.interp(accurate_peak_bins, np.arange(len(freqs)), freqs)
    return observed_peak_freqs

def _get_calibrated_peaks(peak_freqs: np.ndarray, k_target_ln2: float = math.log(2.0)) -> np.ndarray:
    """Calibrates peaks using 'Single-Factor Calibration' to ln(2)."""
    if len(peak_freqs) == 0: return np.array([])
    scaling_factor_S = k_target_ln2 / peak_freqs[0]
    return peak_freqs * scaling_factor_S

def _compute_sse(observed_peaks: np.ndarray, targets: list) -> float:
    """Calculates the Sum of Squared Errors (SSE)."""
    num_targets = min(len(observed_peaks), len(targets))
    if num_targets == 0: return 996.0  # Sentinel for no peaks to match
    squared_errors = (observed_peaks[:num_targets] - targets[:num_targets])**2
    return np.sum(squared_errors)

def prime_log_sse(rho_final_state: np.ndarray) -> Dict:
    """Main function to compute SSE and run null tests."""
    results = {}
    prime_targets = LOG_PRIME_TARGETS


    # --- Treatment (Real SSE) ---
    try:
        freq_bins, spectrum = _get_multi_ray_spectrum(rho_final_state)
        peaks_freqs_main = _find_spectral_peaks(freq_bins, spectrum)
        calibrated_peaks_main = _get_calibrated_peaks(peaks_freqs_main)
        
        if len(calibrated_peaks_main) == 0:
            raise ValueError("No peaks found in main signal")
            
        sse_main = _compute_sse(calibrated_peaks_main, prime_targets)
        results.update({
            "log_prime_sse": sse_main,
            "n_peaks_found_main": len(calibrated_peaks_main),
        })
    except Exception as e:
        results.update({"log_prime_sse": 999.0, "failure_reason_main": str(e)})


    # --- Null A (Phase Scramble) ---
    try:
        scrambled_rho = _null_a_phase_scramble(rho_final_state)
        freq_bins_a, spectrum_a = _get_multi_ray_spectrum(scrambled_rho)
        peaks_freqs_a = _find_spectral_peaks(freq_bins_a, spectrum_a)
        calibrated_peaks_a = _get_calibrated_peaks(peaks_freqs_a)
        sse_null_a = _compute_sse(calibrated_peaks_a, prime_targets)
        results.update({"sse_null_phase_scramble": sse_null_a})
    except Exception as e:
        results.update({"sse_null_phase_scramble": 999.0, "failure_reason_null_a": str(e)})


    # --- Null B (Target Shuffle) ---
    try:
        shuffled_targets = _null_b_target_shuffle(prime_targets)
        sse_null_b = _compute_sse(calibrated_peaks_main, shuffled_targets)
        results.update({"sse_null_target_shuffle": sse_null_b})
    except Exception as e:
        results.update({"sse_null_target_shuffle": 999.0, "failure_reason_null_b": str(e)})


    return results
