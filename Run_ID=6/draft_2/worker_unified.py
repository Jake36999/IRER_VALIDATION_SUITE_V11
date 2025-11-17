#!/usr/bin/env python3
"""
worker_unified.py
CLASSIFICATION: JAX Physics Engine (ASTE V10.1 - S-NCGL Core)
GOAL: Execute the Sourced Non-Local Complex Ginzburg-Landau (S-NCGL) simulation.
      This component is architected to be called by an orchestrator,
      is optimized for GPU execution, and adheres to the jax.lax.scan HPC mandate.
"""

import os
import sys
import json
import time
import argparse
import traceback
import h5py
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from functools import partial
from typing import Dict, Any, Tuple, NamedTuple

# Import Core Physics Bridge
try:
    from gravity.unified_omega import jnp_derive_metric_from_rho
except ImportError:
    print("Error: Cannot import jnp_derive_metric_from_rho from gravity.unified_omega", file=sys.stderr)
    print("Please ensure 'gravity/unified_omega.py' and 'gravity/__init__.py' (even if empty) exist.", file=sys.stderr)
    sys.exit(1)

# Define the explicit state carrier for the simulation
class SimState(NamedTuple):
    A_field: jnp.ndarray
    rho: jnp.ndarray
    k_squared: jnp.ndarray
    K_fft: jnp.ndarray
    key: jnp.ndarray


def precompute_kernels(grid_size: int, sigma_k: float) -> Tuple[jnp.ndarray, jnp.ndarray]:
    k_vals = 2 * jnp.pi * jnp.fft.fftfreq(grid_size, d=1.0 / grid_size)
    kx, ky, kz = jnp.meshgrid(k_vals, k_vals, k_vals, indexing='ij')
    k_squared = kx**2 + ky**2 + kz**2
    K_fft = jnp.exp(-k_squared / (2.0 * (sigma_k**2)))
    return k_squared, K_fft


def s_ncgl_simulation_step(state: SimState, _, dt: float, alpha: float, kappa: float, c_diffusion: float, c_nonlinear: float) -> Tuple[SimState, jnp.ndarray]:
    A_field, rho, k_squared, K_fft, key = state
    step_key, next_key = jax.random.split(key)


    # S-NCGL Equation Terms
    A_fft = jnp.fft.fftn(A_field)


    # Linear Operator (Diffusion)
    linear_op = -(c_diffusion + 1j * alpha) * k_squared
    A_linear_fft = A_fft * jnp.exp(linear_op * dt)
    A_linear = jnp.fft.ifftn(A_linear_fft)


    # Non-Local Splash Term (Convolution in Fourier space)
    rho_fft = jnp.fft.fftn(rho)
    non_local_term_fft = K_fft * rho_fft
    non_local_term = jnp.fft.ifftn(non_local_term_fft).real


    # Non-Linear Term
    nonlinear_term = (1 + 1j * c_nonlinear) * jnp.abs(A_linear)**2 * A_linear


    # Step forward
    A_new = A_linear + dt * (kappa * non_local_term * A_linear - nonlinear_term)
    rho_new = jnp.abs(A_new)**2


    new_state = SimState(A_field=A_new, rho=rho_new, k_squared=k_squared, K_fft=K_fft, key=next_key)
    return new_state, rho_new  # (carry, history_slice)


def np_find_collapse_points(rho_state: np.ndarray, threshold: float, max_points: int) -> np.ndarray:
    points = np.argwhere(rho_state > threshold)
    if len(points) > max_points:
        indices = np.random.choice(len(points), max_points, replace=False)
        points = points[indices]
    return points


def run_simulation(config: Dict[str, Any], config_hash: str, output_dir: str) -> bool:
    try:
        params = config['params']
        grid_size = config.get('grid_size', 32)
        num_steps = config.get('T_steps', 500)
        dt = 0.01


        print(f"[Worker] Run {config_hash[:10]}... Initializing.")


        # 1. Initialize Simulation
        key = jax.random.PRNGKey(config.get("global_seed", 0))
        initial_A = jax.random.normal(key, (grid_size, grid_size, grid_size), dtype=jnp.complex64) * 0.1
        initial_rho = jnp.abs(initial_A)**2


        # 2. Precompute Kernels from parameters
        k_squared, K_fft = precompute_kernels(grid_size, params['param_sigma_k'])


        # 3. Create Initial State
        initial_state = SimState(A_field=initial_A, rho=initial_rho, k_squared=k_squared, K_fft=K_fft, key=key)


        # 4. Create a partial function to handle static arguments for JIT
        step_fn_jitted = partial(s_ncgl_simulation_step,
                                 dt=dt,
                                 alpha=params['param_alpha'],
                                 kappa=params['param_kappa'],
                                 c_diffusion=params.get('param_c_diffusion', 0.1),
                                 c_nonlinear=params.get('param_c_nonlinear', 1.0))


        # 5. Run the Simulation using jax.lax.scan
        print(f"[Worker] JAX: Compiling and running scan for {num_steps} steps...")
        start_run = time.time()
        final_carry, rho_history = jax.lax.scan(jax.jit(step_fn_jitted), initial_state, None, length=num_steps)
        final_carry.rho.block_until_ready()
        run_time = time.time() - start_run
        print(f"[Worker] JAX: Scan complete in {run_time:.4f}s")


        final_rho_state = np.asarray(final_carry.rho)


        # --- Artifact 1: HDF5 History ---
        h5_path = os.path.join(output_dir, f"rho_history_{config_hash}.h5")
        print(f"[Worker] Saving HDF5 artifact to: {h5_path}")
        with h5py.File(h5_path, 'w') as f:
            f.create_dataset('rho_history', data=np.asarray(rho_history), compression="gzip")
            f.create_dataset('final_rho', data=final_rho_state)


        # --- Artifact 2: TDA Point Cloud ---
        csv_path = os.path.join(output_dir, f"{config_hash}_quantule_events.csv")
        print(f"[Worker] Generating TDA point cloud...")
        collapse_points_np = np_find_collapse_points(final_rho_state, threshold=0.1, max_points=2000)


        print(f"[Worker] Found {len(collapse_points_np)} collapse points for TDA.")
        if len(collapse_points_np) > 0:
            int_indices = tuple(collapse_points_np.astype(int).T)
            magnitudes = final_rho_state[int_indices]
            df = pd.DataFrame(collapse_points_np, columns=['x', 'y', 'z'])
            df['magnitude'] = magnitudes
            df['quantule_id'] = range(len(df))
            df = df[['quantule_id', 'x', 'y', 'z', 'magnitude']]
            df.to_csv(csv_path, index=False)
            print(f"[Worker] Saved TDA artifact to: {csv_path}")
        else:
            pd.DataFrame(columns=['quantule_id', 'x', 'y', 'z', 'magnitude']).to_csv(csv_path, index=False)
            print(f"[Worker] No collapse points found. Saved empty TDA artifact.")


        print(f"[Worker] Run {config_hash[:10]}... SUCCEEDED.")
        return True
    except Exception as e:
        print(f"[Worker] CRITICAL_FAIL: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ASTE JAX Simulation Worker (V10.1)")
    parser.add_argument("--params", type=str, required=True, help="Path to the input config JSON file.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save artifacts.")


    args = parser.parse_args()


    try:
        with open(args.params, 'r') as f:
            config = json.load(f)
        config_hash = config['config_hash']
    except Exception as e:
        print(f"[Worker Error] Failed to load or parse params file: {e}", file=sys.stderr)
        sys.exit(1)


    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)


    success = run_simulation(config, config_hash, args.output_dir)
    sys.exit(0 if success else 1)
