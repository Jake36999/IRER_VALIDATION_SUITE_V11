# V11.0: S-NCGL Physics Worker (Phase 2 Core Upgrade)
# Mandate: Implement S-NCGL EOM coupled with SDG solver, using received UUID.

import jax
import jax.numpy as jnp
import numpy as np
import json
import argparse
import os
import h5py
import settings
from functools import partial

# Ensure necessary physics components are available
from solver_sdg import solve_sdg_geometry, calculate_informational_stress_energy, apply_complex_diffusion

# Placeholder for complex physics logic (Non-Local Kernel K)
@jax.jit
def apply_non_local_term(Psi: jnp.ndarray, sncgl_g_nonlocal: float) -> jnp.ndarray:
    """Placeholder for the Non-Local 'Splash' Term Phi(A). Derived from L_Non_Local."""
    rho = jnp.abs(Psi)**2
    # Simplified non-local interaction (mean-field coupling)
    mean_rho = jnp.mean(rho)
    # Phi(A) = g * A * Integral(...)
    non_local_contribution = sncgl_g_nonlocal * Psi * mean_rho
    return non_local_contribution

# The core evolution function, structured for JAX JIT compilation
@partial(jax.jit, static_argnames=('sncgl_epsilon', 'sncgl_lambda', 'sncgl_g_nonlocal', 'sdg_kappa', 'sdg_eta', 'spatial_resolution', 'sdg_alpha', 'sdg_rho_vac'))
def _evolve_sncgl_step(carry, _, sncgl_epsilon: float, sncgl_lambda: float, sncgl_g_nonlocal: float, sdg_kappa: float, sdg_eta: float, spatial_resolution: int, sdg_alpha: float, sdg_rho_vac: float):
    """
    One step of the coupled S-NCGL/SDG co-evolution.
    """
    Psi, rho_s, g_mu_nu = carry

    # --- 1. S-NCGL EOM Terms ---
    L_term = sncgl_epsilon * Psi
    NL_term = (1.0 + 1j * 0.0) * jnp.abs(Psi)**2 * Psi * sncgl_lambda
    Diff_term = apply_complex_diffusion(Psi, sncgl_epsilon, g_mu_nu)
    NonL_term = apply_non_local_term(Psi, sncgl_g_nonlocal)
    dPsi_dt = L_term + Diff_term - NL_term - NonL_term

    dt = 0.01
    Psi_new = Psi + dt * dPsi_dt

    # --- 2. Geometric Feedback Loop (Source -> Solve -> Feedback) ---
    T_info = calculate_informational_stress_energy(Psi_new, sdg_kappa, sdg_eta)
    rho_s_new, g_mu_nu_new = solve_sdg_geometry(T_info, rho_s, spatial_resolution, sdg_alpha, sdg_rho_vac)

    return (Psi_new, rho_s_new, g_mu_nu_new), None # No output from scan for now

def run_sncgl_sdg_coevolution(run_uuid: str, config_path: str):
    with open(config_path, 'r') as f:
        params = json.load(f)

    print(f"Starting co-evolution for UUID: {run_uuid}")

    N = params["spatial_resolution"]
    key = jax.random.PRNGKey(42)

    # Generate complex initial field for Psi
    key, subkey1, subkey2 = jax.random.split(key, 3)
    real_part = jax.random.normal(subkey1, (N, N), dtype=jnp.float32) * 0.1
    imag_part = jax.random.normal(subkey2, (N, N), dtype=jnp.float32) * 0.1
    Psi = real_part + 1j * imag_part

    rho_s = jnp.ones((N, N)) * params["sdg_rho_vac"]
    eta_mu_nu = jnp.diag(jnp.array([-1.0, 1.0, 1.0, 1.0]))
    g_mu_nu = jnp.tile(eta_mu_nu[:, :, None, None], (1, 1, N, N))

    # Extract all fixed parameters for partial application to _evolve_sncgl_step
    sncgl_epsilon = params["sncgl_epsilon"]
    sncgl_lambda = params["sncgl_lambda"]
    sncgl_g_nonlocal = params["sncgl_g_nonlocal"]
    sdg_kappa = params["sdg_kappa"]
    sdg_eta = params["sdg_eta"]
    spatial_resolution = params["spatial_resolution"]
    sdg_alpha = params["sdg_alpha"]
    sdg_rho_vac = params["sdg_rho_vac"]
    time_steps = params["time_steps"]

    # Partial apply fixed parameters to the JIT-compiled step function
    evolve_fn = partial(
        _evolve_sncgl_step,
        sncgl_epsilon=sncgl_epsilon,
        sncgl_lambda=sncgl_lambda,
        sncgl_g_nonlocal=sncgl_g_nonlocal,
        sdg_kappa=sdg_kappa,
        sdg_eta=sdg_eta,
        spatial_resolution=spatial_resolution,
        sdg_alpha=sdg_alpha,
        sdg_rho_vac=sdg_rho_vac
    )

    initial_carry = (Psi, rho_s, g_mu_nu)
    
    # Use jax.lax.scan for a performant, JIT-compiled loop
    # Note: jax.lax.scan returns (final_carry, y) where y is the stacked results of the second return value of fn.
    # Since _evolve_sncgl_step returns (new_carry, None), the y will be None.
    final_carry, _ = jax.lax.scan(evolve_fn, initial_carry, None, length=time_steps)
    final_Psi, final_rho_s, final_g_mu_nu = final_carry
    
    # Ensure computation is finished before saving
    final_Psi.block_until_ready()

    # Placeholder for calculating SSE and H-norm from final state
    sse_metric = 0.0 # Replace with actual calculation
    h_norm = 0.0 # Replace with actual calculation

    # --- Save Artifact (MANDATE: Must use the received UUID) ---
    data_dir = settings.DATA_DIR
    os.makedirs(data_dir, exist_ok=True)
    rho_path = os.path.join(data_dir, f"rho_history_{run_uuid}.h5")

    print(f"Saving artifact to {rho_path}...")
    with h5py.File(rho_path, 'w') as f:
        f.create_dataset('final_psi', data=np.array(final_Psi))
        f.create_dataset('final_rho_s', data=np.array(final_rho_s))
        f.create_dataset('final_g_mu_nu', data=np.array(final_g_mu_nu))
        
        # Save metrics as attributes for easy access by validator
        f.attrs['uuid'] = run_uuid
        f.attrs['time_steps'] = time_steps
        f.attrs[settings.SSE_METRIC_KEY] = sse_metric
        f.attrs[settings.STABILITY_METRIC_KEY] = h_norm

    print(f"Run {run_uuid} finished and artifact saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IRER V11.0 S-NCGL/SDG Worker.")
    # MANDATE: Worker must receive the hash from the Orchestrator.
    parser.add_argument("--config_hash", required=True, help="Deterministic UUID for the run.")
    parser.add_argument("--config_path", required=True, help="Path to the configuration JSON file.")
    args = parser.parse_args()
    run_sncgl_sdg_coevolution(args.config_hash, args.config_path)
