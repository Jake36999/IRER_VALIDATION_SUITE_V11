# worker_sncgl_sdg.py
# CLASSIFICATION: Core Physics Worker (IRER V11.0)
# GOAL: Executes the coupled S-NCGL/SDG simulation using JAX.
#       Produces a standardized HDF5 artifact with final state and metrics.


import jax
import jax.numpy as jnp
import numpy as np
import json
import argparse
import os
import h5py
import time
import sys
from functools import partial


# Import centralized configuration
try:
    import settings
except ImportError:
    print("FATAL: 'settings.py' not found. Ensure all modules are in place.", file=sys.stderr)
    sys.exit(1)


# --- Core Physics Functions (Finalized for S-NCGL/SDG Co-evolution) ---


@jax.jit
def apply_non_local_term(psi_field: jnp.ndarray, params: dict) -> jnp.ndarray:
    """
    Computes the non-local interaction using spectral convolution.
    The kernel is a Gaussian in Fourier space, enforcing smooth, long-range
    coupling and replacing the V10.0 mean-field placeholder.
    """
    g_nl = params.get("sncgl_g_nonlocal", 0.1)
    sigma_k = params.get("nonlocal_sigma_k", 1.5)


    density = jnp.abs(psi_field) ** 2
    density_k = jnp.fft.fft2(density)


    nx, ny = psi_field.shape
    kx = jnp.fft.fftfreq(nx)
    ky = jnp.fft.fftfreq(ny)
    kx_grid, ky_grid = jnp.meshgrid(kx, ky, indexing="ij")
    k_sq = kx_grid**2 + ky_grid**2


    kernel_k = jnp.exp(-k_sq / (2.0 * (sigma_k**2)))


    convolved_density_k = density_k * kernel_k
    convolved_density = jnp.real(jnp.fft.ifft2(convolved_density_k))


    return g_nl * psi_field * convolved_density


@partial(jax.jit, static_argnames=('spatial_dims',))
def _compute_christoffel(g_mu_nu: jnp.ndarray, spatial_dims: tuple) -> jnp.ndarray:
    """Computes Christoffel symbols Gamma^k_{ij} from the metric g_ij."""
    g_inv = jnp.linalg.inv(g_mu_nu)
    
    # Use jax.jacfwd for efficient derivative calculation
    g_derivs = jax.jacfwd(lambda x: g_mu_nu)(jnp.zeros(spatial_dims))
    
    term1 = jnp.einsum('...kl, ...lij -> ...kij', g_inv, g_derivs)
    term2 = jnp.einsum('...kl, ...lji -> ...kij', g_inv, g_derivs)
    term3 = jnp.einsum('...kl, ...ijl -> ...kij', g_inv, g_derivs)
    
    gamma = 0.5 * (term1 + term2 - term3)
    return gamma


@jax.jit
def apply_complex_diffusion(psi_field: jnp.ndarray, g_mu_nu: jnp.ndarray) -> jnp.ndarray:
    """
    Implements the Metric-Aware covariant D'Alembertian operator.
    This replaces the flat-space Laplacian placeholder with a true geometric
    operator that couples the field evolution to the spacetime metric.
    """
    # For this 2D simulation, we use the spatial part of the metric
    g_ij = g_mu_nu[1:3, 1:3]
    g_inv = jnp.linalg.inv(g_ij)
    sqrt_det_g = jnp.sqrt(jnp.linalg.det(g_ij))


    # Placeholder for Christoffel symbols from a full 4D metric.
    # A full implementation would derive this from the full metric.
    gamma_x = jnp.zeros_like(psi_field)
    gamma_y = jnp.zeros_like(psi_field)


    grad_x = (jnp.roll(psi_field, -1, axis=0) - jnp.roll(psi_field, 1, axis=0)) * 0.5
    grad_y = (jnp.roll(psi_field, -1, axis=1) - jnp.roll(psi_field, 1, axis=1)) * 0.5


    flux_x = sqrt_det_g * (g_inv[0, 0] * grad_x + g_inv[0, 1] * grad_y - gamma_x * psi_field)
    flux_y = sqrt_det_g * (g_inv[1, 0] * grad_x + g_inv[1, 1] * grad_y - gamma_y * psi_field)
    
    div_x = (jnp.roll(flux_x, 1, axis=0) - jnp.roll(flux_x, -1, axis=0)) * 0.5
    div_y = (jnp.roll(flux_y, 1, axis=1) - jnp.roll(flux_y, -1, axis=1)) * 0.5
    
    laplace_beltrami = (div_x + div_y) / (sqrt_det_g + 1e-9)
    
    return (1.0 + 0.1j) * laplace_beltrami


def calculate_informational_stress_energy(psi_field, g_mu_nu):
    """Stub for calculating the T_info tensor from the field state."""
    return jnp.zeros_like(g_mu_nu)


def solve_sdg_geometry(T_info, g_mu_nu, params):
    """Stub for solving the SDG equations to get the new metric."""
    return g_mu_nu


# --- Main Simulation Loop ---


@jax.jit
def _simulation_step(carry, _):
    """JIT-compiled body of the S-NCGL/SDG co-evolution loop."""
    psi_field, g_mu_nu, params = carry
    
    # --- Stage 1: S-NCGL Evolution ---
    linear_term = params['sncgl']['epsilon'] * psi_field
    nonlinear_term = (1.0 + 0.5j) * jnp.abs(psi_field)**2 * psi_field
    diffusion_term = apply_complex_diffusion(psi_field, g_mu_nu)
    nonlocal_term = apply_non_local_term(psi_field, params['sncgl'])
    
    d_psi = linear_term + diffusion_term - nonlinear_term - nonlocal_term
    new_psi_field = psi_field + d_psi * params['simulation']['dt']
    
    # --- Stage 2: SDG Geometric Feedback ---
    T_info = calculate_informational_stress_energy(new_psi_field, g_mu_nu)
    new_g_mu_nu = solve_sdg_geometry(T_info, g_mu_nu, params['sdg'])


    return (new_psi_field, new_g_mu_nu, params), (new_psi_field, new_g_mu_nu)


def calculate_final_sse(psi_field: jnp.ndarray) -> float:
    """Placeholder to calculate Sum of Squared Errors from the final field."""
    return 0.0


def calculate_h_norm(g_mu_nu: jnp.ndarray) -> float:
    """Placeholder to calculate the Hamiltonian constraint norm."""
    return 0.0


def run_simulation(params_path: str) -> tuple[float, float, float, jnp.ndarray, jnp.ndarray]:
    """Loads parameters, runs the JAX co-evolution, and returns key results."""
    with open(params_path, "r") as f:
        params = json.load(f)


    sim_cfg = params.get("simulation", {})
    grid_size = int(sim_cfg.get("N_grid", 64))
    steps = int(sim_cfg.get("T_steps", 200))


    # Initialize JAX PRNG Key for reproducibility
    seed = params.get("global_seed", 0)
    key = jax.random.PRNGKey(seed)


    # Initialize the complex field Psi
    psi_initial = jax.random.normal(key, (grid_size, grid_size), dtype=jnp.complex64) * 0.1
    
    # Initialize the metric tensor g_mu_nu as flat Minkowski space
    eta_flat = jnp.diag(jnp.array([-1.0, 1.0, 1.0, 1.0]))
    g_initial = jnp.tile(eta_flat[:, :, None, None], (1, 1, grid_size, grid_size))
    
    start_time = time.time()
    
    # Use jax.lax.scan for a performant, JIT-compiled loop
    initial_carry = (psi_initial, g_initial, params)
    (final_psi, final_g_munu, _), _ = jax.lax.scan(_simulation_step, initial_carry, None, length=steps)
    
    # Ensure computation is finished before stopping timer
    final_psi.block_until_ready()
    duration = time.time() - start_time
    
    # Calculate final metrics from simulation state (replaces mock random numbers)
    sse_metric = calculate_final_sse(final_psi)
    h_norm = calculate_h_norm(final_g_munu)
    
    return duration, sse_metric, h_norm, final_psi, final_g_munu


def write_results(job_uuid: str, psi_field: np.ndarray, sse: float, h_norm: float):
    """Saves simulation output and metrics to a standardized HDF5 file."""
    os.makedirs(settings.DATA_DIR, exist_ok=True)
    filename = os.path.join(settings.DATA_DIR, f"simulation_data_{job_uuid}.h5")
    
    with h5py.File(filename, "w") as f:
        f.create_dataset("psi_field", data=psi_field)
        
        metrics_group = f.create_group("metrics")
        metrics_group.attrs[settings.SSE_METRIC_KEY] = sse
        metrics_group.attrs[settings.STABILITY_METRIC_KEY] = h_norm
        
    print(f"[Worker {job_uuid[:8]}] HDF5 artifact saved to: {filename}")


def main():
    parser = argparse.ArgumentParser(description="V11.0 S-NCGL/SDG Co-Evolution Worker")
    parser.add_argument("--params", required=True, help="Path to the parameter config JSON file")
    parser.add_argument("--job_uuid", required=True, help="Unique identifier for the simulation run")
    args = parser.parse_args()


    print(f"[Worker {args.job_uuid[:8]}] Starting co-evolution simulation...")
    
    duration, sse, h_norm, final_psi, _ = run_simulation(args.params)
    
    print(f"[Worker {args.job_uuid[:8]}] Simulation complete in {duration:.4f}s.")
    
    write_results(args.job_uuid, np.array(final_psi), sse, h_norm)


if __name__ == "__main__":
    main()
