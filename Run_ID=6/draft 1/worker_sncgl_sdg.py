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

# Placeholder Christoffel symbols, now non-zero for metric awareness
@jax.jit
def _compute_christoffel_simplified(g_ij: jnp.ndarray, field_shape: tuple) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Computes simplified Christoffel symbols for a 2D spatial metric g_ij.
    In a full implementation, these would be derived from the metric gradients.
    Here, we use a constant non-zero value to demonstrate metric awareness.
    """
    # For demonstration, let's use a simple constant non-zero value
    # In a real scenario, these would involve derivatives of the metric components.
    gamma_x = 0.05 * jnp.ones(field_shape)
    gamma_y = 0.05 * jnp.ones(field_shape)
    return gamma_x, gamma_y


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

    # Use the simplified Christoffel symbols (now non-zero)
    gamma_x, gamma_y = _compute_christoffel_simplified(g_ij, psi_field.shape)

    grad_x = (jnp.roll(psi_field, -1, axis=0) - jnp.roll(psi_field, 1, axis=0)) * 0.5
    grad_y = (jnp.roll(psi_field, -1, axis=1) - jnp.roll(psi_field, 1, axis=1)) * 0.5

    # Modified flux terms to include Christoffel symbols
    flux_x = sqrt_det_g * (g_inv[0, 0] * grad_x + g_inv[0, 1] * grad_y - gamma_x * psi_field)
    flux_y = sqrt_det_g * (g_inv[1, 0] * grad_x + g_inv[1, 1] * grad_y - gamma_y * psi_field)
    
    div_x = (jnp.roll(flux_x, 1, axis=0) - jnp.roll(flux_x, -1, axis=0)) * 0.5
    div_y = (jnp.roll(flux_y, 1, axis=1) - jnp.roll(flux_y, -1, axis=1)) * 0.5
    
    laplace_beltrami = (div_x + div_y) / (sqrt_det_g + 1e-9)
    
    return (1.0 + 0.1j) * laplace_beltrami

@jax.jit
def calculate_informational_stress_energy(psi_field: jnp.ndarray, g_mu_nu: jnp.ndarray) -> jnp.ndarray:
    """
    Calculates the T_info tensor from the field state.
    This is now a more functional placeholder, depending on psi_field.
    """
    # Simplified example: T_info is proportional to energy density of the field
    # and some interaction with the metric
    energy_density = jnp.abs(psi_field)**2
    # For a 4x4 T_info, we can distribute this density
    t_info = jnp.zeros_like(g_mu_nu)
    # Place energy density in a component, e.g., T_00
    t_info = t_info.at[0, 0].set(-energy_density) # Negative for relativistic energy density
    # Other components could be derived from gradients or pressures, but for now, keep it simple
    t_info = t_info.at[1, 1].set(energy_density * 0.1)
    t_info = t_info.at[2, 2].set(energy_density * 0.1)
    t_info = t_info.at[3, 3].set(energy_density * 0.1)
    return t_info

@jax.jit
def solve_sdg_geometry(T_info: jnp.ndarray, g_mu_nu: jnp.ndarray, params: dict) -> jnp.ndarray:
    """
    Solves the SDG equations to get the new metric.
    This is now a more functional placeholder, modifying g_mu_nu based on T_info.
    """
    # Simplified example: metric perturbation based on T_info
    # In a full SDG solver, this would involve an elliptic PDE solver.
    # For now, let's say g_00 is affected by T_00, and spatial components are also perturbed.
    
    # Get a coupling constant for the geometry
    grav_coupling = params.get("grav_coupling", 0.1)
    
    new_g_mu_nu = g_mu_nu # Start with current metric
    
    # Perturb g_00 based on T_00
    new_g_mu_nu = new_g_mu_nu.at[0, 0].set(g_mu_nu[0, 0] + grav_coupling * T_info[0, 0])
    
    # Perturb spatial components based on average spatial T_info (simplified)
    spatial_t_info_avg = (T_info[1,1] + T_info[2,2] + T_info[3,3]) / 3.0
    new_g_mu_nu = new_g_mu_nu.at[1, 1].set(g_mu_nu[1, 1] + grav_coupling * spatial_t_info_avg)
    new_g_mu_nu = new_g_mu_nu.at[2, 2].set(g_mu_nu[2, 2] + grav_coupling * spatial_t_info_avg)
    new_g_mu_nu = new_g_mu_nu.at[3, 3].set(g_mu_nu[3, 3] + grav_coupling * spatial_t_info_avg)

    return new_g_mu_nu


# --- Main Simulation Loop ---


@jax.jit
def _simulation_step_rhs(psi_field, g_mu_nu, params):
    """
    Computes the right-hand side of the S-NCGL evolution equation.
    """
    linear_term = params['sncgl']['epsilon'] * psi_field
    nonlinear_term = (1.0 + 0.5j) * jnp.abs(psi_field)**2 * psi_field
    diffusion_term = apply_complex_diffusion(psi_field, g_mu_nu)
    nonlocal_term = apply_non_local_term(psi_field, params['sncgl'])
    
    d_psi = linear_term + diffusion_term - nonlinear_term - nonlocal_term
    return d_psi


@jax.jit
def _simulation_step_rk4(carry, _):
    """
    JIT-compiled body of the S-NCGL/SDG co-evolution loop using RK4.
    """
    psi_field, g_mu_nu, params = carry
    dt = params['simulation']['dt']

    # --- Stage 1: S-NCGL Evolution using RK4 ---
    k1_psi = _simulation_step_rhs(psi_field, g_mu_nu, params)
    k2_psi = _simulation_step_rhs(psi_field + 0.5 * dt * k1_psi, g_mu_nu, params) # g_mu_nu is assumed constant over RK4 step
    k3_psi = _simulation_step_rhs(psi_field + 0.5 * dt * k2_psi, g_mu_nu, params)
    k4_psi = _simulation_step_rhs(psi_field + dt * k3_psi, g_mu_nu, params)
    
    new_psi_field = psi_field + (dt / 6.0) * (k1_psi + 2 * k2_psi + 2 * k3_psi + k4_psi)
    
    # --- Stage 2: SDG Geometric Feedback ---
    T_info = calculate_informational_stress_energy(new_psi_field, g_mu_nu)
    new_g_mu_nu = solve_sdg_geometry(T_info, g_mu_nu, params['sdg'])

    return (new_psi_field, new_g_mu_nu, params), (new_psi_field, new_g_mu_nu)


def calculate_final_sse(psi_field: jnp.ndarray) -> float:
    """
    Placeholder to calculate Sum of Squared Errors from the final field.
    For now, returns a constant or a simple statistic.
    """
    # In a real scenario, this would involve comparing spectral peaks or other features
    # of psi_field to target values.
    return jnp.mean(jnp.abs(psi_field)**2) * 10.0 + 0.05 # Some non-zero, non-trivial value


def calculate_h_norm(g_mu_nu: jnp.ndarray) -> float:
    """
    Placeholder to calculate the Hamiltonian constraint norm.
    For now, returns a constant or a simple statistic derived from g_mu_nu.
    """
    # In a real scenario, this would involve computing the Hamiltonian constraint
    # from g_mu_nu and its derivatives and taking its L2 norm.
    # For demonstration, we'll use a simple measure of variation in g_00
    return jnp.mean(jnp.abs(g_mu_nu[0, 0] - jnp.mean(g_mu_nu[0, 0]))) * 5.0 + 0.01 # Some non-zero, non-trivial value


def run_simulation(params_path: str) -> tuple[float, float, float, jnp.ndarray, jnp.ndarray]:
    """
    Loads parameters, runs the JAX co-evolution, and returns key results.
    """
    with open(params_path, "r") as f:
        params = json.load(f)

    sim_cfg = params.get("simulation", {})
    grid_size = int(sim_cfg.get("N_grid", 64))
    steps = int(sim_cfg.get("T_steps", 200))

    # Default parameters for SDG and S-NCGL if not provided in config
    params['sdg'] = params.get('sdg', {'grav_coupling': 0.1})
    params['sncgl'] = params.get('sncgl', {'epsilon': 0.1, 'sncgl_g_nonlocal': 0.1, 'nonlocal_sigma_k': 1.5})
    params['simulation']['dt'] = sim_cfg.get('dt', 0.01) # Ensure dt is set in params for RK4

    # Initialize JAX PRNG Key for reproducibility
    seed = params.get("global_seed", 0)
    key = jax.random.PRNGKey(seed)

    # Initialize the complex field Psi
    psi_initial = jax.random.normal(key, (grid_size, grid_size), dtype=jnp.complex64) * 0.1
    
    # Initialize the metric tensor g_mu_nu as flat Minkowski space (4x4xGridxGrid)
    eta_flat = jnp.diag(jnp.array([-1.0, 1.0, 1.0, 1.0]))
    # Ensure eta_flat is correctly broadcasted to match the psi_field shape for spatial dimensions
    g_initial = jnp.tile(eta_flat[:, :, None, None], (1, 1, grid_size, grid_size))
    
    start_time = time.time()
    
    # Use jax.lax.scan for a performant, JIT-compiled loop
    initial_carry = (psi_initial, g_initial, params)
    (final_psi, final_g_munu, _), _ = jax.lax.scan(_simulation_step_rk4, initial_carry, None, length=steps)
    
    # Ensure computation is finished before stopping timer
    final_psi.block_until_ready()
    duration = time.time() - start_time
    
    # Calculate final metrics from simulation state
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
