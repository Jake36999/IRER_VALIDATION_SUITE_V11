"""
solver_sdg.py
V11.0: The JAX-native Spacetime-Density Gravity (SDG) solver library.
This module contains the axiomatically-derived physics kernels that form the
new "law-keeper" for the IRER framework, resolving the V10.x Geometric Crisis.
"""
import jax
import jax.numpy as jnp
from functools import partial

@partial(jax.jit, static_argnames=('iterations', 'omega', 'dx'))
def _jacobi_poisson_solver(source: jnp.ndarray, x: jnp.ndarray, dx: float, iterations: int, omega: float) -> jnp.ndarray:
    """A JAX-jitted Jacobi-Poisson solver for the SDG geometry."""
    d_sq = dx * dx
    for _ in range(iterations):
        x_new = (
            jnp.roll(x, 1, axis=0) + jnp.roll(x, -1, axis=0) +
            jnp.roll(x, 1, axis=1) + jnp.roll(x, -1, axis=1) +
            source * d_sq
        ) / 4.0
        x = (1.0 - omega) * x + omega * x_new
    return x

@jax.jit
def _compute_spatial_christoffel(g_ij: jnp.ndarray, dx: float) -> jnp.ndarray:
    """
    Computes spatial Christoffel symbols Gamma^k_{ij} for a 2D metric g_ij.
    g_ij has shape (N, N, 2, 2) and represents g_xx, g_xy, g_yx, g_yy at each grid point.
    Returns Gamma^k_{ij} of shape (N, N, 2, 2, 2).
    Indices: (grid_x, grid_y, upper_k, lower_i, lower_j)
    """
    N = g_ij.shape[0] # Assuming g_ij is (N, N, 2, 2)
    g_inv_ij = jnp.linalg.inv(g_ij) # (N, N, 2, 2)

    Gamma = jnp.zeros((N, N, 2, 2, 2)) # (N, N, upper_k, lower_i, lower_j)

    # Compute numerical derivatives of the metric components
    # dg_p_mn: derivative of g_mn with respect to p-th coordinate (p=0 for x, p=1 for y)
    # Shape: (p_idx, m_idx, n_idx, grid_x, grid_y)
    dg_p_mn = jnp.zeros((2, 2, 2, N, N))

    for m in range(2):
        for n in range(2):
            # Derivative with respect to x (p_idx = 0)
            dg_p_mn = dg_p_mn.at[0, m, n].set(
                (jnp.roll(g_ij[:,:,m,n], -1, axis=0) - jnp.roll(g_ij[:,:,m,n], 1, axis=0)) / (2 * dx)
            )
            # Derivative with respect to y (p_idx = 1)
            dg_p_mn = dg_p_mn.at[1, m, n].set(
                (jnp.roll(g_ij[:,:,m,n], -1, axis=1) - jnp.roll(g_ij[:,:,m,n], 1, axis=1)) / (2 * dx)
            )
    
    # Christoffel symbols Gamma^k_{ij} = 0.5 * g^{kl} (partial_i g_{jl} + partial_j g_{il} - partial_l g_{ij})
    # Loop over spatial grid points (N, N) is implicit in JAX array operations
    for k in range(2): # upper index for Gamma
        for i in range(2): # lower first index for Gamma
            for j in range(2): # lower second index for Gamma
                # Sum over 'l'
                term_sum = jnp.zeros((N, N))
                for l in range(2):
                    # (partial_i g_jl) -> dg_p_mn[i, j, l] (using convention p=i)
                    t1 = dg_p_mn[i, j, l]
                    # (partial_j g_il) -> dg_p_mn[j, i, l] (using convention p=j)
                    t2 = dg_p_mn[j, i, l]
                    # (partial_l g_ij) -> dg_p_mn[l, i, j] (using convention p=l)
                    t3 = dg_p_mn[l, i, j]
                    
                    # g_inv_ij[:,:,k,l] is g^{kl} for each grid point
                    term_sum += g_inv_ij[:,:,k,l] * (t1 + t2 - t3)
                Gamma = Gamma.at[:,:,k,i,j].set(0.5 * term_sum)
    return Gamma

@jax.jit
def calculate_informational_stress_energy(Psi: jnp.ndarray, sdg_kappa: float, sdg_eta: float) -> jnp.ndarray:
    """
    The "Bridge": Calculates the Informational Stress-Energy Tensor (T_info).
    This tensor is formally derived from the Fields of Minimal Informational
    Action (L_FMIA) Lagrangian via the standard variational principle and
    serves as the source term for the emergent gravitational field.
    """
    rho = jnp.abs(Psi)**2
    phi = jnp.angle(Psi)
    sqrt_rho = jnp.sqrt(jnp.maximum(rho, 1e-9)) # Add epsilon for stability

    # Calculate spatial gradients of the core fields
    grad_phi_y, grad_phi_x = jnp.gradient(phi)
    grad_sqrt_rho_y, grad_sqrt_rho_x = jnp.gradient(sqrt_rho)

    # --- T_munu components from T_info = k*rho*(d_mu phi)(d_nu phi) + eta*(d_mu sqrt(rho))(d_nu sqrt(rho)) ---

    # T_00: Energy Density (sum over spatial components)
    energy_density = (sdg_kappa * rho * (grad_phi_x**2 + grad_phi_y**2) +
                      sdg_eta * (grad_sqrt_rho_x**2 + grad_sqrt_rho_y**2))

    # T_ij: Spatial Stress components (2D simulation)
    stress_xx = sdg_kappa * rho * (grad_phi_x**2) + sdg_eta * (grad_sqrt_rho_x**2)
    stress_yy = sdg_kappa * rho * (grad_phi_y**2) + sdg_eta * (grad_sqrt_rho_y**2)
    stress_xy = sdg_kappa * rho * (grad_phi_x * grad_phi_y) + sdg_eta * (grad_sqrt_rho_x * grad_sqrt_rho_y)

    # Assemble the 4x4 tensor for each grid point
    tensor_shape = (4, 4) + Psi.shape
    t_info = jnp.zeros(tensor_shape, dtype=jnp.complex64)

    # Populate tensor components (assuming a 2+1D system embedded in 4D tensor)
    # T_0i components (momentum density) are ignored in this simplified model.
    t_info = t_info.at[0, 0].set(energy_density)
    t_info = t_info.at[1, 1].set(stress_xx)
    t_info = t_info.at[2, 2].set(stress_yy)
    t_info = t_info.at[1, 2].set(stress_xy)
    t_info = t_info.at[2, 1].set(stress_xy) # Tensor must be symmetric

    return t_info

@partial(jax.jit, static_argnames=('spatial_resolution', 'sdg_alpha', 'sdg_rho_vac'))
def solve_sdg_geometry(T_info: jnp.ndarray, current_rho_s: jnp.ndarray, spatial_resolution: int, sdg_alpha: float, sdg_rho_vac: float) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    The "Engine": Solves for the new spacetime geometry using the SDG model.
    This function solves a Poisson-like equation for the conformal factor (Omega),
    where the emergent metric is defined as g_munu = Omega^2 * eta_munu.
    """
    # Solver parameters
    dx = 1.0 / spatial_resolution
    iterations = 50
    omega = 1.8 # Relaxation parameter for Jacobi solver

    # Use the real part of the energy density as the Poisson source
    T_00_source = jnp.real(T_info[0, 0])

    # Solve for the new spacetime density scalar field
    rho_s_new = _jacobi_poisson_solver(T_00_source, current_rho_s, dx, iterations, omega)
    rho_s_new = jnp.clip(rho_s_new, 1e-6, None) # Enforce positivity

    # Calculate the emergent metric via conformal scaling of Minkowski metric
    eta = jnp.diag(jnp.array([-1.0, 1.0, 1.0, 1.0]))
    Omega_sq = (sdg_rho_vac / rho_s_new) ** sdg_alpha

    # g_mu_nu_new will be (4, 4, N, N)
    g_mu_nu_new = jnp.einsum('ab,xy->abxy', eta, Omega_sq)

    return rho_s_new, g_mu_nu_new

@partial(jax.jit, static_argnames=('sncgl_epsilon', 'spatial_resolution'))
def apply_complex_diffusion(Psi: jnp.ndarray, sncgl_epsilon: float, g_mu_nu: jnp.ndarray, spatial_resolution: int) -> jnp.ndarray:
    """
    Applies metric-aware complex diffusion using the covariant Laplacian.
    Delta_g Psi = g^{ij} (partial_i partial_j Psi - Gamma^k_{ij} partial_k Psi)
    This addresses the audit's requirement for Christoffel symbol calculation.
    """
    N = Psi.shape[0]
    dx = 1.0 / spatial_resolution

    # Extract spatial part of the metric (g_ij from g_mu_nu, where i,j = 1,2 for x,y)
    # g_mu_nu has shape (4, 4, N, N). g_ij will be (2, 2, N, N)
    g_ij = g_mu_nu[1:3, 1:3, :, :]
    # Reshape g_ij to (N, N, 2, 2) for jnp.linalg.inv and _compute_spatial_christoffel
    g_ij_reshaped = jnp.transpose(g_ij, (2, 3, 0, 1))
    g_inv_ij = jnp.linalg.inv(g_ij_reshaped) # (N, N, 2, 2)

    # Compute Christoffel symbols Gamma^k_{ij}
    # Gamma has shape (N, N, upper_k, lower_i, lower_j)
    Gamma = _compute_spatial_christoffel(g_ij_reshaped, dx)

    # Compute first and second partial derivatives of Psi
    # First derivatives
    d_psi_dx = (jnp.roll(Psi, -1, axis=0) - jnp.roll(Psi, 1, axis=0)) / (2 * dx)
    d_psi_dy = (jnp.roll(Psi, -1, axis=1) - jnp.roll(Psi, 1, axis=1)) / (2 * dx)
    
    # Second derivatives (central difference)
    d2_psi_dx2 = (jnp.roll(Psi, -1, axis=0) + jnp.roll(Psi, 1, axis=0) - 2 * Psi) / (dx**2)
    d2_psi_dy2 = (jnp.roll(Psi, -1, axis=1) + jnp.roll(Psi, 1, axis=1) - 2 * Psi) / (dx**2)
    # Mixed partial derivative, assuming Psi is complex
    d2_psi_dxdy = (
        jnp.roll(jnp.roll(Psi, -1, axis=0), -1, axis=1)
        - jnp.roll(jnp.roll(Psi, -1, axis=0), 1, axis=1)
        - jnp.roll(jnp.roll(Psi, 1, axis=0), -1, axis=1)
        + jnp.roll(jnp.roll(Psi, 1, axis=0), 1, axis=1)
    ) / (4 * dx**2)

    # Stack first partials for Gamma term: (N, N, 2)
    partial_psi = jnp.stack([d_psi_dx, d_psi_dy], axis=-1)
    
    # Stack second partials for g^{ij} partial_i partial_j Psi term: (N,N,2,2)
    partial2_psi_tensor = jnp.array([
        [d2_psi_dx2, d2_psi_dxdy],
        [d2_psi_dxdy, d2_psi_dy2]
    ]) # (2, 2, N, N)
    partial2_psi_tensor = jnp.transpose(partial2_psi_tensor, (2, 3, 0, 1)) # (N, N, 2, 2)

    covariant_laplacian = jnp.zeros_like(Psi)

    # Delta_g Psi = g^{ij} (partial_i partial_j Psi - Gamma^k_{ij} partial_k Psi)
    for i in range(2):
        for j in range(2):
            term_partial2 = partial2_psi_tensor[:,:,i,j] # partial_i partial_j Psi

            term_gamma_gradient_sum = jnp.zeros_like(Psi)
            for k in range(2):
                # Gamma[:,:,k,i,j] is Gamma^k_{ij} for each grid point
                # partial_psi[:,:,k] is partial_k Psi for each grid point
                term_gamma_gradient_sum += Gamma[:,:,k,i,j] * partial_psi[:,:,k]
            
            # g_inv_ij[:,:,i,j] is g^{ij} for each grid point
            covariant_laplacian += g_inv_ij[:,:,i,j] * (term_partial2 - term_gamma_gradient_sum)

    # Combine with complex diffusion coefficients
    return (sncgl_epsilon * 0.5 + 1j * sncgl_epsilon * 0.8) * covariant_laplacian
