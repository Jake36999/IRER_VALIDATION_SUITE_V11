"""
JAX-native implementations for the IRER SDG solver components.

Provides:
- calculate_informational_stress_energy: informational stress-energy tensor.
- solve_sdg_geometry: elliptic SDG solver for rho_s and emergent metric.
- apply_complex_diffusion: metric-aware covariant diffusion operator.
"""

import jax
import jax.numpy as jnp


def _central_gradient(field: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute periodic central differences along x/y axes."""
    grad_x = 0.5 * (jnp.roll(field, -1, axis=0) - jnp.roll(field, 1, axis=0))
    grad_y = 0.5 * (jnp.roll(field, -1, axis=1) - jnp.roll(field, 1, axis=1))
    return grad_x, grad_y


def _laplacian_flat(field: jnp.ndarray) -> jnp.ndarray:
    """Flat-space Laplacian with periodic boundaries."""
    return (
        jnp.roll(field, 1, axis=0)
        + jnp.roll(field, -1, axis=0)
        + jnp.roll(field, 1, axis=1)
        + jnp.roll(field, -1, axis=1)
        - 4.0 * field
    )


def _metric_spatial_components(g_mu_nu: jnp.ndarray) -> jnp.ndarray:
    """Extract the 2x2 spatial block (x, y) from the 4x4 metric."""
    return g_mu_nu[1:3, 1:3]


def _inverse_and_det_2x2(matrix: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute inverse and determinant for a batched 2x2 matrix."""
    a = matrix[0, 0]
    b = matrix[0, 1]
    c = matrix[1, 0]
    d = matrix[1, 1]
    det = a * d - b * c
    inv = jnp.stack([jnp.stack([d, -b], axis=0), jnp.stack([-c, a], axis=0)], axis=0) / det
    return inv, det


def _covariant_laplacian_scalar(scalar: jnp.ndarray, g_mu_nu: jnp.ndarray) -> jnp.ndarray:
    """Apply the covariant Laplacian to a scalar field in 2D using the provided metric."""
    g_spatial = _metric_spatial_components(g_mu_nu)
    g_inv, det_g = _inverse_and_det_2x2(g_spatial)

    dpsi_dx, dpsi_dy = _central_gradient(scalar)
    grad = jnp.stack([dpsi_dx, dpsi_dy], axis=0)

    g_xx_x = 0.5 * (jnp.roll(g_spatial[0, 0], -1, axis=0) - jnp.roll(g_spatial[0, 0], 1, axis=0))
    g_xx_y = 0.5 * (jnp.roll(g_spatial[0, 0], -1, axis=1) - jnp.roll(g_spatial[0, 0], 1, axis=1))
    g_xy_x = 0.5 * (jnp.roll(g_spatial[0, 1], -1, axis=0) - jnp.roll(g_spatial[0, 1], 1, axis=0))
    g_xy_y = 0.5 * (jnp.roll(g_spatial[0, 1], -1, axis=1) - jnp.roll(g_spatial[0, 1], 1, axis=1))
    g_yy_x = 0.5 * (jnp.roll(g_spatial[1, 1], -1, axis=0) - jnp.roll(g_spatial[1, 1], 1, axis=0))
    g_yy_y = 0.5 * (jnp.roll(g_spatial[1, 1], -1, axis=1) - jnp.roll(g_spatial[1, 1], 1, axis=1))

    Gamma = jnp.zeros((2, 2, 2) + scalar.shape)

    Gamma = Gamma.at[0, 0, 0].set(0.5 * g_inv[0, 0] * (2 * g_xx_x - g_xx_x))
    Gamma = Gamma.at[0, 0, 1].set(0.5 * g_inv[0, 0] * (g_xx_y))
    Gamma = Gamma.at[0, 1, 0].set(0.5 * g_inv[0, 0] * (g_xy_x))
    Gamma = Gamma.at[0, 1, 1].set(0.5 * g_inv[0, 0] * (2 * g_xy_y - g_yy_x))

    Gamma = Gamma.at[1, 0, 0].set(0.5 * g_inv[1, 1] * (2 * g_xy_x - g_xx_y))
    Gamma = Gamma.at[1, 0, 1].set(0.5 * g_inv[1, 1] * (g_yy_x))
    Gamma = Gamma.at[1, 1, 0].set(0.5 * g_inv[1, 1] * (g_xy_y))
    Gamma = Gamma.at[1, 1, 1].set(0.5 * g_inv[1, 1] * (2 * g_yy_y - g_xy_x))

    def second_derivative(axis):
        return 0.5 * (jnp.roll(grad[axis], -1, axis=axis + 1) - jnp.roll(grad[axis], 1, axis=axis + 1))

    d2psi_dx2 = second_derivative(0)
    d2psi_dy2 = second_derivative(1)
    d2psi_dxdy = 0.25 * (
        jnp.roll(scalar, (-1, -1), axis=(0, 1))
        + jnp.roll(scalar, (1, 1), axis=(0, 1))
        - jnp.roll(scalar, (-1, 1), axis=(0, 1))
        - jnp.roll(scalar, (1, -1), axis=(0, 1))
    )

    cov_hessian = jnp.stack(
        [
            jnp.stack([d2psi_dx2, d2psi_dxdy], axis=0),
            jnp.stack([d2psi_dxdy, d2psi_dy2], axis=0),
        ],
        axis=0,
    )

    connection_term = jnp.einsum("kij...,k...->ij...", Gamma, grad)
    cov_hessian = cov_hessian - connection_term

    lap = jnp.einsum("ij...,ij...->...", g_inv, cov_hessian)
    return lap / jnp.sqrt(jnp.abs(det_g))


@jax.jit
def calculate_informational_stress_energy(
    Psi: jnp.ndarray, params: dict, g_mu_nu: jnp.ndarray
) -> jnp.ndarray:
    """Compute the informational stress-energy tensor T_info[mu,nu]."""
    mass = params.get("sncgl_mass", 1.0)
    lam = params.get("sncgl_lambda", 1.0)

    grad_x, grad_y = _central_gradient(Psi)
    kinetic = 0.5 * (jnp.abs(grad_x) ** 2 + jnp.abs(grad_y) ** 2)
    potential = 0.5 * mass**2 * jnp.abs(Psi) ** 2 + 0.25 * lam * jnp.abs(Psi) ** 4
    energy_density = kinetic + potential

    g_spatial = _metric_spatial_components(g_mu_nu)
    g_inv, _ = _inverse_and_det_2x2(g_spatial)

    momentum_flux_x = jnp.real(jnp.conj(Psi) * grad_x)
    momentum_flux_y = jnp.real(jnp.conj(Psi) * grad_y)

    stress_xx = jnp.real(g_inv[0, 0]) * kinetic - potential
    stress_yy = jnp.real(g_inv[1, 1]) * kinetic - potential
    stress_xy = jnp.real(g_inv[0, 1]) * kinetic

    T_info = jnp.zeros((4, 4) + Psi.shape, dtype=Psi.dtype)
    T_info = T_info.at[0, 0].set(energy_density)
    T_info = T_info.at[0, 1].set(momentum_flux_x)
    T_info = T_info.at[0, 2].set(momentum_flux_y)
    T_info = T_info.at[1, 0].set(momentum_flux_x)
    T_info = T_info.at[2, 0].set(momentum_flux_y)
    T_info = T_info.at[1, 1].set(stress_xx)
    T_info = T_info.at[2, 2].set(stress_yy)
    T_info = T_info.at[1, 2].set(stress_xy)
    T_info = T_info.at[2, 1].set(stress_xy)
    return T_info


@jax.jit
def solve_sdg_geometry(
    T_info: jnp.ndarray, current_rho_s: jnp.ndarray, params: dict
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Iteratively solve the SDG scalar field and reconstruct the metric."""
    alpha = params.get("sdg_alpha", 1.0)
    rho_vac = params.get("sdg_rho_vac", 1.0)
    relaxation = params.get("sdg_relaxation", 0.25)
    iterations = params.get("sdg_solver_iterations", 50)
    mass_term = params.get("sdg_mass", 0.1)

    source = jnp.real(T_info[0, 0])

    def body_fun(_, state):
        rho = state
        lap = _laplacian_flat(rho)
        residual = source - (lap - mass_term * rho)
        rho_next = rho + relaxation * residual
        return rho_next

    rho_s = jax.lax.fori_loop(0, iterations, body_fun, current_rho_s)
    rho_s = jnp.clip(rho_s, 1e-5, None)

    scale = (rho_vac / rho_s) ** alpha
    eta = jnp.diag(jnp.array([-1.0, 1.0, 1.0, 1.0]))
    g_mu_nu = jnp.einsum("ij,...->ij...", eta, scale)
    return rho_s, g_mu_nu


@jax.jit
def apply_complex_diffusion(
    Psi: jnp.ndarray, params: dict, g_mu_nu: jnp.ndarray
) -> jnp.ndarray:
    """Metric-aware complex diffusion term using the covariant Laplacian."""
    D_real = params.get("sncgl_epsilon", 1.0) * 0.5
    c1_imag = params.get("sncgl_epsilon", 1.0) * 0.8

    cov_lap = _covariant_laplacian_scalar(Psi, g_mu_nu)
    return (D_real + 1j * c1_imag) * cov_lap


__all__ = [
    "calculate_informational_stress_energy",
    "solve_sdg_geometry",
    "apply_complex_diffusion",
]
