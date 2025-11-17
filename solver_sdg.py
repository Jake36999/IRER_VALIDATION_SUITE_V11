"""
SDG Geometric Solver and SNCGL coupling kernels.

This module provides the differentiable, JAX-native implementations for
informational stress-energy, SDG geometry solve, and metric-aware diffusion.
"""

import jax
import jax.numpy as jnp


def _central_diff(field: jnp.ndarray, axis: int) -> jnp.ndarray:
    """Compute a central finite difference along an axis with periodic boundaries."""
    return 0.5 * (jnp.roll(field, -1, axis=axis) - jnp.roll(field, 1, axis=axis))


def _laplacian(field: jnp.ndarray, axes: tuple[int, ...]) -> jnp.ndarray:
    """Compute a finite-difference Laplacian with periodic boundaries."""
    lap = jnp.zeros_like(field)
    for ax in axes:
        lap += jnp.roll(field, -1, axis=ax) - 2.0 * field + jnp.roll(field, 1, axis=ax)
    return lap


@jax.jit
def calculate_informational_stress_energy(
    Psi: jnp.ndarray, params: dict, g_mu_nu: jnp.ndarray
) -> jnp.ndarray:
    """Compute the informational stress-energy tensor T_info.

    The tensor is built from gradient energy and potential terms of the
    informational field ``Psi``. Spatial momentum fluxes use a simple symmetric
    construction so the output is fully differentiable and suitable for JAX
    autodiff.
    """

    lambda_coupling = params.get("sncgl_lambda", 1.0)
    spatial_axes = tuple(range(-2, 0))

    # Field gradients for kinetic contributions
    gradients = [
        _central_diff(Psi, axis=ax) for ax in spatial_axes
    ]
    grad_norm = sum(jnp.abs(g) ** 2 for g in gradients)

    potential = lambda_coupling * jnp.abs(Psi) ** 4 + 0.5 * jnp.abs(Psi) ** 2
    energy_density = potential + 0.5 * grad_norm

    # Initialize tensor and populate symmetric components
    T_info = jnp.zeros((4, 4) + Psi.shape, dtype=Psi.dtype)
    T_info = T_info.at[0, 0].set(energy_density)

    # Spatial stress terms: T_ij = grad_i * grad_j - delta_ij * (grad_norm/2 - potential)
    for idx_i, grad_i in enumerate(gradients):
        for idx_j, grad_j in enumerate(gradients):
            tensor_idx = (idx_i + 1, idx_j + 1)
            if idx_i == idx_j:
                stress = grad_i * jnp.conj(grad_j) - 0.5 * grad_norm + potential
            else:
                stress = grad_i * jnp.conj(grad_j)
            T_info = T_info.at[tensor_idx].set(stress)

    return T_info


@jax.jit
def solve_sdg_geometry(
    T_info: jnp.ndarray, current_rho_s: jnp.ndarray, params: dict
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Solve the SDG elliptic equations for the spacetime density and metric.

    A Jacobi-style relaxation solves a Helmholtz-like equation using the
    informational energy density as the source. The resulting scalar density is
    then mapped to a conformally scaled metric ansatz.
    """

    spatial_axes = tuple(range(-2, 0))
    source = jnp.real(T_info[0, 0])

    diffusion_coeff = params.get("sdg_diffusion", 1.0)
    mass_term = params.get("sdg_mass", 1.0)
    iterations = params.get("sdg_iterations", 50)
    relaxation = params.get("sdg_relaxation", 0.7)

    rho = current_rho_s
    for _ in range(iterations):
        neighbor_sum = jnp.zeros_like(rho)
        for ax in spatial_axes:
            neighbor_sum += jnp.roll(rho, 1, axis=ax) + jnp.roll(rho, -1, axis=ax)
        denom = mass_term + 2.0 * diffusion_coeff * len(spatial_axes)
        update = (source + diffusion_coeff * neighbor_sum) / denom
        rho = rho + relaxation * (update - rho)

    rho = jnp.clip(rho, 1e-6, None)

    sdg_alpha = params.get("sdg_alpha", 1.0)
    sdg_rho_vac = params.get("sdg_rho_vac", 1.0)
    scale_factor = (sdg_rho_vac / rho) ** sdg_alpha

    eta = jnp.diag(jnp.array([-1.0, 1.0, 1.0, 1.0]))
    g_mu_nu = jnp.zeros((4, 4) + rho.shape)
    for i in range(4):
        for j in range(4):
            component = eta[i, j]
            if i == j:
                component = component * scale_factor
            g_mu_nu = g_mu_nu.at[i, j].set(component)

    return rho, g_mu_nu


@jax.jit
def apply_complex_diffusion(
    Psi: jnp.ndarray, params: dict, g_mu_nu: jnp.ndarray
) -> jnp.ndarray:
    """Apply a metric-aware complex diffusion operator to ``Psi``.

    Uses the conformal metric produced by :func:`solve_sdg_geometry`. The
    operator implements ``(D + i c1) * \nabla_g^2 Psi`` with periodic boundary
    conditions. The conformal structure allows a closed-form inverse metric and
    determinant, enabling fully differentiable Christoffel-aware diffusion.
    """

    D = params.get("diffusion_coefficient", 1.0)
    c1 = params.get("complex_c1", 0.0)

    # For the conformal metric g_mu_nu = A * eta, the spatial inverse metric is (1/A)
    # and sqrt|g| = A**2. We extract A from a spatial component of the metric.
    scale_factor = g_mu_nu[1, 1]
    inv_scale = 1.0 / scale_factor
    sqrt_det = jnp.sqrt(jnp.abs((-scale_factor) * scale_factor ** 3))

    spatial_axes = tuple(range(-2, 0))

    covariant_flux = []
    for ax in spatial_axes:
        grad = _central_diff(Psi, axis=ax)
        flux = sqrt_det * inv_scale * grad
        covariant_flux.append(flux)

    divergence_terms = []
    for ax, flux in zip(spatial_axes, covariant_flux):
        divergence = 0.5 * (jnp.roll(flux, -1, axis=ax) - jnp.roll(flux, 1, axis=ax))
        divergence_terms.append(divergence)

    covariant_laplacian = sum(divergence_terms) / sqrt_det

    return (D + 1j * c1) * covariant_laplacian
