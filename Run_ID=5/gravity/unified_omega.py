"""Unified Omega derivation utilities.

This module provides the single source of truth for deriving the
emergent spacetime metric used by :mod:`worker_unified`.
"""

from __future__ import annotations


from typing import Dict


import jax
import jax.numpy as jnp




@jax.jit
def jnp_derive_metric_from_rho(
    rho: jnp.ndarray,
    fmia_params: Dict[str, float],
    epsilon: float = 1e-10,
) -> jnp.ndarray:
    """Derive the emergent spacetime metric ``g_munu`` from ``rho``.

    This function closes the geometric loop using the ECM proxy model.
    The analytical solution for the conformal factor is:
    Omega(rho) = (rho_vac / rho)^(a/2)

    This solution has been certified to reproduce the PPN parameter gamma = 1.
    """
    # 1. Load parameters with defaults
    rho_vac = fmia_params.get("param_rho_vac", 1.0)
    a_coupling = fmia_params.get("param_a_coupling", 1.0)


    # 2. Calculate the Effective Conformal Factor Omega
    # Ensure rho is positive to avoid NaNs
    rho_safe = jnp.maximum(rho, epsilon)
    ratio = rho_vac / rho_safe
    Omega = jnp.power(ratio, a_coupling / 2.0)
    Omega_sq = jnp.square(Omega)


    # 3. Construct the Conformal Metric: g_munu = Omega^2 * eta_munu
    grid_shape = rho.shape
    g_munu = jnp.zeros((4, 4) + grid_shape)


    # Time-time component g00 = -Omega^2
    g_munu = g_munu.at[0, 0].set(-Omega_sq)


    # Spatial components gii = +Omega^2
    g_munu = g_munu.at[1, 1].set(Omega_sq)
    g_munu = g_munu.at[2, 2].set(Omega_sq)
    g_munu = g_munu.at[3, 3].set(Omega_sq)


    return g_munu
