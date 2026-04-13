"""
Differentiable 1D Fokker-Planck equation solver.

Solves: ∂ρ/∂t = -∂_x(μ(x)ρ) + (1/2)∂_xx ρ
with absorbing BCs: ρ(a,t) = ρ(b,t) = 0

Strategy: precompute the CN step matrix S = (I - r*L)^{-1} (I + r*L)
so that ρ^{n+1}_int = S @ ρ^n_int. For uniform sub-step dt this S is
reused across all sub-steps → only matrix-vector products in the loop.
"""

import torch


def build_drift(theta, x_grid):
    """Drift μ(x) = θ₁x - θ₂x³"""
    return theta[0] * x_grid - theta[1] * x_grid ** 3


def _build_L(mu, dx, n_int):
    """Build tridiagonal operator L (n_int x n_int) on interior nodes."""
    M = n_int + 1
    a = mu[0:M - 1] / (2 * dx) + 1.0 / (2 * dx ** 2)
    b = -1.0 / (dx ** 2) * torch.ones(n_int, dtype=mu.dtype, device=mu.device)
    c = -mu[2:M + 1] / (2 * dx) + 1.0 / (2 * dx ** 2)
    L = torch.diag(b) + torch.diag(a[1:], -1) + torch.diag(c[:-1], 1)
    return L


def solve_fpe(theta, x_grid, t_output, rho0, n_substeps=20):
    """
    Solve 1D FPE with Crank-Nicolson using precomputed step matrix.

    Uses a uniform fine time grid with n_substeps steps from t=0 to t=max(t_output),
    then interpolates to get output at requested times.

    Args:
        theta: (2,) drift parameters
        x_grid: (M+1,) spatial grid
        t_output: (K,) output times (>0)
        rho0: (M+1,) initial density
        n_substeps: number of uniform time steps from 0 to T_max

    Returns:
        rho_out: (K, M+1) density at each output time
    """
    M = x_grid.shape[0] - 1
    n_int = M - 1
    dx = (x_grid[-1] - x_grid[0]).item() / M

    mu = build_drift(theta, x_grid)
    L = _build_L(mu, dx, n_int)
    I = torch.eye(n_int, dtype=L.dtype, device=L.device)

    T_max = t_output.max().item()
    dt = T_max / n_substeps
    r = dt / 2.0

    # Precompute step matrix: S = (I - r*L)^{-1} @ (I + r*L)
    A = I - r * L
    B = I + r * L
    S = torch.linalg.solve(A, B)  # (n_int, n_int), computed once

    # Time-step: store all interior states
    rho_int = rho0[1:M].clone()  # (n_int,)
    # Store snapshots at each substep: t_k = k*dt, k=0..n_substeps
    all_rho_int = [rho_int]

    for _ in range(n_substeps):
        rho_int = S @ rho_int
        rho_int = torch.clamp(rho_int, min=0.0)
        all_rho_int.append(rho_int)

    all_rho_int = torch.stack(all_rho_int, dim=0)  # (n_substeps+1, n_int)

    # Interpolate to get output at requested times
    # t_k = k * dt, find fractional index for each t in t_output
    frac_idx = t_output / dt  # (K,)
    idx_lo = frac_idx.long().clamp(0, n_substeps - 1)
    idx_hi = (idx_lo + 1).clamp(0, n_substeps)
    alpha = (frac_idx - idx_lo.float()).unsqueeze(-1)  # (K, 1)

    rho_lo = all_rho_int[idx_lo]  # (K, n_int)
    rho_hi = all_rho_int[idx_hi]  # (K, n_int)
    rho_int_out = (1 - alpha) * rho_lo + alpha * rho_hi  # (K, n_int)

    # Reconstruct full solution with BCs
    K = t_output.shape[0]
    rho_out = torch.zeros(K, M + 1, dtype=rho0.dtype, device=rho0.device)
    rho_out[:, 1:M] = rho_int_out

    return rho_out
