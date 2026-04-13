"""
Generate ground truth data: unlabeled density snapshots from FPE with true drift.
"""

import torch
import numpy as np
from fpe_solver import solve_fpe

torch.set_default_dtype(torch.float64)


def generate_observations(theta_true, x_grid, rho0, N=50, seed=0):
    """
    Generate N unlabeled density snapshots.

    Returns:
        rho_obs: (N, M+1) density snapshots (time labels discarded)
        t_obs: (N,) the hidden times (for validation only)
    """
    torch.manual_seed(seed)
    t_obs = torch.rand(N)  # Unif(0,1)

    with torch.no_grad():
        rho_obs = solve_fpe(theta_true, x_grid, t_obs, rho0)

    return rho_obs, t_obs


def make_setup(a=-3.0, b=3.0, M=200, sigma0=0.5):
    """Create standard grid and initial condition."""
    x_grid = torch.linspace(a, b, M + 1)
    dx = (b - a) / M

    # Truncated Gaussian IC
    rho0 = torch.exp(-0.5 * (x_grid / sigma0) ** 2) / (sigma0 * np.sqrt(2 * np.pi))
    rho0[0] = 0.0
    rho0[-1] = 0.0
    mass = torch.trapezoid(rho0, dx=dx)
    rho0 = rho0 / mass
    rho0[0] = 0.0
    rho0[-1] = 0.0

    return x_grid, rho0


if __name__ == '__main__':
    x_grid, rho0 = make_setup()
    theta_true = torch.tensor([1.0, 1.0])
    rho_obs, t_obs = generate_observations(theta_true, x_grid, rho0, N=50)
    print(f"Generated {rho_obs.shape[0]} snapshots on grid of {rho_obs.shape[1]} points")
    print(f"Hidden times range: [{t_obs.min():.3f}, {t_obs.max():.3f}]")

    torch.save({'rho_obs': rho_obs, 't_obs': t_obs, 'x_grid': x_grid, 'rho0': rho0,
                'theta_true': theta_true}, 'data.pt')
    print("Saved to data.pt")
