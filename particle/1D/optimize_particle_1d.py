"""
1D drift recovery from time-unlabeled particle observations.

Observation model:
  t_i ~ Unif(0,1), i=1..N  (unknown)
  X_{i,1},...,X_{i,m} ~ rho(·, t_i)  (observed particles)

Loss: negative marginal log-likelihood (integrate out latent time)
  L(mu) = -sum_i log( sum_n w_n * prod_r rho_mu(X_{i,r}, t_n) )
        = -sum_i LSE_n( log w_n + sum_r log rho_mu(X_{i,r}, t_n) )

Uses existing FPE solver for forward model.
"""

import sys
sys.path.insert(0, '../../density/1D')

import torch
import numpy as np
import matplotlib.pyplot as plt
from fpe_solver import solve_fpe, build_drift
from generate_data import make_setup

torch.set_default_dtype(torch.float64)


def sample_particles_from_density(rho, x_grid, m, rng=None):
    """
    Sample m particles from density rho(x) on x_grid.
    Uses inverse CDF sampling.

    Args:
        rho: (M+1,) density values on grid
        x_grid: (M+1,) spatial grid
        m: number of particles to sample
        rng: torch Generator

    Returns:
        particles: (m,) sampled positions
    """
    dx = (x_grid[-1] - x_grid[0]).item() / (x_grid.shape[0] - 1)
    # Compute CDF using trapezoidal weights
    weights = rho.clone()
    weights[0] *= 0.5
    weights[-1] *= 0.5
    weights = weights * dx
    weights = torch.clamp(weights, min=0.0)
    total = weights.sum()
    if total < 1e-15:
        # Degenerate: return uniform samples
        return x_grid[0] + (x_grid[-1] - x_grid[0]) * torch.rand(m, generator=rng)
    weights = weights / total
    cdf = torch.cumsum(weights, dim=0)
    cdf[-1] = 1.0  # Ensure exact 1

    # Sample uniform, invert CDF with linear interpolation
    u = torch.rand(m, generator=rng)
    # Find indices: cdf[idx-1] < u <= cdf[idx]
    idx = torch.searchsorted(cdf, u)
    idx = idx.clamp(1, x_grid.shape[0] - 1)

    # Linear interpolation within the cell
    x_lo = x_grid[idx - 1]
    x_hi = x_grid[idx]
    cdf_lo = cdf[idx - 1]
    cdf_hi = cdf[idx]
    frac = (u - cdf_lo) / (cdf_hi - cdf_lo + 1e-30)
    particles = x_lo + frac * (x_hi - x_lo)

    return particles


def generate_particle_observations(theta_true, x_grid, rho0, N=50, m=10,
                                    n_substeps=20, seed=0):
    """
    Generate N groups of m particles from the true SDE.

    Returns:
        particles: (N, m) particle positions
        t_obs: (N,) hidden times (for validation only)
    """
    torch.manual_seed(seed)
    t_obs = torch.rand(N)

    with torch.no_grad():
        rho_all = solve_fpe(theta_true, x_grid, t_obs, rho0, n_substeps=n_substeps)

    rng = torch.Generator()
    rng.manual_seed(seed + 1000)
    particles = torch.zeros(N, m)
    for i in range(N):
        particles[i] = sample_particles_from_density(rho_all[i], x_grid, m, rng=rng)

    return particles, t_obs


def marginal_nll(theta, x_grid, rho0, particles, N_t=50, n_substeps=20, eps=1e-30):
    """
    Compute negative marginal log-likelihood.

    L = -sum_i LSE_n( log w_n + sum_r log rho(X_{i,r}, t_n) )

    Args:
        theta: (2,) drift parameters
        x_grid: (M+1,) spatial grid
        rho0: (M+1,) initial density
        particles: (N, m) observed particle positions
        N_t: number of quadrature points for time integration
        n_substeps: substeps for FPE solver

    Returns:
        loss: scalar negative log-likelihood
    """
    N, m = particles.shape
    dx = (x_grid[-1] - x_grid[0]).item() / (x_grid.shape[0] - 1)

    # Time quadrature: uniform grid on (0, 1)
    # Use midpoints for better quadrature
    t_grid = torch.linspace(0.5 / N_t, 1.0 - 0.5 / N_t, N_t)
    log_w = -torch.log(torch.tensor(float(N_t)))  # uniform weights: w_n = 1/N_t

    # Forward solve: get rho at all quadrature times
    rho_all = solve_fpe(theta, x_grid, t_grid, rho0, n_substeps=n_substeps)
    # rho_all: (N_t, M+1)

    # Evaluate rho at particle positions via linear interpolation
    # particles: (N, m), x_grid: (M+1,)
    # We need rho(X_{i,r}, t_n) for all i, r, n
    x_min = x_grid[0].item()
    x_max = x_grid[-1].item()
    M = x_grid.shape[0] - 1

    # Normalize particle positions to [0, M] index space
    particles_flat = particles.reshape(-1)  # (N*m,)
    frac_idx = (particles_flat - x_min) / dx  # (N*m,)
    frac_idx = frac_idx.clamp(0, M - 1e-6)
    idx_lo = frac_idx.long()
    idx_hi = (idx_lo + 1).clamp(max=M)
    alpha = frac_idx - idx_lo.float()  # (N*m,)

    # Interpolate: rho_all is (N_t, M+1)
    # rho_at_particles: (N_t, N*m)
    rho_lo = rho_all[:, idx_lo]  # (N_t, N*m)
    rho_hi = rho_all[:, idx_hi]  # (N_t, N*m)
    rho_at_particles = (1 - alpha).unsqueeze(0) * rho_lo + alpha.unsqueeze(0) * rho_hi
    # (N_t, N*m)

    # Reshape to (N_t, N, m)
    rho_at_particles = rho_at_particles.reshape(N_t, N, m)

    # Clamp to avoid log(0)
    rho_at_particles = torch.clamp(rho_at_particles, min=eps)

    # log rho: (N_t, N, m)
    log_rho = torch.log(rho_at_particles)

    # Sum over particles in each group: (N_t, N)
    log_prod = log_rho.sum(dim=2)

    # Add log weights: (N_t, N)
    log_terms = log_w + log_prod

    # Log-sum-exp over time: (N,)
    log_marginal = torch.logsumexp(log_terms, dim=0)

    # Negative log-likelihood
    loss = -log_marginal.mean()

    return loss


def optimize(n_iter=300, lr=0.01, N_obs=50, m=10, N_t=50, n_substeps=20, seed=0):
    """
    Recover drift parameters from particle observations.
    """
    x_grid, rho0 = make_setup()
    theta_true = torch.tensor([1.0, 1.0])

    # Generate particle observations
    particles, t_obs = generate_particle_observations(
        theta_true, x_grid, rho0, N=N_obs, m=m, n_substeps=n_substeps, seed=seed
    )
    print(f"Generated {N_obs} groups of {m} particles")
    print(f"Particle range: [{particles.min():.2f}, {particles.max():.2f}]")

    # Initial guess
    theta = torch.tensor([0.5, 0.5], requires_grad=True)
    optimizer = torch.optim.Adam([theta], lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_iter, eta_min=lr/10)

    loss_history = []
    theta_history = []

    print(f"\nOptimizing with marginal NLL, N_t={N_t}")
    print(f"True theta: {theta_true.numpy()}")
    print(f"Initial theta: {theta.data.numpy()}")
    print("-" * 50)

    for it in range(n_iter):
        optimizer.zero_grad()
        loss = marginal_nll(theta, x_grid, rho0, particles,
                           N_t=N_t, n_substeps=n_substeps)
        loss.backward()
        optimizer.step()
        scheduler.step()

        loss_history.append(loss.item())
        theta_history.append(theta.data.clone().numpy())

        if it % 20 == 0 or it == n_iter - 1:
            print(f"  iter {it:4d}: loss={loss.item():.4e}, "
                  f"theta=[{theta.data[0]:.4f}, {theta.data[1]:.4f}]")

    theta_history = np.array(theta_history)

    # ===== Plotting =====
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Loss curve
    axes[0, 0].plot(loss_history)
    axes[0, 0].set_xlabel('Iteration')
    axes[0, 0].set_ylabel('NLL')
    axes[0, 0].set_title('Marginal NLL Loss')
    axes[0, 0].grid(True)

    # Parameter trajectories
    axes[0, 1].plot(theta_history[:, 0], label='θ₁')
    axes[0, 1].plot(theta_history[:, 1], label='θ₂')
    axes[0, 1].axhline(1.0, color='k', linestyle='--', alpha=0.5, label='true')
    axes[0, 1].set_xlabel('Iteration')
    axes[0, 1].set_ylabel('Parameter')
    axes[0, 1].set_title('Parameter Trajectories')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # Parameter space trajectory
    axes[1, 0].plot(theta_history[:, 0], theta_history[:, 1], 'b.-', alpha=0.3, markersize=2)
    axes[1, 0].plot(1.0, 1.0, 'r*', markersize=15, label='true')
    axes[1, 0].plot(0.5, 0.5, 'go', markersize=8, label='init')
    axes[1, 0].set_xlabel('θ₁')
    axes[1, 0].set_ylabel('θ₂')
    axes[1, 0].set_title('Parameter Space')
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # Recovered drift vs true
    with torch.no_grad():
        x_plot = x_grid.numpy()
        mu_true = build_drift(theta_true, x_grid).numpy()
        mu_learned = build_drift(theta.data, x_grid).numpy()
    axes[1, 1].plot(x_plot, mu_true, 'k-', lw=2, label='true μ(x)')
    axes[1, 1].plot(x_plot, mu_learned, 'r--', lw=2, label='learned μ(x)')
    axes[1, 1].set_xlabel('x')
    axes[1, 1].set_ylabel('μ(x)')
    axes[1, 1].set_title('Drift Recovery')
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    plt.suptitle(f'Particle-based Inversion: N={N_obs}, m={m}', fontsize=14)
    plt.tight_layout()
    plt.savefig('optimize_particle_1d.png', dpi=150)
    plt.close()

    print(f"\nFinal theta: [{theta.data[0]:.4f}, {theta.data[1]:.4f}]")
    print(f"True theta:  [1.0000, 1.0000]")
    print(f"Error: [{abs(theta.data[0]-1):.4f}, {abs(theta.data[1]-1):.4f}]")
    print("Saved optimize_particle_1d.png")

    return theta.data.clone(), loss_history, theta_history


def ablation_m(m_list=[1, 5, 10, 20, 50], N_obs=50, n_iter=300, lr=0.01):
    """Ablation study on particle count m."""
    results = {}
    for m in m_list:
        print(f"\n{'='*60}")
        print(f"Running m={m}")
        theta_final, loss_hist, theta_hist = optimize(
            n_iter=n_iter, lr=lr, N_obs=N_obs, m=m, seed=0
        )
        results[m] = {
            'theta': theta_final.numpy(),
            'error': np.abs(theta_final.numpy() - 1.0),
            'loss': loss_hist,
            'trajectory': theta_hist,
        }

    # Summary plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for m in m_list:
        r = results[m]
        axes[0].plot(r['loss'], label=f'm={m}', alpha=0.8)
        axes[1].plot(r['trajectory'][:, 0], label=f'm={m}', alpha=0.8)

    axes[0].set_xlabel('Iteration'); axes[0].set_ylabel('NLL')
    axes[0].set_title('Loss vs m'); axes[0].legend(); axes[0].grid(True)

    axes[1].axhline(1.0, color='k', ls='--', alpha=0.5)
    axes[1].set_xlabel('Iteration'); axes[1].set_ylabel('θ₁')
    axes[1].set_title('θ₁ Recovery vs m'); axes[1].legend(); axes[1].grid(True)

    m_vals = list(results.keys())
    err1 = [results[m]['error'][0] for m in m_vals]
    err2 = [results[m]['error'][1] for m in m_vals]
    axes[2].semilogy(m_vals, err1, 'bo-', label='|θ₁ error|')
    axes[2].semilogy(m_vals, err2, 'rs-', label='|θ₂ error|')
    axes[2].set_xlabel('m (particles per group)')
    axes[2].set_ylabel('Parameter error')
    axes[2].set_title('Error vs m')
    axes[2].legend(); axes[2].grid(True)

    plt.tight_layout()
    plt.savefig('ablation_m.png', dpi=150)
    plt.close()
    print("\nSaved ablation_m.png")

    print(f"\n{'='*60}")
    print("Summary:")
    print(f"{'m':>5} {'θ₁':>8} {'θ₂':>8} {'err₁':>10} {'err₂':>10}")
    for m in m_vals:
        r = results[m]
        print(f"{m:5d} {r['theta'][0]:8.4f} {r['theta'][1]:8.4f} "
              f"{r['error'][0]:10.4f} {r['error'][1]:10.4f}")


if __name__ == '__main__':
    # Single run with default settings
    optimize(n_iter=300, lr=0.01, N_obs=50, m=10, N_t=50)
