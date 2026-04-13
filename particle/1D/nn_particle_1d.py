"""
1D NN drift recovery from time-unlabeled particle observations.

Architecture: μ(x) = a₁x + a₃x³ + NN(x)  (poly+NN hybrid)
Loss: negative marginal log-likelihood (integrate out latent time)

True drift: μ*(x) = x - x³
"""

import sys
sys.path.insert(0, '../../density/1D')

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from fpe_solver import _build_L
from generate_data import make_setup
from optimize_particle_1d import sample_particles_from_density

torch.set_default_dtype(torch.float64)


class DriftNet(nn.Module):
    """μ(x) = c₀ + c₁x + c₂x² + c₃x³ + NN(x), NN initialized near zero."""
    def __init__(self, hidden=16):
        super().__init__()
        self.coeffs = nn.Parameter(torch.zeros(4))  # [1, x, x², x³]
        self.net = nn.Sequential(
            nn.Linear(1, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )
        with torch.no_grad():
            self.net[-1].weight.mul_(0.01)
            self.net[-1].bias.mul_(0.01)

    def forward(self, x):
        c = self.coeffs
        poly = c[0] + c[1] * x + c[2] * x**2 + c[3] * x**3
        nn_out = self.net(x.unsqueeze(-1)).squeeze(-1)
        return poly + nn_out


def solve_fpe_with_drift(mu, x_grid, t_output, rho0, n_substeps=20):
    """FPE solver taking drift vector directly."""
    M = x_grid.shape[0] - 1
    n_int = M - 1
    dx = (x_grid[-1] - x_grid[0]).item() / M

    L = _build_L(mu, dx, n_int)
    I = torch.eye(n_int, dtype=L.dtype, device=L.device)

    T_max = t_output.max().item()
    dt = T_max / n_substeps
    r = dt / 2.0

    S = torch.linalg.solve(I - r * L, I + r * L)

    rho_int = rho0[1:M].clone()
    all_rho_int = [rho_int]
    for _ in range(n_substeps):
        rho_int = S @ rho_int
        rho_int = torch.clamp(rho_int, min=0.0)
        all_rho_int.append(rho_int)
    all_rho_int = torch.stack(all_rho_int, dim=0)

    frac_idx = t_output / dt
    idx_lo = frac_idx.long().clamp(0, n_substeps - 1)
    idx_hi = (idx_lo + 1).clamp(0, n_substeps)
    alpha = (frac_idx - idx_lo.float()).unsqueeze(-1)
    rho_int_out = (1 - alpha) * all_rho_int[idx_lo] + alpha * all_rho_int[idx_hi]

    K = t_output.shape[0]
    rho_out = torch.zeros(K, M + 1, dtype=rho0.dtype, device=rho0.device)
    rho_out[:, 1:M] = rho_int_out
    return rho_out


def generate_particle_obs(x_grid, rho0, N=100, m=1000, n_substeps=20, seed=0):
    """Generate particle observations from true drift μ*(x) = x - x³."""
    mu_true = x_grid - x_grid**3
    torch.manual_seed(seed)
    t_obs = torch.rand(N)
    with torch.no_grad():
        rho_all = solve_fpe_with_drift(mu_true, x_grid, t_obs, rho0, n_substeps=n_substeps)

    rng = torch.Generator()
    rng.manual_seed(seed + 1000)
    particles = torch.zeros(N, m)
    for i in range(N):
        particles[i] = sample_particles_from_density(rho_all[i], x_grid, m, rng=rng)
    return particles, t_obs


def marginal_nll(mu, x_grid, rho0, particles, N_t=50, n_substeps=20, eps=1e-30):
    """Negative marginal log-likelihood with log-sum-exp."""
    N, m = particles.shape
    dx = (x_grid[-1] - x_grid[0]).item() / (x_grid.shape[0] - 1)
    M = x_grid.shape[0] - 1

    t_grid = torch.linspace(0.5 / N_t, 1.0 - 0.5 / N_t, N_t)
    log_w = -torch.log(torch.tensor(float(N_t)))

    rho_all = solve_fpe_with_drift(mu, x_grid, t_grid, rho0, n_substeps=n_substeps)

    x_min = x_grid[0].item()
    particles_flat = particles.reshape(-1)
    frac_idx = (particles_flat - x_min) / dx
    frac_idx = frac_idx.clamp(0, M - 1e-6)
    idx_lo = frac_idx.long()
    idx_hi = (idx_lo + 1).clamp(max=M)
    alpha = frac_idx - idx_lo.float()

    rho_lo = rho_all[:, idx_lo]
    rho_hi = rho_all[:, idx_hi]
    rho_at_particles = (1 - alpha).unsqueeze(0) * rho_lo + alpha.unsqueeze(0) * rho_hi
    rho_at_particles = rho_at_particles.reshape(N_t, N, m)
    rho_at_particles = torch.clamp(rho_at_particles, min=eps)

    log_rho = torch.log(rho_at_particles)
    log_prod = log_rho.sum(dim=2)  # (N_t, N)
    log_terms = log_w + log_prod   # (N_t, N)
    log_marginal = torch.logsumexp(log_terms, dim=0)  # (N,)

    return -log_marginal.mean()


def run(n_iter=500, lr=0.01, N_obs=100, m=1000, N_t=50,
        M=200, n_substeps=20, seed=0):
    """NN drift recovery from particle observations."""
    x_grid, rho0 = make_setup(M=M)
    mu_true = x_grid - x_grid**3

    particles, t_obs = generate_particle_obs(
        x_grid, rho0, N=N_obs, m=m, n_substeps=n_substeps, seed=seed
    )
    print(f"Generated {N_obs} groups of {m} particles")
    print(f"Particle range: [{particles.min():.2f}, {particles.max():.2f}]")

    torch.manual_seed(seed + 100)
    model = DriftNet(hidden=16).double()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_iter, eta_min=lr / 10
    )

    loss_history = []
    drift_mse_history = []

    print(f"\nNN drift recovery from particles, N_t={N_t}")
    print(f"N={N_obs}, m={m}, M={M}")
    print("-" * 60)

    for it in range(n_iter):
        optimizer.zero_grad()
        mu_pred = model(x_grid)
        loss = marginal_nll(mu_pred, x_grid, rho0, particles,
                           N_t=N_t, n_substeps=n_substeps)
        loss.backward()
        optimizer.step()
        scheduler.step()

        loss_history.append(loss.item())
        with torch.no_grad():
            mu_err = ((model(x_grid) - mu_true)**2).mean().item()
        drift_mse_history.append(mu_err)

        if it % 25 == 0 or it == n_iter - 1:
            c = model.coeffs.data
            print(f"  iter {it:4d}: loss={loss.item():.4e}  "
                  f"drift_mse={mu_err:.4e}  c={c.numpy().round(3)}")

    # ===== Plotting =====
    with torch.no_grad():
        mu_learned = model(x_grid).numpy()

    xn = x_grid.numpy()
    mu_t = mu_true.numpy()

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Drift comparison
    axes[0, 0].plot(xn, mu_t, 'k-', lw=2, label='true: x - x³')
    axes[0, 0].plot(xn, mu_learned, 'r--', lw=2, label='learned (poly+NN)')
    axes[0, 0].set_xlabel('x'); axes[0, 0].set_ylabel('μ(x)')
    axes[0, 0].set_title('Drift Recovery')
    axes[0, 0].legend(); axes[0, 0].grid(True)

    # Drift error
    axes[0, 1].plot(xn, mu_learned - mu_t, 'b-', lw=1.5)
    axes[0, 1].axhline(0, color='k', ls='--', alpha=0.3)
    axes[0, 1].set_xlabel('x'); axes[0, 1].set_ylabel('μ_learned - μ_true')
    axes[0, 1].set_title(f'Drift Error (MSE={drift_mse_history[-1]:.4e})')
    axes[0, 1].grid(True)

    # Loss curve
    axes[1, 0].plot(loss_history)
    axes[1, 0].set_xlabel('Iteration'); axes[1, 0].set_ylabel('NLL')
    axes[1, 0].set_title('Marginal NLL Loss')
    axes[1, 0].grid(True)

    # Drift MSE curve
    axes[1, 1].semilogy(drift_mse_history)
    axes[1, 1].set_xlabel('Iteration'); axes[1, 1].set_ylabel('MSE(μ)')
    axes[1, 1].set_title('Drift MSE')
    axes[1, 1].grid(True)

    plt.suptitle(f'NN Particle Inversion: N={N_obs}, m={m}', fontsize=14)
    plt.tight_layout()
    plt.savefig('nn_particle_1d.png', dpi=150)
    plt.close()

    print(f"\nFinal drift MSE: {drift_mse_history[-1]:.4e}")
    print(f"Final coeffs: {model.coeffs.data.numpy().round(4)}")
    print("Saved nn_particle_1d.png")

    return model, loss_history, drift_mse_history


if __name__ == '__main__':
    run(n_iter=500, lr=0.01, N_obs=100, m=1000, N_t=50)
