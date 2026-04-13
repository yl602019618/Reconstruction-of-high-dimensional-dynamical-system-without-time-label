"""
1D joint inversion of drift + diffusion from time-unlabeled particle observations.

True: μ*(x) = x - x³,  D*(x) = 1.0 (constant)
Two-phase training: Phase 1 drift only, Phase 2 joint.
Loss: negative marginal log-likelihood.
"""

import sys
sys.path.insert(0, '../../density/1D')

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from joint_inversion import _build_L_variable_D
from generate_data import make_setup
from optimize_particle_1d import sample_particles_from_density

torch.set_default_dtype(torch.float64)


class DriftNet(nn.Module):
    """μ(x) = c₀ + c₁x + c₂x² + c₃x³ + NN(x)"""
    def __init__(self, hidden=16):
        super().__init__()
        self.coeffs = nn.Parameter(torch.zeros(4))
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
        poly = c[0] + c[1]*x + c[2]*x**2 + c[3]*x**3
        return poly + self.net(x.unsqueeze(-1)).squeeze(-1)


class DiffusionNet(nn.Module):
    """D(x) = softplus(b₀ + b₁x + b₂x² + NN(x))"""
    def __init__(self, hidden=8):
        super().__init__()
        self.coeffs = nn.Parameter(torch.zeros(3))  # [1, x, x²]
        self.net = nn.Sequential(
            nn.Linear(1, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )
        with torch.no_grad():
            self.net[-1].weight.mul_(0.01)
            self.net[-1].bias.mul_(0.01)

    def forward(self, x):
        c = self.coeffs
        raw = c[0] + c[1]*x + c[2]*x**2 + self.net(x.unsqueeze(-1)).squeeze(-1)
        return torch.nn.functional.softplus(raw)


def solve_fpe_varD(mu, D, x_grid, t_output, rho0, n_substeps=20):
    """FPE solver with variable diffusion."""
    M = x_grid.shape[0] - 1
    n_int = M - 1
    dx = (x_grid[-1] - x_grid[0]).item() / M

    L = _build_L_variable_D(mu, D, dx, n_int)
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


def generate_particle_obs(x_grid, rho0, D_true, N=100, m=1000, n_substeps=20, seed=0):
    """Generate particle observations from true drift and diffusion."""
    mu_true = x_grid - x_grid**3
    torch.manual_seed(seed)
    t_obs = torch.rand(N)
    with torch.no_grad():
        rho_all = solve_fpe_varD(mu_true, D_true, x_grid, t_obs, rho0, n_substeps=n_substeps)

    rng = torch.Generator()
    rng.manual_seed(seed + 1000)
    particles = torch.zeros(N, m)
    for i in range(N):
        particles[i] = sample_particles_from_density(rho_all[i], x_grid, m, rng=rng)
    return particles, t_obs


def marginal_nll(mu, D, x_grid, rho0, particles, N_t=50, n_substeps=20, eps=1e-30):
    """Negative marginal log-likelihood with variable diffusion."""
    N, m = particles.shape
    dx = (x_grid[-1] - x_grid[0]).item() / (x_grid.shape[0] - 1)
    M_grid = x_grid.shape[0] - 1

    t_grid = torch.linspace(0.5 / N_t, 1.0 - 0.5 / N_t, N_t)
    log_w = -torch.log(torch.tensor(float(N_t)))

    rho_all = solve_fpe_varD(mu, D, x_grid, t_grid, rho0, n_substeps=n_substeps)

    x_min = x_grid[0].item()
    particles_flat = particles.reshape(-1)
    frac_idx = (particles_flat - x_min) / dx
    frac_idx = frac_idx.clamp(0, M_grid - 1e-6)
    idx_lo = frac_idx.long()
    idx_hi = (idx_lo + 1).clamp(max=M_grid)
    alpha = frac_idx - idx_lo.float()

    rho_lo = rho_all[:, idx_lo]
    rho_hi = rho_all[:, idx_hi]
    rho_at_particles = (1 - alpha).unsqueeze(0) * rho_lo + alpha.unsqueeze(0) * rho_hi
    rho_at_particles = rho_at_particles.reshape(N_t, N, m)
    rho_at_particles = torch.clamp(rho_at_particles, min=eps)

    log_rho = torch.log(rho_at_particles)
    log_prod = log_rho.sum(dim=2)
    log_terms = log_w + log_prod
    log_marginal = torch.logsumexp(log_terms, dim=0)

    return -log_marginal.mean()


def run(N_obs=100, m=1000, N_t=50, M=200, n_substeps=20, seed=0,
        phase1_iter=400, phase2_iter=400, lr1=0.03, lr2_drift=0.005, lr2_diff=0.01):

    x_grid, rho0 = make_setup(M=M)
    mu_true = x_grid - x_grid**3
    D_true = 0.5 * torch.ones_like(x_grid)

    particles, t_obs = generate_particle_obs(
        x_grid, rho0, D_true, N=N_obs, m=m, n_substeps=n_substeps, seed=seed
    )
    print(f"Generated {N_obs} groups of {m} particles")

    torch.manual_seed(seed + 100)
    drift_net = DriftNet(hidden=16).double()
    diff_net = DiffusionNet(hidden=8).double()

    loss_hist, mu_mse_hist, D_mse_hist = [], [], []

    # ===== Phase 1: drift only =====
    print("=" * 60)
    print(f"Phase 1: drift only ({phase1_iter} iters, lr={lr1})")
    print("=" * 60)

    opt1 = torch.optim.Adam(drift_net.parameters(), lr=lr1)
    sch1 = torch.optim.lr_scheduler.CosineAnnealingLR(opt1, T_max=phase1_iter, eta_min=lr1/10)

    for it in range(phase1_iter):
        opt1.zero_grad()
        mu_pred = drift_net(x_grid)
        with torch.no_grad():
            D_init = diff_net(x_grid)

        loss = marginal_nll(mu_pred, D_init, x_grid, rho0, particles,
                           N_t=N_t, n_substeps=n_substeps)
        loss.backward()
        opt1.step()
        sch1.step()

        loss_hist.append(loss.item())
        with torch.no_grad():
            mu_mse = ((drift_net(x_grid) - mu_true)**2).mean().item()
        mu_mse_hist.append(mu_mse)
        D_mse_hist.append(0.0)

        if it % 50 == 0 or it == phase1_iter - 1:
            print(f"  iter {it:4d}: loss={loss.item():.4e}  μ_mse={mu_mse:.4e}")

    # ===== Phase 2: joint =====
    print(f"\n{'='*60}")
    print(f"Phase 2: joint ({phase2_iter} iters)")
    print("=" * 60)

    opt2 = torch.optim.Adam([
        {'params': drift_net.parameters(), 'lr': lr2_drift},
        {'params': diff_net.parameters(), 'lr': lr2_diff},
    ])
    sch2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=phase2_iter, eta_min=1e-4)

    for it in range(phase2_iter):
        opt2.zero_grad()
        mu_pred = drift_net(x_grid)
        D_pred = diff_net(x_grid)

        loss = marginal_nll(mu_pred, D_pred, x_grid, rho0, particles,
                           N_t=N_t, n_substeps=n_substeps)
        loss.backward()
        opt2.step()
        sch2.step()

        loss_hist.append(loss.item())
        with torch.no_grad():
            mu_mse = ((drift_net(x_grid) - mu_true)**2).mean().item()
            D_mse = ((diff_net(x_grid) - D_true)**2).mean().item()
        mu_mse_hist.append(mu_mse)
        D_mse_hist.append(D_mse)

        if it % 50 == 0 or it == phase2_iter - 1:
            print(f"  iter {phase1_iter+it:4d}: loss={loss.item():.4e}  "
                  f"μ_mse={mu_mse:.4e}  D_mse={D_mse:.4e}")

    # ===== Plotting =====
    with torch.no_grad():
        mu_learned = drift_net(x_grid).numpy()
        D_learned = diff_net(x_grid).numpy()

    xn = x_grid.numpy()
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    # Drift
    axes[0, 0].plot(xn, mu_true.numpy(), 'k-', lw=2, label='true: x-x³')
    axes[0, 0].plot(xn, mu_learned, 'r--', lw=2, label='learned')
    axes[0, 0].set_title('Drift Recovery'); axes[0, 0].legend(); axes[0, 0].grid(True)

    axes[0, 1].plot(xn, mu_learned - mu_true.numpy(), 'b-', lw=1.5)
    axes[0, 1].axhline(0, color='k', ls='--', alpha=0.3)
    axes[0, 1].set_title(f'Drift Error (MSE={mu_mse_hist[-1]:.3e})'); axes[0, 1].grid(True)

    axes[0, 2].semilogy(mu_mse_hist)
    axes[0, 2].axvline(phase1_iter, color='grey', ls='--', alpha=0.5, label='phase 2')
    axes[0, 2].set_title('Drift MSE'); axes[0, 2].legend(); axes[0, 2].grid(True)

    # Diffusion
    axes[1, 0].plot(xn, D_true.numpy(), 'k-', lw=2, label=f'true: D={D_true[0].item():.1f}')
    axes[1, 0].plot(xn, D_learned, 'r--', lw=2, label='learned')
    axes[1, 0].set_title('Diffusion Recovery'); axes[1, 0].legend(); axes[1, 0].grid(True)

    axes[1, 1].plot(xn, D_learned - D_true.numpy(), 'b-', lw=1.5)
    axes[1, 1].axhline(0, color='k', ls='--', alpha=0.3)
    axes[1, 1].set_title(f'Diffusion Error (MSE={D_mse_hist[-1]:.3e})'); axes[1, 1].grid(True)

    axes[1, 2].semilogy(loss_hist)
    axes[1, 2].axvline(phase1_iter, color='grey', ls='--', alpha=0.5, label='phase 2')
    axes[1, 2].set_title('NLL Loss'); axes[1, 2].legend(); axes[1, 2].grid(True)

    plt.suptitle(f'Joint Particle Inversion: N={N_obs}, m={m}', fontsize=14)
    plt.tight_layout()
    plt.savefig('joint_particle_1d.png', dpi=150)
    plt.close()

    print(f"\n{'='*60}")
    print(f"FINAL: μ_mse={mu_mse_hist[-1]:.4e}, D_mse={D_mse_hist[-1]:.4e}")
    print("Saved joint_particle_1d.png")


if __name__ == '__main__':
    run(N_obs=100, m=1000, phase1_iter=400, phase2_iter=400)
