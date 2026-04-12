"""
Joint inversion of drift μ(x) and diffusion D(x) from unlabeled density snapshots.

FPE: ∂ρ/∂t = -∂_x(μρ) + ∂_xx(Dρ)
where D(x) = σ²(x)/2.

True: μ*(x) = x - x³,  D*(x) = 1/2 (constant)

Parameterization:
  μ(x) = a₁x + a₂x² + a₃x³ + NN_μ(x)
  D(x) = softplus(b₀ + b₁x + NN_D(x))   [ensures D>0]
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from losses import pointwise_w1_loss

torch.set_default_dtype(torch.float64)


# ---- FPE solver with variable diffusion ----

def _build_L_variable_D(mu, D, dx, n_int):
    """
    Build operator matrix L for: L[ρ] = -∂_x(μρ) + ∂_xx(Dρ)

    Discretization on interior nodes j=1..M-1:
      ∂_xx(Dρ)_j ≈ [D_{j+1}ρ_{j+1} - 2D_jρ_j + D_{j-1}ρ_{j-1}] / dx²
      -∂_x(μρ)_j ≈ [μ_{j-1}ρ_{j-1} - μ_{j+1}ρ_{j+1}] / (2dx)

    Coefficients of ρ_{j-1}, ρ_j, ρ_{j+1}:
      a_j = μ_{j-1}/(2dx) + D_{j-1}/dx²
      b_j = -2D_j/dx²
      c_j = -μ_{j+1}/(2dx) + D_{j+1}/dx²
    """
    M = n_int + 1
    dx2 = dx ** 2

    a = mu[0:M-1] / (2*dx) + D[0:M-1] / dx2      # (n_int,)
    b = -2 * D[1:M] / dx2                           # (n_int,)
    c = -mu[2:M+1] / (2*dx) + D[2:M+1] / dx2      # (n_int,)

    L = torch.diag(b) + torch.diag(a[1:], -1) + torch.diag(c[:-1], 1)
    return L


def solve_fpe_variable(mu, D, x_grid, t_output, rho0, n_substeps=20):
    """
    Solve FPE with spatially varying drift mu and diffusion D.

    mu: (M+1,) drift
    D:  (M+1,) diffusion coefficient (must be >0)
    """
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


# ---- Neural network models ----

class DriftNet(nn.Module):
    """μ(x) = a₁x + a₂x² + a₃x³ + NN(x)"""
    def __init__(self, hidden=16):
        super().__init__()
        self.a1 = nn.Parameter(torch.tensor(0.0))
        self.a2 = nn.Parameter(torch.tensor(0.0))
        self.a3 = nn.Parameter(torch.tensor(0.0))
        self.net = nn.Sequential(
            nn.Linear(1, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )
        with torch.no_grad():
            self.net[-1].weight.mul_(0.01)
            self.net[-1].bias.mul_(0.01)

    def forward(self, x):
        poly = self.a1 * x + self.a2 * x**2 + self.a3 * x**3
        return poly + self.net(x.unsqueeze(-1)).squeeze(-1)


class DiffusionNet(nn.Module):
    """D(x) = softplus(b₀ + b₁x + NN(x)), ensures D>0"""
    def __init__(self, hidden=16):
        super().__init__()
        self.b0 = nn.Parameter(torch.tensor(0.0))  # will learn ~ log(0.5) ≈ -0.69
        self.b1 = nn.Parameter(torch.tensor(0.0))
        self.net = nn.Sequential(
            nn.Linear(1, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )
        with torch.no_grad():
            self.net[-1].weight.mul_(0.01)
            self.net[-1].bias.mul_(0.01)

    def forward(self, x):
        raw = self.b0 + self.b1 * x + self.net(x.unsqueeze(-1)).squeeze(-1)
        return torch.nn.functional.softplus(raw)


# ---- Experiment ----

def run_joint_inversion(n_iter=800, lr=0.03, N_obs=50, N_sim=50,
                        M=50, n_substeps=15, seed=0):
    # Setup
    x_grid = torch.linspace(-3, 3, M + 1)
    dx = 6.0 / M
    sigma0 = 0.5
    rho0 = torch.exp(-0.5 * (x_grid / sigma0) ** 2) / (sigma0 * np.sqrt(2 * np.pi))
    rho0[0] = 0.0; rho0[-1] = 0.0
    mass = torch.trapezoid(rho0, dx=dx)
    rho0 = rho0 / mass
    rho0[0] = 0.0; rho0[-1] = 0.0

    # True parameters
    mu_true = x_grid - x_grid ** 3
    D_true = 1.0 * torch.ones_like(x_grid)

    # Generate observations
    torch.manual_seed(seed)
    t_obs = torch.rand(N_obs)
    with torch.no_grad():
        rho_obs = solve_fpe_variable(mu_true, D_true, x_grid, t_obs, rho0, n_substeps=n_substeps)

    # Models
    torch.manual_seed(seed + 100)
    drift_net = DriftNet(hidden=16).double()
    diff_net = DiffusionNet(hidden=16).double()

    optimizer = torch.optim.Adam(
        list(drift_net.parameters()) + list(diff_net.parameters()), lr=lr
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_iter, eta_min=lr/10)

    loss_history = []
    drift_mse_history = []
    diff_mse_history = []

    print(f"Joint inversion: drift + diffusion (W1 loss)")
    print(f"Grid M={M}, N_obs={N_obs}, N_sim={N_sim}, substeps={n_substeps}, iters={n_iter}")
    print(f"True: μ(x)=x-x³, D(x)=1.0")
    print("-" * 60)

    for it in range(n_iter):
        optimizer.zero_grad()

        mu_pred = drift_net(x_grid)
        D_pred = diff_net(x_grid)

        torch.manual_seed(2000 + it)
        t_sim = torch.rand(N_sim)

        rho_sim = solve_fpe_variable(mu_pred, D_pred, x_grid, t_sim, rho0, n_substeps=n_substeps)

        loss = pointwise_w1_loss(rho_sim, rho_obs)
        loss.backward()
        optimizer.step()
        scheduler.step()

        loss_val = loss.item()
        loss_history.append(loss_val)

        with torch.no_grad():
            mu_err = torch.mean((drift_net(x_grid) - mu_true) ** 2).item()
            D_err = torch.mean((diff_net(x_grid) - D_true) ** 2).item()
        drift_mse_history.append(mu_err)
        diff_mse_history.append(D_err)

        if it % 40 == 0 or it == n_iter - 1:
            a1 = drift_net.a1.item()
            a2 = drift_net.a2.item()
            a3 = drift_net.a3.item()
            b0 = diff_net.b0.item()
            b1 = diff_net.b1.item()
            print(f"  iter {it:4d}: loss={loss_val:.4e}  "
                  f"μ_mse={mu_err:.4e}  D_mse={D_err:.4e}  "
                  f"a=[{a1:.3f},{a2:.3f},{a3:.3f}]  b=[{b0:.3f},{b1:.3f}]")

    # ---- Plotting ----
    with torch.no_grad():
        mu_learned = drift_net(x_grid).numpy()
        D_learned = diff_net(x_grid).numpy()

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    # Row 1: drift
    axes[0, 0].plot(x_grid.numpy(), mu_true.numpy(), 'k-', lw=2, label='true: x-x³')
    axes[0, 0].plot(x_grid.numpy(), mu_learned, 'r--', lw=2, label='learned')
    axes[0, 0].set_xlabel('x'); axes[0, 0].set_ylabel('μ(x)')
    axes[0, 0].set_title('Drift Recovery'); axes[0, 0].legend(); axes[0, 0].grid(True)

    axes[0, 1].semilogy(drift_mse_history)
    axes[0, 1].set_xlabel('Iteration'); axes[0, 1].set_ylabel('MSE')
    axes[0, 1].set_title('Drift MSE'); axes[0, 1].grid(True)

    axes[0, 2].plot(x_grid.numpy(), mu_learned - mu_true.numpy(), 'b-', lw=1)
    axes[0, 2].axhline(0, color='k', ls='--', alpha=0.3)
    axes[0, 2].set_xlabel('x'); axes[0, 2].set_ylabel('μ_learned - μ_true')
    axes[0, 2].set_title('Drift Error'); axes[0, 2].grid(True)

    # Row 2: diffusion
    axes[1, 0].plot(x_grid.numpy(), D_true.numpy(), 'k-', lw=2, label=f'true: D={D_true[0].item():.1f}')
    axes[1, 0].plot(x_grid.numpy(), D_learned, 'r--', lw=2, label='learned')
    axes[1, 0].set_xlabel('x'); axes[1, 0].set_ylabel('D(x)')
    axes[1, 0].set_title('Diffusion Recovery'); axes[1, 0].legend(); axes[1, 0].grid(True)

    axes[1, 1].semilogy(diff_mse_history)
    axes[1, 1].set_xlabel('Iteration'); axes[1, 1].set_ylabel('MSE')
    axes[1, 1].set_title('Diffusion MSE'); axes[1, 1].grid(True)

    axes[1, 2].semilogy(loss_history)
    axes[1, 2].set_xlabel('Iteration'); axes[1, 2].set_ylabel('W1 Loss')
    axes[1, 2].set_title('Loss Curve'); axes[1, 2].grid(True)

    plt.tight_layout()
    plt.savefig('joint_inversion.png', dpi=150)
    plt.close()

    print(f"\nFinal drift MSE: {drift_mse_history[-1]:.6e}")
    print(f"Final diff  MSE: {diff_mse_history[-1]:.6e}")
    print("Saved joint_inversion.png")

    return drift_net, diff_net, loss_history, drift_mse_history, diff_mse_history


if __name__ == '__main__':
    run_joint_inversion(n_iter=800, lr=0.03, N_obs=200, N_sim=200, M=50, n_substeps=15)
