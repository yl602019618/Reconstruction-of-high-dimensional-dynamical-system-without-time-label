"""
Neural network parameterized drift for FPE inverse problem.

Replaces the 2-parameter drift μ(x)=θ₁x-θ₂x³ with a small MLP.
The FPE solver is modified to accept a drift vector directly.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from fpe_solver import _build_L


class DriftNet(nn.Module):
    """
    MLP + polynomial basis: μ(x) = a₁x + a₃x³ + NN(x)

    The polynomial part provides good initial signal for gradient-based
    optimization; the NN part captures residual nonlinearity.
    NN is initialized near zero so initial drift ≈ a₁x + a₃x³.
    """
    def __init__(self, hidden=16):
        super().__init__()
        # Polynomial coefficients (learnable)
        self.a1 = nn.Parameter(torch.tensor(0.0))
        self.a3 = nn.Parameter(torch.tensor(0.0))
        # Residual NN (initialized small)
        self.net = nn.Sequential(
            nn.Linear(1, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )
        # Initialize last layer small
        with torch.no_grad():
            self.net[-1].weight.mul_(0.01)
            self.net[-1].bias.mul_(0.01)

    def forward(self, x):
        """x: (M+1,) -> mu: (M+1,)"""
        poly = self.a1 * x + self.a3 * x ** 3
        nn_out = self.net(x.unsqueeze(-1)).squeeze(-1)
        return poly + nn_out


def solve_fpe_with_drift(mu, x_grid, t_output, rho0, n_substeps=20):
    """
    FPE solver that takes a drift vector directly (instead of theta).

    mu: (M+1,) drift values on grid
    x_grid: (M+1,)
    t_output: (K,)
    rho0: (M+1,)
    n_substeps: time steps

    Returns: (K, M+1)
    """
    M = x_grid.shape[0] - 1
    n_int = M - 1
    dx = (x_grid[-1] - x_grid[0]).item() / M

    L = _build_L(mu, dx, n_int)
    I = torch.eye(n_int, dtype=L.dtype, device=L.device)

    T_max = t_output.max().item()
    dt = T_max / n_substeps
    r = dt / 2.0

    A = I - r * L
    B = I + r * L
    S = torch.linalg.solve(A, B)

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


def run_nn_experiment(loss_type='w1', n_iter=200, lr=1e-3, N_obs=20, N_sim=20,
                      M=50, n_substeps=15, seed=0, reg_weight=0.0):
    """
    Run drift recovery with NN parameterization.

    True drift: μ*(x) = x - x³
    """
    from losses import mmd_loss, pointwise_w1_loss

    torch.set_default_dtype(torch.float64)

    # Setup
    x_grid = torch.linspace(-3, 3, M + 1)
    dx = 6.0 / M
    sigma0 = 0.5
    rho0 = torch.exp(-0.5 * (x_grid / sigma0) ** 2) / (sigma0 * np.sqrt(2 * np.pi))
    rho0[0] = 0.0; rho0[-1] = 0.0
    mass = torch.trapezoid(rho0, dx=dx)
    rho0 = rho0 / mass
    rho0[0] = 0.0; rho0[-1] = 0.0

    # True drift and observations
    mu_true = x_grid - x_grid ** 3
    torch.manual_seed(seed)
    t_obs = torch.rand(N_obs)
    with torch.no_grad():
        rho_obs = solve_fpe_with_drift(mu_true, x_grid, t_obs, rho0, n_substeps=n_substeps)

    # NN model
    torch.manual_seed(seed + 100)
    model = DriftNet(hidden=16).double()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_iter, eta_min=lr/10)

    loss_history = []
    drift_errors = []

    print(f"NN drift recovery with {loss_type.upper()} loss")
    print(f"Grid M={M}, N_obs={N_obs}, N_sim={N_sim}, substeps={n_substeps}")
    print("-" * 50)

    for it in range(n_iter):
        optimizer.zero_grad()

        mu_pred = model(x_grid)

        torch.manual_seed(2000 + it)
        t_sim = torch.rand(N_sim)

        rho_sim = solve_fpe_with_drift(mu_pred, x_grid, t_sim, rho0, n_substeps=n_substeps)

        if loss_type == 'mmd':
            loss = mmd_loss(rho_sim, rho_obs)
        else:
            loss = pointwise_w1_loss(rho_sim, rho_obs)

        # Smoothness regularization: penalize |μ''(x)|² via finite diff on grid
        if reg_weight > 0:
            mu_dd = (mu_pred[2:] - 2 * mu_pred[1:-1] + mu_pred[:-2]) / (dx ** 2)
            loss = loss + reg_weight * torch.mean(mu_dd ** 2)

        loss.backward()

        # Gradient diagnostics
        if it % 50 == 0:
            grad_norms = [p.grad.norm().item() for p in model.parameters() if p.grad is not None]
            print(f"    grad norms: {[f'{g:.2e}' for g in grad_norms]}")

        optimizer.step()
        scheduler.step()

        loss_val = loss.item()
        loss_history.append(loss_val)

        with torch.no_grad():
            mu_err = torch.mean((model(x_grid) - mu_true) ** 2).item()
        drift_errors.append(mu_err)

        if it % 20 == 0 or it == n_iter - 1:
            a1 = model.a1.item()
            a3 = model.a3.item()
            print(f"  iter {it:4d}: loss={loss_val:.6e}, drift_mse={mu_err:.6e}, a1={a1:.4f}, a3={a3:.4f}")

    # Final drift comparison plot
    with torch.no_grad():
        mu_learned = model(x_grid).numpy()

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Drift comparison
    axes[0].plot(x_grid.numpy(), mu_true.numpy(), 'k-', linewidth=2, label='true: x-x³')
    axes[0].plot(x_grid.numpy(), mu_learned, 'r--', linewidth=2, label='NN learned')
    axes[0].set_xlabel('x'); axes[0].set_ylabel('μ(x)')
    axes[0].set_title('Drift Function Recovery')
    axes[0].legend(); axes[0].grid(True)

    # Loss curve
    axes[1].semilogy(loss_history)
    axes[1].set_xlabel('Iteration'); axes[1].set_ylabel('Loss')
    axes[1].set_title(f'{loss_type.upper()} Loss')
    axes[1].grid(True)

    # Drift MSE
    axes[2].semilogy(drift_errors)
    axes[2].set_xlabel('Iteration'); axes[2].set_ylabel('MSE(μ)')
    axes[2].set_title('Drift MSE vs True')
    axes[2].grid(True)

    plt.tight_layout()
    plt.savefig(f'nn_drift_{loss_type}.png', dpi=150)
    plt.close()
    print(f"\nFinal drift MSE: {drift_errors[-1]:.6e}")
    print(f"Saved nn_drift_{loss_type}.png")

    return loss_history, drift_errors, model


if __name__ == '__main__':
    print("=" * 60)
    print("NN Drift Recovery: W1 loss")
    print("=" * 60)
    loss_w1, err_w1, model_w1 = run_nn_experiment('w1', n_iter=800, lr=0.03, N_obs=50, N_sim=50)

    print("\n" + "=" * 60)
    print("NN Drift Recovery: MMD loss")
    print("=" * 60)
    loss_mmd, err_mmd, model_mmd = run_nn_experiment('mmd', n_iter=800, lr=0.03, N_obs=50, N_sim=50)

    # Comparison
    x_grid = torch.linspace(-3, 3, 51)
    mu_true = x_grid - x_grid ** 3

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    with torch.no_grad():
        mu_w1 = model_w1(x_grid).numpy()
        mu_mmd = model_mmd(x_grid).numpy()

    axes[0].plot(x_grid.numpy(), mu_true.numpy(), 'k-', linewidth=2, label='true')
    axes[0].plot(x_grid.numpy(), mu_w1, 'r--', linewidth=2, label='W1')
    axes[0].plot(x_grid.numpy(), mu_mmd, 'b:', linewidth=2, label='MMD')
    axes[0].set_xlabel('x'); axes[0].set_ylabel('μ(x)')
    axes[0].set_title('Drift Recovery Comparison')
    axes[0].legend(); axes[0].grid(True)

    axes[1].semilogy(err_w1, label='W1')
    axes[1].semilogy(err_mmd, label='MMD')
    axes[1].set_xlabel('Iteration'); axes[1].set_ylabel('Drift MSE')
    axes[1].set_title('Drift MSE Comparison')
    axes[1].legend(); axes[1].grid(True)

    plt.tight_layout()
    plt.savefig('nn_drift_comparison.png', dpi=150)
    plt.close()

    print(f"\nFinal drift MSE - W1: {err_w1[-1]:.6e}, MMD: {err_mmd[-1]:.6e}")
    print("Saved nn_drift_comparison.png")
