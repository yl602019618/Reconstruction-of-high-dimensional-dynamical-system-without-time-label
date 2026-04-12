"""
2D FPE inversion with complete cubic polynomial drift basis.

μ₁(x,y) = Σ c₁ₖ φₖ(x,y),  μ₂(x,y) = Σ c₂ₖ φₖ(x,y)
where {φₖ} = {1, x, y, x², xy, y², x³, x²y, xy², y³}  (10 terms)

True: μ₁ = x - x³, μ₂ = y - y³  (only 4 of 20 coefficients nonzero)
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from fpe_solver_2d import solve_fpe_2d

torch.set_default_dtype(torch.float64)


def poly_basis_2d(X, Y):
    """
    Complete cubic polynomial basis on 2D grid.
    X, Y: (Mx+1, My+1)
    Returns: (10, Mx+1, My+1)
    """
    return torch.stack([
        torch.ones_like(X),  # 0: 1
        X,                    # 1: x
        Y,                    # 2: y
        X**2,                 # 3: x²
        X * Y,                # 4: xy
        Y**2,                 # 5: y²
        X**3,                 # 6: x³
        X**2 * Y,             # 7: x²y
        X * Y**2,             # 8: xy²
        Y**3,                 # 9: y³
    ], dim=0)


BASIS_NAMES = ['1', 'x', 'y', 'x²', 'xy', 'y²', 'x³', 'x²y', 'xy²', 'y³']

# True coefficients: μ₁ = x - x³ → c1=[0,1,0,0,0,0,-1,0,0,0]
#                    μ₂ = y - y³ → c2=[0,0,1,0,0,0,0,0,0,-1]
TRUE_C1 = torch.tensor([0, 1, 0, 0, 0, 0, -1, 0, 0, 0], dtype=torch.float64)
TRUE_C2 = torch.tensor([0, 0, 1, 0, 0, 0, 0, 0, 0, -1], dtype=torch.float64)


def build_drift_full_poly(c1, c2, basis):
    """c1, c2: (10,), basis: (10, Mx+1, My+1) -> mu1, mu2: (Mx+1, My+1)"""
    mu1 = torch.einsum('k,kij->ij', c1, basis)
    mu2 = torch.einsum('k,kij->ij', c2, basis)
    return mu1, mu2


def pointwise_w1_loss_2d(rho_sim, rho_obs):
    N = rho_sim.shape[0]
    sim_flat = rho_sim.reshape(N, -1)
    obs_flat = rho_obs.reshape(N, -1)
    sim_sorted = torch.sort(sim_flat, dim=0)[0]
    obs_sorted = torch.sort(obs_flat, dim=0)[0]
    return torch.abs(sim_sorted - obs_sorted).mean()


def make_setup(M=30, L=3.0):
    x = torch.linspace(-L, L, M + 1)
    y = torch.linspace(-L, L, M + 1)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    dx = 2 * L / M
    sigma0 = 0.5
    rho0 = torch.exp(-0.5 * (X**2 + Y**2) / sigma0**2) / (2 * np.pi * sigma0**2)
    rho0[0, :] = 0; rho0[-1, :] = 0; rho0[:, 0] = 0; rho0[:, -1] = 0
    mass = rho0.sum() * dx * dx
    rho0 = rho0 / mass
    rho0[0, :] = 0; rho0[-1, :] = 0; rho0[:, 0] = 0; rho0[:, -1] = 0
    return x, y, X, Y, rho0


def run(n_iter=200, lr=0.01, N_obs=30, N_sim=30, M=30, n_substeps=20, seed=0):
    x, y, X, Y, rho0 = make_setup(M=M)
    D = 0.5
    basis = poly_basis_2d(X, Y)  # (10, Mx+1, My+1)

    # True drift
    mu1_true, mu2_true = build_drift_full_poly(TRUE_C1, TRUE_C2, basis)

    # Observations
    torch.manual_seed(seed)
    t_obs = torch.rand(N_obs)
    with torch.no_grad():
        rho_obs = solve_fpe_2d(mu1_true, mu2_true, D, x, y, t_obs, rho0, n_substeps=n_substeps)

    # Init all coefficients at zero
    c1 = torch.zeros(10, requires_grad=True)
    c2 = torch.zeros(10, requires_grad=True)
    optimizer = torch.optim.Adam([c1, c2], lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_iter, eta_min=lr/10)

    loss_history = []
    c1_history = []
    c2_history = []

    print(f"Full cubic poly drift inversion (20 params)")
    print(f"M={M}, N={N_obs}, substeps={n_substeps}")
    print("-" * 60)

    for it in range(n_iter):
        optimizer.zero_grad()
        mu1, mu2 = build_drift_full_poly(c1, c2, basis)

        torch.manual_seed(1000 + it)
        t_sim = torch.rand(N_sim)

        rho_sim = solve_fpe_2d(mu1, mu2, D, x, y, t_sim, rho0, n_substeps=n_substeps)
        loss = pointwise_w1_loss_2d(rho_sim, rho_obs)
        loss.backward()
        optimizer.step()
        scheduler.step()

        loss_history.append(loss.item())
        c1_history.append(c1.data.clone().numpy())
        c2_history.append(c2.data.clone().numpy())

        if it % 20 == 0 or it == n_iter - 1:
            print(f"  iter {it:4d}: loss={loss.item():.4e}")

    c1_history = np.array(c1_history)
    c2_history = np.array(c2_history)

    # Print final coefficients
    print(f"\nμ₁ coefficients (true → learned):")
    for k in range(10):
        t = TRUE_C1[k].item()
        l = c1.data[k].item()
        flag = " ***" if abs(t) > 0.01 or abs(l) > 0.05 else ""
        print(f"  {BASIS_NAMES[k]:5s}: {t:7.3f} → {l:7.4f}{flag}")

    print(f"\nμ₂ coefficients (true → learned):")
    for k in range(10):
        t = TRUE_C2[k].item()
        l = c2.data[k].item()
        flag = " ***" if abs(t) > 0.01 or abs(l) > 0.05 else ""
        print(f"  {BASIS_NAMES[k]:5s}: {t:7.3f} → {l:7.4f}{flag}")

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Loss
    axes[0, 0].semilogy(loss_history)
    axes[0, 0].set_title('W1 Loss'); axes[0, 0].grid(True)
    axes[0, 0].set_xlabel('Iteration')

    # c1 trajectories
    for k in range(10):
        style = '-' if abs(TRUE_C1[k]) > 0.01 else '--'
        alpha = 1.0 if abs(TRUE_C1[k]) > 0.01 else 0.3
        axes[0, 1].plot(c1_history[:, k], style, alpha=alpha, label=BASIS_NAMES[k])
    axes[0, 1].set_title('μ₁ coefficients'); axes[0, 1].legend(fontsize=7, ncol=2)
    axes[0, 1].grid(True); axes[0, 1].set_xlabel('Iteration')

    # c2 trajectories
    for k in range(10):
        style = '-' if abs(TRUE_C2[k]) > 0.01 else '--'
        alpha = 1.0 if abs(TRUE_C2[k]) > 0.01 else 0.3
        axes[1, 1].plot(c2_history[:, k], style, alpha=alpha, label=BASIS_NAMES[k])
    axes[1, 1].set_title('μ₂ coefficients'); axes[1, 1].legend(fontsize=7, ncol=2)
    axes[1, 1].grid(True); axes[1, 1].set_xlabel('Iteration')

    # Bar chart: final coefficients vs true
    x_pos = np.arange(10)
    width = 0.35
    axes[1, 0].bar(x_pos - width/2, TRUE_C1.numpy(), width, label='true c₁', color='steelblue', alpha=0.7)
    axes[1, 0].bar(x_pos + width/2, c1.data.numpy(), width, label='learned c₁', color='coral', alpha=0.7)
    axes[1, 0].bar(x_pos - width/2, TRUE_C2.numpy(), width, label='true c₂', color='steelblue', alpha=0.3)
    axes[1, 0].bar(x_pos + width/2, c2.data.numpy(), width, label='learned c₂', color='coral', alpha=0.3)
    axes[1, 0].set_xticks(x_pos); axes[1, 0].set_xticklabels(BASIS_NAMES, fontsize=8)
    axes[1, 0].set_title('Coefficients: true vs learned')
    axes[1, 0].legend(fontsize=7); axes[1, 0].grid(True, axis='y')

    plt.tight_layout()
    plt.savefig('optimize_2d_full_poly.png', dpi=150)
    plt.close()
    print("\nSaved optimize_2d_full_poly.png")


if __name__ == '__main__':
    run(n_iter=200, lr=0.01, N_obs=30, N_sim=30, M=30, n_substeps=20)
