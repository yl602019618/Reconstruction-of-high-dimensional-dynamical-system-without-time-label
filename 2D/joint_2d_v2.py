"""
2D joint inversion v2: improved training strategy.

Key improvements over v1:
  1. More data: N=200
  2. Two-phase training: Phase 1 trains drift only (D fixed at softplus(0)≈0.69),
     Phase 2 jointly fine-tunes both with smaller lr for diffusion
  3. More iterations: 400+400=800 total
  4. Smaller diffusion NN (less overfitting for a near-constant target)
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from joint_2d import (
    solve_fpe_2d_varD, poly_basis_2d, pointwise_w1,
    DriftNet2D, make_setup
)

torch.set_default_dtype(torch.float64)


class DiffusionNet2D_small(nn.Module):
    """Smaller D network: softplus(b₀ + b₁x + b₂y + small_NN)"""
    def __init__(self, hidden=8):
        super().__init__()
        self.b0 = nn.Parameter(torch.tensor(0.0))
        self.b1 = nn.Parameter(torch.tensor(0.0))
        self.b2 = nn.Parameter(torch.tensor(0.0))
        self.net = nn.Sequential(
            nn.Linear(2, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )
        with torch.no_grad():
            self.net[-1].weight.mul_(0.01)
            self.net[-1].bias.mul_(0.01)

    def forward(self, X, Y):
        raw = self.b0 + self.b1 * X + self.b2 * Y
        xy = torch.stack([X, Y], dim=-1)
        nn_out = self.net(xy.reshape(-1, 2)).reshape(X.shape)
        return torch.nn.functional.softplus(raw + nn_out)


def run(N_obs=200, N_sim=200, M=30, n_substeps=20, seed=0,
        phase1_iter=400, phase2_iter=400, lr1=0.01, lr2_drift=0.003, lr2_diff=0.005):

    x, y, X, Y, rho0 = make_setup(M=M)
    basis = poly_basis_2d(X, Y)

    mu1_true = X - X**3
    mu2_true = Y - Y**3
    D_true = 0.5 * torch.ones_like(X)

    # Observations
    torch.manual_seed(seed)
    t_obs = torch.rand(N_obs)
    with torch.no_grad():
        rho_obs = solve_fpe_2d_varD(mu1_true, mu2_true, D_true, x, y, t_obs, rho0, n_substeps=n_substeps)

    # Models
    torch.manual_seed(seed + 100)
    drift_net = DriftNet2D(hidden=32).double()
    diff_net = DiffusionNet2D_small(hidden=8).double()

    total_iter = phase1_iter + phase2_iter
    loss_hist, mu_mse_hist, D_mse_hist = [], [], []

    # ============ Phase 1: drift only, D fixed ============
    print("=" * 60)
    print(f"Phase 1: drift only ({phase1_iter} iters, lr={lr1})")
    print(f"N={N_obs}, M={M}, substeps={n_substeps}")
    print("=" * 60)

    opt1 = torch.optim.Adam(drift_net.parameters(), lr=lr1)
    sch1 = torch.optim.lr_scheduler.CosineAnnealingLR(opt1, T_max=phase1_iter, eta_min=lr1/10)

    for it in range(phase1_iter):
        opt1.zero_grad()
        mu1, mu2 = drift_net(X, Y, basis)
        # Use diff_net's initial output as fixed D (no oracle)
        with torch.no_grad():
            D_fixed = diff_net(X, Y)

        torch.manual_seed(1000 + it)
        t_sim = torch.rand(N_sim)
        rho_sim = solve_fpe_2d_varD(mu1, mu2, D_fixed, x, y, t_sim, rho0, n_substeps=n_substeps)
        loss = pointwise_w1(rho_sim, rho_obs)
        loss.backward()
        opt1.step()
        sch1.step()

        loss_hist.append(loss.item())
        with torch.no_grad():
            m1p, m2p = drift_net(X, Y, basis)
            mu_mse = ((m1p - mu1_true)**2 + (m2p - mu2_true)**2).mean().item() / 2
        mu_mse_hist.append(mu_mse)
        D_mse_hist.append(0.0)  # not training D yet

        if it % 40 == 0 or it == phase1_iter - 1:
            print(f"  iter {it:4d}: loss={loss.item():.4e}  μ_mse={mu_mse:.4e}")

    # ============ Phase 2: joint fine-tune ============
    print(f"\n{'='*60}")
    print(f"Phase 2: joint ({phase2_iter} iters, lr_drift={lr2_drift}, lr_diff={lr2_diff})")
    print("=" * 60)

    opt2 = torch.optim.Adam([
        {'params': drift_net.parameters(), 'lr': lr2_drift},
        {'params': diff_net.parameters(), 'lr': lr2_diff},
    ])
    sch2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=phase2_iter, eta_min=1e-4)

    for it in range(phase2_iter):
        opt2.zero_grad()
        mu1, mu2 = drift_net(X, Y, basis)
        D_pred = diff_net(X, Y)

        torch.manual_seed(2000 + it)
        t_sim = torch.rand(N_sim)
        rho_sim = solve_fpe_2d_varD(mu1, mu2, D_pred, x, y, t_sim, rho0, n_substeps=n_substeps)
        loss = pointwise_w1(rho_sim, rho_obs)
        loss.backward()
        opt2.step()
        sch2.step()

        loss_hist.append(loss.item())
        with torch.no_grad():
            m1p, m2p = drift_net(X, Y, basis)
            Dp = diff_net(X, Y)
            mu_mse = ((m1p - mu1_true)**2 + (m2p - mu2_true)**2).mean().item() / 2
            D_mse = ((Dp - D_true)**2).mean().item()
        mu_mse_hist.append(mu_mse)
        D_mse_hist.append(D_mse)

        if it % 40 == 0 or it == phase2_iter - 1:
            print(f"  iter {phase1_iter+it:4d}: loss={loss.item():.4e}  "
                  f"μ_mse={mu_mse:.4e}  D_mse={D_mse:.4e}  b0={diff_net.b0.item():.3f}")

    # ============ Plotting ============
    with torch.no_grad():
        m1p, m2p = drift_net(X, Y, basis)
        Dp = diff_net(X, Y)

    Xn, Yn = X.numpy(), Y.numpy()
    xn = x.numpy()
    m1t, m2t = mu1_true.numpy(), mu2_true.numpy()
    m1l, m2l = m1p.numpy(), m2p.numpy()
    Dt, Dl = D_true.numpy(), Dp.numpy()
    idx = M // 2

    fig = plt.figure(figsize=(18, 20))

    # Row 1: μ₁ contour
    vmax = max(abs(m1t).max(), abs(m1l).max())
    kw = dict(levels=30, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    for col, (data, title) in enumerate([(m1t, 'True μ₁'), (m1l, 'Learned μ₁'),
                                          (m1l-m1t, f'Error μ₁ (max={abs(m1l-m1t).max():.2f})')]):
        ax = fig.add_subplot(5, 3, col+1)
        kw2 = kw if col < 2 else dict(levels=30, cmap='RdBu_r')
        im = ax.contourf(Xn, Yn, data, **kw2); plt.colorbar(im, ax=ax)
        ax.set_title(title); ax.set_aspect('equal')

    # Row 2: μ₂ contour
    for col, (data, title) in enumerate([(m2t, 'True μ₂'), (m2l, 'Learned μ₂'),
                                          (m2l-m2t, f'Error μ₂ (max={abs(m2l-m2t).max():.2f})')]):
        ax = fig.add_subplot(5, 3, col+4)
        kw2 = kw if col < 2 else dict(levels=30, cmap='RdBu_r')
        im = ax.contourf(Xn, Yn, data, **kw2); plt.colorbar(im, ax=ax)
        ax.set_title(title); ax.set_aspect('equal')

    # Row 3: D contour
    ax = fig.add_subplot(5, 3, 7)
    im = ax.contourf(Xn, Yn, Dt, levels=30, cmap='viridis', vmin=0.3, vmax=0.7)
    plt.colorbar(im, ax=ax); ax.set_title('True D=0.5'); ax.set_aspect('equal')

    ax = fig.add_subplot(5, 3, 8)
    im = ax.contourf(Xn, Yn, Dl, levels=30, cmap='viridis', vmin=0.3, vmax=0.7)
    plt.colorbar(im, ax=ax); ax.set_title('Learned D'); ax.set_aspect('equal')

    ax = fig.add_subplot(5, 3, 9)
    im = ax.contourf(Xn, Yn, Dl-Dt, levels=30, cmap='RdBu_r')
    plt.colorbar(im, ax=ax); ax.set_title(f'Error D (max={abs(Dl-Dt).max():.3f})'); ax.set_aspect('equal')

    # Row 4: 1D slices
    ax = fig.add_subplot(5, 3, 10)
    ax.plot(xn, m1t[:, idx], 'k-', lw=2, label='true')
    ax.plot(xn, m1l[:, idx], 'r--', lw=2, label='learned')
    ax.set_title('μ₁(x, y=0)'); ax.legend(); ax.grid(True)

    ax = fig.add_subplot(5, 3, 11)
    ax.plot(xn, m2t[idx, :], 'k-', lw=2, label='true')
    ax.plot(xn, m2l[idx, :], 'r--', lw=2, label='learned')
    ax.set_title('μ₂(x=0, y)'); ax.legend(); ax.grid(True)

    ax = fig.add_subplot(5, 3, 12)
    ax.plot(xn, Dt[:, idx], 'k-', lw=2, label='true')
    ax.plot(xn, Dl[:, idx], 'r--', lw=2, label='learned')
    ax.set_title('D(x, y=0)'); ax.legend(); ax.grid(True)

    # Row 5: loss and MSE curves
    ax = fig.add_subplot(5, 3, 13)
    ax.semilogy(loss_hist, alpha=0.7)
    ax.axvline(phase1_iter, color='grey', ls='--', alpha=0.5, label='phase 2 start')
    ax.set_title('W1 Loss'); ax.grid(True); ax.legend(); ax.set_xlabel('Iteration')

    ax = fig.add_subplot(5, 3, 14)
    ax.semilogy(mu_mse_hist, alpha=0.7)
    ax.axvline(phase1_iter, color='grey', ls='--', alpha=0.5)
    ax.set_title('Drift MSE'); ax.grid(True); ax.set_xlabel('Iteration')

    ax = fig.add_subplot(5, 3, 15)
    ax.semilogy(D_mse_hist[phase1_iter:], alpha=0.7)
    ax.set_title('Diffusion MSE (phase 2)'); ax.grid(True); ax.set_xlabel('Iteration')

    plt.tight_layout()
    plt.savefig('joint_2d_v2.png', dpi=150)
    plt.close()

    print(f"\n{'='*60}")
    print(f"FINAL: μ_mse={mu_mse_hist[-1]:.4e}, D_mse={D_mse_hist[-1]:.4e}")
    print("Saved joint_2d_v2.png")


if __name__ == '__main__':
    run()
