"""
2D drift recovery from time-unlabeled particle observations.

Observation: N groups of m particles, each group sampled at unknown time t_i.
Loss: negative marginal log-likelihood with log-sum-exp.

Experiments:
  1. Polynomial drift (20 params): complete cubic basis
  2. NN drift (poly+NN hybrid)
  3. Joint drift + diffusion inversion
"""

import sys
sys.path.insert(0, '../../density/2D')

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from fpe_solver_2d import solve_fpe_2d, _build_L_2d_vectorized

torch.set_default_dtype(torch.float64)


# ========== Setup ==========

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


def poly_basis_2d(X, Y):
    """Complete cubic polynomial basis: 10 terms."""
    return torch.stack([
        torch.ones_like(X), X, Y, X**2, X*Y, Y**2,
        X**3, X**2*Y, X*Y**2, Y**3,
    ], dim=0)


BASIS_NAMES = ['1', 'x', 'y', 'x²', 'xy', 'y²', 'x³', 'x²y', 'xy²', 'y³']

# True: μ₁=x-x³, μ₂=y-y³
TRUE_C1 = torch.tensor([0, 1, 0, 0, 0, 0, -1, 0, 0, 0], dtype=torch.float64)
TRUE_C2 = torch.tensor([0, 0, 1, 0, 0, 0, 0, 0, 0, -1], dtype=torch.float64)


# ========== Particle sampling from 2D density ==========

def sample_particles_2d(rho, x, y, m, rng=None):
    """
    Sample m particles from 2D density rho(x,y) on grid.
    Returns: (m, 2) positions.
    """
    Mx = x.shape[0] - 1
    My = y.shape[0] - 1
    dx = (x[-1] - x[0]).item() / Mx
    dy = (y[-1] - y[0]).item() / My

    # Flatten density as weights
    weights = rho.reshape(-1).clone()
    weights = torch.clamp(weights, min=0.0)
    total = weights.sum()
    if total < 1e-15:
        px = x[0] + (x[-1] - x[0]) * torch.rand(m, generator=rng)
        py = y[0] + (y[-1] - y[0]) * torch.rand(m, generator=rng)
        return torch.stack([px, py], dim=1)

    # Sample cell indices
    indices = torch.multinomial(weights, m, replacement=True, generator=rng)
    # Convert flat index to (i, j)
    i_idx = indices // (My + 1)
    j_idx = indices % (My + 1)

    # Add uniform jitter within cell
    px = x[i_idx] + (torch.rand(m, generator=rng) - 0.5) * dx
    py = y[j_idx] + (torch.rand(m, generator=rng) - 0.5) * dy

    # Clamp to domain
    px = px.clamp(x[0].item(), x[-1].item())
    py = py.clamp(y[0].item(), y[-1].item())

    return torch.stack([px, py], dim=1)


def generate_particle_obs_2d(mu1, mu2, D, x, y, rho0, N=100, m=1000,
                              n_substeps=20, seed=0):
    """Generate N groups of m particles."""
    torch.manual_seed(seed)
    t_obs = torch.rand(N)
    with torch.no_grad():
        rho_all = solve_fpe_2d(mu1, mu2, D, x, y, t_obs, rho0, n_substeps=n_substeps)

    rng = torch.Generator()
    rng.manual_seed(seed + 1000)
    particles = torch.zeros(N, m, 2)
    for i in range(N):
        particles[i] = sample_particles_2d(rho_all[i], x, y, m, rng=rng)

    return particles, t_obs


# ========== Marginal NLL for 2D ==========

def marginal_nll_2d(mu1, mu2, D, x, y, rho0, particles,
                     N_t=50, n_substeps=20, eps=1e-30):
    """
    Negative marginal log-likelihood for 2D particle observations.

    particles: (N, m, 2) positions
    """
    N, m, _ = particles.shape
    Mx = x.shape[0] - 1
    My = y.shape[0] - 1
    dx = (x[-1] - x[0]).item() / Mx
    dy = (y[-1] - y[0]).item() / My

    t_grid = torch.linspace(0.5 / N_t, 1.0 - 0.5 / N_t, N_t)
    log_w = -torch.log(torch.tensor(float(N_t)))

    rho_all = solve_fpe_2d(mu1, mu2, D, x, y, t_grid, rho0, n_substeps=n_substeps)
    # rho_all: (N_t, Mx+1, My+1)

    # Bilinear interpolation at particle positions
    px = particles[:, :, 0].reshape(-1)  # (N*m,)
    py = particles[:, :, 1].reshape(-1)

    fi = (px - x[0].item()) / dx
    fj = (py - y[0].item()) / dy
    fi = fi.clamp(0, Mx - 1e-6)
    fj = fj.clamp(0, My - 1e-6)

    i0 = fi.long()
    j0 = fj.long()
    i1 = (i0 + 1).clamp(max=Mx)
    j1 = (j0 + 1).clamp(max=My)
    ai = fi - i0.float()
    aj = fj - j0.float()

    # rho_all: (N_t, Mx+1, My+1), evaluate at all N*m points
    r00 = rho_all[:, i0, j0]  # (N_t, N*m)
    r10 = rho_all[:, i1, j0]
    r01 = rho_all[:, i0, j1]
    r11 = rho_all[:, i1, j1]

    rho_at_p = (r00 * (1-ai) * (1-aj) + r10 * ai * (1-aj) +
                r01 * (1-ai) * aj + r11 * ai * aj)
    # (N_t, N*m) -> (N_t, N, m)
    rho_at_p = rho_at_p.reshape(N_t, N, m)
    rho_at_p = torch.clamp(rho_at_p, min=eps)

    log_rho = torch.log(rho_at_p)
    log_prod = log_rho.sum(dim=2)  # (N_t, N)
    log_terms = log_w + log_prod
    log_marginal = torch.logsumexp(log_terms, dim=0)  # (N,)

    return -log_marginal.mean()


# ========== Experiment 1: Polynomial drift ==========

def run_poly(n_iter=200, lr=0.01, N_obs=50, m=1000, N_t=50,
             M=30, n_substeps=20, seed=0):
    """Complete cubic polynomial drift inversion from particles."""
    x, y, X, Y, rho0 = make_setup(M=M)
    D = 0.5
    basis = poly_basis_2d(X, Y)

    mu1_true = X - X**3
    mu2_true = Y - Y**3

    particles, t_obs = generate_particle_obs_2d(
        mu1_true, mu2_true, D, x, y, rho0, N=N_obs, m=m,
        n_substeps=n_substeps, seed=seed
    )
    print(f"Generated {N_obs} groups of {m} particles (2D)")

    c1 = torch.zeros(10, requires_grad=True)
    c2 = torch.zeros(10, requires_grad=True)
    optimizer = torch.optim.Adam([c1, c2], lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_iter, eta_min=lr/10)

    loss_hist = []
    c1_hist, c2_hist = [], []

    print(f"Polynomial drift inversion (20 params), M={M}, N_t={N_t}")
    print("-" * 60)

    for it in range(n_iter):
        optimizer.zero_grad()
        mu1 = torch.einsum('k,kij->ij', c1, basis)
        mu2 = torch.einsum('k,kij->ij', c2, basis)

        loss = marginal_nll_2d(mu1, mu2, D, x, y, rho0, particles,
                               N_t=N_t, n_substeps=n_substeps)
        loss.backward()
        optimizer.step()
        scheduler.step()

        loss_hist.append(loss.item())
        c1_hist.append(c1.data.clone().numpy())
        c2_hist.append(c2.data.clone().numpy())

        if it % 20 == 0 or it == n_iter - 1:
            print(f"  iter {it:4d}: loss={loss.item():.4e}")

    c1_hist = np.array(c1_hist)
    c2_hist = np.array(c2_hist)

    # Print coefficients
    print(f"\nμ₁ coefficients (true → learned):")
    for k in range(10):
        t = TRUE_C1[k].item(); l = c1.data[k].item()
        flag = " ***" if abs(t) > 0.01 or abs(l) > 0.05 else ""
        print(f"  {BASIS_NAMES[k]:5s}: {t:7.3f} → {l:7.4f}{flag}")

    print(f"\nμ₂ coefficients (true → learned):")
    for k in range(10):
        t = TRUE_C2[k].item(); l = c2.data[k].item()
        flag = " ***" if abs(t) > 0.01 or abs(l) > 0.05 else ""
        print(f"  {BASIS_NAMES[k]:5s}: {t:7.3f} → {l:7.4f}{flag}")

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    axes[0, 0].plot(loss_hist)
    axes[0, 0].set_title('NLL Loss'); axes[0, 0].grid(True); axes[0, 0].set_xlabel('Iteration')

    for k in range(10):
        s = '-' if abs(TRUE_C1[k]) > 0.01 else '--'
        a = 1.0 if abs(TRUE_C1[k]) > 0.01 else 0.3
        axes[0, 1].plot(c1_hist[:, k], s, alpha=a, label=BASIS_NAMES[k])
    axes[0, 1].set_title('μ₁ coefficients'); axes[0, 1].legend(fontsize=7, ncol=2)
    axes[0, 1].grid(True)

    for k in range(10):
        s = '-' if abs(TRUE_C2[k]) > 0.01 else '--'
        a = 1.0 if abs(TRUE_C2[k]) > 0.01 else 0.3
        axes[1, 1].plot(c2_hist[:, k], s, alpha=a, label=BASIS_NAMES[k])
    axes[1, 1].set_title('μ₂ coefficients'); axes[1, 1].legend(fontsize=7, ncol=2)
    axes[1, 1].grid(True)

    x_pos = np.arange(10)
    w = 0.35
    axes[1, 0].bar(x_pos - w/2, TRUE_C1.numpy(), w, label='true c₁', color='steelblue', alpha=0.7)
    axes[1, 0].bar(x_pos + w/2, c1.data.numpy(), w, label='learned c₁', color='coral', alpha=0.7)
    axes[1, 0].bar(x_pos - w/2, TRUE_C2.numpy(), w, label='true c₂', color='steelblue', alpha=0.3)
    axes[1, 0].bar(x_pos + w/2, c2.data.numpy(), w, label='learned c₂', color='coral', alpha=0.3)
    axes[1, 0].set_xticks(x_pos); axes[1, 0].set_xticklabels(BASIS_NAMES, fontsize=8)
    axes[1, 0].set_title('Coefficients: true vs learned')
    axes[1, 0].legend(fontsize=7); axes[1, 0].grid(True, axis='y')

    plt.suptitle(f'2D Polynomial Particle Inversion: N={N_obs}, m={m}', fontsize=14)
    plt.tight_layout()
    plt.savefig('particle_2d_poly.png', dpi=150)
    plt.close()
    print("\nSaved particle_2d_poly.png")


# ========== Variable-D FPE solver ==========

def _build_L_2d_varD(mu1, mu2, D, dx, dy, Mx, My):
    """Build 2D FPE operator with spatially varying D(x,y)."""
    nx, ny = Mx - 1, My - 1
    n = nx * ny

    ii = torch.arange(1, Mx, device=mu1.device)
    jj = torch.arange(1, My, device=mu1.device)
    I, J = torch.meshgrid(ii, jj, indexing='ij')
    K = (I - 1) * ny + (J - 1)

    L = torch.zeros(n, n, dtype=mu1.dtype, device=mu1.device)
    k_flat = K.reshape(-1)
    i_flat = I.reshape(-1)
    j_flat = J.reshape(-1)

    # Center
    L[k_flat, k_flat] = -2 * D[i_flat, j_flat] / dx**2 - 2 * D[i_flat, j_flat] / dy**2

    # (i-1,j)
    mask = i_flat >= 2
    src = k_flat[mask]; dst = (i_flat[mask] - 2) * ny + (j_flat[mask] - 1)
    L[src, dst] = mu1[i_flat[mask]-1, j_flat[mask]] / (2*dx) + D[i_flat[mask]-1, j_flat[mask]] / dx**2

    # (i+1,j)
    mask = i_flat <= Mx - 2
    src = k_flat[mask]; dst = i_flat[mask] * ny + (j_flat[mask] - 1)
    L[src, dst] = -mu1[i_flat[mask]+1, j_flat[mask]] / (2*dx) + D[i_flat[mask]+1, j_flat[mask]] / dx**2

    # (i,j-1)
    mask = j_flat >= 2
    src = k_flat[mask]; dst = (i_flat[mask] - 1) * ny + (j_flat[mask] - 2)
    L[src, dst] = mu2[i_flat[mask], j_flat[mask]-1] / (2*dy) + D[i_flat[mask], j_flat[mask]-1] / dy**2

    # (i,j+1)
    mask = j_flat <= My - 2
    src = k_flat[mask]; dst = (i_flat[mask] - 1) * ny + j_flat[mask]
    L[src, dst] = -mu2[i_flat[mask], j_flat[mask]+1] / (2*dy) + D[i_flat[mask], j_flat[mask]+1] / dy**2

    return L


def solve_fpe_2d_varD(mu1, mu2, D, x, y, t_output, rho0, n_substeps=20):
    """Solve 2D FPE with spatially varying D(x,y)."""
    Mx = x.shape[0] - 1
    My = y.shape[0] - 1
    nx, ny_int = Mx - 1, My - 1
    n_int = nx * ny_int
    dx = (x[-1] - x[0]).item() / Mx
    dy = (y[-1] - y[0]).item() / My

    L = _build_L_2d_varD(mu1, mu2, D, dx, dy, Mx, My)
    I_mat = torch.eye(n_int, dtype=L.dtype, device=L.device)

    T_max = t_output.max().item()
    dt = T_max / n_substeps
    r = dt / 2.0
    S = torch.linalg.solve(I_mat - r * L, I_mat + r * L)

    rho_int = rho0[1:Mx, 1:My].reshape(-1).clone()
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

    Kt = t_output.shape[0]
    rho_out = torch.zeros(Kt, Mx + 1, My + 1, dtype=rho0.dtype, device=rho0.device)
    rho_out[:, 1:Mx, 1:My] = rho_int_out.reshape(Kt, nx, ny_int)
    return rho_out


def marginal_nll_2d_varD(mu1, mu2, D, x, y, rho0, particles,
                          N_t=50, n_substeps=20, eps=1e-30):
    """Marginal NLL with variable D."""
    N, m, _ = particles.shape
    Mx = x.shape[0] - 1; My = y.shape[0] - 1
    dx = (x[-1] - x[0]).item() / Mx
    dy = (y[-1] - y[0]).item() / My

    t_grid = torch.linspace(0.5 / N_t, 1.0 - 0.5 / N_t, N_t)
    log_w = -torch.log(torch.tensor(float(N_t)))

    rho_all = solve_fpe_2d_varD(mu1, mu2, D, x, y, t_grid, rho0, n_substeps=n_substeps)

    px = particles[:, :, 0].reshape(-1)
    py = particles[:, :, 1].reshape(-1)
    fi = ((px - x[0].item()) / dx).clamp(0, Mx - 1e-6)
    fj = ((py - y[0].item()) / dy).clamp(0, My - 1e-6)
    i0 = fi.long(); j0 = fj.long()
    i1 = (i0 + 1).clamp(max=Mx); j1 = (j0 + 1).clamp(max=My)
    ai = fi - i0.float(); aj = fj - j0.float()

    r00 = rho_all[:, i0, j0]; r10 = rho_all[:, i1, j0]
    r01 = rho_all[:, i0, j1]; r11 = rho_all[:, i1, j1]
    rho_at_p = (r00*(1-ai)*(1-aj) + r10*ai*(1-aj) + r01*(1-ai)*aj + r11*ai*aj)
    rho_at_p = rho_at_p.reshape(N_t, N, m)
    rho_at_p = torch.clamp(rho_at_p, min=eps)

    log_rho = torch.log(rho_at_p)
    log_prod = log_rho.sum(dim=2)
    log_terms = log_w + log_prod
    log_marginal = torch.logsumexp(log_terms, dim=0)
    return -log_marginal.mean()


# ========== NN Models for 2D ==========

class DriftNet2D(nn.Module):
    """Poly+NN for 2D drift: μ_k = Σ c_k φ_j(x,y) + NN_k(x,y)"""
    def __init__(self, hidden=32):
        super().__init__()
        self.c1 = nn.Parameter(torch.zeros(10))
        self.c2 = nn.Parameter(torch.zeros(10))
        self.net1 = nn.Sequential(
            nn.Linear(2, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )
        self.net2 = nn.Sequential(
            nn.Linear(2, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )
        with torch.no_grad():
            for net in [self.net1, self.net2]:
                net[-1].weight.mul_(0.01)
                net[-1].bias.mul_(0.01)

    def forward(self, X, Y, basis):
        mu1 = torch.einsum('k,kij->ij', self.c1, basis)
        mu2 = torch.einsum('k,kij->ij', self.c2, basis)
        xy = torch.stack([X, Y], dim=-1)
        nn1 = self.net1(xy.reshape(-1, 2)).reshape(X.shape)
        nn2 = self.net2(xy.reshape(-1, 2)).reshape(X.shape)
        return mu1 + nn1, mu2 + nn2


class DiffusionNet2D(nn.Module):
    """D(x,y) = softplus(b₀ + b₁x + b₂y + small_NN(x,y))"""
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


# ========== Experiment 2: NN drift ==========

def run_nn(n_iter=500, lr=0.02, N_obs=100, m=1000, N_t=50,
           M=30, n_substeps=20, seed=0):
    """NN drift inversion from 2D particles."""
    x, y, X, Y, rho0 = make_setup(M=M)
    D = 0.5
    basis = poly_basis_2d(X, Y)
    mu1_true = X - X**3
    mu2_true = Y - Y**3

    particles, t_obs = generate_particle_obs_2d(
        mu1_true, mu2_true, D, x, y, rho0, N=N_obs, m=m,
        n_substeps=n_substeps, seed=seed
    )
    print(f"Generated {N_obs} groups of {m} particles (2D)")

    torch.manual_seed(seed + 100)
    model = DriftNet2D(hidden=32).double()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_iter, eta_min=lr/10)

    loss_hist, mu_mse_hist = [], []

    print(f"NN drift inversion, M={M}, N_t={N_t}")
    print("-" * 60)

    for it in range(n_iter):
        optimizer.zero_grad()
        mu1, mu2 = model(X, Y, basis)
        loss = marginal_nll_2d(mu1, mu2, D, x, y, rho0, particles,
                               N_t=N_t, n_substeps=n_substeps)
        loss.backward()
        optimizer.step()
        scheduler.step()

        loss_hist.append(loss.item())
        with torch.no_grad():
            m1, m2 = model(X, Y, basis)
            mu_mse = ((m1 - mu1_true)**2 + (m2 - mu2_true)**2).mean().item() / 2
        mu_mse_hist.append(mu_mse)

        if it % 50 == 0 or it == n_iter - 1:
            print(f"  iter {it:4d}: loss={loss.item():.4e}  μ_mse={mu_mse:.4e}")

    # Plot
    with torch.no_grad():
        m1p, m2p = model(X, Y, basis)
    Xn, Yn = X.numpy(), Y.numpy()
    xn = x.numpy()
    m1t, m2t = mu1_true.numpy(), mu2_true.numpy()
    m1l, m2l = m1p.numpy(), m2p.numpy()
    idx = M // 2

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    vmax = max(abs(m1t).max(), abs(m1l).max())
    kw = dict(levels=30, cmap='RdBu_r', vmin=-vmax, vmax=vmax)

    for col, (data, title) in enumerate([
        (m1t, 'True μ₁'), (m1l, 'Learned μ₁'),
        (m1l-m1t, f'Error μ₁ (max={abs(m1l-m1t).max():.2f})')]):
        ax = axes[0, col]
        kw2 = kw if col < 2 else dict(levels=30, cmap='RdBu_r')
        im = ax.contourf(Xn, Yn, data, **kw2); plt.colorbar(im, ax=ax)
        ax.set_title(title); ax.set_aspect('equal')

    # 1D slices + loss + MSE
    axes[1, 0].plot(xn, m1t[:, idx], 'k-', lw=2, label='true')
    axes[1, 0].plot(xn, m1l[:, idx], 'r--', lw=2, label='learned')
    axes[1, 0].set_title('μ₁(x, y=0)'); axes[1, 0].legend(); axes[1, 0].grid(True)

    axes[1, 1].plot(loss_hist)
    axes[1, 1].set_title('NLL Loss'); axes[1, 1].grid(True); axes[1, 1].set_xlabel('Iteration')

    axes[1, 2].semilogy(mu_mse_hist)
    axes[1, 2].set_title('Drift MSE'); axes[1, 2].grid(True); axes[1, 2].set_xlabel('Iteration')

    plt.suptitle(f'2D NN Particle Inversion: N={N_obs}, m={m}', fontsize=14)
    plt.tight_layout()
    plt.savefig('particle_2d_nn.png', dpi=150)
    plt.close()
    print(f"\nFinal drift MSE: {mu_mse_hist[-1]:.4e}")
    print("Saved particle_2d_nn.png")


# ========== Experiment 3: Joint drift + diffusion ==========

def run_joint(N_obs=100, m=1000, N_t=50, M=30, n_substeps=20, seed=0,
              phase1_iter=50, phase2_iter=800, lr1=0.02,
              lr2_drift=0.01, lr2_diff=0.005):
    """Joint drift+diffusion inversion from 2D particles.
    Phase 1: very short drift warmup with constant D=softplus(0).
    Phase 2: joint training from early on.
    """
    x, y, X, Y, rho0 = make_setup(M=M)
    basis = poly_basis_2d(X, Y)
    mu1_true = X - X**3
    mu2_true = Y - Y**3
    D_true = 0.5 * torch.ones_like(X)

    particles, t_obs = generate_particle_obs_2d(
        mu1_true, mu2_true, 0.5, x, y, rho0, N=N_obs, m=m,
        n_substeps=n_substeps, seed=seed
    )
    print(f"Generated {N_obs} groups of {m} particles (2D)")

    torch.manual_seed(seed + 100)
    drift_net = DriftNet2D(hidden=32).double()
    diff_net = DiffusionNet2D(hidden=8).double()

    loss_hist, mu_mse_hist, D_mse_hist = [], [], []

    # Phase 1: short drift warmup with D = softplus(0) ≈ 0.69
    D_init = torch.nn.functional.softplus(torch.tensor(0.0)).item()
    print("=" * 60)
    print(f"Phase 1: drift warmup ({phase1_iter} iters, D_fixed={D_init:.3f})")
    print("=" * 60)

    opt1 = torch.optim.Adam(drift_net.parameters(), lr=lr1)
    sch1 = torch.optim.lr_scheduler.CosineAnnealingLR(opt1, T_max=phase1_iter, eta_min=lr1/10)

    for it in range(phase1_iter):
        opt1.zero_grad()
        mu1, mu2 = drift_net(X, Y, basis)

        loss = marginal_nll_2d(mu1, mu2, D_init, x, y, rho0, particles,
                               N_t=N_t, n_substeps=n_substeps)
        loss.backward()
        opt1.step()
        sch1.step()

        loss_hist.append(loss.item())
        with torch.no_grad():
            m1, m2 = drift_net(X, Y, basis)
            mu_mse = ((m1-mu1_true)**2 + (m2-mu2_true)**2).mean().item() / 2
        mu_mse_hist.append(mu_mse)
        D_mse_hist.append(0.0)

        if it % 50 == 0 or it == phase1_iter - 1:
            print(f"  iter {it:4d}: loss={loss.item():.4e}  μ_mse={mu_mse:.4e}")

    # Phase 2
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
        mu1, mu2 = drift_net(X, Y, basis)
        D_pred = diff_net(X, Y)

        loss = marginal_nll_2d_varD(mu1, mu2, D_pred, x, y, rho0, particles,
                                     N_t=N_t, n_substeps=n_substeps)
        loss.backward()
        opt2.step()
        sch2.step()

        loss_hist.append(loss.item())
        with torch.no_grad():
            m1, m2 = drift_net(X, Y, basis)
            Dp = diff_net(X, Y)
            mu_mse = ((m1-mu1_true)**2 + (m2-mu2_true)**2).mean().item() / 2
            D_mse = ((Dp - D_true)**2).mean().item()
        mu_mse_hist.append(mu_mse)
        D_mse_hist.append(D_mse)

        if it % 50 == 0 or it == phase2_iter - 1:
            print(f"  iter {phase1_iter+it:4d}: loss={loss.item():.4e}  "
                  f"μ_mse={mu_mse:.4e}  D_mse={D_mse:.4e}")

    # Plot
    with torch.no_grad():
        m1p, m2p = drift_net(X, Y, basis)
        Dp = diff_net(X, Y)

    Xn, Yn = X.numpy(), Y.numpy()
    xn = x.numpy()
    m1t, m2t = mu1_true.numpy(), mu2_true.numpy()
    m1l, m2l = m1p.numpy(), m2p.numpy()
    Dt, Dl = D_true.numpy(), Dp.numpy()
    idx = M // 2

    fig = plt.figure(figsize=(18, 16))

    # Row 1: μ₁
    vmax = max(abs(m1t).max(), abs(m1l).max())
    kw = dict(levels=30, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    for col, (data, title) in enumerate([
        (m1t, 'True μ₁'), (m1l, 'Learned μ₁'),
        (m1l-m1t, f'Error μ₁ (max={abs(m1l-m1t).max():.2f})')]):
        ax = fig.add_subplot(4, 3, col+1)
        kw2 = kw if col < 2 else dict(levels=30, cmap='RdBu_r')
        im = ax.contourf(Xn, Yn, data, **kw2); plt.colorbar(im, ax=ax)
        ax.set_title(title); ax.set_aspect('equal')

    # Row 2: D
    ax = fig.add_subplot(4, 3, 4)
    im = ax.contourf(Xn, Yn, Dt, levels=30, cmap='viridis', vmin=0.3, vmax=0.7)
    plt.colorbar(im, ax=ax); ax.set_title(f'True D={D_true[0,0].item():.1f}'); ax.set_aspect('equal')
    ax = fig.add_subplot(4, 3, 5)
    im = ax.contourf(Xn, Yn, Dl, levels=30, cmap='viridis', vmin=0.3, vmax=0.7)
    plt.colorbar(im, ax=ax); ax.set_title('Learned D'); ax.set_aspect('equal')
    ax = fig.add_subplot(4, 3, 6)
    im = ax.contourf(Xn, Yn, Dl-Dt, levels=30, cmap='RdBu_r')
    plt.colorbar(im, ax=ax); ax.set_title(f'Error D (max={abs(Dl-Dt).max():.3f})'); ax.set_aspect('equal')

    # Row 3: 1D slices
    ax = fig.add_subplot(4, 3, 7)
    ax.plot(xn, m1t[:, idx], 'k-', lw=2, label='true')
    ax.plot(xn, m1l[:, idx], 'r--', lw=2, label='learned')
    ax.set_title('μ₁(x, y=0)'); ax.legend(); ax.grid(True)

    ax = fig.add_subplot(4, 3, 8)
    ax.plot(xn, m2t[idx, :], 'k-', lw=2, label='true')
    ax.plot(xn, m2l[idx, :], 'r--', lw=2, label='learned')
    ax.set_title('μ₂(x=0, y)'); ax.legend(); ax.grid(True)

    ax = fig.add_subplot(4, 3, 9)
    ax.plot(xn, Dt[:, idx], 'k-', lw=2, label='true')
    ax.plot(xn, Dl[:, idx], 'r--', lw=2, label='learned')
    ax.set_title('D(x, y=0)'); ax.legend(); ax.grid(True)

    # Row 4: curves
    ax = fig.add_subplot(4, 3, 10)
    ax.plot(loss_hist)
    ax.axvline(phase1_iter, color='grey', ls='--', alpha=0.5, label='phase 2')
    ax.set_title('NLL Loss'); ax.grid(True); ax.legend()

    ax = fig.add_subplot(4, 3, 11)
    ax.semilogy(mu_mse_hist)
    ax.axvline(phase1_iter, color='grey', ls='--', alpha=0.5)
    ax.set_title('Drift MSE'); ax.grid(True)

    ax = fig.add_subplot(4, 3, 12)
    ax.semilogy(D_mse_hist[phase1_iter:])
    ax.set_title('Diffusion MSE (phase 2)'); ax.grid(True)

    plt.suptitle(f'2D Joint Particle Inversion: N={N_obs}, m={m}', fontsize=14)
    plt.tight_layout()
    plt.savefig('particle_2d_joint.png', dpi=150)
    plt.close()

    print(f"\n{'='*60}")
    print(f"FINAL: μ_mse={mu_mse_hist[-1]:.4e}, D_mse={D_mse_hist[-1]:.4e}")
    print("Saved particle_2d_joint.png")


if __name__ == '__main__':
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else 'poly'
    if mode == 'poly':
        run_poly(n_iter=500, lr=0.02, N_obs=100, m=1000, N_t=50, M=30)
    elif mode == 'nn':
        run_nn(n_iter=500, lr=0.02, N_obs=100, m=1000, N_t=50, M=30)
    elif mode == 'joint':
        run_joint(N_obs=100, m=1000, phase1_iter=400, phase2_iter=400)
