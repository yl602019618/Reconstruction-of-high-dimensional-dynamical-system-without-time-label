"""
Tests for 2D FPE solver.

Test 1: Pure diffusion vs analytic (2D Gaussian spreading)
Test 2: Mass non-increasing (absorbing BCs)
Test 3: Double well evolution (visual)
Test 4: Gradient check
Test 5: Speed benchmark
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from fpe_solver_2d import solve_fpe_2d, build_drift_2d_poly

torch.set_default_dtype(torch.float64)

L_DOMAIN = 3.0
M = 30  # grid points per dimension


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


def test1_pure_diffusion():
    print("=" * 60)
    print("Test 1: Pure diffusion vs 2D analytic Gaussian")
    x, y, X, Y, rho0 = make_setup(M=40, L=5.0)
    dx = (x[1] - x[0]).item()
    D = 0.5
    mu1 = torch.zeros_like(X)
    mu2 = torch.zeros_like(Y)

    t_out = torch.tensor([0.1, 0.3])
    rho_num = solve_fpe_2d(mu1, mu2, D, x, y, t_out, rho0, n_substeps=30)

    sigma0 = 0.5
    errors = []
    for i, t_val in enumerate(t_out):
        sigma_t = np.sqrt(sigma0**2 + 2 * D * t_val.item())
        rho_exact = torch.exp(-0.5 * (X**2 + Y**2) / sigma_t**2) / (2 * np.pi * sigma_t**2)
        # Normalize both
        m_num = rho_num[i].sum() * dx * dx
        m_ex = rho_exact.sum() * dx * dx
        err = torch.max(torch.abs(rho_num[i]/m_num - rho_exact/m_ex)).item()
        errors.append(err)
    print(f"  L∞ errors: {[f'{e:.2e}' for e in errors]}")
    print(f"  PASS: {all(e < 0.05 for e in errors)}")


def test2_mass():
    print("=" * 60)
    print("Test 2: Mass non-increasing")
    x, y, X, Y, rho0 = make_setup()
    dx = (x[1] - x[0]).item()
    D = 0.5
    theta = torch.tensor([1.0, -1.0, 1.0, -1.0])
    mu1, mu2 = build_drift_2d_poly(theta, X, Y)

    t_out = torch.linspace(0.05, 1.0, 20)
    rho_out = solve_fpe_2d(mu1, mu2, D, x, y, t_out, rho0, n_substeps=30)

    masses = rho_out.sum(dim=(1, 2)) * dx * dx
    diffs = masses[1:] - masses[:-1]
    print(f"  Initial mass: {masses[0].item():.6f}")
    print(f"  Final mass:   {masses[-1].item():.6f}")
    print(f"  Max increase: {diffs.max().item():.2e}")
    print(f"  PASS: {diffs.max().item() < 1e-4}")


def test3_double_well():
    print("=" * 60)
    print("Test 3: Double well evolution (visual)")
    x, y, X, Y, rho0 = make_setup()
    D = 0.5
    theta = torch.tensor([1.0, -1.0, 1.0, -1.0])
    mu1, mu2 = build_drift_2d_poly(theta, X, Y)

    t_out = torch.tensor([0.0001, 0.2, 0.5, 1.0])
    rho_out = solve_fpe_2d(mu1, mu2, D, x, y, t_out, rho0, n_substeps=30)

    fig, axes = plt.subplots(1, 4, figsize=(16, 3.5))
    xn, yn = x.numpy(), y.numpy()
    for i, t_val in enumerate(t_out):
        ax = axes[i]
        im = ax.contourf(xn, yn, rho_out[i].detach().numpy().T, levels=20, cmap='hot')
        ax.set_title(f't={t_val.item():.2f}')
        ax.set_xlabel('x'); ax.set_ylabel('y')
        ax.set_aspect('equal')
        plt.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    plt.savefig('test3_2d_double_well.png', dpi=150)
    plt.close()
    print("  Saved test3_2d_double_well.png")
    print("  PASS: visual (should develop 4 modes)")


def test4_gradient():
    print("=" * 60)
    print("Test 4: Gradient check")
    x, y, X, Y, rho0 = make_setup(M=20)
    D = 0.5
    theta = torch.tensor([0.8, -1.2, 0.9, -0.8], requires_grad=True)
    mu1, mu2 = build_drift_2d_poly(theta, X, Y)
    t_out = torch.tensor([0.3, 0.7])
    rho = solve_fpe_2d(mu1, mu2, D, x, y, t_out, rho0, n_substeps=15)
    loss = rho.sum()
    loss.backward()
    grad_auto = theta.grad.clone()

    eps = 1e-5
    grad_fd = torch.zeros(4)
    base = torch.tensor([0.8, -1.2, 0.9, -0.8])
    for k in range(4):
        tp = base.clone(); tm = base.clone()
        tp[k] += eps; tm[k] -= eps
        m1p, m2p = build_drift_2d_poly(tp, X, Y)
        m1m, m2m = build_drift_2d_poly(tm, X, Y)
        rp = solve_fpe_2d(m1p, m2p, D, x, y, t_out, rho0, n_substeps=15)
        rm = solve_fpe_2d(m1m, m2m, D, x, y, t_out, rho0, n_substeps=15)
        grad_fd[k] = (rp.sum() - rm.sum()) / (2 * eps)

    rel_err = torch.abs(grad_auto - grad_fd) / (torch.abs(grad_fd) + 1e-12)
    print(f"  Autodiff:   {grad_auto.numpy()}")
    print(f"  Finite diff: {grad_fd.numpy()}")
    print(f"  Rel error:  {rel_err.numpy()}")
    print(f"  PASS: {(rel_err < 0.05).all().item()}")


def test5_speed():
    import time
    print("=" * 60)
    print("Test 5: Speed benchmark")
    x, y, X, Y, rho0 = make_setup(M=30)
    D = 0.5
    theta = torch.tensor([1.0, -1.0, 1.0, -1.0], requires_grad=True)
    mu1, mu2 = build_drift_2d_poly(theta, X, Y)
    t_out = torch.rand(30) * 0.98 + 0.01

    # Warmup
    rho = solve_fpe_2d(mu1, mu2, D, x, y, t_out, rho0, n_substeps=20)
    rho.sum().backward()

    N = 20
    t0 = time.time()
    for _ in range(N):
        theta.grad = None
        mu1, mu2 = build_drift_2d_poly(theta, X, Y)
        rho = solve_fpe_2d(mu1, mu2, D, x, y, t_out, rho0, n_substeps=20)
        rho.sum().backward()
    elapsed = (time.time() - t0) / N
    print(f"  M=30, substeps=20: {elapsed*1000:.0f} ms/iter")
    print(f"  Estimated 300 iters: {elapsed*300:.0f} s")


if __name__ == '__main__':
    test1_pure_diffusion()
    test2_mass()
    test3_double_well()
    test4_gradient()
    test5_speed()
    print("=" * 60)
    print("All 2D tests done.")
