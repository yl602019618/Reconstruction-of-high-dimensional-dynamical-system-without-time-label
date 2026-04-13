"""Tests for the FPE forward solver."""

import torch
import numpy as np
import matplotlib.pyplot as plt
from fpe_solver import solve_fpe, build_drift

torch.set_default_dtype(torch.float64)

def make_grid(a, b, M):
    return torch.linspace(a, b, M + 1)

def gaussian_ic(x_grid, mu=0.0, sigma=0.5):
    rho = torch.exp(-0.5 * ((x_grid - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
    rho[0] = 0.0; rho[-1] = 0.0
    dx = (x_grid[-1] - x_grid[0]).item() / (x_grid.shape[0] - 1)
    mass = torch.trapezoid(rho, dx=dx)
    rho = rho / mass
    rho[0] = 0.0; rho[-1] = 0.0
    return rho

def test1_pure_diffusion():
    print("=" * 60)
    print("Test 1: Pure diffusion (μ=0) vs analytic solution")
    a, b, M = -5.0, 5.0, 200
    x = make_grid(a, b, M)
    dx = (b - a) / M
    sigma0 = 0.5
    rho0 = gaussian_ic(x, 0.0, sigma0)

    theta = torch.tensor([0.0, 0.0])
    t_out = torch.tensor([0.05, 0.1, 0.2, 0.5])
    rho_num = solve_fpe(theta, x, t_out, rho0, n_substeps=50)

    fig, axes = plt.subplots(1, 4, figsize=(16, 3))
    errors = []
    for i, t_val in enumerate(t_out):
        sigma_t = np.sqrt(sigma0 ** 2 + t_val.item())
        rho_exact = torch.exp(-0.5 * (x / sigma_t) ** 2) / (sigma_t * np.sqrt(2 * np.pi))
        mass_num = torch.trapezoid(rho_num[i], dx=dx)
        mass_exact = torch.trapezoid(rho_exact, dx=dx)
        err = torch.max(torch.abs(rho_num[i] / mass_num - rho_exact / mass_exact)).item()
        errors.append(err)
        axes[i].plot(x.numpy(), (rho_exact / mass_exact).numpy(), 'k-', label='exact')
        axes[i].plot(x.numpy(), (rho_num[i] / mass_num).detach().numpy(), 'r--', label='numerical')
        axes[i].set_title(f't={t_val.item():.2f}, err={err:.2e}')
        axes[i].legend(fontsize=8)
    plt.tight_layout(); plt.savefig('test1_pure_diffusion.png', dpi=150); plt.close()
    print(f"  L∞ errors: {[f'{e:.2e}' for e in errors]}")
    print(f"  PASS: {all(e < 0.05 for e in errors)}")

def test2_mass_conservation():
    print("=" * 60)
    print("Test 2: Mass non-increasing (absorbing BCs)")
    a, b, M = -3.0, 3.0, 100
    x = make_grid(a, b, M)
    dx = (b - a) / M
    rho0 = gaussian_ic(x)
    theta = torch.tensor([1.0, 1.0])
    t_out = torch.linspace(0.02, 1.0, 50)
    rho_out = solve_fpe(theta, x, t_out, rho0, n_substeps=50)
    masses = torch.trapezoid(rho_out, dx=dx, dim=1)
    diffs = masses[1:] - masses[:-1]
    print(f"  Initial mass: {masses[0].item():.6f}")
    print(f"  Final mass:   {masses[-1].item():.6f}")
    print(f"  Max increase: {diffs.max().item():.2e}")
    print(f"  PASS: {diffs.max().item() < 1e-4}")
    plt.figure(figsize=(6, 3))
    plt.plot(t_out.numpy(), masses.detach().numpy())
    plt.xlabel('t'); plt.ylabel('mass'); plt.title('Mass over time')
    plt.tight_layout(); plt.savefig('test2_mass.png', dpi=150); plt.close()

def test3_steady_state():
    print("=" * 60)
    print("Test 3: Double well evolution")
    a, b, M = -3.0, 3.0, 100
    x = make_grid(a, b, M)
    rho0 = gaussian_ic(x)
    theta = torch.tensor([1.0, 1.0])
    t_out = torch.tensor([0.1, 0.3, 0.5, 0.8, 1.0])
    rho_out = solve_fpe(theta, x, t_out, rho0, n_substeps=50)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(x.numpy(), rho0.numpy(), 'k-', label='t=0', linewidth=2)
    for i, t_val in enumerate(t_out):
        ax.plot(x.numpy(), rho_out[i].detach().numpy(), label=f't={t_val.item():.1f}')
    ax.set_xlabel('x'); ax.set_ylabel('ρ'); ax.set_title('Double well evolution')
    ax.legend(); plt.tight_layout(); plt.savefig('test3_double_well.png', dpi=150); plt.close()
    print("  PASS: visual")

def test4_nonuniform_time():
    print("=" * 60)
    print("Test 4: Non-uniform time grid robustness")
    a, b, M = -3.0, 3.0, 100
    x = make_grid(a, b, M)
    rho0 = gaussian_ic(x)
    theta = torch.tensor([1.0, 1.0])
    t_eval = torch.tensor([0.2, 0.5, 0.8])
    rho_coarse = solve_fpe(theta, x, t_eval, rho0, n_substeps=20)
    rho_fine = solve_fpe(theta, x, t_eval, rho0, n_substeps=200)
    errors = []
    for i in range(len(t_eval)):
        e = torch.max(torch.abs(rho_coarse[i] - rho_fine[i])).item()
        errors.append(e)
    print(f"  Errors (20 vs 200 substeps): {[f'{e:.2e}' for e in errors]}")
    print(f"  PASS: {all(e < 0.01 for e in errors)}")

def test5_gradient_check():
    print("=" * 60)
    print("Test 5: Gradient verification")
    a, b, M = -3.0, 3.0, 50
    x = make_grid(a, b, M)
    rho0 = gaussian_ic(x)
    theta = torch.tensor([0.8, 1.2], requires_grad=True)
    t_out = torch.tensor([0.3, 0.7])
    rho_out = solve_fpe(theta, x, t_out, rho0, n_substeps=20)
    loss = rho_out.sum()
    loss.backward()
    grad_auto = theta.grad.clone()

    eps = 1e-5
    grad_fd = torch.zeros(2)
    for k in range(2):
        tp = torch.tensor([0.8, 1.2]); tm = torch.tensor([0.8, 1.2])
        tp[k] += eps; tm[k] -= eps
        rp = solve_fpe(tp, x, t_out, rho0, n_substeps=20)
        rm = solve_fpe(tm, x, t_out, rho0, n_substeps=20)
        grad_fd[k] = (rp.sum() - rm.sum()) / (2 * eps)
    rel_err = torch.abs(grad_auto - grad_fd) / (torch.abs(grad_fd) + 1e-12)
    print(f"  Autodiff:  {grad_auto.numpy()}")
    print(f"  Finite diff: {grad_fd.numpy()}")
    print(f"  Rel error: {rel_err.numpy()}")
    print(f"  PASS: {(rel_err < 0.05).all().item()}")

def test_speed():
    import time
    print("=" * 60)
    print("Speed benchmark")
    x = make_grid(-3, 3, 100)
    rho0 = gaussian_ic(x)
    theta = torch.tensor([1.0, 1.0], requires_grad=True)
    t_out = torch.rand(50) * 0.98 + 0.01
    # Warmup
    solve_fpe(theta, x, t_out, rho0, n_substeps=20)
    # Benchmark
    t0 = time.time()
    N = 50
    for _ in range(N):
        theta.grad = None
        rho = solve_fpe(theta, x, t_out, rho0, n_substeps=20)
        rho.sum().backward()
    elapsed = (time.time() - t0) / N
    print(f"  Fwd+bwd per iter: {elapsed*1000:.1f} ms")
    print(f"  Est 300 iters: {elapsed*300:.1f} s")

if __name__ == '__main__':
    test1_pure_diffusion()
    test2_mass_conservation()
    test3_steady_state()
    test4_nonuniform_time()
    test5_gradient_check()
    test_speed()
    print("=" * 60)
    print("All tests done.")
