"""
Optimization loop: recover drift parameters from unlabeled density snapshots.
Supports both MMD and pointwise W1 loss.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from fpe_solver import solve_fpe
from losses import mmd_loss, pointwise_w1_loss
from generate_data import make_setup, generate_observations

torch.set_default_dtype(torch.float64)


def optimize(loss_type='mmd', n_iter=300, lr=0.01, N_obs=50, N_sim=50, seed=0):
    """
    Run optimization to recover theta_true = (1, 1).

    Args:
        loss_type: 'mmd' or 'w1'
        n_iter: number of iterations
        lr: learning rate
        N_obs: number of observed snapshots
        N_sim: number of simulated snapshots per iteration
    """
    x_grid, rho0 = make_setup()
    theta_true = torch.tensor([1.0, 1.0])

    # Generate observations (fixed)
    rho_obs, _ = generate_observations(theta_true, x_grid, rho0, N=N_obs, seed=seed)

    # Initial guess
    theta = torch.tensor([0.5, 0.5], requires_grad=True)
    optimizer = torch.optim.Adam([theta], lr=lr)

    loss_history = []
    theta_history = []

    print(f"Optimizing with {loss_type.upper()} loss")
    print(f"True theta: {theta_true.numpy()}")
    print(f"Initial theta: {theta.data.numpy()}")
    print("-" * 50)

    for it in range(n_iter):
        optimizer.zero_grad()

        # Sample simulation times
        torch.manual_seed(1000 + it)
        t_sim = torch.rand(N_sim)

        # Forward solve
        rho_sim = solve_fpe(theta, x_grid, t_sim, rho0)

        # Compute loss
        if loss_type == 'mmd':
            loss = mmd_loss(rho_sim, rho_obs)
        else:
            loss = pointwise_w1_loss(rho_sim, rho_obs)

        loss.backward()
        optimizer.step()

        loss_val = loss.item()
        loss_history.append(loss_val)
        theta_history.append(theta.data.clone().numpy())

        if it % 20 == 0 or it == n_iter - 1:
            print(f"  iter {it:4d}: loss={loss_val:.6e}, theta=[{theta.data[0]:.4f}, {theta.data[1]:.4f}]")

    theta_history = np.array(theta_history)

    # Plot loss curve
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].semilogy(loss_history)
    axes[0].set_xlabel('Iteration')
    axes[0].set_ylabel('Loss')
    axes[0].set_title(f'{loss_type.upper()} Loss Curve')

    axes[1].plot(theta_history[:, 0], label='θ₁')
    axes[1].plot(theta_history[:, 1], label='θ₂')
    axes[1].axhline(1.0, color='k', linestyle='--', alpha=0.5, label='true')
    axes[1].set_xlabel('Iteration')
    axes[1].set_ylabel('Parameter value')
    axes[1].set_title('Parameter Trajectories')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(f'optimize_{loss_type}.png', dpi=150)
    plt.close()

    print(f"\nFinal theta: [{theta.data[0]:.4f}, {theta.data[1]:.4f}]")
    print(f"True theta:  [1.0000, 1.0000]")
    print(f"Error: [{abs(theta.data[0]-1):.4f}, {abs(theta.data[1]-1):.4f}]")

    return theta.data.clone(), loss_history, theta_history


if __name__ == '__main__':
    print("=" * 60)
    theta_mmd, loss_mmd, traj_mmd = optimize('mmd', n_iter=300)
    print("\n" + "=" * 60)
    theta_w1, loss_w1, traj_w1 = optimize('w1', n_iter=300)

    # Comparison plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].semilogy(loss_mmd, label='MMD')
    axes[0].semilogy(loss_w1, label='W1')
    axes[0].set_xlabel('Iteration')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss Comparison')
    axes[0].legend()

    axes[1].plot(traj_mmd[:, 0], traj_mmd[:, 1], 'b.-', alpha=0.3, markersize=2, label='MMD')
    axes[1].plot(traj_w1[:, 0], traj_w1[:, 1], 'r.-', alpha=0.3, markersize=2, label='W1')
    axes[1].plot(1.0, 1.0, 'k*', markersize=15, label='true')
    axes[1].plot(0.5, 0.5, 'go', markersize=8, label='init')
    axes[1].set_xlabel('θ₁')
    axes[1].set_ylabel('θ₂')
    axes[1].set_title('Parameter Trajectories')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig('optimize_comparison.png', dpi=150)
    plt.close()
    print("\nSaved comparison plot to optimize_comparison.png")
