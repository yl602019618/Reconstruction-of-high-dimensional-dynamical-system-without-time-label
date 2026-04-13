"""
Two loss functions for comparing unlabeled density snapshot distributions.

Loss A: MMD with Gaussian RBF kernel
Loss B: Pointwise 1D Wasserstein (W1) averaged over space
"""

import torch


def mmd_loss(rho_sim, rho_obs, sigma=None):
    """
    MMD² between two sets of density snapshots.

    rho_sim: (N_s, M+1) simulated snapshots
    rho_obs: (N, M+1) observed snapshots
    sigma: kernel bandwidth; if None, use median heuristic

    Returns: scalar MMD²
    """
    if sigma is None:
        # Median heuristic on all pairwise distances
        all_rho = torch.cat([rho_sim, rho_obs], dim=0)
        dists = torch.cdist(all_rho, all_rho, p=2)
        sigma = torch.median(dists[dists > 0]).item()
        if sigma < 1e-10:
            sigma = 1.0

    def rbf(X, Y):
        # (N1, N2)
        dist_sq = torch.cdist(X, Y, p=2) ** 2
        return torch.exp(-dist_sq / (2 * sigma ** 2))

    Kxx = rbf(rho_sim, rho_sim)
    Kyy = rbf(rho_obs, rho_obs)
    Kxy = rbf(rho_sim, rho_obs)

    N_s = rho_sim.shape[0]
    N = rho_obs.shape[0]

    mmd2 = Kxx.sum() / (N_s * N_s) + Kyy.sum() / (N * N) - 2 * Kxy.sum() / (N_s * N)
    return mmd2


def pointwise_w1_loss(rho_sim, rho_obs):
    """
    Pointwise 1D Wasserstein distance averaged over spatial grid.

    For each spatial point x_j, sort the simulated and observed density values,
    then compute W1 = (1/N) Σ|sorted_sim - sorted_obs|.

    rho_sim: (N_s, M+1)
    rho_obs: (N, M+1)

    If N_s != N, subsample the larger set to match.
    Returns: scalar average W1
    """
    N_s = rho_sim.shape[0]
    N = rho_obs.shape[0]

    if N_s != N:
        n = min(N_s, N)
        if N_s > N:
            idx = torch.randperm(N_s)[:n]
            rho_sim = rho_sim[idx]
        else:
            idx = torch.randperm(N)[:n]
            rho_obs = rho_obs[idx]
    else:
        n = N

    # Sort along sample dimension (dim=0) at each spatial point
    sim_sorted = torch.sort(rho_sim, dim=0)[0]  # (n, M+1)
    obs_sorted = torch.sort(rho_obs, dim=0)[0]  # (n, M+1)

    # W1 at each spatial point, then average
    w1_per_point = torch.abs(sim_sorted - obs_sorted).mean(dim=0)  # (M+1,)
    return w1_per_point.mean()
