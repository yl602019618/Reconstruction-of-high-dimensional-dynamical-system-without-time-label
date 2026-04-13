"""
Differentiable 2D Fokker-Planck equation solver (Crank-Nicolson).

Solves: ∂ρ/∂t = -∇·(μρ) + D∇²ρ   on Ω=[-L,L]²
with absorbing BCs: ρ=0 on ∂Ω.

μ(x,y) = (μ₁(x,y), μ₂(x,y)) is the 2D drift vector field.
D is a known constant diffusion coefficient.

Interior nodes are indexed by flattened (i,j) -> k = (i-1)*ny + (j-1)
where i=1..Mx-1, j=1..My-1, and ny = My-1.
"""

import torch


def _build_L_2d(mu1, mu2, D, dx, dy, Mx, My):
    """
    Build the 2D FPE operator matrix L on interior nodes.

    mu1, mu2: (Mx+1, My+1) drift components on full grid
    D: scalar diffusion
    dx, dy: grid spacings
    Mx, My: number of intervals in x, y

    Interior: i=1..Mx-1, j=1..My-1
    n_int = nx * ny where nx=Mx-1, ny=My-1

    L[ρ]_{i,j} = -(μ₁_{i+1,j}ρ_{i+1,j} - μ₁_{i-1,j}ρ_{i-1,j})/(2dx)
                 -(μ₂_{i,j+1}ρ_{i,j+1} - μ₂_{i,j-1}ρ_{i,j-1})/(2dy)
                 + D(ρ_{i+1,j}-2ρ_{i,j}+ρ_{i-1,j})/dx²
                 + D(ρ_{i,j+1}-2ρ_{i,j}+ρ_{i,j-1})/dy²

    Coefficients for ρ at neighbors:
      (i-1,j): μ₁_{i-1,j}/(2dx) + D/dx²
      (i+1,j): -μ₁_{i+1,j}/(2dx) + D/dx²
      (i,j-1): μ₂_{i,j-1}/(2dy) + D/dy²
      (i,j+1): -μ₂_{i,j+1}/(2dy) + D/dy²
      (i,j):   -2D/dx² - 2D/dy²

    Only include connections to other interior nodes; boundary nodes have ρ=0.
    """
    nx = Mx - 1
    ny = My - 1
    n = nx * ny

    def idx(i, j):
        """Interior (i,j) -> flat index. i in 1..Mx-1, j in 1..My-1."""
        return (i - 1) * ny + (j - 1)

    # Build L as dense matrix
    L = torch.zeros(n, n, dtype=mu1.dtype, device=mu1.device)

    for i in range(1, Mx):
        for j in range(1, My):
            k = idx(i, j)
            # Center
            L[k, k] = -2 * D / dx**2 - 2 * D / dy**2

            # (i-1,j)
            if i - 1 >= 1:  # interior
                L[k, idx(i-1, j)] = mu1[i-1, j] / (2*dx) + D / dx**2
            # (i+1,j)
            if i + 1 <= Mx - 1:
                L[k, idx(i+1, j)] = -mu1[i+1, j] / (2*dx) + D / dx**2
            # (i,j-1)
            if j - 1 >= 1:
                L[k, idx(i, j-1)] = mu2[i, j-1] / (2*dy) + D / dy**2
            # (i,j+1)
            if j + 1 <= My - 1:
                L[k, idx(i, j+1)] = -mu2[i, j+1] / (2*dy) + D / dy**2

    return L


def _build_L_2d_vectorized(mu1, mu2, D, dx, dy, Mx, My):
    """
    Vectorized version of _build_L_2d for speed.
    """
    nx = Mx - 1
    ny = My - 1
    n = nx * ny

    # Interior grid indices: i in [1,Mx-1], j in [1,My-1]
    # Flat index: k = (i-1)*ny + (j-1)
    ii = torch.arange(1, Mx, device=mu1.device)
    jj = torch.arange(1, My, device=mu1.device)
    I, J = torch.meshgrid(ii, jj, indexing='ij')  # (nx, ny)
    K = (I - 1) * ny + (J - 1)  # flat indices (nx, ny)

    L = torch.zeros(n, n, dtype=mu1.dtype, device=mu1.device)

    k_flat = K.reshape(-1)
    i_flat = I.reshape(-1)
    j_flat = J.reshape(-1)

    # Center coefficient
    center_val = -2 * D / dx**2 - 2 * D / dy**2
    L[k_flat, k_flat] = center_val

    # (i-1, j): exists when i >= 2
    mask = i_flat >= 2
    src = k_flat[mask]
    dst = ((i_flat[mask] - 2) * ny + (j_flat[mask] - 1))
    vals = mu1[i_flat[mask] - 1, j_flat[mask]] / (2*dx) + D / dx**2
    L[src, dst] = vals

    # (i+1, j): exists when i <= Mx-2
    mask = i_flat <= Mx - 2
    src = k_flat[mask]
    dst = (i_flat[mask]) * ny + (j_flat[mask] - 1)
    vals = -mu1[i_flat[mask] + 1, j_flat[mask]] / (2*dx) + D / dx**2
    L[src, dst] = vals

    # (i, j-1): exists when j >= 2
    mask = j_flat >= 2
    src = k_flat[mask]
    dst = (i_flat[mask] - 1) * ny + (j_flat[mask] - 2)
    vals = mu2[i_flat[mask], j_flat[mask] - 1] / (2*dy) + D / dy**2
    L[src, dst] = vals

    # (i, j+1): exists when j <= My-2
    mask = j_flat <= My - 2
    src = k_flat[mask]
    dst = (i_flat[mask] - 1) * ny + j_flat[mask]
    vals = -mu2[i_flat[mask], j_flat[mask] + 1] / (2*dy) + D / dy**2
    L[src, dst] = vals

    return L


def solve_fpe_2d(mu1, mu2, D, x_grid, y_grid, t_output, rho0, n_substeps=20):
    """
    Solve 2D FPE with Crank-Nicolson.

    Args:
        mu1: (Mx+1, My+1) x-drift on full grid
        mu2: (Mx+1, My+1) y-drift on full grid
        D: scalar diffusion coefficient
        x_grid: (Mx+1,) x coordinates
        y_grid: (My+1,) y coordinates
        t_output: (K,) output times
        rho0: (Mx+1, My+1) initial density (zero on boundary)
        n_substeps: number of uniform time steps

    Returns:
        rho_out: (K, Mx+1, My+1) density at each output time
    """
    Mx = x_grid.shape[0] - 1
    My = y_grid.shape[0] - 1
    nx, ny = Mx - 1, My - 1
    n_int = nx * ny
    dx = (x_grid[-1] - x_grid[0]).item() / Mx
    dy = (y_grid[-1] - y_grid[0]).item() / My

    L = _build_L_2d_vectorized(mu1, mu2, D, dx, dy, Mx, My)
    I = torch.eye(n_int, dtype=L.dtype, device=L.device)

    T_max = t_output.max().item()
    dt = T_max / n_substeps
    r = dt / 2.0

    S = torch.linalg.solve(I - r * L, I + r * L)

    # Extract interior of rho0
    rho_int = rho0[1:Mx, 1:My].reshape(-1).clone()

    all_rho_int = [rho_int]
    for _ in range(n_substeps):
        rho_int = S @ rho_int
        rho_int = torch.clamp(rho_int, min=0.0)
        all_rho_int.append(rho_int)

    all_rho_int = torch.stack(all_rho_int, dim=0)  # (n_substeps+1, n_int)

    # Interpolate to output times
    frac_idx = t_output / dt
    idx_lo = frac_idx.long().clamp(0, n_substeps - 1)
    idx_hi = (idx_lo + 1).clamp(0, n_substeps)
    alpha = (frac_idx - idx_lo.float()).unsqueeze(-1)

    rho_int_out = (1 - alpha) * all_rho_int[idx_lo] + alpha * all_rho_int[idx_hi]

    # Reconstruct full grid with BCs
    K = t_output.shape[0]
    rho_out = torch.zeros(K, Mx + 1, My + 1, dtype=rho0.dtype, device=rho0.device)
    rho_out[:, 1:Mx, 1:My] = rho_int_out.reshape(K, nx, ny)

    return rho_out


def build_drift_2d_poly(theta, X, Y):
    """
    2D polynomial drift: μ = (θ₁x + θ₂x³, θ₃y + θ₄y³)

    True drift for double well: θ* = (1, -1, 1, -1)
    Potential: V(x,y) = -(x²+y²)/2 + (x⁴+y⁴)/4
    μ = -∇V = (x-x³, y-y³)

    theta: (4,)
    X, Y: (Mx+1, My+1) meshgrid
    Returns mu1, mu2: (Mx+1, My+1)
    """
    mu1 = theta[0] * X + theta[1] * X**3
    mu2 = theta[2] * Y + theta[3] * Y**3
    return mu1, mu2
