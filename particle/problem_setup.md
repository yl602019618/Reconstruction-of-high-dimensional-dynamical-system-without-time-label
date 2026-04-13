# Drift Recovery from Unlabeled Particle Snapshots

Recovering unknown drift (and diffusion) in Fokker–Planck equations from **time-unlabeled particle observations** via PDE-constrained optimization.

---

## 1. Problem Setup

### 1.1 Stochastic Dynamics

Consider the 1D SDE:
\[
dX_t = \mu(X_t)\,dt + \sigma(X_t)\,dW_t,
\]
with known diffusion (we take $\sigma(x)\equiv 1$), hence
\[
D = \tfrac12.
\]

The probability density $\rho(x,t)$ satisfies the Fokker–Planck equation:
\[
\partial_t \rho(x,t)
=
-\partial_x\!\big(\mu(x)\rho(x,t)\big)
+\frac12\,\partial_{xx}\rho(x,t),
\quad (x,t)\in \Omega\times(0,T],
\]
with

- initial condition:
\[
\rho(x,0)=\rho_0(x),
\]

- absorbing boundary condition:
\[
\rho|_{\partial\Omega}=0.
\]

---

### 1.2 Observation Model (Particle-based, Time-unlabeled)

We do **not** observe density snapshots directly.

Instead, each observation consists of a **set of particles sampled at a single (unknown) time**:

- Observation times:
\[
t_i \overset{\text{i.i.d.}}{\sim} \nu_t = \text{Unif}(0,T),
\quad i=1,\dots,N,
\]

- Particle sampling:
\[
X_{i,1},\dots,X_{i,m}
\overset{\text{i.i.d.}}{\sim}
\rho(\cdot,t_i),
\]

- Observed data:
\[
Y_i = \{X_{i,1},\dots,X_{i,m}\},
\quad i=1,\dots,N,
\]

but the time labels \(t_i\) are **unknown**.

---

### 1.3 Key Structure

Each observation is a **group of particles sharing the same latent time**.

This induces a hierarchical model:

\[
t_i \sim \nu_t,
\quad
Y_i \mid t_i \sim \rho(\cdot,t_i)^{\otimes m}.
\]

This grouping is critical — it carries information about the time evolution.

---

## 2. Forward Model

Given drift $\mu$, define:

1. Solve FPE:
\[
t \mapsto \rho_\mu(\cdot,t)
\]

2. Generate particle groups:
\[
Y \mid t \sim \rho_\mu(\cdot,t)^{\otimes m}
\]

3. Marginalize over time:
\[
\mathbb P_\mu(Y)
=
\int_0^T
\prod_{r=1}^m \rho_\mu(X_r,t)\,\nu_t(dt)
\]

---

## 3. Inverse Problem

Given observations:
\[
\mathcal D = \{Y_i\}_{i=1}^N,
\]

recover the unknown drift:
\[
\mu(x).
\]

---

## 4. Method: Marginal Likelihood (Route B)

### 4.1 Core Idea

Treat observation times \(t_i\) as latent variables and **integrate them out**.

For each observation group:
\[
Y_i = \{X_{i,1},\dots,X_{i,m}\},
\]
the likelihood is:
\[
\mathbb P_\mu(Y_i)
=
\int_0^T
\prod_{r=1}^m \rho_\mu(X_{i,r},t)\,\nu_t(dt).
\]

---

### 4.2 Objective Function

We minimize the negative log-likelihood:
\[
\mathcal L(\mu)
=
-\sum_{i=1}^N
\log
\left(
\int_0^T
\prod_{r=1}^m \rho_\mu(X_{i,r},t)\,\nu_t(dt)
\right).
\]

---

### 4.3 Discrete Approximation

Let $\{t_n\}_{n=1}^{N_t}$ be a time grid, with weights $w_n$.

Approximate:
\[
\int_0^T
\prod_{r=1}^m \rho_\mu(X_{i,r},t)\,\nu_t(dt)
\approx
\sum_{n=1}^{N_t}
w_n \prod_{r=1}^m \rho_\mu(X_{i,r},t_n).
\]

---

### 4.4 Stable Implementation (log-sum-exp)

To avoid numerical underflow:

\[
\log \sum_n w_n \prod_r \rho
=
\operatorname{LSE}_n
\left(
\log w_n + \sum_r \log \rho_\mu(X_{i,r},t_n)
\right).
\]

Final loss:
\[
\mathcal L(\mu)
=
-\sum_{i=1}^N
\operatorname{LSE}_n
\left(
\log w_n + \sum_{r=1}^m \log \rho_\mu(X_{i,r},t_n)
\right).
\]

---

## 5. Numerical Pipeline

### Step 1: Forward Solve

- Use Crank–Nicolson scheme
- Compute $\rho_\mu(x,t_n)$ for all time grid points

### Step 2: Interpolation

- Evaluate $\rho_\mu(X_{i,r},t_n)$ via interpolation

### Step 3: Loss Computation

- Compute per-sample log-likelihood using log-sum-exp

### Step 4: Optimization

- Backpropagate through solver
- Update $\mu$ parameters via gradient descent

---

## 6. Parameterization

Same as density-based version:

- Polynomial:
\[
\mu(x) = \theta_1 x + \theta_2 x^3
\]

- Hybrid:
\[
\mu(x) = \text{poly}(x) + \text{MLP}(x)
\]

- Joint inversion:
\[
(\mu, D) \text{ with two-phase training}
\]

---

## 7. Important Properties

### 7.1 Effect of Particle Count \(m\)

- \(m=1\):
  only recovers time-averaged density
  → severely ill-posed

- \(m \gg 1\):
  strong temporal signal
  → identifiable

---

### 7.2 Drift–Diffusion Tradeoff

As in density-based inversion:

- $\mu$ and $D$ can compensate each other
- requires:
  - larger $N$
  - larger $m$
  - two-phase training

---

### 7.3 Comparison to Density Observation

| Setting | Information |
|--------|------------|
| Density snapshot | full $\rho(x,t)$ |
| Particle snapshot | empirical measure |
| Particle (m small) | noisy / weak signal |
| Particle (m large) | approximates density |

---

## 8. Extensions

### 8.1 EM Interpretation (optional)

Define soft time assignment:
\[
\gamma_{in}
\propto
w_n \prod_r \rho_\mu(X_{i,r},t_n),
\]

then alternate:

- E-step: compute $\gamma_{in}$
- M-step: weighted likelihood optimization

---

### 8.2 Connection to Original Problem

Original problem:
\[
\mu \to \rho(\cdot,t) \to \text{distribution over functions}
\]

New problem:
\[
\mu \to \rho(\cdot,t)
\to \text{distribution over empirical measures}
\]

---

## 9. Summary

We extend time-unlabeled Fokker–Planck inversion from **density observations** to **particle observations**.

Key idea:

> Treat observation time as a latent variable and integrate it out via marginal likelihood.

This leads to a principled, differentiable, PDE-constrained optimization framework that works directly on particle data.
