#!/usr/bin/env python3
"""
H4: Synchronization Critical Exponent

Measures the critical exponent at the sync/unsync transition.
"""

import numpy as np
from scipy.integrate import odeint
from scipy import stats

# Lorenz parameters
SIGMA = 10.0
RHO = 28.0
BETA = 8.0 / 3.0
DT = 0.005


def lorenz(state, t):
    x, y, z = state
    return [SIGMA * (y - x), x * (RHO - z) - y, x * y - BETA * z]


def generate_lorenz(n_samples, warmup=1000):
    initial = [1 + np.random.randn()*0.1, 1 + np.random.randn()*0.1,
               1 + np.random.randn()*0.1]
    t = np.arange(0, (warmup + n_samples) * DT, DT)
    traj = odeint(lorenz, initial, t)
    return traj[warmup:warmup+n_samples]


def pecora_carroll_sync(x_received, dt, y0, z0):
    n = len(x_received)
    y_recv = np.zeros(n)
    z_recv = np.zeros(n)
    y_recv[0] = y0
    z_recv[0] = z0

    for i in range(1, n):
        x = x_received[i-1]
        y = y_recv[i-1]
        z = z_recv[i-1]
        dy = x * (RHO - z) - y
        dz = x * y - BETA * z
        y_recv[i] = y + dy * dt
        z_recv[i] = z + dz * dt

    return y_recv, z_recv


def get_correlation(g, n_samples=200, n_trials=10):
    """Get average sync correlation at coupling g."""
    corrs = []
    for _ in range(n_trials):
        traj = generate_lorenz(n_samples)
        x_true, y_true = traj[:, 0], traj[:, 1]
        noise = np.random.randn(n_samples) * np.std(x_true)
        x_drive = g * x_true + (1 - g) * noise
        y_sync, _ = pecora_carroll_sync(x_drive, DT, traj[0, 1], traj[0, 2])
        c = np.corrcoef(y_true[50:], y_sync[50:])[0, 1]
        if not np.isnan(c):
            corrs.append(c)
    return np.mean(corrs) if corrs else 0


def test_critical_exponent():
    print("=" * 60)
    print("H4: SYNCHRONIZATION CRITICAL EXPONENT")
    print("=" * 60)

    n_samples = 200
    n_trials = 50

    # Phase 1: Find g_c via binary search
    print("\nPhase 1: Finding critical point g_c...")

    g_c_estimates = []
    for run in range(5):
        g_low, g_high = 0.5, 1.0
        for _ in range(12):
            g_mid = (g_low + g_high) / 2
            if get_correlation(g_mid) < 0.5:
                g_low = g_mid
            else:
                g_high = g_mid
        g_c = (g_low + g_high) / 2
        g_c_estimates.append(g_c)
        print(f"  Run {run+1}: g_c = {g_c:.5f}")

    g_c_mean = np.mean(g_c_estimates)
    g_c_std = np.std(g_c_estimates)
    print(f"\n  g_c = {g_c_mean:.5f} ± {g_c_std:.5f}")

    # Phase 2: Dense sampling above g_c
    print("\nPhase 2: Measuring exponent...")

    delta_g_values = np.unique(np.concatenate([
        np.linspace(0.005, 0.03, 6),
        np.linspace(0.04, 0.1, 7),
        np.linspace(0.12, 0.25, 5),
    ]))

    data = []
    for delta_g in delta_g_values:
        g = g_c_mean + delta_g
        if g > 1.0:
            continue

        corrs = []
        for _ in range(n_trials):
            traj = generate_lorenz(n_samples)
            x_true, y_true = traj[:, 0], traj[:, 1]
            noise = np.random.randn(n_samples) * np.std(x_true)
            x_drive = g * x_true + (1 - g) * noise
            y_sync, _ = pecora_carroll_sync(x_drive, DT, traj[0, 1], traj[0, 2])
            c = np.corrcoef(y_true[50:], y_sync[50:])[0, 1]
            if not np.isnan(c):
                corrs.append(c)

        mean_corr = np.mean(corrs)
        std_corr = np.std(corrs) / np.sqrt(len(corrs))
        data.append({'delta_g': delta_g, 'corr': mean_corr, 'err': std_corr})
        print(f"  Δg={delta_g:.4f}: r={mean_corr:.4f} ± {std_corr:.4f}")

    # Phase 3: Power law fit
    print("\nPhase 3: Fitting power law...")

    delta_gs = np.array([d['delta_g'] for d in data])
    corrs = np.array([d['corr'] for d in data])

    valid = corrs > 0.1
    log_x = np.log(delta_gs[valid])
    log_y = np.log(corrs[valid])

    slope, intercept = np.polyfit(log_x, log_y, 1)
    beta = slope

    # Bootstrap
    n_bootstrap = 1000
    beta_samples = []
    for _ in range(n_bootstrap):
        idx = np.random.choice(len(log_x), size=len(log_x), replace=True)
        s, _ = np.polyfit(log_x[idx], log_y[idx], 1)
        beta_samples.append(s)

    beta_std = np.std(beta_samples)
    beta_ci = (np.percentile(beta_samples, 2.5), np.percentile(beta_samples, 97.5))

    print(f"\n  β = {beta:.4f} ± {beta_std:.4f}")
    print(f"  95% CI: [{beta_ci[0]:.4f}, {beta_ci[1]:.4f}]")

    # Compare to universality classes
    print("\nUniversality class comparison:")
    classes = {
        'Mean-field': 0.5,
        '2D Ising': 0.125,
        '3D Ising': 0.326,
        '2D Percolation': 0.139,
        'Directed percolation': 0.276,
    }

    for name, beta_th in sorted(classes.items(), key=lambda x: abs(x[1] - beta)):
        delta = abs(beta - beta_th)
        match = "***" if delta < 2 * beta_std else ""
        print(f"  {name:25s} β={beta_th:.3f}  Δ={delta:.3f} {match}")

    return beta, beta_std, g_c_mean


if __name__ == '__main__':
    test_critical_exponent()
