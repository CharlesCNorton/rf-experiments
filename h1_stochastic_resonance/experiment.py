#!/usr/bin/env python3
"""
H1: Stochastic Resonance Phase Transition

Tests whether chaos synchronization shows a sharp phase transition
at optimal noise level.
"""

import numpy as np
from scipy.integrate import odeint

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


def test_stochastic_resonance():
    print("=" * 60)
    print("H1: STOCHASTIC RESONANCE PHASE TRANSITION")
    print("=" * 60)

    n_samples = 200
    traj = generate_lorenz(n_samples)
    x_true, y_true, z_true = traj[:, 0], traj[:, 1], traj[:, 2]

    # Fine noise sampling
    noise_levels = np.concatenate([
        np.linspace(0, 1, 11),
        np.linspace(1.5, 5, 8),
        np.linspace(6, 15, 6)
    ])

    print("\nTesting WHITE noise...")
    results_white = []

    for sigma in noise_levels:
        corrs = []
        for _ in range(3):
            noise = np.random.randn(n_samples) * sigma
            x_noisy = x_true + noise
            y_sync, z_sync = pecora_carroll_sync(x_noisy, DT, y_true[0], z_true[0])
            skip = 30
            corr = np.corrcoef(y_true[skip:], y_sync[skip:])[0, 1]
            if not np.isnan(corr):
                corrs.append(corr)
        avg_corr = np.mean(corrs) if corrs else 0
        results_white.append({'sigma': sigma, 'corr': avg_corr})
        print(f"  σ={sigma:5.1f}: corr={avg_corr:.4f}")

    print("\nTesting COLORED noise (AR(1), ρ=0.8)...")
    results_colored = []

    for sigma in noise_levels:
        corrs = []
        for _ in range(3):
            colored = np.zeros(n_samples)
            ar_coef = 0.8
            for i in range(1, n_samples):
                colored[i] = ar_coef * colored[i-1] + np.random.randn() * np.sqrt(1 - ar_coef**2)
            colored *= sigma

            x_noisy = x_true + colored
            y_sync, z_sync = pecora_carroll_sync(x_noisy, DT, y_true[0], z_true[0])
            skip = 30
            corr = np.corrcoef(y_true[skip:], y_sync[skip:])[0, 1]
            if not np.isnan(corr):
                corrs.append(corr)
        avg_corr = np.mean(corrs) if corrs else 0
        results_colored.append({'sigma': sigma, 'corr': avg_corr})

    # Analysis
    sigmas_w = np.array([r['sigma'] for r in results_white])
    corrs_w = np.array([r['corr'] for r in results_white])
    sigmas_c = np.array([r['sigma'] for r in results_colored])
    corrs_c = np.array([r['corr'] for r in results_colored])

    peak_idx_w = np.argmax(corrs_w)
    peak_idx_c = np.argmax(corrs_c)

    print(f"\nRESULTS:")
    print(f"  WHITE noise peak: σ={sigmas_w[peak_idx_w]:.2f}, corr={corrs_w[peak_idx_w]:.4f}")
    print(f"  COLORED noise peak: σ={sigmas_c[peak_idx_c]:.2f}, corr={corrs_c[peak_idx_c]:.4f}")

    # Measure sharpness
    if 1 < peak_idx_w < len(corrs_w) - 1:
        slopes = np.diff(corrs_w)
        max_slope_change = np.max(np.abs(np.diff(slopes)))
        print(f"\n  Max slope change (white): {max_slope_change:.4f}")

        if max_slope_change > 0.05:
            print("  -> SHARP TRANSITION detected")
        else:
            print("  -> Smooth transition")

    has_sr_white = (peak_idx_w > 0) and (corrs_w[peak_idx_w] > corrs_w[0])
    has_sr_colored = (peak_idx_c > 0) and (corrs_c[peak_idx_c] > corrs_c[0])

    print(f"\n  SR detected (white):   {'YES' if has_sr_white else 'NO'}")
    print(f"  SR detected (colored): {'YES' if has_sr_colored else 'NO'}")

    return results_white, results_colored


if __name__ == '__main__':
    test_stochastic_resonance()
