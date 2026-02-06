#!/usr/bin/env python3
"""
H8: Generalized Synchronization

Tests whether sync persists with parameter mismatch via functional relationship.
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


def test_generalized_sync():
    print("=" * 60)
    print("H8: GENERALIZED SYNCHRONIZATION")
    print("=" * 60)

    n_samples = 150

    # Transmitter with standard parameters
    traj_tx = generate_lorenz(n_samples)
    x_tx, y_tx, z_tx = traj_tx[:, 0], traj_tx[:, 1], traj_tx[:, 2]

    # Receiver with MISMATCHED parameters
    sigma_rx = SIGMA * 1.2
    rho_rx = RHO * 0.9
    beta_rx = BETA * 1.1

    print(f"\nParameter mismatch:")
    print(f"  TX: σ={SIGMA}, ρ={RHO}, β={BETA:.3f}")
    print(f"  RX: σ={sigma_rx}, ρ={rho_rx}, β={beta_rx:.3f}")

    # Drive receiver with transmitted x
    def lorenz_rx(state, t, x_drive_func):
        _, y, z = state
        x = x_drive_func(t)
        return [0, x * (rho_rx - z) - y, x * y - beta_rx * z]

    t = np.arange(0, n_samples * DT, DT)

    def x_interp(t_val):
        idx = int(t_val / DT)
        if idx < 0:
            return x_tx[0]
        if idx >= len(x_tx):
            return x_tx[-1]
        return x_tx[idx]

    traj_rx = odeint(lambda s, t: lorenz_rx(s, t, x_interp),
                     [0, y_tx[0], z_tx[0]], t)
    y_rx, z_rx = traj_rx[:, 1], traj_rx[:, 2]

    skip = 30

    # Direct correlation
    direct_corr = np.corrcoef(y_tx[skip:], y_rx[skip:])[0, 1]
    print(f"\nDirect y correlation: {direct_corr:.4f}")

    # Test for functional relationship
    # If y_rx = F(y_tx), then conditional variance should be low
    n_bins = 20
    bins = np.linspace(y_tx[skip:].min(), y_tx[skip:].max(), n_bins + 1)
    bin_idx = np.digitize(y_tx[skip:], bins)

    conditional_stds = []
    for b in range(1, n_bins + 1):
        mask = bin_idx == b
        if np.sum(mask) > 3:
            conditional_stds.append(np.std(y_rx[skip:][mask]))

    mean_cond_std = np.mean(conditional_stds)
    total_std = np.std(y_rx[skip:])
    variance_ratio = mean_cond_std / total_std

    print(f"\nFunctional relationship test:")
    print(f"  Total std(y_rx):       {total_std:.4f}")
    print(f"  Mean conditional std:  {mean_cond_std:.4f}")
    print(f"  Ratio (cond/total):    {variance_ratio:.4f}")

    # Interpretation
    print("\nInterpretation:")
    if variance_ratio < 0.3:
        print(f"  Ratio < 0.3: STRONG functional relationship")
        print(f"  y_rx is nearly a deterministic function of y_tx")
        determinism = (1 - variance_ratio**2) * 100
        print(f"  Relationship is ~{determinism:.0f}% deterministic")
    elif variance_ratio < 0.6:
        print(f"  Ratio < 0.6: WEAK functional relationship")
    else:
        print(f"  Ratio > 0.6: No clear functional relationship")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Parameter mismatch: 20%")
    print(f"  Direct correlation: {direct_corr:.3f}")
    print(f"  Variance ratio:     {variance_ratio:.3f}")

    if direct_corr > 0.9 and variance_ratio < 0.3:
        print("\n  -> GENERALIZED SYNC CONFIRMED")

    return direct_corr, variance_ratio


if __name__ == '__main__':
    test_generalized_sync()
