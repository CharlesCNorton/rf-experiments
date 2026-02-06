#!/usr/bin/env python3
"""
H3: Lorenz Reservoir Computing

Tests whether Lorenz dynamics can classify patterns with linear readout.
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


def test_reservoir_computing():
    print("=" * 60)
    print("H3: LORENZ RESERVOIR COMPUTING")
    print("=" * 60)

    n_samples = 50
    n_trials = 40

    features = []
    labels = []

    print(f"\nGenerating {n_trials} samples...")

    for trial in range(n_trials):
        traj = generate_lorenz(n_samples + 20)

        # Create input pattern
        if trial % 2 == 0:
            # Class A: increasing modulation
            modulation = np.linspace(0, 2, n_samples)
            label = 0
        else:
            # Class B: decreasing modulation
            modulation = np.linspace(2, 0, n_samples)
            label = 1

        x_input = traj[20:20+n_samples, 0] + modulation

        # Run through reservoir
        y_out, z_out = pecora_carroll_sync(x_input, DT, traj[20, 1], traj[20, 2])

        # Extract features
        feat = [
            np.mean(y_out[20:]),
            np.std(y_out[20:]),
            np.mean(z_out[20:]),
            np.std(z_out[20:]),
            y_out[-1] - y_out[20],
            z_out[-1] - z_out[20],
            np.corrcoef(y_out[20:], z_out[20:])[0, 1]
        ]
        features.append(feat)
        labels.append(label)

    features = np.array(features)
    labels = np.array(labels)

    # Split train/test
    n_train = 30
    X_train, X_test = features[:n_train], features[n_train:]
    y_train, y_test = labels[:n_train], labels[n_train:]

    # Add bias term
    X_train_b = np.hstack([X_train, np.ones((n_train, 1))])
    X_test_b = np.hstack([X_test, np.ones((n_trials - n_train, 1))])

    # Least squares
    w, _, _, _ = np.linalg.lstsq(X_train_b, y_train, rcond=None)

    # Predict
    y_pred_train = (X_train_b @ w > 0.5).astype(int)
    y_pred_test = (X_test_b @ w > 0.5).astype(int)

    train_acc = np.mean(y_pred_train == y_train)
    test_acc = np.mean(y_pred_test == y_test)

    print(f"\nRESULTS:")
    print(f"  Training accuracy: {train_acc:.1%}")
    print(f"  Test accuracy:     {test_acc:.1%}")
    print(f"  Features used:     {features.shape[1]}")

    if test_acc >= 0.8:
        print(f"\n  -> HYPOTHESIS SUPPORTED (>80% accuracy)")
    else:
        print(f"\n  -> Hypothesis not supported")

    return train_acc, test_acc


if __name__ == '__main__':
    test_reservoir_computing()
