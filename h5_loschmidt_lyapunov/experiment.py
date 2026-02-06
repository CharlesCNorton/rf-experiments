#!/usr/bin/env python3
"""
H5: Loschmidt Echo = Lyapunov Exponent

Tests whether trajectory separation growth rate equals the Lyapunov exponent.
"""

import numpy as np
from scipy.integrate import odeint
from scipy import stats

# Lorenz parameters
SIGMA = 10.0
RHO = 28.0
BETA = 8.0 / 3.0
DT = 0.01  # Finer timestep for accuracy


def lorenz(state, t):
    x, y, z = state
    return [SIGMA * (y - x), x * (RHO - z) - y, x * y - BETA * z]


def test_loschmidt_lyapunov():
    print("=" * 60)
    print("H5: LOSCHMIDT ECHO = LYAPUNOV EXPONENT")
    print("=" * 60)

    lambda_1_theory = 0.906
    print(f"\nTheoretical λ₁ = {lambda_1_theory}")

    n_samples = 500

    # Reference trajectory (on attractor)
    initial_ref = [1.0, 1.0, 1.0]
    t = np.arange(0, (1000 + n_samples) * DT, DT)
    traj_full = odeint(lorenz, initial_ref, t)
    traj_ref = traj_full[1000:1000+n_samples]

    # Characteristic scale
    sigma_avg = np.mean([np.std(traj_ref[:, i]) for i in range(3)])
    print(f"Attractor scale: σ ≈ {sigma_avg:.2f}")

    perturbations = [1e-10, 1e-8, 1e-6, 1e-4]

    print("\nMeasuring separation growth rate:")
    results = []

    for eps in perturbations:
        print(f"\n  ε = {eps}:")

        # Perturbed trajectory
        initial_pert = initial_ref.copy()
        initial_pert[0] += eps
        traj_pert_full = odeint(lorenz, initial_pert, t)
        traj_pert = traj_pert_full[1000:1000+n_samples]

        # Separation over time
        separation = np.linalg.norm(traj_ref - traj_pert, axis=1)
        times = np.arange(n_samples) * DT
        log_sep = np.log(separation + 1e-20)

        # Find exponential regime
        valid = (separation > eps * 10) & (separation < sigma_avg * 0.5) & (times > 0.5)

        if np.sum(valid) >= 20:
            t_valid = times[valid]
            log_sep_valid = log_sep[valid]

            slope, intercept, r_value, _, std_err = stats.linregress(t_valid, log_sep_valid)

            print(f"    Growth rate: {slope:.4f} ± {std_err:.4f}")
            print(f"    Expected:    {lambda_1_theory:.4f}")
            print(f"    Ratio:       {slope/lambda_1_theory:.3f}")
            print(f"    R²:          {r_value**2:.4f}")
            print(f"    Valid pts:   {np.sum(valid)}")

            if abs(slope - lambda_1_theory) / lambda_1_theory < 0.15:
                print(f"    -> MATCH within 15%")

            results.append({
                'eps': eps,
                'rate': slope,
                'ratio': slope/lambda_1_theory,
                'r2': r_value**2
            })
        else:
            print(f"    Could not find exponential regime")
            print(f"    Sep range: [{separation.min():.2e}, {separation.max():.2e}]")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    best = None
    best_error = float('inf')
    for r in results:
        error = abs(r['ratio'] - 1.0)
        if error < best_error:
            best_error = error
            best = r

    if best:
        print(f"\nBest result: ε = {best['eps']}")
        print(f"  Growth rate = {best['rate']:.4f}")
        print(f"  λ₁ theory   = {lambda_1_theory:.4f}")
        print(f"  Error       = {abs(best['rate'] - lambda_1_theory)/lambda_1_theory*100:.1f}%")

    return results


if __name__ == '__main__':
    test_loschmidt_lyapunov()
