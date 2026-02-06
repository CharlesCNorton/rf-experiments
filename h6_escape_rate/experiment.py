#!/usr/bin/env python3
"""
H6: Transient Chaos Escape Rate

Tests the distribution of escape times from a chaotic saddle.
"""

import numpy as np
from scipy.integrate import odeint
from scipy import stats

# Lorenz parameters (reduced ρ for transient chaos)
SIGMA = 10.0
BETA = 8.0 / 3.0
DT = 0.005


def test_escape_rate():
    print("=" * 60)
    print("H6: TRANSIENT CHAOS ESCAPE RATE")
    print("=" * 60)

    rho_test = 14.0
    n_trials = 500

    print(f"\nUsing ρ = {rho_test} (transient chaos regime)")
    print(f"Collecting {n_trials} escape times...")

    def lorenz_transient(state, t):
        x, y, z = state
        return [SIGMA * (y - x), x * (rho_test - z) - y, x * y - BETA * z]

    escape_times = []

    for trial in range(n_trials):
        initial = [
            5 + 10 * np.random.randn(),
            5 + 10 * np.random.randn(),
            rho_test + 10 * np.abs(np.random.randn())
        ]

        t = np.arange(0, 200, DT)
        traj = odeint(lorenz_transient, initial, t)
        x = traj[:, 0]

        # Escape detection via rolling variance
        window = 200
        activity = np.array([np.std(x[max(0,i-window):i+1]) for i in range(len(x))])

        threshold = 0.5
        escaped = np.where(activity[window:] < threshold)[0]

        if len(escaped) > 0:
            escape_time = (escaped[0] + window) * DT
            if escape_time > 1.0:
                escape_times.append(escape_time)

        if (trial + 1) % 100 == 0:
            print(f"  {trial+1}/{n_trials} trials, {len(escape_times)} escapes")

    escape_times = np.array(escape_times)

    print(f"\nCollected {len(escape_times)} escape times")
    print(f"Range: [{np.min(escape_times):.2f}, {np.max(escape_times):.2f}]")
    print(f"Mean: {np.mean(escape_times):.2f}, Median: {np.median(escape_times):.2f}")

    # Fit distributions
    print("\nFitting distributions...")
    results = {}

    # Exponential
    loc, scale = stats.expon.fit(escape_times)
    ks_stat, p_value = stats.kstest(escape_times, 'expon', args=(loc, scale))
    aic = 2 * 1 - 2 * np.sum(stats.expon.logpdf(escape_times, loc, scale))
    results['Exponential'] = {'aic': aic, 'p': p_value, 'params': f'λ={1/scale:.3f}'}
    print(f"  Exponential: λ={1/scale:.4f}, p={p_value:.4f}")

    # Weibull
    c, loc, scale = stats.weibull_min.fit(escape_times, floc=0)
    ks_stat, p_value = stats.kstest(escape_times, 'weibull_min', args=(c, loc, scale))
    aic = 2 * 2 - 2 * np.sum(stats.weibull_min.logpdf(escape_times, c, loc, scale))
    results['Weibull'] = {'aic': aic, 'p': p_value, 'params': f'k={c:.3f}, λ={scale:.3f}'}
    print(f"  Weibull: k={c:.3f}, λ={scale:.3f}, p={p_value:.4f}")

    # Gamma
    a, loc, scale = stats.gamma.fit(escape_times, floc=0)
    ks_stat, p_value = stats.kstest(escape_times, 'gamma', args=(a, loc, scale))
    aic = 2 * 2 - 2 * np.sum(stats.gamma.logpdf(escape_times, a, loc, scale))
    results['Gamma'] = {'aic': aic, 'p': p_value, 'params': f'k={a:.3f}, θ={scale:.3f}'}
    print(f"  Gamma: k={a:.3f}, θ={scale:.3f}, p={p_value:.4f}")

    # Log-normal
    s, loc, scale = stats.lognorm.fit(escape_times, floc=0)
    ks_stat, p_value = stats.kstest(escape_times, 'lognorm', args=(s, loc, scale))
    aic = 2 * 2 - 2 * np.sum(stats.lognorm.logpdf(escape_times, s, loc, scale))
    results['Log-normal'] = {'aic': aic, 'p': p_value, 'params': f'σ={s:.3f}, μ={np.log(scale):.3f}'}
    print(f"  Log-normal: σ={s:.3f}, μ={np.log(scale):.3f}, p={p_value:.4f}")

    # Rank by AIC
    print("\nRanking by AIC (lower is better):")
    ranked = sorted(results.items(), key=lambda x: x[1]['aic'])

    for name, data in ranked:
        print(f"  {name:15s} AIC={data['aic']:8.1f}  p={data['p']:.4f}  {data['params']}")

    best = ranked[0][0]
    print(f"\nBest fit: {best}")

    if best == 'Weibull':
        c, _, scale = stats.weibull_min.fit(escape_times, floc=0)
        if c > 1:
            print(f"\nk = {c:.2f} > 1: INCREASING hazard rate (aging dynamics)")
        elif c < 1:
            print(f"\nk = {c:.2f} < 1: DECREASING hazard rate")
        else:
            print(f"\nk ≈ 1: Constant hazard (exponential)")

    return escape_times, results


if __name__ == '__main__':
    test_escape_rate()
