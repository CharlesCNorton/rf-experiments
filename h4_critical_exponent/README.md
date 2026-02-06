# H4: Synchronization Transition Analysis

## Hypothesis

The sync/unsync transition (varying coupling strength g) exhibits critical scaling:

```
r ~ |g - g_c|^β
```

with β belonging to a known universality class.

## Method

1. Mix true Lorenz x(t) with Gaussian noise: `x_drive = g·x + (1-g)·noise`
2. Drive Pecora-Carroll receiver with x_drive
3. Measure correlation between true y(t) and synchronized y(t)
4. Sweep coupling g from 0.5 to 1.0
5. Fit functional forms to correlation vs coupling data
6. Test for signatures of criticality

## Results

### Functional Form

| Model | R² |
|-------|-----|
| Power law | 0.96 |
| Exponential approach | 0.98 |
| **Tanh/sigmoid** | **0.9995** |

The sigmoid `r(g) = A·tanh(B·(g - g₀)) + C` provides the best fit.

### Threshold Dependence of g_c

| Threshold (r =) | g_c |
|-----------------|-----|
| 0.3 | 0.59 |
| 0.5 | 0.67 |
| 0.7 | 0.76 |
| 0.9 | 0.88 |

The crossing point shifts continuously with threshold choice.

### Variance Scaling

Variance of correlation measurements peaks at g ≈ 0.64, not at the r = 0.5 crossing (g ≈ 0.67). In critical phenomena, fluctuations peak at the critical point.

### Exponent Stability

Fitting power laws to different runs yields β ranging from 0.09 to 0.19, exceeding statistical uncertainty.

## Interpretation

The data are consistent with a **smooth crossover** rather than a phase transition:

- Correlation increases monotonically with signal fraction g
- No unique critical point exists independent of measurement threshold
- Fluctuation maximum does not coincide with order parameter threshold
- Sigmoid saturation behavior reflects bounded correlation (0 to 1)

This differs from synchronization transitions in coupled map lattices, which exhibit genuine criticality in the Multiplicative Noise or Directed Percolation universality classes. Those systems have spatial extent and a thermodynamic limit; a single driven oscillator does not.

The correlation-vs-coupling relationship is determined by signal-to-noise ratio in the drive signal, which produces smooth interpolation between desynchronized (g → 0) and synchronized (g → 1) regimes.

## Run

```bash
python3 experiment.py
```
