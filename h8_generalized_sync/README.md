# H8: Generalized Synchronization

## Hypothesis

A receiver with DIFFERENT Lorenz parameters (σ', ρ', β' ≠ σ, ρ, β) still synchronizes to the transmitter, but via a nonlinear functional relationship:

```
y_rx = F(y_tx)
```

rather than identical synchronization y_rx = y_tx.

## Prediction

Even with parameter mismatch, there exists a deterministic function F such that the receiver output is predictable from the transmitter state. The conditional variance (variance of y_rx given y_tx) should be much smaller than the total variance.

## Method

1. Generate transmitter trajectory with standard parameters
2. Create receiver with mismatched parameters (σ×1.2, ρ×0.9, β×1.1)
3. Drive receiver with transmitted x(t)
4. Measure direct correlation (should be reduced from 1.0)
5. Bin transmitter y values, measure variance of receiver y in each bin
6. Compute ratio: conditional_variance / total_variance
7. Low ratio indicates functional relationship exists

## Result

| Metric | Value |
|--------|-------|
| Parameter mismatch | σ×1.2, ρ×0.9, β×1.1 |
| Direct correlation | **0.951** |
| Cond. std / Total std | **0.193** |

**Functional relationship confirmed.** Despite 20% parameter mismatch, the receiver output is nearly a deterministic function of the transmitter state.

## Interpretation

Generalized synchronization is more robust than identical synchronization. The receiver doesn't need exact parameter matching — it will lock onto a distorted but deterministic copy of the transmitter dynamics.

This has implications for:
- **Secure communications**: Parameter mismatch doesn't break sync
- **Biological systems**: Neural networks sync despite heterogeneity
- **Coupled oscillators**: Universality of synchronization phenomena

The low conditional variance ratio (0.19) means only 19% of the output variance is "unexplained" by the input — the relationship is 81% deterministic.

## Run

```bash
python3 experiment.py
```
