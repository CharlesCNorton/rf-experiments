# H6: Transient Chaos Escape Rate

## Hypothesis

When the Lorenz system is in the transient chaos regime (below the critical ρ), trajectories escape the chaotic saddle with exponentially distributed escape times.

## Prediction

The escape time distribution follows:

```
P(t) ~ exp(-κt)
```

where κ is the escape rate.

## Method

1. Set ρ = 14.0 (below critical value ~24.7)
2. Initialize trajectories on approximate chaotic saddle
3. Detect escape via activity threshold (rolling variance drops)
4. Collect escape times from 500 trials
5. Fit to candidate distributions: Exponential, Weibull, Gamma, Log-normal
6. Use AIC and KS test to determine best fit

## Result

| Distribution | AIC | KS p-value | Status |
|-------------|-----|------------|--------|
| **Weibull** | 1863 | **0.082** | Best fit |
| Gamma | 1977 | 0.000 | Rejected |
| Log-normal | 2068 | 0.000 | Rejected |
| Exponential | 2380 | 0.000 | Rejected |

**Weibull parameters:**
- Shape k = **3.76**
- Scale λ = 5.63

**Escape times are NOT exponential.** They follow a Weibull distribution with k > 1.

## Interpretation

The finding k > 1 indicates an **increasing hazard rate** — trajectories become MORE likely to escape the longer they stay. This is "aging" dynamics:

- k < 1: Decreasing hazard (infant mortality)
- k = 1: Constant hazard (memoryless, exponential)
- k > 1: Increasing hazard (aging/wear-out)

Physical interpretation: The chaotic saddle has structure. Trajectories that haven't escaped are not uniformly distributed — they're in increasingly precarious regions of phase space. The longer they survive, the closer they are to the saddle's "edge."

This challenges the simple exponential escape model and suggests richer dynamics in transient chaos.

## Run

```bash
python3 experiment.py
```
