# H3: Lorenz Reservoir Computing

## Hypothesis

The Lorenz system, driven by input signals, can perform nonlinear classification without explicit training of the dynamics — only a linear readout layer is needed.

## Prediction

Feeding different input patterns through the Lorenz dynamics, then applying linear regression on the output state, achieves >80% classification accuracy.

## Method

1. Generate base Lorenz trajectory
2. Create two input classes:
   - Class A: Increasing modulation added to x
   - Class B: Decreasing modulation added to x
3. Drive Pecora-Carroll sync with modulated x
4. Extract features from synchronized y, z (mean, std, trend, correlation)
5. Train linear classifier (least squares regression)
6. Test on held-out samples

## Result

| Metric | Value |
|--------|-------|
| Training accuracy | **100%** |
| Test accuracy | **100%** |
| Features used | 7 (statistics of y, z output) |
| Samples | 40 (30 train, 10 test) |

**Hypothesis strongly supported.** The Lorenz dynamics serve as an effective computational reservoir.

## Interpretation

The chaotic attractor's nonlinear dynamics transform input patterns into separable representations. The reservoir computing paradigm works: complex dynamics + simple readout = powerful computation.

This suggests physical chaotic systems (electronic, optical, RF) could serve as analog computers for pattern recognition tasks.

## Run

```bash
python3 experiment.py
```
