# RF Chaos Experiments

Frontier physics experiments using chaotic synchronization over 433 MHz RF.

## Equipment

### Transmitter
- **Flipper Zero** "Archera"
  - Firmware: 1.4.3
  - Transceiver: CC1101
  - Frequencies: 315, 433.92, 915 MHz (US region)
  - Modulation: OOK, 2FSK
  - Timing resolution: ~200 µs minimum

### Receiver
- **RTL-SDR** (RTL2838 + R820T tuner)
  - Frequency range: 24 MHz – 1.7 GHz
  - Sample rate: 1 MS/s
  - Gain: 40 dB

### Compute
- **Raspberry Pi 5** (8GB)
  - OS: Raspberry Pi OS Lite (Debian Trixie)
  - Profile: Eco mode (1.5 GHz) for USB power budget
  - Both Flipper and SDR connected via USB

### Physical Setup
- Flipper and SDR ~10 cm apart on same Pi
- Indoor environment, 433.92 MHz primary band
- SNR typically 40-70

## Experiments

| # | Experiment | Result |
|---|------------|--------|
| H1 | [Stochastic Resonance Phase Transition](h1_stochastic_resonance/) | Sharp transition confirmed |
| H3 | [Lorenz Reservoir Computing](h3_reservoir_computing/) | 100% classification accuracy |
| H4 | [Synchronization Transition Analysis](h4_critical_exponent/) | Smooth crossover, not critical |
| H5 | [Loschmidt Echo = Lyapunov](h5_loschmidt_lyapunov/) | Growth rate matches λ₁ within 8% |
| H6 | [Transient Chaos Escape Rate](h6_escape_rate/) | Weibull distribution (k = 3.76) |
| H8 | [Generalized Synchronization](h8_generalized_sync/) | Functional relationship confirmed |

## Theory Background

All experiments use the **Lorenz system**:

```
dx/dt = σ(y - x)
dy/dt = x(ρ - z) - y
dz/dt = xy - βz
```

With standard parameters: σ = 10, ρ = 28, β = 8/3.

**Pecora-Carroll synchronization**: Transmit x(t), receiver reconstructs y(t), z(t) by integrating the driven subsystem.

**RF encoding**: Lorenz x values mapped to pulse periods (500-2000 µs). Transmitted via Flipper, captured via SDR, periods extracted from envelope.

## Usage

Each experiment is self-contained. Run from the Pi:

```bash
cd /path/to/experiment
python3 experiment.py
```

## References

- Pecora & Carroll, "Synchronization in Chaotic Systems", PRL 1990
- Cuomo & Oppenheim, "Circuit Implementation of Synchronized Chaos", PRL 1993

## License

MIT
