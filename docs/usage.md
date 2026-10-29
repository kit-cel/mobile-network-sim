# Usage

Requires Python 3.12 or newer. From the repository root:

```bash
python -m pip install -e .
```

Use `.[plot]` for Plotly helpers or `.[gui]` for the Qt GUI.

## Command line

```bash
mobile-network-sim radio --config examples/radio.yaml --output results/radio
mobile-network-sim radio-rrc --config examples/radio_rrc.yaml --output results/summary.json
```

`radio` writes radio samples to an HDF5 file in the output directory. `radio-rrc` writes a JSON summary with the final RRC states and serving cells; use the Python runner for every step's radio and RRC results. Omitting `--output` runs without writing those results.

## Python

```python
from mobile_network_sim.runners import RadioRRCSimulationConfig, RadioRRCSimulationRunner

config = RadioRRCSimulationConfig.from_yaml("examples/radio_rrc.yaml")
runner = RadioRRCSimulationRunner(config)
try:
    step = runner.step()
    print(step.radio.rsrp_dbm.shape, step.states, step.serving_cells)
finally:
    runner.close()
```

For radio-only episodes, use `SimulationConfig.from_yaml(...)` and `RadioSimulationRunner(config).run(...)`. To drive RRC from existing measurements without a radio simulation, see [`examples/rrc_from_measurements.py`](../examples/rrc_from_measurements.py).

Generated radio fields are cached under `XDG_CACHE_HOME/mobile-network-sim` when `XDG_CACHE_HOME` is set, otherwise under `LOCALAPPDATA/mobile-network-sim` on Windows or `~/.cache/mobile-network-sim` elsewhere.
