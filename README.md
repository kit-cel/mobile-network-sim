# Mobile Network Simulator

Step-based Python simulator for mobile radio geometry, mobility, propagation, and RRC handover. It provides radio-only and combined radio/RRC runs.

[Documentation](docs/README.md) · [Examples](examples/)

## Install

```bash
python -m pip install -e .
```

Add plotting or GUI dependencies with `python -m pip install -e ".[plot]"` or `python -m pip install -e ".[gui]"`.

## Run

```bash
mobile-network-sim radio --config examples/radio.yaml --output results/radio
mobile-network-sim radio-rrc --config examples/radio_rrc.yaml --output results/summary.json
```

## License

MIT. See [LICENSE](LICENSE).
