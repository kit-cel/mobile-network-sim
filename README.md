# Mobile Network Simulator

[![Pylint score](https://img.shields.io/endpoint?url=https://kit-cel.github.io/mobile-network-sim/pylint-badge.json)](https://github.com/kit-cel/mobile-network-sim/actions/workflows/pylint.yml)
[![Test coverage](https://img.shields.io/endpoint?url=https://kit-cel.github.io/mobile-network-sim/coverage-badge.json)](https://github.com/kit-cel/mobile-network-sim/actions/workflows/pylint.yml)
[![Docs](https://github.com/kit-cel/mobile-network-sim/actions/workflows/docs.yml/badge.svg?branch=main)](https://github.com/kit-cel/mobile-network-sim/actions/workflows/docs.yml)
[![Python](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)

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
