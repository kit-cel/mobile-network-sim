"""Command-line interface for standalone simulations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .radio import RadioSimulationRunner, SimulationConfig
from .runners import RadioRRCSimulationConfig, RadioRRCSimulationRunner


def _parser() -> argparse.ArgumentParser:
    """Create the command-line argument parser for the simulator."""
    parser = argparse.ArgumentParser(prog="mobile-network-sim")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("radio", "radio-rrc"):
        command = commands.add_parser(name)
        command.add_argument("--config", required=True, type=Path)
        command.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the simulator from the command line."""
    args = _parser().parse_args(argv)
    if args.command == "radio":
        config = SimulationConfig.from_yaml(args.config)
        runner = RadioSimulationRunner(config)
        runner.run(
            n_ep=1,
            n_steps=config.num_steps,
            save_results=args.output is not None,
            output_dir=args.output,
        )
        return 0

    config = RadioRRCSimulationConfig.from_yaml(args.config)
    results = RadioRRCSimulationRunner(config).run()
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        final = results[-1]
        args.output.write_text(
            json.dumps(
                {
                    "num_steps": len(results),
                    "states": final.states.tolist(),
                    "serving_cells": final.serving_cells.tolist(),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
