from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load_config
from .data import generate
from .modeling import train
from .monitoring import monitor
from .scenario import run_scenario


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("generate")
    commands.add_parser("train")
    commands.add_parser("evaluate")
    commands.add_parser("monitor")
    scenario = commands.add_parser("scenario")
    scenario.add_argument("--file", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.command == "generate":
        print({"rows": len(generate(config)), "synthetic": True})
    elif args.command == "train":
        print(json.dumps(train(config), indent=2))
    elif args.command == "evaluate":
        print((Path(config["paths"]["artifacts"]) / "metadata.json").read_text())
    elif args.command == "scenario":
        print(json.dumps(run_scenario(config, args.file), indent=2))
    else:
        print(json.dumps(monitor(config), indent=2))


if __name__ == "__main__":
    main()
