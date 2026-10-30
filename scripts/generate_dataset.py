import argparse
from pathlib import Path

import yaml

from aquasentinel.sim.generate import generate_dataset


def main():
    parser = argparse.ArgumentParser(
        description="Generate AquaSentinel AI water-network simulation data."
    )
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Path to YAML configuration."
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Generate a smaller dataset for a fast smoke test."
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    print(f"Loading config: {config_path}")
    print(f"Quick mode: {args.quick}")
    generate_dataset(cfg, quick=args.quick)
    print("Dataset generation finished.")


if __name__ == "__main__":
    main()
