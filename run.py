from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from src import evaluate, train, utils  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Heart disease ML pipeline")
    parser.add_argument("--mode", choices=["train", "eval"], required=True)
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Path to YAML configuration file",
    )
    return parser.parse_args()


def resolve_config(path: str) -> Path:
    cfg_path = Path(path)
    if cfg_path.exists():
        return cfg_path
    candidate = PROJECT_ROOT / path
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"Configuration file not found: {path}")


def main() -> None:
    args = parse_args()
    config_path = resolve_config(args.config)

    if args.mode == "train":
        results = train.run(config_path)
        print("\nArtifacts saved:")
        for label, rel_path in results.get("artifact_paths", {}).items():
            print(f"  - {label}: {rel_path}")
    else:
        results = evaluate.run(config_path)
        print("\nEvaluation complete. Artifacts updated:")
        for label, rel_path in results.get("artifact_paths", {}).items():
            print(f"  - {label}: {rel_path}")


if __name__ == "__main__":
    main()
