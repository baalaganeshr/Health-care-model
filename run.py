from __future__ import annotations

import argparse
import sys
import os
import random
import numpy as np
import torch
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from src import evaluate, train, utils  # noqa: E402
from src.report import make_tables  # noqa: E402


def set_seeds(seed: int = 42):
    """Set seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # For deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Healthcare ML + IoT pipeline")
    parser.add_argument("--mode", choices=["train", "eval", "report", "sim", "train_sadnn", "reproduce"], required=True)
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--paper_tables",
        choices=["true", "false"],
        default="false",
        help="Include paper-reported static tables in report mode",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
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
    
    # Set global seed for reproducibility
    set_seeds(args.seed)
    print(f"🌱 Global seed set to: {args.seed}")

    if args.mode == "train":
        results = train.run(config_path)
        print("\nArtifacts saved:")
        for label, rel_path in results.get("artifact_paths", {}).items():
            print(f"  - {label}: {rel_path}")
    elif args.mode == "eval":
        results = evaluate.run(config_path)
        print("\nEvaluation complete. Artifacts updated:")
        for label, rel_path in results.get("artifact_paths", {}).items():
            print(f"  - {label}: {rel_path}")
    elif args.mode == "report":
        paper_tables = args.paper_tables.lower() == "true" or os.getenv("PAPER_TABLES") == "1"
        make_tables.main(paper_tables=paper_tables)
    elif args.mode == "sim":
        from src import simulation
        results = simulation.run(config_path, seed=args.seed)
        print(f"\n🌐 IoT Simulation completed for nodes: {results.get('nodes_tested', [])}")
        print("Artifacts saved:")
        for label, rel_path in results.get("artifact_paths", {}).items():
            print(f"  - {label}: {rel_path}")
    elif args.mode == "train_sadnn":
        from src import train_sadnn
        results = train_sadnn.run(config_path, seed=args.seed)
        print(f"\n🧠 SA-DNN training completed.")
        print("Artifacts saved:")
        for label, rel_path in results.get("artifact_paths", {}).items():
            print(f"  - {label}: {rel_path}")
    elif args.mode == "reproduce":
        from src import reproduce
        paper_tables = args.paper_tables.lower() == "true" or os.getenv("PAPER_TABLES") == "1"
        results = reproduce.run(config_path, paper_tables=paper_tables, seed=args.seed)
        print(f"\n📊 Paper reproduction completed.")
        print("Artifacts saved:")
        for label, rel_path in results.get("artifact_paths", {}).items():
            print(f"  - {label}: {rel_path}")


if __name__ == "__main__":
    main()
