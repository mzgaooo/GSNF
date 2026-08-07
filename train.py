import argparse
from pathlib import Path

import torch

from training.config import load_config
from training.engine import run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.seed is not None:
        config.seed = args.seed
    if args.epochs is not None:
        config.epochs = args.epochs
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    result = run(config, Path(args.data_root), torch.device(args.device))
    print(f"best_epoch={result['best_epoch']}")
    print(f"test_auroc={result['test']['auroc']:.6f}")
    print(f"test_auprc={result['test']['auprc']:.6f}")


if __name__ == "__main__":
    main()
