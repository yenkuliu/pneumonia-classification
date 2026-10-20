"""Run deterministic inference with default threshold 0.5 and batch size 16."""

import sys

import train
import inference


def main():
    train.set_seed(820, deterministic=True)
    if not any(arg == "--threshold" or arg.startswith("--threshold=") for arg in sys.argv):
        sys.argv.extend(["--threshold", "0.5"])
    if not any(arg == "--batch-size" or arg.startswith("--batch-size=") for arg in sys.argv):
        sys.argv.extend(["--batch-size", "16"])
    inference.main()


if __name__ == "__main__":
    main()
