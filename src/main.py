from pathlib import Path
import argparse
import logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.data.exists():
        raise FileNotFoundError(args.data)
    logging.getLogger(__name__).debug("Ready: %s", args.data)


if __name__ == "__main__":
    main()