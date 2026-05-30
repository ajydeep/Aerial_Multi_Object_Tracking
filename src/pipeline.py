from __future__ import annotations

import argparse
from pathlib import Path

from src.tracker import ByteTrackTracker
from src.render import render_sequence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.source.exists():
        raise FileNotFoundError(args.source)

    tracker = ByteTrackTracker()
    render_sequence(args.source, args.output, tracker)
    print(args.output)


if __name__ == "__main__":
    main()