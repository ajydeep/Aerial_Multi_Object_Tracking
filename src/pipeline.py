from __future__ import annotations

import argparse
from pathlib import Path

from src.tracker import ByteTrackTracker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.source.exists():
        raise FileNotFoundError(args.source)

    tracker = ByteTrackTracker()
    print(tracker.config.tracker_yaml)


if __name__ == "__main__":
    main()