from __future__ import annotations

import argparse
from pathlib import Path

from src.tracker import ByteTrackTracker
from src.render import render_sequence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("result.mp4"))
    return parser.parse_args()


def next_output_path(output_file: Path) -> Path:
    if not output_file.exists():
        return output_file

    suffix = output_file.suffix
    stem = output_file.stem
    parent = output_file.parent
    index = 1
    while True:
        candidate = parent / f"{stem}{index}{suffix}"
        if not candidate.exists():
            return candidate
        index += 1


def main() -> None:
    args = parse_args()
    if not args.source.exists():
        raise FileNotFoundError(args.source)

    tracker = ByteTrackTracker()
    # If the user didn't provide an explicit output, default to outputs/<source_basename>.mp4
    if args.output == Path("result.mp4"):
        default_out = Path("outputs") / f"{args.source.name}.mp4"
        output_file = next_output_path(default_out)
    else:
        output_file = next_output_path(args.output)
    render_sequence(args.source, output_file, tracker)
    print(output_file)


if __name__ == "__main__":
    main()