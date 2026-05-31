from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from src.data.visdrone import build_person_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--classes", type=int, nargs="+", default=[1, 2])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.data.exists():
        raise FileNotFoundError(args.data)

    manifest = build_person_manifest(args.data, set(args.classes))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2))
    logging.getLogger(__name__).debug("Wrote %s", args.output)


if __name__ == "__main__":
    main()