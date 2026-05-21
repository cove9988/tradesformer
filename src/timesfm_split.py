from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import re

WEEKLY_CSV_PATTERN = re.compile(r"^(?P<symbol>[A-Z]+)_(?P<iso_year>\d{4})_(?P<iso_week>\d{1,2})\.csv$")


def parse_weekly_csv_path(csv_path: str | Path) -> dict[str, int | str]:
    path = Path(csv_path)
    match = WEEKLY_CSV_PATTERN.match(path.name)
    if not match:
        raise ValueError(f"Invalid weekly CSV filename: {path.name}")

    return {
        "symbol": match.group("symbol"),
        "iso_year": int(match.group("iso_year")),
        "iso_week": int(match.group("iso_week")),
        "file_name": path.name,
    }


def _sorted_week_files(data_directory: str | Path, symbol: str | None = None) -> list[Path]:
    data_directory = Path(data_directory)
    csv_files = []
    for path in data_directory.glob("*.csv"):
        parsed = parse_weekly_csv_path(path)
        if symbol and parsed["symbol"] != symbol:
            continue
        csv_files.append((int(parsed["iso_year"]), int(parsed["iso_week"]), path))

    csv_files.sort(key=lambda item: (item[0], item[1], item[2].name))
    return [item[2] for item in csv_files]


def build_split_manifest(
    data_directory: str | Path,
    symbol: str,
    validation_weeks: int = 6,
    test_weeks: int = 6,
) -> dict[str, object]:
    if validation_weeks < 0 or test_weeks < 0:
        raise ValueError("validation_weeks and test_weeks must be non-negative")

    week_files = _sorted_week_files(data_directory, symbol=symbol)
    total_weeks = len(week_files)
    if total_weeks == 0:
        raise ValueError(f"No weekly CSV files found for {symbol} in {data_directory}")
    if validation_weeks + test_weeks >= total_weeks:
        raise ValueError("validation_weeks + test_weeks must leave at least one training week")

    train_end = total_weeks - validation_weeks - test_weeks
    val_end = total_weeks - test_weeks

    all_weeks = [path.name for path in week_files]
    return {
        "symbol": symbol,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "data_directory": str(Path(data_directory)),
        "all_weeks": all_weeks,
        "splits": {
            "train": all_weeks[:train_end],
            "val": all_weeks[train_end:val_end],
            "test": all_weeks[val_end:],
        },
    }


def write_split_manifest(manifest_path: str | Path, manifest: dict[str, object]) -> Path:
    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest_path


def load_split_manifest(manifest_path: str | Path) -> dict[str, object]:
    manifest_path = Path(manifest_path)
    return json.loads(manifest_path.read_text())


def _default_train_directory(symbol: str) -> Path:
    return Path(f"./data/split/{symbol}/weekly/train")


def _default_manifest_path(symbol: str) -> Path:
    return Path(f"./data/split/{symbol}/weekly/manifest.json")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a chronological TimeFM split manifest")
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--data-directory", default=None)
    parser.add_argument("--manifest-path", default=None)
    parser.add_argument("--validation-weeks", type=int, default=6)
    parser.add_argument("--test-weeks", type=int, default=6)
    parser.add_argument("--write-manifest", action="store_true")
    args = parser.parse_args()

    data_directory = Path(args.data_directory) if args.data_directory else _default_train_directory(args.symbol)
    manifest = build_split_manifest(
        data_directory=data_directory,
        symbol=args.symbol,
        validation_weeks=args.validation_weeks,
        test_weeks=args.test_weeks,
    )

    if args.write_manifest:
        manifest_path = Path(args.manifest_path) if args.manifest_path else _default_manifest_path(args.symbol)
        write_split_manifest(manifest_path, manifest)
        print(manifest_path)
    else:
        print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
