#!/usr/bin/env python3
"""Gating guard: the model-price maps must not contain duplicate JSON keys.

``json.load`` silently keeps the LAST occurrence of a repeated key, so a duplicated
model entry parses cleanly, resolves deterministically, and still ships the wrong
pricing: whichever entry a human reads in the file may not be the one that takes
effect. Two of these have already been inherited from upstream
(``gemini-omni-flash-preview``, ``jp.anthropic.claude-sonnet-4-6``), and each cost
real time to diagnose because every ordinary validator -- ``jq empty`` included --
accepts them.

This check re-parses each price map with an ``object_pairs_hook`` that inspects the
raw key/value pairs BEFORE they collapse into a dict, so a repeat is caught instead
of swallowed. It walks every nesting level, not just the top-level model map.

Usage:
    python scripts/check_model_prices_duplicate_keys.py [FILE ...]

With no arguments it checks the two tracked price maps. Exits 1 on any duplicate.
Stdlib only.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, NamedTuple

REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_TARGETS: tuple[str, ...] = (
    "model_prices_and_context_window.json",
    "litellm/model_prices_and_context_window_backup.json",
)


class DuplicateKey(NamedTuple):
    """A key that appeared more than once inside a single JSON object."""

    key: str
    count: int
    lines: tuple[int, ...]


def find_duplicate_keys(path: Path) -> list[DuplicateKey]:
    """Return every key repeated within one JSON object in ``path``.

    Duplicates are read off the pair list handed to ``object_pairs_hook``, which
    still holds both occurrences; the dict it returns does not.
    """
    raw = path.read_text()
    repeats: Counter[str] = Counter()

    def hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        counts = Counter(key for key, _ in pairs)
        for key, count in counts.items():
            if count > 1:
                # Keep the highest count seen for this key across all objects.
                repeats[key] = max(repeats[key], count)
        return dict(pairs)

    json.loads(raw, object_pairs_hook=hook)

    duplicates: list[DuplicateKey] = []
    for key, count in sorted(repeats.items()):
        needle = f'"{key}":'
        lines = tuple(
            number
            for number, line in enumerate(raw.splitlines(), start=1)
            if needle in line
        )
        duplicates.append(DuplicateKey(key=key, count=count, lines=lines))
    return duplicates


def check(path: Path) -> bool:
    """Check one file. True when clean, False when duplicates (or errors) found."""
    try:
        duplicates = find_duplicate_keys(path)
    except FileNotFoundError:
        print(f"FAIL {path}: file not found", file=sys.stderr)
        return False
    except json.JSONDecodeError as exc:
        print(f"FAIL {path}: invalid JSON - {exc}", file=sys.stderr)
        return False

    if not duplicates:
        print(f"OK   {path}: no duplicate keys")
        return True

    print(f"FAIL {path}: {len(duplicates)} duplicate key(s)", file=sys.stderr)
    for dup in duplicates:
        where = ", ".join(str(line) for line in dup.lines) or "unknown"
        print(
            f"       {dup.key!r} appears {dup.count}x in the same object "
            f"(lines: {where})",
            file=sys.stderr,
        )
    print(
        "\n  json.load keeps the LAST occurrence, so the earlier entry is dead text\n"
        "  that still reads as authoritative. Delete whichever entry is stale and\n"
        "  keep the one that matches the other price map and its sibling models.",
        file=sys.stderr,
    )
    return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Fail if a model-price JSON file repeats a key inside an object."
        ),
    )
    parser.add_argument(
        "files",
        nargs="*",
        help=f"JSON files to check (default: {', '.join(DEFAULT_TARGETS)})",
    )
    args = parser.parse_args(argv)

    paths = (
        [Path(f) for f in args.files]
        if args.files
        else [REPO_ROOT / target for target in DEFAULT_TARGETS]
    )

    results = [check(path) for path in paths]
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
