#!/usr/bin/env python3
"""Check the live tree against docs/frozen-evidence-manifest.json.

An evidence directory under docs/evidence/ is an atomic record keyed by its name:
the recorders refuse an existing evidence path, so its file set is part of the
record. This entry point makes that rule executable rather than prose. It fails
with the offending directory and file named for:

  * a file added to, removed from, or byte-modified inside a recorded directory,
  * a recorded directory that is missing entirely,
  * a directory under docs/evidence/ that no manifest entry records,
  * a loose file at the evidence root that is missing, modified or unregistered.

Digests cover content with CRLF normalized to LF, so a checkout's line-ending
policy is not drift while every content change still fails. The manifest is the
record: this checker never writes or re-baselines it, because regenerating the
manifest must be a reviewed commit rather than a silent side effect. Pure
hashlib; no network, no test suite, no non-standard dependency.

Usage, from the checkout root:
    .venv/Scripts/python.exe tools/check_frozen_evidence.py

Exit codes: 0 clean, 1 drift/missing/unregistered entries, 2 unusable manifest.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs" / "evidence"
MANIFEST = ROOT / "docs" / "frozen-evidence-manifest.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def relative_files(directory: Path) -> dict[str, Path]:
    return {str(path.relative_to(directory)).replace(os.sep, "/"): path
            for path in directory.rglob("*") if path.is_file()}


def main() -> int:
    try:
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        directories = manifest["directories"]
        root_files = manifest["root_files"]
    except (OSError, KeyError, json.JSONDecodeError) as error:
        print("ERROR unusable manifest %s: %s" % (MANIFEST, error))
        return 2
    if not EVIDENCE.is_dir():
        print("ERROR missing evidence root %s" % EVIDENCE)
        return 2

    problems: list[str] = []
    verified_files = 0
    for name, record in sorted(directories.items()):
        directory = EVIDENCE / name
        if not directory.is_dir():
            problems.append("MISSING DIRECTORY %s" % name)
            continue
        expected = record["files"]
        actual = relative_files(directory)
        for relative in sorted(set(actual) - set(expected)):
            problems.append("ADDED FILE %s/%s" % (name, relative))
        for relative in sorted(set(expected) - set(actual)):
            problems.append("REMOVED FILE %s/%s" % (name, relative))
        for relative in sorted(set(expected) & set(actual)):
            digest = sha256(actual[relative])
            if digest != expected[relative]:
                problems.append("MODIFIED FILE %s/%s expected=%s actual=%s"
                                % (name, relative, expected[relative][:12], digest[:12]))
            verified_files += 1

    for name, digest in sorted(root_files.items()):
        path = EVIDENCE / name
        if not path.is_file():
            problems.append("MISSING FILE %s" % name)
        elif sha256(path) != digest:
            problems.append("MODIFIED FILE %s expected=%s actual=%s"
                            % (name, digest[:12], sha256(path)[:12]))
        else:
            verified_files += 1

    for entry in sorted(EVIDENCE.iterdir()):
        if entry.is_dir() and entry.name not in directories:
            problems.append("UNREGISTERED DIRECTORY %s" % entry.name)
        elif entry.is_file() and entry.name not in root_files:
            problems.append("UNREGISTERED FILE %s" % entry.name)

    if problems:
        for problem in problems:
            print("FAIL %s" % problem)
        print("FAIL: %d problem(s) against %s (%d directories, %d files verified)"
              % (len(problems), MANIFEST.name, len(directories), verified_files))
        return 1
    print("OK: %d directories and %d files match %s (format %s)"
          % (len(directories), verified_files, MANIFEST.name, manifest.get("format")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
