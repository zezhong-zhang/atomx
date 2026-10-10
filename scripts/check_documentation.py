#!/usr/bin/env python3
"""Check repository link targets in documentation entry points.

Checks paths against Git's tracked snapshot, so an ignored local artifact cannot
hide a broken clean-checkout link. Does not check external URLs, heading anchors,
scientific claims, or archived report contents.
"""
from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path
from urllib.parse import unquote, urlsplit

LINK = re.compile(r"\]\((<[^>]+>|[^\s)]+)(?:\s+[^)]*)?\)")
FENCE = re.compile(r"^\s*(`{3,}|~{3,})")


def broken_links(root: Path, documents: list[Path], tracked: set[str]) -> list[str]:
    root = root.resolve()
    errors = []
    for document in documents:
        fenced = None
        for number, line in enumerate(document.read_text(encoding="utf-8").splitlines(), 1):
            fence = FENCE.match(line)
            if fence:
                marker = fence[1][0]
                if fenced is None:
                    fenced = marker
                elif fenced == marker:
                    fenced = None
                continue
            if fenced:
                continue
            for match in LINK.finditer(line):
                destination = match[1].strip("<>")
                url = urlsplit(destination)
                if url.scheme or url.netloc or not url.path:
                    continue
                target = (document.parent / unquote(url.path)).resolve()
                try:
                    relative = target.relative_to(root).as_posix()
                except ValueError:
                    errors.append(f"{document.relative_to(root)}:{number}: outside repository: {destination}")
                    continue
                published = relative in tracked or any(
                    name.startswith(relative.rstrip("/") + "/") for name in tracked
                )
                if not target.exists() or not published:
                    errors.append(f"{document.relative_to(root)}:{number}: missing tracked target: {destination}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", help="Markdown paths; defaults to root and docs entry points")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    tracked = set(subprocess.check_output(["git", "-C", str(root), "ls-files", "-z"]).decode().split("\0"))
    documents = ([root / path for path in args.paths] if args.paths else
                 [root / "README.md", root / "AGENTS.md", *sorted((root / "docs").glob("*.md"))])
    errors = broken_links(root, documents, tracked)
    for error in errors:
        print(error)
    print(f"Checked {len(documents)} documentation entry points; {len(errors)} broken repository links.")
    return int(bool(errors))


if __name__ == "__main__":
    raise SystemExit(main())
