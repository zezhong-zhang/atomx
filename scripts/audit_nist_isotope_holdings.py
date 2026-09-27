#!/usr/bin/env python3
"""Inventory NIST's isotope-specific line and level holdings by element."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import sys
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from atomx.constants import element_symbol  # noqa: E402

DEFAULT_OUT = ROOT / "output/nist_spectra"
ANCHOR = re.compile(r"<a class=['\"]bib['\"] id=\"aa\"[^>]*>([^<]+)</a>")
CELL = re.compile(r"<td[^>]*>(.*?)</td>", re.S)
TAG = re.compile(r"<[^>]*>")
ISOTOPE = re.compile(r"^(?:\d+[A-Z][a-z]?|D|T)\s+[IVXLCDM]+$")


def parse_holdings(body: bytes, kind: str) -> list[dict]:
    source = body.decode("utf-8", errors="replace")
    anchors = list(ANCHOR.finditer(source))
    if not anchors:
        heading = "No. of lines" if kind == "lines" else "No. of levels"
        if heading in source and "NIST ASD Holdings" in source:
            return []
        raise ValueError("NIST holdings page has no spectrum links or holdings table")
    rows = []
    for match in anchors:
        end = source.find("<tr", match.end())
        segment = source[match.end():end if end >= 0 else None]
        cells = [html.unescape(TAG.sub("", cell)).strip()
                 for cell in CELL.findall(segment)]
        required = 6 if kind == "lines" else 2
        if len(cells) < required:
            raise ValueError(f"incomplete holdings row: {match.group(1)}")
        label = html.unescape(match.group(1)).strip()
        count = int(cells[1].replace(",", ""))
        row = {"spectrum": label, "count": count,
               "isotope_specific": bool(ISOTOPE.fullmatch(label))}
        if kind == "lines":
            row.update(with_transition_probability=int(cells[3].replace(",", "")),
                       with_level_designation=int(cells[5].replace(",", "")))
        rows.append(row)
    return rows


def fetch_one(out: Path, kind: str, symbol: str, offline: bool) -> dict:
    stem = "lines_hold" if kind == "lines" else "levels_hold"
    url = "https://physics.nist.gov/cgi-bin/ASD/" + stem + ".pl?" + \
        urllib.parse.urlencode({"el": symbol})
    path = out / "official_isotope_holdings" / kind / f"{symbol}.html"
    if path.exists():
        body = path.read_bytes()
    elif offline:
        raise FileNotFoundError(path)
    else:
        request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(request, timeout=60) as response:
            body = response.read()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.with_suffix(".part").write_bytes(body)
        path.with_suffix(".part").replace(path)
    return {"element": symbol, "url": url,
            "sha256": hashlib.sha256(body).hexdigest(),
            "rows": parse_holdings(body, kind)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--kind", choices=("lines", "levels"), default="lines")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--max-z", type=int, default=118)
    args = parser.parse_args()
    elements = [element_symbol(z) for z in range(1, args.max_z + 1)]
    pages = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(fetch_one, args.output, args.kind, symbol,
                               args.offline): symbol for symbol in elements}
        for future in as_completed(futures):
            page = future.result()
            pages.append(page)
            print(f"{page['element']}: {len(page['rows'])} spectra", flush=True)
    pages.sort(key=lambda page: elements.index(page["element"]))
    isotope_rows = [dict(element=page["element"], **row)
                    for page in pages for row in page["rows"]
                    if row["isotope_specific"]]
    summary = {"kind": args.kind, "elements": len(pages),
               "all_rows": sum(len(page["rows"]) for page in pages),
               "all_count": sum(row["count"] for page in pages for row in page["rows"]),
               "isotope_spectra": len(isotope_rows),
               "isotope_count": sum(row["count"] for row in isotope_rows),
               "isotope_rows": isotope_rows, "pages": pages}
    path = args.output / f"official_isotope_{args.kind}_holdings.json"
    path.write_text(json.dumps(summary, indent=2))
    print({key: summary[key] for key in ("kind", "elements", "all_count",
                                         "isotope_spectra", "isotope_count")})


if __name__ == "__main__":
    main()
