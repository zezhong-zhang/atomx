#!/usr/bin/env python
"""Download the NIST Atomic Spectra Database panel used by the level tests.

The NIST ASD v5.12 is mirrored as plain ASCII (XHTML, actually) by the
Cloudy/Nublado project.  Two endpoints matter here::

    https://data.nublado.org/nist/asd_v5.12/levels/asd.ZZZSSS
    https://data.nublado.org/nist/asd_v5.12/lines/asd.ZZZSSS

``ZZZ`` is the 3-digit nuclear charge and ``SSS`` the 3-digit **ionic
charge** (``000`` = neutral, ``001`` = singly ionised, ...), i.e. the
number of electrons is ``Z - SSS``.  The convention is self-verifying:
each file carries its spectroscopic label (``Fe  XVII``) and its
isoelectronic sequence (``Z = 26, Ne isoelectronic sequence``) in the
HTML preamble, and this script *checks both* against the declared panel
— a mislabelled ``ZZZSSS`` therefore fails loudly instead of silently
feeding the wrong ion into the verification.

Usage::

    python benchmarks/fetch_nist_asd.py            # fetch what is missing
    python benchmarks/fetch_nist_asd.py --force    # re-fetch everything
    python benchmarks/fetch_nist_asd.py --check     # offline integrity check
    python benchmarks/fetch_nist_asd.py --list      # print the declared panel

Idempotency and integrity
-------------------------
A download is written to a ``.part`` file, then validated (HTTP 200,
minimum byte count, expected NIST table header, expected ion label,
balanced ``</html>``), then atomically renamed into place.  A truncated
or error page can therefore never be cached.  Re-running the script
with an intact tree is a no-op: files whose size *and* SHA-256 match the
manifest are skipped.  ``--check`` re-verifies sizes and hashes without
touching the network — that is what the offline test suite uses.

The manifest ``benchmarks/nist_asd/manifest.json`` records, per file,
the URL, the retrieval date, the byte count and the SHA-256, so any
quoted number in ``benchmarks/NIST_VERIFICATION.md`` can be retraced to
an exact byte sequence.

The data are US-government public-domain facts and are committed to the
repository so the verification runs offline in CI.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import date
from pathlib import Path

BASE_URL = "https://data.nublado.org/nist/asd_v5.12"
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = REPO_ROOT / "benchmarks" / "nist_asd"
MANIFEST_PATH = DATA_ROOT / "manifest.json"

#: A short body is a truncated body.  The smallest real panel file is
#: ~34 kB (Fe XVII levels), the largest is ~2.2 MB (Fe I lines); 4 kB
#: rejects an empty file, a 404 page and a Cloudflare interstitial.
MIN_BYTES = 4096

USER_AGENT = "atomx-nist-verify/1.0 (+https://github.com/; benchmark data fetch)"


@dataclass(frozen=True)
class PanelEntry:
    """One ``(ion, kind)`` artefact of the declared panel.

    ``label`` is the NIST spectroscopic label (``"Fe XVII"``); the file
    must carry it.  ``sequence`` is the isoelectronic sequence NIST
    states for the ion (``"Ne"``), checked against ``Z`` and the ionic
    charge: ``n_electrons = Z - charge`` and the sequence is the
    ``n_electrons``-electron isoelectronic sequence.
    """

    label: str
    z: int
    charge: int
    sequence: str
    kinds: tuple[str, ...] = ("levels", "lines")

    @property
    def zzz(self) -> str:
        return f"{self.z:03d}"

    @property
    def sss(self) -> str:
        return f"{self.charge:03d}"

    @property
    def n_electrons(self) -> int:
        return self.z - self.charge

    def url(self, kind: str) -> str:
        return f"{BASE_URL}/{kind}/asd.{self.zzz}{self.sss}"

    def relpath(self, kind: str) -> str:
        return f"{kind}/asd.{self.zzz}{self.sss}"


def _p(label: str, z: int, charge: int, sequence: str,
       kinds: tuple[str, ...] = ("levels", "lines")) -> PanelEntry:
    return PanelEntry(label, z, charge, sequence, kinds)


#: The declared panel.  It is chosen to span the theory ladder rather
#: than to flatter it: neutral atoms from H to Fe, the H-like and
#: He-like isoelectronic sequences (where QED is measurable), a Li-like
#: and a Na-like ion (one valence electron), Ne-like Fe XVII
#: (closed shell + strong correlation) and Fe I (open 3d shell).
PANEL: tuple[PanelEntry, ...] = (
    # --- neutral / low-charge atoms -------------------------------------
    _p("H I", 1, 0, "H"),
    _p("He I", 2, 0, "He"),
    _p("Li I", 3, 0, "Li"),
    _p("Be I", 4, 0, "Be"),
    _p("C I", 6, 0, "C"),
    _p("N I", 7, 0, "N"),
    _p("O I", 8, 0, "O"),
    _p("Ne I", 10, 0, "Ne"),
    _p("Na I", 11, 0, "Na"),
    _p("Ar I", 18, 0, "Ar"),
    _p("Fe I", 26, 0, "Fe"),
    # --- H-like sequence (1 electron) -----------------------------------
    _p("C VI", 6, 5, "H"),
    _p("Ne X", 10, 9, "H"),
    _p("Ar XVIII", 18, 17, "H"),
    _p("Fe XXVI", 26, 25, "H"),
    # --- He-like sequence (2 electrons) ---------------------------------
    _p("C V", 6, 4, "He"),
    _p("Ne IX", 10, 8, "He"),
    _p("Ar XVII", 18, 16, "He"),
    _p("Fe XXV", 26, 24, "He"),
    # --- few-valence / closed-shell highly charged ions ------------------
    _p("Ne VIII", 10, 7, "Li"),
    _p("Ar VIII", 18, 7, "Na"),
    _p("Fe XVII", 26, 16, "Ne"),
)

#: ``Fe XVII`` in the task brief was pointed at ``asd.026017``, which is
#: in fact ``Fe XVIII`` (F-like).  The fetch below checks the label
#: embedded in the file, so the panel above carries the correct index.
#: The levels pages announce the ion as ``<sup></sup>Fe  XVII &nbsp;...``;
#: the lines pages as ``<u>Fe XVII</u>: ... Lines of Data Found``.
TITLE_RES = (
    re.compile(r"<sup>\s*</sup>\s*(?P<label>[A-Za-z]{1,2}\s+[IVX]+)\s*&nbsp;"),
    re.compile(r"<u>(?P<label>[A-Za-z]{1,2}\s+[IVX]+)</u>\s*:"),
)
SEQUENCE_RE = re.compile(
    r"<i>Z</i>\s*=\s*(?P<z>\d+),\s*(?P<seq>[A-Za-z]{1,2})\s+"
    r"isoelectronic sequence")
LEVELS_HEADER_RE = re.compile(
    r"Configuration\s*\|\s*Term\s*\|\s*J\s*\|\s*Level \(cm-1\)")
#: The lines pages differ in column *count* between ions (Li I has no
#: observed-wavelength uncertainty column, He I does), but every one of
#: them carries "Observed ... | ... Aki" on a single header line.
LINES_HEADER_RE = re.compile(r"Observed[^\n]*\|[^\n]*Aki")


class FetchError(RuntimeError):
    """Raised for every condition that must not be cached silently."""


def _normalise_label(text: str) -> str:
    return " ".join(text.split()).upper()


def _download(url: str, *, retries: int = 3, timeout: int = 120) -> bytes:
    """Fetch ``url``, raising :class:`FetchError` on any anomaly."""
    last: Exception | None = None
    for attempt in range(retries):
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                status = response.status
                body = response.read()
        except urllib.error.HTTPError as exc:  # includes 404 / 5xx
            if exc.code in (429, 500, 502, 503, 504) and attempt + 1 < retries:
                last = exc
                time.sleep(2 ** attempt)
                continue
            raise FetchError(f"HTTP {exc.code} for {url}") from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last = exc
            if attempt + 1 < retries:
                time.sleep(2 ** attempt)
                continue
            raise FetchError(f"network error for {url}: {exc}") from exc

        if status != 200:
            raise FetchError(f"HTTP {status} for {url} (expected 200)")
        if len(body) < MIN_BYTES:
            raise FetchError(
                f"{url}: body is {len(body)} bytes < MIN_BYTES={MIN_BYTES} "
                f"— refusing to cache a truncated/empty file")
        return body
    raise FetchError(f"exhausted retries for {url}: {last}")


def _validate(body: bytes, entry: PanelEntry, kind: str) -> None:
    """Structural + label validation of a freshly downloaded body."""
    try:
        text = body.decode("utf-8", errors="replace")
    except Exception as exc:  # pragma: no cover - decode is lossy already
        raise FetchError(
            f"{entry.label} {kind}: undecodable body ({exc})") from exc

    if "</html>" not in text:
        raise FetchError(
            f"{entry.label} {kind}: no closing </html> — truncated download")

    header_re = LEVELS_HEADER_RE if kind == "levels" else LINES_HEADER_RE
    if not header_re.search(text):
        raise FetchError(
            f"{entry.label} {kind}: the expected NIST {kind} table header "
            f"is missing — this is not a {kind} file")

    title = next((m for m in (rx.search(text) for rx in TITLE_RES)
                  if m is not None), None)
    if title is None:
        raise FetchError(
            f"{entry.label} {kind}: could not read the spectroscopic label")
    found = _normalise_label(title.group("label"))
    if found != _normalise_label(entry.label):
        raise FetchError(
            f"{entry.label} {kind}: file declares ion {found!r}; the panel "
            f"index asd.{entry.zzz}{entry.sss} does not match its label")

    seq = SEQUENCE_RE.search(text)
    if seq is None:
        raise FetchError(
            f"{entry.label} {kind}: could not read the isoelectronic sequence")
    seq_name = seq.group("seq")
    if seq_name != entry.sequence:
        raise FetchError(
            f"{entry.label} {kind}: file declares the {seq_name} "
            f"isoelectronic sequence, panel says {entry.sequence}")
    if int(seq.group("z")) != entry.z:
        raise FetchError(
            f"{entry.label} {kind}: file declares Z={seq.group('z')}, "
            f"panel says Z={entry.z}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest() -> dict:
    if MANIFEST_PATH.exists():
        return json.loads(MANIFEST_PATH.read_text())
    return {"source": {}, "files": []}


def _manifest_index(manifest: dict) -> dict[str, dict]:
    return {item["path"]: item for item in manifest.get("files", [])}


def _entry_record(entry: PanelEntry, kind: str) -> dict:
    return {
        "path": entry.relpath(kind),
        "kind": kind,
        "label": entry.label,
        "Z": entry.z,
        "charge": entry.charge,
        "n_electrons": entry.n_electrons,
        "isoelectronic_sequence": entry.sequence,
        "url": entry.url(kind),
    }


def fetch(*, force: bool = False, verbose: bool = True) -> dict:
    """Fetch the declared panel; return the updated manifest."""
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest()
    old = _manifest_index(manifest)
    records: list[dict] = []
    downloaded = skipped = 0

    for entry in PANEL:
        for kind in entry.kinds:
            target = DATA_ROOT / entry.relpath(kind)
            target.parent.mkdir(parents=True, exist_ok=True)
            previous = old.get(entry.relpath(kind))
            record = _entry_record(entry, kind)

            if (not force and target.exists() and previous is not None
                    and previous.get("bytes") == target.stat().st_size
                    and previous.get("sha256") == _sha256(target)):
                record.update({
                    "bytes": previous["bytes"],
                    "sha256": previous["sha256"],
                    "retrieved": previous["retrieved"],
                    "http_status": previous.get("http_status", 200),
                    "note": "verified against manifest (not re-downloaded)",
                })
                records.append(record)
                skipped += 1
                continue

            body = _download(entry.url(kind))
            _validate(body, entry, kind)
            part = target.with_suffix(target.suffix + ".part")
            part.write_bytes(body)
            part.replace(target)
            record.update({
                "bytes": len(body),
                "sha256": _sha256(target),
                "retrieved": date.today().isoformat(),
                "http_status": 200,
                "note": "downloaded and validated",
            })
            records.append(record)
            downloaded += 1
            if verbose:
                print(f"  fetched {entry.label:>10s} {kind:<6s} "
                      f"{len(body):>9,d} B  {entry.url(kind)}")

    manifest = {
        "source": {
            "database": "NIST Atomic Spectra Database (ASD) v5.12",
            "mirror": BASE_URL,
            "levels_url_template": f"{BASE_URL}/levels/asd.ZZZSSS",
            "lines_url_template": f"{BASE_URL}/lines/asd.ZZZSSS",
            "index_convention": ("ZZZ = nuclear charge, SSS = ionic charge "
                                 "(000 = neutral)"),
            "citation": (
                "Kramida, A., Ralchenko, Yu., Reader, J., and NIST ASD Team "
                "(2024). NIST Atomic Spectra Database (ver. 5.12), "
                "[Online]. Available: https://physics.nist.gov/asd "
                "[2024, October]. National Institute of Standards and "
                "Technology, Gaithersburg, MD. DOI: "
                "https://doi.org/10.18434/T4W30F"),
            "licence": ("US Government work — public domain "
                        "(17 U.S.C. §105)"),
        },
        "generated_by": "benchmarks/fetch_nist_asd.py",
        "retrieved": date.today().isoformat(),
        "n_files": len(records),
        "total_bytes": sum(r["bytes"] for r in records),
        "files": sorted(records, key=lambda r: r["path"]),
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n")
    if verbose:
        print(f"manifest: {MANIFEST_PATH}  "
              f"({len(records)} files, {downloaded} downloaded, "
              f"{skipped} already present)")
    return manifest


def check() -> int:
    """Offline integrity check of the cached panel.  Returns an exit code."""
    if not MANIFEST_PATH.exists():
        print(f"MISSING manifest {MANIFEST_PATH}\n"
              f"  -> run: python benchmarks/fetch_nist_asd.py",
              file=sys.stderr)
        return 1
    manifest = load_manifest()
    problems: list[str] = []
    seen: set[str] = set()
    for record in manifest["files"]:
        seen.add(record["path"])
        path = DATA_ROOT / record["path"]
        if not path.exists():
            problems.append(f"missing file {record['path']}")
            continue
        size = path.stat().st_size
        if size != record["bytes"]:
            problems.append(
                f"{record['path']}: {size} B on disk, manifest says "
                f"{record['bytes']} B")
            continue
        if _sha256(path) != record["sha256"]:
            problems.append(f"{record['path']}: SHA-256 mismatch")
    for entry in PANEL:
        for kind in entry.kinds:
            if entry.relpath(kind) not in seen:
                problems.append(f"panel member absent from manifest: "
                                f"{entry.relpath(kind)}")
    if problems:
        print("NIST ASD cache is NOT intact:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        print("  -> re-run: python benchmarks/fetch_nist_asd.py",
              file=sys.stderr)
        return 1
    print(f"NIST ASD cache OK: {manifest['n_files']} files, "
          f"{manifest['total_bytes']:,d} B, "
          f"retrieved {manifest['retrieved']}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--force", action="store_true",
                        help="re-download every file, even if cached")
    parser.add_argument("--check", action="store_true",
                        help="offline integrity check; no network access")
    parser.add_argument("--list", action="store_true",
                        help="print the declared panel and exit")
    args = parser.parse_args(argv)

    if args.list:
        for entry in PANEL:
            print(f"{entry.label:>10s}  Z={entry.z:3d}  charge={entry.charge:3d}"
                  f"  N={entry.n_electrons:3d}  seq={entry.sequence:<3s}"
                  f"  {entry.relpath('levels')}")
        return 0
    if args.check:
        return check()
    try:
        fetch(force=args.force)
    except FetchError as exc:
        print(f"FETCH FAILED: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
