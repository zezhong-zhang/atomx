#!/usr/bin/env python3
"""Compare archived line rows with direct NIST queries across light/heavy ions.

    .venv/bin/python scripts/verify_nist_live_samples.py
    .venv/bin/python scripts/verify_nist_live_samples.py --offline

The exhaustive per-ion count audit is in build_nist_spectra.py. This check
adds field-by-field numeric verification of representative live NIST results.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))
sys.path.insert(0, str(ROOT / "scripts"))
import build_nist_spectra as catalog  # noqa: E402
from atomx.reference_data import nist as nist_asd  # noqa: E402

SAMPLES = ((1, 0), (2, 0), (3, 0), (6, 4), (10, 7), (11, 0),
           (12, 0), (12, 1), (18, 7), (19, 0), (19, 1),
           (26, 16), (92, 1))
PARAMETERS = {
    "limits_type": "0", "unit": "1", "submit": "Retrieve Data",
    "de": "0", "format": "1", "line_out": "0", "remove_js": "on",
    "en_unit": "0", "output": "0", "bibrefs": "1", "show_obs_wl": "1",
    "show_calc_wl": "1", "unc_out": "1", "show_av": "3", "A_out": "0",
    "allowed_out": "1", "forbid_out": "1", "conf_out": "on",
    "term_out": "on", "enrg_out": "on", "J_out": "on",
}
FIELDS = (
    "medium", "observed_nm", "observed_unc_nm", "ritz_nm", "ritz_unc_nm",
    "aki", "acc", "ei_cm1", "ek_cm1",
    "lower_configuration", "lower_term", "lower_j_text",
    "upper_configuration", "upper_term", "upper_j_text",
    "transition_type", "line_reference", "tp_reference",
)


def check_one(out: Path, z: int, charge: int, *, offline: bool) -> dict:
    label = catalog.ion_label(z, charge)
    name = f"asd.{z:03d}{charge:03d}"
    url = "https://physics.nist.gov/cgi-bin/ASD/lines1.pl?" + urllib.parse.urlencode(
        {"spectra": label, **PARAMETERS})
    live_path = out / "live_checks" / name
    if not offline:
        request = urllib.request.Request(url, headers={
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/120.0 Safari/537.36"})
        with urllib.request.urlopen(request, timeout=60) as response:
            body = response.read()
        catalog._valid(body, "lines", z, charge)
        live_path.parent.mkdir(parents=True, exist_ok=True)
        live_path.write_bytes(body)
    body = live_path.read_bytes()
    catalog._valid(body, "lines", z, charge)
    archive_path = out / "raw" / "lines" / name
    if not archive_path.exists():
        archive_path = catalog.PANEL / "lines" / name
    archive_body = archive_path.read_bytes()
    live, live_problems = nist_asd.parse_lines_text(
        body.decode("utf-8", errors="replace"), ion=label, z=z, charge=charge)
    archive, archive_problems = nist_asd.parse_lines_text(
        archive_body.decode("utf-8", errors="replace"), ion=label, z=z, charge=charge)
    mismatches = [
        {"row": i, "field": field, "live": getattr(left, field),
         "archive": getattr(right, field)}
        for i, (left, right) in enumerate(zip(live, archive, strict=False))
        for field in FIELDS if getattr(left, field) != getattr(right, field)
    ]
    return {
        "label": label, "z": z, "charge": charge, "url": url,
        "live_sha256": hashlib.sha256(body).hexdigest(),
        "archive_sha256": hashlib.sha256(archive_body).hexdigest(),
        "live_rows": len(live), "archive_rows": len(archive),
        "live_parse_problems": live_problems,
        "archive_parse_problems": archive_problems,
        "field_mismatches": mismatches,
        "matched": (len(live) == len(archive) and not live_problems
                    and not archive_problems and not mismatches),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=catalog.DEFAULT_OUT)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    out = args.out.resolve()
    results = [check_one(out, z, charge, offline=args.offline)
               for z, charge in SAMPLES]
    report = {
        "checked_utc": datetime.now(timezone.utc).isoformat(),
        "all_matched": all(item["matched"] for item in results),
        "samples": results,
    }
    (out / "live_sample_verification.json").write_text(
        json.dumps(report, indent=2) + "\n")
    for item in results:
        print(f"{item['label']}: {item['live_rows']} direct NIST rows, "
              f"{item['archive_rows']} archive rows, matched={item['matched']}")
    return 0 if report["all_matched"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
