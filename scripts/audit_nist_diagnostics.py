#!/usr/bin/env python3
"""Probe the original NIST ASD plasma-diagnostics line output.

ASD's HTML form exposes ``diag_out`` and ``line_out=4``.  This small source
audit retains the raw responses and distinguishes a source query error from
an empty diagnostic dataset.  It does not alter the spectral archive.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from import_nist_line_details import BASE_URL, PARAMETERS

ROOT = Path(__file__).resolve().parents[1]
PROBES = (
    ("c_iii_minimal_diag_html", "C III", "0", "4", True),
    ("c_iii_control_html", "C III", "0", "0", False),
    ("c_iii_include_html", "C III", "0", "0", True),
    ("c_iii_browser_defaults_html", "C III", "0", "0", True),
    ("c_iii_only_html", "C III", "0", "4", False),
    ("c_iii_include_tsv", "C III", "3", "0", True),
    ("fe_xii_include_html", "Fe XII", "0", "0", True),
)
BROWSER_DEFAULTS = {
    # Checked fields and selected options read from NIST's live Lines form.
    "output_type": "0", "unit": "1", "de": "0", "plot_out": "0",
    "I_scale_type": "1", "format": "0", "line_out": "0",
    "en_unit": "0", "output": "0", "bibrefs": "1", "page_size": "15",
    "show_obs_wl": "1", "show_calc_wl": "1", "unc_out": "1",
    "order_out": "0", "show_av": "2", "tsb_value": "0",
    "A_out": "0", "allowed_out": "1", "forbid_out": "1",
}


def probe_url(spectrum: str, fmt: str, line_out: str, include: bool) -> str:
    params = {"spectra": spectrum, **PARAMETERS,
              "format": fmt, "line_out": line_out}
    if include:
        params["diag_out"] = "1"
    return BASE_URL + "?" + urllib.parse.urlencode(params)


def run(out: Path, offline: bool = False, refresh: bool = False) -> dict:
    raw_dir = out / "official_diagnostics_probe"
    raw_dir.mkdir(parents=True, exist_ok=True)
    report_path = out / "nist_diagnostics_audit.json"
    previous = (json.loads(report_path.read_text())
                if report_path.exists() else {})
    previous_probes = {row["name"]: row
                       for row in previous.get("probes", [])}
    records = []
    for name, spectrum, fmt, line_out, include in PROBES:
        url = (BASE_URL + "?" + urllib.parse.urlencode({
            **BROWSER_DEFAULTS, "spectra": spectrum,
            "submit": "Retrieve Data", "diag_out": "1"})
            if name == "c_iii_browser_defaults_html" else
            BASE_URL + "?" + urllib.parse.urlencode({
                "spectra": spectrum, "line_out": line_out,
                "diag_out": "1", "format": fmt, "unit": "1"})
            if name == "c_iii_minimal_diag_html" else
            probe_url(spectrum, fmt, line_out, include))
        raw_path = raw_dir / f"{name}.{'tsv' if fmt == '3' else 'html'}"
        http_status = None
        if offline or (raw_path.exists() and not refresh):
            body = raw_path.read_bytes()
            prior = previous_probes.get(name, {})
            if prior.get("sha256") == hashlib.sha256(body).hexdigest():
                http_status = prior.get("http_status")
        else:
            request = urllib.request.Request(url, headers={
                "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) "
                              "AppleWebKit/537.36 Chrome/120.0 Safari/537.36"})
            try:
                with urllib.request.urlopen(request, timeout=120) as response:
                    http_status = response.status
                    body = response.read()
            except urllib.error.HTTPError as exc:
                http_status = exc.code
                body = exc.read()
            raw_path.write_bytes(body)
        content = body.decode("utf-8", errors="replace")
        plain = re.sub(r"\s+", " ", html.unescape(
            re.sub(r"<[^>]+>", " ", content)))
        message = ("Query error." if "Error Message: Query error." in plain
                   else None)
        records.append({
            "name": name, "spectrum": spectrum, "format": fmt,
            "line_out": line_out, "diag_out": include,
            "url": url, "http_status": http_status,
            "sha256": hashlib.sha256(body).hexdigest(),
            "bytes": len(body), "error_message": message,
            "input_error": "Input Error" in content,
            "raw_path": str(raw_path.relative_to(out)),
        })
    summary = {
        "audited_at_utc": datetime.now(timezone.utc).isoformat(),
        "nist_lines_form": "https://physics.nist.gov/PhysRefData/ASD/lines_form.html",
        "nist_lines_help": "https://physics.nist.gov/PhysRefData/ASD/Html/lineshelp.html",
        "diagnostics_imported": False,
        "probes": records,
    }
    report_path.write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path,
                        default=ROOT / "output/nist_spectra")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    summary = run(args.out, args.offline, args.refresh)
    for row in summary["probes"]:
        print(row["name"], row["http_status"], row["bytes"],
              row["error_message"] or
              ("HTTP error" if row["http_status"] not in (None, 200) else None) or
              ("cached; HTTP status unknown" if row["http_status"] is None
               else None) or
              ("Input Error" if row["input_error"] else "data response"))


if __name__ == "__main__":
    main()
