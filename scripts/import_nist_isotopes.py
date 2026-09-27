#!/usr/bin/env python
"""Select a documented reference isotope for finite-mass atomx calculations.

The source is NIST's isotope mass and abundance table. An element with a
natural abundance uses its most abundant isotope. For elements without a
reported abundance, the first tabulated isotope is a provisional choice.
The selected isotope is not necessarily the isotope of every ASD line.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import subprocess
from pathlib import Path

from atomx.constants import CODATA, DEFAULT_CODATA_VERSION, PROTON_ELECTRON_MASS_RATIO

URL = "https://physics.nist.gov/cgi-bin/Compositions/stand_alone.pl?all=all&ascii=ascii2"
ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output/nist_spectra"


def parse_all_isotopes(source: str) -> list[dict]:
    """Retain each NIST isotope mass for mass-resolved spectrum comparisons."""
    text = html.unescape(source)
    records = []
    for block in re.split(r"(?=Atomic Number\s*=\s*\d+)", text):
        fields = {key.strip(): value.strip() for key, value in
                  re.findall(r"(?m)^([A-Za-z ]+)[ \t]*=[ \t]*([^\n]*)", block)}
        if "Atomic Number" not in fields:
            continue
        try:
            z = int(fields["Atomic Number"])
            mass_number = int(fields["Mass Number"])
            atomic_mass_u = float(fields["Relative Atomic Mass"].split("(")[0])
        except (KeyError, ValueError):
            continue
        abundance_text = fields.get("Isotopic Composition", "").strip()
        abundance = float(abundance_text.split("(")[0]) if abundance_text else None
        records.append(dict(z=z, symbol=fields.get("Atomic Symbol", "").strip(),
                            a=mass_number, atomic_mass_u=atomic_mass_u,
                            abundance=abundance))
    constants = CODATA[DEFAULT_CODATA_VERSION]
    amu_per_me = constants["_amu"] / constants["_me"]
    for record in records:
        record["nuclear_mass_me"] = (PROTON_ELECTRON_MASS_RATIO
            if record["z"] == 1 and record["a"] == 1 else
            record["atomic_mass_u"] * amu_per_me - record["z"])
    return records


def parse_isotopes(source: str) -> dict[int, dict]:
    """Select the most abundant isotope while retaining its exact mass."""
    records = parse_all_isotopes(source)
    by_z: dict[int, list[dict]] = {}
    for record in records:
        by_z.setdefault(record["z"], []).append(record)
    selected = {}
    for z, isotopes in by_z.items():
        natural = [item for item in isotopes if item["abundance"] is not None]
        choice = (max(natural, key=lambda item: item["abundance"])
                  if natural else isotopes[0])
        selected[z] = dict(choice,
                           choice_basis=("most_abundant" if natural
                                         else "first_tabulated_no_abundance"))
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    source_path = args.output / "nist_isotopes_source.html"
    if not args.offline or not source_path.exists():
        if args.offline:
            parser.error(f"missing {source_path}")
        subprocess.run(["curl", "-fLsS", "-A", "Mozilla/5.0", URL,
                        "-o", str(source_path)], check=True)
    source = source_path.read_text()
    isotopes = parse_isotopes(source)
    all_isotopes = parse_all_isotopes(source)
    if set(isotopes) != set(range(1, 119)):
        parser.error(f"expected 118 elements; found {len(isotopes)}")
    payload = dict(source_url=URL, source_sha256=hashlib.sha256(
        source_path.read_bytes()).hexdigest(),
        mass_convention="atomic isotope mass minus Z electron masses; H-1 exact CODATA proton mass; electron binding mass omitted",
        selected={str(z): value for z, value in isotopes.items()},
        all_isotopes={f"{record['z']}:{record['a']}": record
                      for record in all_isotopes})
    (args.output / "isotopes.json").write_text(json.dumps(payload, indent=2))
    print(f"saved {len(isotopes)} isotope choices to {args.output / 'isotopes.json'}")


if __name__ == "__main__":
    main()
