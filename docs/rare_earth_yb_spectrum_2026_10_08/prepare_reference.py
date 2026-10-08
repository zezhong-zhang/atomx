"""Extract explicitly selected NIST handbook data, preserving reference type."""
import hashlib
import json
from pathlib import Path
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parent
REF = ROOT / "references"


def rows(file):
    soup = BeautifulSoup((REF / file).read_text(), "html.parser")
    for tr in soup.select("tr"):
        cells = [td.get_text(" ", strip=True) for td in tr.find_all("td", recursive=False)]
        yield cells


levels = []
configuration = term = ""
for cells in rows("nist_ybii_levels.html"):
    if len(cells) != 5:
        continue
    if cells[0]:
        configuration = cells[0]
    if cells[1]:
        term = cells[1]
    if not configuration.startswith("4 f 1 4") or cells[4] != "M67":
        continue
    if "( 1 S)6 s" in configuration:
        label = "6s_1/2"
    elif "( 1 S)5 d" in configuration:
        label = "5d_" + cells[2]
    elif "( 1 S)6 p" in configuration:
        label = "6p_" + cells[2]
    else:
        continue
    levels.append({"label": label, "configuration_raw": configuration, "term_raw": term,
        "two_J": int(2 * float(cells[2].split('/')[0]) / float(cells[2].split('/')[1])),
        "parity": -1 if "°" in term else 1, "excitation_cm1": float(cells[3]),
        "uncertainty_cm1": None, "uncertainty_status": "not supplied on handbook table; printed digits are not an uncertainty",
        "source_ref": "M67", "source_url": "https://physics.nist.gov/PhysRefData/Handbook/Tables/ytterbiumtable6.htm"})
assert {x["label"] for x in levels} == {"6s_1/2", "5d_3/2", "5d_5/2", "6p_1/2", "6p_3/2"}
energy = {x["label"]: x["excitation_cm1"] for x in levels}
assert energy == {"6s_1/2": 0., "5d_3/2": 22960.8, "5d_5/2": 24332.69, "6p_1/2": 27061.82, "6p_3/2": 30392.23}

observed = {}
strong_text = BeautifulSoup((REF / "nist_yb_air_lines.html").read_text(), "html.parser").get_text(" ", strip=True)
assert "Air" in strong_text
for cells in rows("nist_ybii_persistent.html"):
    if len(cells) != 9 or cells[1] not in ("3289.370", "3694.190"):
        continue
    label = "6p_3/2" if cells[1] == "3289.370" else "6p_1/2"
    assert cells[1] in strong_text
    observed[label] = {"lower": "6s_1/2", "upper": label,
        "observed_wavelength_air_nm": float(cells[1])/10,
        "observed_wavelength_uncertainty_nm": None, "wavelength_ref": cells[7],
        "compiled_A_per_s": float(cells[2])*1e8, "A_ref": cells[8],
        "A_reference_type": "Morton 2000 compilation; not claimed as a new direct experiment",
        "source_url": "https://physics.nist.gov/PhysRefData/Handbook/Tables/ytterbiumtable4.htm"}
assert len(observed) == 2

pairs = [("6s_1/2", "6p_1/2", "E1"), ("6s_1/2", "6p_3/2", "E1"),
    ("5d_3/2", "6p_1/2", "E1"), ("5d_3/2", "6p_3/2", "E1"),
    ("5d_5/2", "6p_3/2", "E1"), ("6s_1/2", "5d_3/2", "E2"),
    ("6s_1/2", "5d_5/2", "E2")]
lines = []
for lower, upper, mp in pairs:
    difference = energy[upper]-energy[lower]
    lines.append({"lower": lower, "upper": upper, "multipole": mp,
        "reference_wavenumber_cm1": difference, "reference_Ritz_vacuum_nm": 1e7/difference,
        "reference_type": "Ritz wavelength derived here from measured handbook level energies; not an observed wavelength",
        "observed_air_record": observed.get(upper) if lower == "6s_1/2" else None})

lifetimes = [
    {"state": "6p_1/2", "isotope": 174, "lifetime_ns": 8.12, "uncertainty_ns": .02,
     "type": "direct single trapped-ion experiment", "source_url": "https://doi.org/10.1103/PhysRevA.80.022502",
     "open_source_url": "https://arxiv.org/abs/0906.0586"},
    {"state": "5d_3/2", "isotope": 174, "lifetime_ms": 54.83, "uncertainty_ms": .18,
     "type": "direct single trapped-ion experiment", "source_url": "https://doi.org/10.1103/PhysRevResearch.5.023193"},
]

output = {"system": "Yb II / Yb+ free ion", "retrieval_date": "2026-10-08",
    "reference_scope": "Explicit five low levels from NIST handbook, two measured air resonance wavelengths, and two direct lifetime experiments; not a complete/current-isotope-resolved ASD atlas",
    "levels": levels, "lines": lines, "lifetimes": lifetimes,
    "reference_file_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in REF.iterdir() if p.is_file()}}
(ROOT / "experimental_reference.json").write_text(json.dumps(output, ensure_ascii=False, indent=2)+"\n")
print(json.dumps({"n_levels": len(levels), "n_lines": len(lines), "n_observed_lines": len(observed), "n_lifetimes": len(lifetimes)}))
