"""Catalog integrity: the atlas must expose missing data and preserve source rows."""

from __future__ import annotations

import importlib.util
import json
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "build_nist_spectra", ROOT / "scripts" / "build_nist_spectra.py")
catalog = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(catalog)


def test_panel_files_build_a_source_traceable_catalog(tmp_path):
    files = []
    index = {"lines": set(), "levels": set()}
    for kind in index:
        name = "asd.001000"
        index[kind].add(name)
        item = catalog._one_file(tmp_path, kind, name, offline=True)
        assert "error" not in item
        files.append(item)

    summary, species = catalog._build_database(tmp_path, index, files)
    assert summary["complete_for_indexed_archive"]
    assert summary["candidate_ions"] == 7021
    assert summary["usable_wavelengths"] > 0
    assert species[0]["label"] == "H I"
    assert species[0]["line_status"] == "parsed"
    assert next(s for s in species if s["z"] == 118 and s["charge"] == 0)[
        "line_status"] == "not_in_archive"

    with sqlite3.connect(tmp_path / "spectra.sqlite") as db:
        assert db.execute("SELECT COUNT(*) FROM lines").fetchone()[0] == summary["lines"]
        assert db.execute("SELECT COUNT(*) FROM levels").fetchone()[0] == summary["levels"]
        assert db.execute("SELECT COUNT(*) FROM lines WHERE wavelength_nm>0").fetchone()[0] == summary["usable_wavelengths"]
        assert db.execute("SELECT COUNT(*) FROM source_files WHERE sha256 IS NULL").fetchone()[0] == 0
    manifest = json.loads((tmp_path / "source_manifest.json").read_text())
    assert all(len(f["sha256"]) == 64 and f["url"].startswith(catalog.BASE)
               for f in manifest)

    holdings = "\n".join(
        f'<a title="{catalog.ion_label(z, charge)}: '
        f'{summary["lines"] if (z, charge) == (1, 0) else 0} lines"></a>'
        for z in range(1, 111) for charge in range(z))
    holdings += "\n<h3>Current total number of lines: 700</h3>"
    holdings += "\n<h3>Current total number of lines with transition probabilities: 300</h3>"
    (tmp_path / "official_lines_holdings.html").write_text(holdings)
    audited = catalog.verify_official_holdings(tmp_path, summary, offline=True)
    assert audited["line_holdings_verified"]
    assert audited["official_ion_cells"] == 6105
    assert audited["official_headline_minus_cell_sum"] == 700 - summary["lines"]


def test_invalid_nist_identity_is_rejected(tmp_path):
    name = "asd.001000"
    body = (ROOT / "benchmarks" / "nist_asd" / "lines" / name).read_bytes()
    try:
        catalog._valid(body, "lines", 2, 0)
    except ValueError as exc:
        assert "mismatch" in str(exc)
    else:
        raise AssertionError("source label mismatch was accepted")


def test_hydrogen_wavelength_and_scientific_aki_columns_are_aligned():
    lines, problems = catalog.nist_asd.parse_lines_text(
        (ROOT / "benchmarks" / "nist_asd" / "lines" / "asd.001000").read_text(),
        ion="H I", z=1, charge=0)
    assert not problems
    assert any(abs(line.observed_nm - 121.56699) < 1e-6 and
               6.2e8 < line.aki < 6.3e8
               for line in lines if line.observed_nm and line.aki)
    assert any(abs(line.observed_nm - 656.46) < 0.01
               for line in lines if line.observed_nm)
    assert any(line.ritz_nm and line.ritz_unc_nm and not line.observed_nm
               for line in lines)


def test_unclassified_nist_line_table_is_retained():
    # This NIST table has no configuration columns; its wavelengths are still
    # part of the measured line holdings and must survive the import.
    text = """<pre>
Observed | Ritz | Aki | Acc. | Ei Ek |Type| TP | Line |
Wavelength | Wavelength | (s^-1) | |(cm-1)| |Ref.|Ref.|
Vac (nm) | Vac (nm) | | | | | | |
96.926 | | | | | | | L7485 |
</pre>"""
    lines, problems = catalog.nist_asd.parse_lines_text(
        text, ion="Cr III", z=24, charge=2)
    assert not problems
    assert len(lines) == 1
    assert lines[0].observed_nm == 96.926
    assert lines[0].line_reference == "L7485"
    assert lines[0].lower_configuration == ""


def test_nist_configuration_table_without_j_columns_is_retained():
    text = """<pre>
Observed | Unc. | Ritz | Aki | Acc. | Ei Ek | Lower | Upper |Type| TP | Line |
Wavelength | (nm) | Wavelength | (s^-1) | |(cm-1)|---|---| | Ref. | Ref. |
Vac (nm) | | Vac (nm) | | | |Conf.|Term|Conf.|Term| | | |
0.9786 | 0.0005 | | | | | 3d10.4s2.4p2 | | 3d9.4s2.4p2.6f | * | | | L11790 |
</pre>"""
    lines, problems = catalog.nist_asd.parse_lines_text(
        text, ion="Ba XXV", z=56, charge=24)
    assert not problems
    assert len(lines) == 1
    assert lines[0].observed_nm == 0.9786
    assert lines[0].upper_term == "*"


def test_official_holdings_use_per_ion_cells_even_when_headline_differs():
    page = """<a title="H I: 568 lines"></a>
    <a title="Fe XVII: 99 lines"></a>
    <h3>Current total number of lines: 700</h3>
    <h3>Current total number of lines with transition probabilities: 300</h3>"""
    cells, headline, headline_aki = catalog.parse_official_holdings(
        page, min_cells=2)
    assert cells == {(1, 0): 568, (26, 16): 99}
    assert sum(cells.values()) == 667
    assert headline == 700
    assert headline_aki == 300


def test_level_import_keeps_nist_provenance_and_composition_columns():
    source = """<PRE>
Configuration | Term | J | Level (cm-1) | Uncertainty (cm-1) | Land&eacute; | Leading percentages | Reference
--------------|------|---|--------------|--------------------|--------------|---------------------|----------
1s2           | 1S   | 0 | 0            | 0                  | 0.000        | 96 : 4 other        | L1
1s.2s         | 3S   | 1 | [100.2]      | 0.5                | 2.01:        | 77 : 23 other       | L2
1s.3s         | 1S   | 0 | (200.4)      | 1.0                |              |                     | L3
</PRE>"""
    rows, problems = catalog.nist_asd.parse_levels_text(
        source, ion="He I", z=2, charge=0)
    assert not problems
    assert [row.energy_kind for row in rows] == [
        "evaluated", "semiempirical", "theoretical"]
    assert rows[1].energy_raw == "[100.2]"
    assert rows[1].lande_g_raw == "2.01:"
    assert rows[1].leading_percentages == "77 : 23 other"
