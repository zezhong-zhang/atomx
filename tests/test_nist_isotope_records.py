"""NIST's isotope holdings and direct exports stay distinct from standard ions."""

import csv
import io
import sqlite3

from scripts.audit_nist_isotope_holdings import parse_holdings
from scripts.import_nist_isotope_records import (SCHEMA, import_one,
                                                 isotope_identity, parse_records)
from scripts.view_nist_spectra import isotope_record_page


def test_isotope_holdings_parse_malformed_nist_rows_and_empty_element():
    body = ("<title>NIST ASD Holdings</title>No. of lines"
            "<tr /><td><b><a class='bib' id=\"aa\" name=\"H%20I\">H I</a></b></td>"
            "<td>&nbsp;</td><td align='right'><b>568</b></td><td>&nbsp;</td>"
            "<td>500</td><td>&nbsp;</td><td>568</td>"
            "<tr /><td><b><a class='bib' id=\"aa\" name=\"D%20I\">D I</a></b></td>"
            "<td>&nbsp;</td><td align='right'><b>364</b></td><td>&nbsp;</td>"
            "<td>360</td><td>&nbsp;</td><td>364</td><tr />").encode()
    rows = parse_holdings(body, "lines")
    assert [(r["spectrum"], r["count"], r["isotope_specific"])
            for r in rows] == [("H I", 568, False), ("D I", 364, True)]
    assert parse_holdings(b"<title>NIST ASD Holdings</title>No. of levels", "levels") == []


def test_direct_isotope_rows_match_holdings_and_retain_all_columns(tmp_path):
    assert isotope_identity("D I", {"H": 1}) == (1, 2, 0)
    assert isotope_identity("13C II", {"C": 6}) == (6, 13, 1)
    stream = io.StringIO()
    writer = csv.writer(stream, delimiter="\t")
    writer.writerow(["obs_wl_vac(nm)", "ritz_wl_vac(nm)", "intens", "line_ref"])
    writer.writerow(["121.533", "121.534", "100", "L1"])
    body = stream.getvalue().encode()
    assert len(parse_records(body, "lines")) == 1
    db = sqlite3.connect(tmp_path / "isotopes.sqlite")
    db.row_factory = sqlite3.Row
    db.executescript(SCHEMA)
    ok, _ = import_one(db, "lines", "D I", 1, 2, 0, 1,
                       "https://physics.nist.gov/example", body)
    assert ok
    page = isotope_record_page(db, "lines", "D I", 1, 100)
    assert page["total"] == 1
    assert page["rows"][0]["line_ref"] == "L1"
    assert page["source"]["mass_number"] == 2
    db.close()
