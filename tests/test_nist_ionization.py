"""Parse the official ionization table without dropping its notation."""

import pytest

from scripts.import_nist_ionization import parse


def test_parenthesized_energy_and_uncertainty_are_preserved():
    source = b"""<html><pre>Ionization Energy (eV)
      1 | H I | 0 | Hydrogen | H | 1s | 1s | 2S&lt;1/2> | | <a>(</a>13.5984<a>)</a> | 0.0001 | <a>HDEL</a>
      2 | He I | 0 | Helium | He | 1s2 | 1s2 | 1S0 | 1s | 24.587 | 0.001 | L17714
    </pre></html>"""
    rows = parse(source, min_rows=2)
    assert rows[0][:3] == (1, 0, "H I")
    assert rows[0][7] == "2S<1/2>"
    assert rows[0][9:13] == (pytest.approx(13.5984), "(13.5984)",
                            pytest.approx(0.0001), "0.0001")
    assert rows[0][13] == "HDEL"
    assert rows[1][9] == pytest.approx(24.587)


def test_duplicate_charge_is_rejected():
    source = b"""<pre>Ionization Energy
    1 | H I | 0 | Hydrogen | H | 1s | 1s | 2S | | 13.6 | 0.1 | R
    1 | H I | 0 | Hydrogen | H | 1s | 1s | 2S | | 13.6 | 0.1 | R
    </pre>"""
    with pytest.raises(ValueError, match="duplicate"):
        parse(source, min_rows=1)
