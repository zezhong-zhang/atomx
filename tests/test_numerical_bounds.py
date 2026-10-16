"""Explicit bounds diagnostic; fresh compilation is outside routine CI."""

from test_determinism import _run


def test_panel_has_no_out_of_bounds_reads(tmp_path):
    """Catch reads which otherwise return heap garbage or a poisoned NaN."""
    assert len(_run(tmp_path / "numba-bounds", boundscheck=True)) == 3
