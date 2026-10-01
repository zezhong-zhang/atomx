from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "benchmarks"))

from fac_runner import (  # noqa: E402
    parse_ci,
    radial_grid_command,
    require_sfac_bin,
    resolve_sfac_bin,
    run_sfac,
)


def test_fac_grid_refinement_changes_density_not_only_allocation():
    import ast
    def settings(points, density):
        call = ast.parse(radial_grid_command(points, density_scale=density), mode='eval').body
        return [ast.literal_eval(arg) for arg in call.args]
    base = settings(2000, 1.)
    larger = settings(4000, 1.)
    refined = settings(4000, 2.)
    assert larger[1:] == base[1:]
    assert np.log(refined[1]) == pytest.approx(np.log(base[1])/2)
    assert refined[2] == 2*base[2]
    assert refined[3] == base[3]  # same nuclear-scaled inner radius


def test_parse_ci_uses_third_column_as_cross_section(tmp_path):
    ci = tmp_path / "sample.ci"
    ci.write_text(
        "\n".join([
            "NUSR\t= 2",
            "     0  0      1  3  1.2602E+03 2001",
            " 1.3583E-01  6.9466E-01 -2.8153E-01  4.4758E-01 ",
            " 5.0000E+02  1.0935E-01  2.3628E+00",
            " 9.0000E+02  1.8881E-01  3.3231E+00",
        ])
    )

    rows = parse_ci(ci)

    assert len(rows) == 1
    np.testing.assert_allclose(rows[0]["strength"], [0.10935, 0.18881])
    np.testing.assert_allclose(rows[0]["sigma_1e20cm2"], [2.3628, 3.3231])



def test_sfac_resolution_uses_fac_bin_dir(monkeypatch, tmp_path):
    monkeypatch.delenv("FAC_SFAC_PATH", raising=False)
    monkeypatch.delenv("FAC_BIN", raising=False)
    monkeypatch.setenv("FAC_BIN_DIR", str(tmp_path))
    monkeypatch.setenv("PATH", "")

    assert resolve_sfac_bin() == tmp_path / "sfac"


def test_run_sfac_reports_clear_missing_binary(monkeypatch, tmp_path):
    missing = tmp_path / "missing" / "sfac"
    monkeypatch.setenv("FAC_SFAC_PATH", str(missing))
    monkeypatch.delenv("FAC_BIN", raising=False)
    monkeypatch.delenv("FAC_BIN_DIR", raising=False)
    monkeypatch.setenv("PATH", "")
    script = tmp_path / "run.sf"
    script.write_text("SetAtom('Ne', 10)\n")

    with pytest.raises(FileNotFoundError, match="sfac binary not found") as exc:
        run_sfac(script, workdir=tmp_path)

    assert "FAC_SFAC_PATH" in str(exc.value)
    with pytest.raises(FileNotFoundError, match="sfac binary not found"):
        require_sfac_bin()
