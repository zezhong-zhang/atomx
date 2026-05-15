from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "benchmarks"))

from fac_runner import parse_ci  # noqa: E402


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

