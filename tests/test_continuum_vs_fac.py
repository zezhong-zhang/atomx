"""Side-by-side continuum waves vs FAC WaveFuncTable dumps (2026-07-16).

Fixtures: FAC 1.1.5 free-wave dumps (He-like Fe and Xe ground potentials,
OptimizeRadial defaults) — columns i, r, Vc*r, U*r, P, Q up to ilast.

Physics pinned by the audit:
* the pointwise ratio P_atomx/P_FAC is CONSTANT over r (shape/phase
  agreement; 0.001% in the exact-potential comparison, <= 1% here where
  the (N-1)/N potential differs slightly from FAC's optimized one);
* the ratio equals sqrt(2/pi) at ALL energies (100 eV .. 200 keV):
  FAC's 1/sqrt(k) amplitude convention carries the SAME relativistic
  (E+2c^2) factor as Grant's delta(E) normalization, so the campaign's
  sqrt(pi/2) per-wave conversion is exact at every energy.  The
  pre-2026-07-16 nonrelativistic normalization violated this at high E
  (the closed Z=54 high-energy deficit).
"""

from __future__ import annotations

import contextlib
import re
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from atomx import Atom  # noqa: E402

HEV = 27.211386245988
FIXDIR = Path(__file__).parent / "data" / "facwave"

CASES = [
    ("Fe_k-1_e1000.txt", 26),
    ("Fe_k-1_e40000.txt", 26),
    ("Fe_k-5_e10000.txt", 26),
    ("Xe_k-1_e200000.txt", 54),
]


def _parse(fn):
    hdr, rows = {}, []
    with open(fn) as handle:
        lines = handle.readlines()
    for line in lines:
        if line.startswith("#"):
            m = re.match(r"#\s*(\w+)\s*=\s*(\S+)", line)
            if m:
                with contextlib.suppress(ValueError):
                    hdr[m.group(1)] = float(m.group(2))
            continue
        p = line.split()
        if len(p) >= 6:
            rows.append([float(x) for x in p[:6]])
    il = int(hdr["ilast"])
    a = np.array(rows[:il + 1])
    return hdr, a[:, 1], a[:, 4]


@pytest.fixture(scope="module")
def solved():
    out = {}
    for Z in (26, 54):
        a = Atom(Z=Z, config="1s2", charge=Z - 2, numerics="bspline", breit=False, qed=False)
        a.solve()
        out[Z] = a
    return out


@pytest.mark.parametrize("fname,Z", CASES)
def test_wave_matches_fac(solved, fname, Z):
    from atomx.numerics.quadrature import p_rel
    from atomx.numerics.shooting.dirac_continuum import solve_dirac_continuum_shooting
    from atomx.scattering.distorted_wave.solver import (
        _distorting_potential_fn,
        _shooting_points,
    )

    hdr, r, P = _parse(FIXDIR / fname)
    E = hdr["energy"] / HEV
    kappa = int(hdr["kappa"])
    a = solved[Z]
    V = _distorting_potential_fn(a._result, a._bsp_data, Z, screening="fac")
    r_max = float(a._bsp_data["t"][-1])
    n = _shooting_points(r_max, p_rel(E))
    rs, Ps, _Qs, _ = solve_dirac_continuum_shooting(
        E, kappa, V, r_max=r_max, n_grid=n, Z_residual=Z - 1.0)
    Pi = np.interp(r, rs, Ps)
    m = np.abs(P) > 0.05 * np.abs(P).max()
    ratio = Pi[m] / P[m]
    med = float(np.median(ratio))
    rms = float(np.std(ratio) / abs(med))
    # normalization: sqrt(2/pi) within 1.5% (potential-difference level)
    assert med == pytest.approx(np.sqrt(2.0 / np.pi), rel=0.015)
    # shape: pointwise-constant ratio within 1.5%
    assert rms < 0.015
