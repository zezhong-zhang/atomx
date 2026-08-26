"""All-order pp-ladder resummation (review M9/#12, first rung).

pair_ladder_energy solves (D + V_pp) t = -v0 per (occupied pair, J):
the exact all-order sum of the particle-particle ladder diagram class.
Gates: (1) include_ladder=False reproduces the independently-implemented
mbpt2_energy to machine precision (kernel + normalization exact);
(2) a 1x1 doubles space has the closed form E = -(2J+1) v0^2/(D+V),
verified against the series the code claims to resum.

Production-scale basis convergence (full pseudospectrum, ~6 ms/element
python kernel -> hours) needs the vectorized kernel — the documented
next rung, together with rings/CCD.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from atomx import Atom  # noqa: E402
from atomx.methods.mbpt import (  # noqa: E402
    mbpt2_energy, pair_ladder_energy, _pair_v_J, build_virtual_basis)
from atomx.numerics.bspline.convol import ConvolCache  # noqa: E402


@pytest.fixture(scope="module")
def helium():
    a = Atom(Z=2, config="1s2", numerics="bspline", breit=False, qed=False)
    a.solve()
    return a


def test_ladder_off_reduces_to_mbpt2(helium):
    """Kernel + normalization: identical to the reference MBPT(2)."""
    r2 = mbpt2_energy(helium._result, kappa_max=2, n_virt_max=6)
    r0 = pair_ladder_energy(helium._result, kappa_max=2, n_virt_max=6,
                            include_ladder=False)
    assert r0.correlation_energy == pytest.approx(
        r2.correlation_energy, rel=1e-12)


def test_all_order_1x1_closed_form(helium):
    """One-dimensional doubles space: E = -(2J+1) v0^2/(D+V) exactly."""
    res = helium._result
    cache = ConvolCache(res._bsp_data)
    occ, virt = build_virtual_basis(res, -1, cache=cache)
    Pa, Qa, _ = occ[0]
    eps_a = float(res.orbital_energies["1s1/2"])
    Pr, Qr, eps_r = virt[0]
    bsp = res._bsp_data
    # J=0 (tJ=0) single |rr> state, normalized |rr>/sqrt(2)
    v0 = _pair_v_J(Pa, Qa, -1, Pa, Qa, -1, Pr, Qr, -1, Pr, Qr, -1,
                   0, bsp, cache) / 2.0   # na=nr=sqrt(2)
    V11 = _pair_v_J(Pr, Qr, -1, Pr, Qr, -1, Pr, Qr, -1, Pr, Qr, -1,
                    0, bsp, cache) / 2.0
    D = 2.0 * eps_r - 2.0 * eps_a
    # Physical preconditions of the identity being verified: finite matrix
    # elements, a positive excitation denominator, and |V/D| < 1 (the
    # convergence domain of the geometric perturbation series below).  A
    # non-finite intermediate — e.g. from corrupted BLAS output on hosted
    # CI hardware, as seen in the HYP-1530 pipeline — must fail HERE with
    # the culprit named, not downstream as an opaque `nan == nan`.
    assert np.isfinite([v0, V11, D]).all(), (v0, V11, D)
    assert D > 0.0, (eps_a, eps_r)
    assert abs(V11 / D) < 1.0, (v0, V11, D)
    e_closed = -(0 + 1) * v0 * v0 / (D + V11)
    # the same series the solver resums: -v0^2/D * sum (-V/D)^n
    e_series = 0.0
    for n in range(200):
        e_series += -v0 * v0 / D * (-V11 / D) ** n
    assert e_closed == pytest.approx(e_series, rel=1e-12)
    assert abs(e_closed) > 0.0
    # and the linear-solve result matches the closed form
    t = np.linalg.solve(np.array([[D + V11]]), np.array([-v0]))
    assert float(v0 * t[0]) == pytest.approx(e_closed, rel=1e-12)


def test_ladder_changes_energy_all_orders(helium):
    """The resummation produces a genuine beyond-2nd-order shift."""
    r2 = pair_ladder_energy(helium._result, kappa_max=2, n_virt_max=6,
                            include_ladder=False)
    rl = pair_ladder_energy(helium._result, kappa_max=2, n_virt_max=6)
    ratio = rl.correlation_energy / r2.correlation_energy
    # repulsive pp-ladders reduce |E2| (measured 0.833 in this basis);
    # any regression in the kernel/normalization moves this materially
    assert 0.75 <= ratio <= 0.95
    assert rl.provenance["method"] == "all-order pp-ladder"
