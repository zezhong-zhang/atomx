"""RPA/TDHF gates.

G1 ANALYTIC 1x1: omega^2 = (a+b)(a-b), alpha = (2/3)d^2/(a+b); the
   generalized-eigenvalue trap (ratio instead of product) flips the
   screening sign — this gate pins the symmetric route.
G2 RANDOM 2x2/3x3: RPA equations satisfied columnwise; X^T X - Y^T Y
   = identity; poles real positive.
G3 Ne INTEGRATION: DF workspace, screening direction (alpha_RPA <
   alpha_DF), poles real positive and split.  Absolute alpha is
   virtual-orbital-quality limited (recorded) — NOT gated on
   experiment yet.
"""
import sys

import numpy as np
import pytest

sys.path.insert(0, "/tmp/atomx-cimbpt")

from atomx.methods.rpa import dipole_polarizability, solve_rpa, \
    solve_symmetric_rpa


def test_g1_analytic_one_by_one():
    a, b, d = 0.75, 0.12, 0.5
    w, X, Y = solve_symmetric_rpa(np.array([[a]]), np.array([[b]]))
    assert w[0] == pytest.approx(np.sqrt((a + b) * (a - b)), rel=1e-10)
    amp = float((X[0, 0] + Y[0, 0]) * d)
    alpha = 2.0 / 3.0 * amp ** 2 / w[0]
    assert alpha == pytest.approx(2.0 / 3.0 * d ** 2 / (a + b),
                                  rel=1e-10)
    # screening: b>0 must REDUCE the response vs DF (alpha_DF ~ 2/3 d^2/a)
    assert alpha < 2.0 / 3.0 * d ** 2 / a


def test_g2_random_blocks():
    rng = np.random.default_rng(7)
    for n in (2, 3):
        V = rng.normal(size=(n, n)) * 0.3
        W = rng.normal(size=(n, n)) * 0.3
        A = np.diag(np.linspace(0.5, 1.5, n)) + 0.5 * (V + V.T)
        B = 0.5 * (W + W.T) * 0.12
        w, X, Y = solve_symmetric_rpa(A, B)
        assert np.all(w > 0)
        # RPA equations columnwise: A(X+Y) ... via the defining relations
        x, y = X + Y, X - Y
        for k in range(len(w)):
            lhs = (A - B) @ y[:, k]
            rhs = w[k] * x[:, k]
            assert np.allclose(lhs, rhs, atol=1e-8)
            assert np.allclose((A + B) @ x[:, k], w[k] * y[:, k],
                               atol=1e-8)
        # orthonormalisation X^T X - Y^T Y = I
        assert np.allclose(X.T @ X - Y.T @ Y, np.eye(len(w)),
                           atol=1e-8)


@pytest.fixture(scope="module")
def ne_rpa():
    from atomx.methods.dirac_fock import DFSolver, Subshell
    import re as _re
    occ = [(1, -1), (2, -1), (2, 1), (2, -2)]
    virt = [(3, -1), (3, 1), (3, -2), (3, 2), (3, -3), (4, -1),
            (4, 1), (4, -2), (4, 2), (4, -3), (4, 3), (4, -4),
            (5, -1), (5, 1), (5, -2)]
    subs = [Subshell(n=n, kappa=k,
                     occupation=float(2 * k if k > 0 else -2 * k))
            for n, k in occ]
    subs += [Subshell(n=n, kappa=k, occupation=1e-6) for n, k in virt]
    dhf = DFSolver(Z=10, subshells=subs, ns=40, ks=7,
                   starting_potential="tfda")
    r = dhf.solve()
    od = {(n, k): (P, Q) for (n, k), (P, Q, _o)
          in r.orbitals_by_subshell().items()}
    eps = {}
    for lab, e in r.orbital_energies.items():
        m = _re.fullmatch(r"(\d+)([spdf])(\d+)/(\d+)", lab)
        n = int(m.group(1))
        ll = "spdf".index(m.group(2))
        j2 = int(round(2 * int(m.group(3)) / int(m.group(4))))
        eps[(n, -(ll + 1) if j2 == 2 * ll + 1 else ll)] = e
    occ_keys = [k for k in occ if k in od]
    virt_keys = [k for k in virt if k in od and k in eps]
    res = solve_rpa(od, dhf.bsp_data, eps, occ_keys, virt_keys,
                    twoJ=2, exc_parity=-1)
    return od, dhf.bsp_data, eps, res


def test_g3_ne_screening_direction(ne_rpa):
    od, bsp, eps, res = ne_rpa
    assert np.all(res.omega_au > 0)
    assert len(res.omega_au) >= 10
    alpha_rpa = dipole_polarizability(res, od, bsp)
    from atomx.methods.rpa import reduced_dipole
    d = np.array([reduced_dipole(od, bsp, p, h)
                  for p, h in res.ph_pairs])
    de = np.array([eps[p] - eps[h] for p, h in res.ph_pairs])
    alpha_df = float(np.sum(2.0 / 3.0 * d ** 2 / de))
    assert alpha_rpa < alpha_df, (alpha_rpa, alpha_df)
    assert 0.5 < alpha_rpa / alpha_df < 1.0
