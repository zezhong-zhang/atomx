"""RPA/TDHF gates.

G1 ANALYTIC 1x1: omega^2 = (a+b)(a-b), alpha = (2/3)d^2/(a+b); the
   generalized-eigenvalue trap (ratio instead of product) flips the
   screening sign — this gate pins the symmetric route.
G2 RANDOM 2x2/3x3: RPA equations satisfied columnwise; X^T X - Y^T Y
   = identity; poles real positive.
G3 Ne INTEGRATION: DF workspace, response equations, real positive poles.
   The direction of the response correction is not presumed. Absolute alpha is
   virtual-orbital-quality limited (recorded) — NOT gated on
   experiment yet.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from atomx.methods.rpa import dipole_polarizability, solve_rpa, solve_symmetric_rpa


def test_g1_analytic_one_by_one():
    a, b, d = 0.75, 0.12, 0.5
    w, X, Y = solve_symmetric_rpa(np.array([[a]]), np.array([[b]]))
    assert w[0] == pytest.approx(np.sqrt((a + b) * (a - b)), rel=1e-10)
    amp = float((X[0, 0] + Y[0, 0]) * d)
    alpha = 2.0 / 3.0 * amp**2 / w[0]
    assert alpha == pytest.approx(2.0 / 3.0 * d**2 / (a + b), rel=1e-10)
    # screening: b>0 must REDUCE the response vs DF (alpha_DF ~ 2/3 d^2/a)
    assert alpha < 2.0 / 3.0 * d**2 / a


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
            assert np.allclose((A + B) @ x[:, k], w[k] * y[:, k], atol=1e-8)
        # orthonormalisation X^T X - Y^T Y = I
        assert np.allclose(X.T @ X - Y.T @ Y, np.eye(len(w)), atol=1e-8)


@pytest.fixture(scope="module")
def ne_rpa():
    import re as _re

    from atomx.methods.dirac_fock import DFSolver, Subshell

    occ = [(1, -1), (2, -1), (2, 1), (2, -2)]
    virt = [
        (3, -1),
        (3, 1),
        (3, -2),
        (3, 2),
        (3, -3),
        (4, -1),
        (4, 1),
        (4, -2),
        (4, 2),
        (4, -3),
        (4, 3),
        (4, -4),
        (5, -1),
        (5, 1),
        (5, -2),
    ]
    subs = [
        Subshell(n=n, kappa=k, occupation=float(2 * k if k > 0 else -2 * k))
        for n, k in occ
    ]
    subs += [Subshell(n=n, kappa=k, occupation=1e-6) for n, k in virt]
    dhf = DFSolver(Z=10, subshells=subs, ns=40, ks=7, starting_potential="tfda")
    r = dhf.solve()
    od = {(n, k): (P, Q) for (n, k), (P, Q, _o) in r.orbitals_by_subshell().items()}
    eps = {}
    for lab, e in r.orbital_energies.items():
        m = _re.fullmatch(r"(\d+)([spdf])(\d+)/(\d+)", lab)
        n = int(m.group(1))
        ll = "spdf".index(m.group(2))
        j2 = int(round(2 * int(m.group(3)) / int(m.group(4))))
        eps[(n, -(ll + 1) if j2 == 2 * ll + 1 else ll)] = e
    occ_keys = [k for k in occ if k in od]
    virt_keys = [k for k in virt if k in od and k in eps]
    res = solve_rpa(od, dhf.bsp_data, eps, occ_keys, virt_keys, twoJ=2, exc_parity=-1)
    return od, dhf.bsp_data, eps, res


def test_g3_ne_response_equations(ne_rpa):
    od, bsp, eps, res = ne_rpa
    assert np.all(res.omega_au > 0)
    assert len(res.omega_au) >= 10
    alpha_rpa = dipole_polarizability(res, od, bsp)
    from atomx.methods.rpa import reduced_dipole

    d = np.array([reduced_dipole(od, bsp, p, h) for p, h in res.ph_pairs])
    de = np.array([eps[p] - eps[h] for p, h in res.ph_pairs])
    alpha_df = float(np.sum(2.0 / 3.0 * d**2 / de))
    # Positivity alone does not constrain the sign of the TDHF correction.
    assert alpha_rpa > 0 and alpha_df > 0
    assert np.allclose(res.A @ res.X + res.B @ res.Y, res.X * res.omega_au, atol=1e-10)
    assert np.allclose(-res.B @ res.X - res.A @ res.Y, res.Y * res.omega_au, atol=1e-10)


def test_g4_ne_dense_continuum_completeness():
    """G4: the virtual-orbital fix.  Dense continuum pseudostates in
    the full V^N field restore the response completeness: TRK jumps
    from 0.08 (bound virtuals) to O(10), alpha_DF into the right
    order (experiment 2.669), with a finite-basis response magnitude gate."""
    import re as _re

    from atomx.methods.dirac_fock import DFSolver, Subshell
    from atomx.methods.rpa import (
        dense_continuum_virtuals,
        reduced_dipole,
        solve_rpa,
        trk_sum,
    )

    occ = [(1, -1), (2, -1), (2, 1), (2, -2)]
    subs = [
        Subshell(n=n, kappa=k, occupation=float(2 * k if k > 0 else -2 * k))
        for n, k in occ
    ]
    dhf = DFSolver(Z=10, subshells=subs, ns=40, ks=7, starting_potential="tfda")
    r = dhf.solve()
    od = {(n, k): (P, Q) for (n, k), (P, Q, _o) in r.orbitals_by_subshell().items()}
    eps = {}
    for lab, e in r.orbital_energies.items():
        m = _re.fullmatch(r"(\d+)([spdf])(\d+)/(\d+)", lab)
        n = int(m.group(1))
        ll = "spdf".index(m.group(2))
        j2 = int(round(2 * int(m.group(3)) / int(m.group(4))))
        eps[(n, -(ll + 1) if j2 == 2 * ll + 1 else ll)] = e
    vod, veps = dense_continuum_virtuals(r)
    od.update(vod)
    eps.update(veps)
    assert len(vod) > 40
    res = solve_rpa(
        od,
        dhf.bsp_data,
        eps,
        list({(n, k) for n, k in od if (n, k) in eps} - set(vod)),
        list(vod),
        twoJ=2,
        exc_parity=-1,
    )
    trk = trk_sum(res, od, dhf.bsp_data, eps)
    alpha_rpa = dipole_polarizability(res, od, dhf.bsp_data)
    d = np.array([reduced_dipole(od, dhf.bsp_data, p, h) for p, h in res.ph_pairs])
    de = np.array([eps[p] - eps[h] for p, h in res.ph_pairs])
    alpha_df = float(np.sum(2.0 / 3.0 * d**2 / de))
    assert trk > 8.0, trk  # completeness restored
    assert 1.4 < alpha_df < 2.6, alpha_df  # right order of magnitude
    # Finite ns=40 regression of the corrected particle-hole/V^N calculation.
    # TDHF response increases here; an assumed screening direction hid the bug.
    assert alpha_rpa == pytest.approx(2.374467681, abs=0.005)
    assert np.allclose(res.A, res.A.T, atol=1e-12)
    assert np.allclose(res.B, res.B.T, atol=1e-12)


# ---------------------------------------------------------------------------
# He exact-TDA anchors: use a closed-shell reference and its canonical V^N
# virtual complement. Orbital multipliers from a separately optimized, tiny
# occupation 2s are not eigenvalues of that common Fock operator and cannot
# supply the orbital-energy difference in the canonical TDHF equations.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def he_anchor():
    from atomx.config.csf import enumerate_jj_csfs_many
    from atomx.methods.dirac_fock import DFSolver, Subshell
    from atomx.methods.mbpt import _build_closed_shell_fock, build_virtual_basis
    from atomx.physics.csf import sparse_ci as SC
    from atomx.physics.csf.sparse_ci import patterns_from_csfs

    dhf = DFSolver(
        Z=2,
        subshells=[Subshell(n=1, kappa=-1, occupation=2.0)],
        ns=40,
        ks=7,
        starting_potential="tfda",
    )
    r = dhf.solve()
    od = {(n, k): (P, Q) for (n, k), (P, Q, _o) in r.orbitals_by_subshell().items()}
    _, virtuals = build_virtual_basis(
        r, -1, potential_mode="VN", orthogonal_to_occupied=True
    )
    P, Q, virtual_energy = virtuals[0]
    od[(2, -1)] = (P, Q)
    fock, _ = _build_closed_shell_fock(r, -1, potential_mode="VN")
    eps = {key: float(np.r_[P, Q] @ fock @ np.r_[P, Q]) for key, (P, Q) in od.items()}
    assert eps[(2, -1)] == pytest.approx(virtual_energy, abs=1e-8)
    cfgs = ["1s2", "1s1 2s1"]
    pats = patterns_from_csfs(enumerate_jj_csfs_many(cfgs))
    masks, spinors = SC.enumerate_determinants(pats, od, 0)
    H = SC.assemble_sparse_hamiltonian(masks, spinors, od, dhf.bsp_data, 2, 40)
    w = np.linalg.eigvalsh(H.toarray())
    de = eps[(2, -1)] - eps[(1, -1)]
    return od, dhf.bsp_data, eps, w, de


def test_he_tda_anchor_J0(he_anchor):
    od, bsp, eps, w, de = he_anchor
    res = solve_rpa(od, bsp, eps, [(1, -1)], [(2, -1)], twoJ=0, exc_parity=1)
    assert res.A[0, 0] - de == pytest.approx(w[2] - w[0] - de, abs=1e-9)


def test_he_tda_anchor_J1(he_anchor):
    od, bsp, eps, w, de = he_anchor
    res = solve_rpa(od, bsp, eps, [(1, -1)], [(2, -1)], twoJ=2, exc_parity=1)
    assert res.A[0, 0] - de == pytest.approx(w[1] - w[0] - de, abs=1e-9)


@pytest.mark.parametrize("twoJ", [0, 2])
def test_he_backward_kernel_matches_double_excitation(he_anchor, twoJ):
    """A ph J=1 backward amplitude need not vanish like a pp J=1 pair.

    For these two s subshells, both ph channels recouple the allowed J=0
    ground-to-double-excitation matrix element, with opposite phases.
    """
    from atomx.config.csf import enumerate_jj_csfs_many
    from atomx.physics.csf import sparse_ci as SC

    od, bsp, eps, _, _ = he_anchor
    pats = SC.patterns_from_csfs(enumerate_jj_csfs_many(["1s2", "2s2"]))
    masks, spinors = SC.enumerate_determinants(pats, od, 0)
    H = SC.assemble_sparse_hamiltonian(masks, spinors, od, bsp, 2, 40)
    coupling = abs(float(H[0, 1]))
    assert coupling > 1e-6
    res = solve_rpa(od, bsp, eps, [(1, -1)], [(2, -1)], twoJ, 1)
    assert res.B[0, 0] == pytest.approx(coupling * (1 if twoJ == 0 else -1), abs=1e-12)


def test_ci_direct_matches_canonical_kernel(he_anchor):
    """Cross-gate: on the CANONICAL virtual basis the CG-tensor kernel
    (solve_rpa) and the CI-direct construction (solve_rpa_ci_direct,
    which carries the one-body residual for ARBITRARY virtuals) must
    agree — the CI-direct path also covers non-canonical bases where
    the tensor kernel's precondition fails."""
    from atomx.methods.rpa import solve_rpa_ci_direct
    od, bsp, eps, w, de = he_anchor
    for twoJ, i_ref in ((0, 2), (2, 1)):
        res = solve_rpa(od, bsp, eps, [(1, -1)], [(2, -1)],
                        twoJ=twoJ, exc_parity=1)
        A, _labels, _E0 = solve_rpa_ci_direct(od, bsp, 2, 40,
                                               [(1, -1)], [(2, -1)],
                                               twoJ, 1)
        assert A[0, 0] - de == pytest.approx(res.A[0, 0] - de,
                                             abs=1e-8), twoJ
        assert A[0, 0] - de == pytest.approx(w[i_ref] - w[0] - de,
                                             abs=1e-6), twoJ
