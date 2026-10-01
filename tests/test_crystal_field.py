"""Physics gates for the crystal-field operator (the recorded lesson:
sign/magnitude gates BEFORE anchor runs).

  G1  radial k=0 == 1 on a real DF workspace (fixes the P=rR measure)
  G2  Hermiticity: cf(ia, ib) == conj(cf(ib, ia)) for Hermitian B sets
  G3  per-subshell trace of the CF diagonal == 0 for every k > 0
  G4  digit-exact vs direct angular quadrature: CG-built spinor
      spherical harmonics Omega, Gauss-Legendre x trapezoid — a fully
      independent path (no 3j of the operator route), which also pins
      the reduced_C phase convention for kappa_a != kappa_b elements
  G5  assembler m-mixing channel: python vs numba paths element-identical
      and the assembled H Hermitian
"""
import itertools
import math

import numpy as np
import pytest

from atomx.angular.csf_hamiltonian import _m2_values, twoj
from atomx.methods.csf.crystal_field import build_cf_operator, rpow_grid

SYMPY = pytest.importorskip("sympy")
from sympy.physics.wigner import clebsch_gordan  # noqa: E402


def _l_of_kappa(kappa):
    return kappa if kappa > 0 else -kappa - 1


def _direct_Ckq(kappa_a, m2a, kappa_b, m2b, k, q, nt=48, np_=96):
    """<kappa_a m_a| C^k_q |kappa_b m_b> by direct angular quadrature.

    Vectorised: CG coefficients are angle-independent (computed once);
    Y_lm are cached on the (theta, phi) grid.
    """
    from scipy.special import sph_harm

    x, wx = np.polynomial.legendre.leggauss(nt)
    theta = np.arccos(x)
    phi = 2 * np.pi * (np.arange(np_) + 0.5) / np_
    TH, PH = np.meshgrid(theta, phi, indexing="ij")
    W = wx[:, None] * (2 * np.pi / np_)

    ycache = {}

    def Yv(l, m):
        key = (l, m)
        if key not in ycache:
            if abs(m) > l:
                ycache[key] = np.zeros((nt, np_), dtype=complex)
            else:
                v = sph_harm(abs(int(m)), l, PH, TH)
                if m < 0:
                    v = (-1.0) ** int(-m) * np.conj(v)
                ycache[key] = v
        return ycache[key]

    def cg_pair(kappa, m2):
        l = _l_of_kappa(kappa)
        j = twoj(kappa) / 2.0
        m = m2 / 2.0
        up = float(clebsch_gordan(l, 1 / 2, j, m - 1 / 2, 1 / 2, m))
        dn = float(clebsch_gordan(l, 1 / 2, j, m + 1 / 2, -1 / 2, m))
        return l, m, up, dn

    la, ma, upa, dna = cg_pair(kappa_a, m2a)
    lb, mb, upb, dnb = cg_pair(kappa_b, m2b)
    C = np.sqrt(4 * np.pi / (2 * k + 1)) * Yv(k, q)
    dens = (upa * upb * np.conj(Yv(la, ma - 1 / 2)) * Yv(lb, mb - 1 / 2)
            + dna * dnb * np.conj(Yv(la, ma + 1 / 2)) * Yv(lb, mb + 1 / 2))
    return np.sum(W * C * dens)


# ---- fixture: synthetic orbitals (angular gates) -------------------------

def _synthetic_operator(bkq):
    """cf over a fake 4f-like shell with unit radial."""
    keys = [(4, -4), (4, 3)]  # 4f7/2, 4f5/2
    orbitals = {k: (np.array([1.0]), np.array([0.0])) for k in keys}
    return build_cf_operator(orbitals, None, 4, 50, bkq,
                             radial=lambda a, b, k: 1.0,
                             hermiticity="off")


KAPPAS = [(4, -4), (4, 3)]


# ---- G2/G3: Hermiticity + tracelessness (angular machinery) --------------

@pytest.mark.parametrize("k,q", [(2, 0), (2, 1), (2, 2), (4, 3), (6, 5),
                                 (4, 4)])
def test_g2_hermiticity(k, q):
    cf = _synthetic_operator({(k, q): 1.0, (k, -q): (-1) ** q * 1.0})
    for (na, ka), (nb, kb) in itertools.product(KAPPAS, KAPPAS):
        for ma in _m2_values(twoj(ka)):
            for mb in _m2_values(twoj(kb)):
                v1 = cf(cf.index[(na, ka, ma)], cf.index[(nb, kb, mb)])
                v2 = cf(cf.index[(nb, kb, mb)], cf.index[(na, ka, ma)])
                assert v1 == pytest.approx(np.conj(v2), abs=1e-12), \
                    (ka, ma, kb, mb, v1, v2)


@pytest.mark.parametrize("k", [2, 4, 6])
def test_g3_trace_zero(k):
    cf = _synthetic_operator({(k, 0): 1.0})
    for (n, kap) in KAPPAS:
        tr = sum(cf(cf.index[(n, kap, m)], cf.index[(n, kap, m)])
                 for m in _m2_values(twoj(kap)))
        assert tr == pytest.approx(0.0, abs=1e-12), (kap, tr)


# ---- G4: digit-exact vs direct quadrature --------------------------------

@pytest.mark.parametrize("ka,kb,k,q", [
    (-4, -4, 2, 0), (-4, -4, 4, 2), (-4, -4, 6, 5),
    (3, 3, 2, 1), (3, 3, 4, 3),
    (-4, 3, 2, 0), (-4, 3, 2, 1), (-4, 3, 4, 2), (3, -4, 4, 4),
])
def test_g4_direct_quadrature(ka, kb, k, q):
    # B = CM_PER_HARTREE folds the unit conversion: cf == pure angular
    from atomx.methods.csf.crystal_field import CM_PER_HARTREE
    cf = _synthetic_operator({(k, q): CM_PER_HARTREE})
    checked = 0
    for ma in _m2_values(twoj(ka)):
        for mb in _m2_values(twoj(kb)):
            if ma - mb != 2 * q:
                continue
            got = cf(cf.index[(4, ka, ma)], cf.index[(4, kb, mb)])
            want = _direct_Ckq(ka, ma, kb, mb, k, q)
            assert got == pytest.approx(want, rel=1e-8, abs=1e-10), \
                (ka, ma, kb, mb, k, q, got, want)
            checked += 1
    assert checked > 0


# ---- G1 + G5: real workspace gates ---------------------------------------

@pytest.fixture(scope="module")
def be_workspace():
    from atomx.methods.dirac_fock import DFSolver, Subshell
    dhf = DFSolver(Z=4,
                   subshells=[Subshell(n=1, kappa=-1, occupation=2.0),
                              Subshell(n=2, kappa=-1, occupation=2.0),
                              Subshell(n=2, kappa=1, occupation=1e-6),
                              Subshell(n=2, kappa=-2, occupation=1e-6),
                              Subshell(n=3, kappa=-1, occupation=1e-6),
                              Subshell(n=3, kappa=-2, occupation=1e-6)],
                   ns=50, ks=9, starting_potential="tfda")
    r = dhf.solve()
    od = {(n, k): (P, Q) for (n, k), (P, Q, _o)
          in r.orbitals_by_subshell().items()}
    return od, dhf.bsp_data


def test_g1_radial_k0_norm(be_workspace):
    od, bsp = be_workspace
    rpow = rpow_grid(od, bsp)
    for key in od:
        assert rpow(key, key, 0) == pytest.approx(1.0, abs=1e-10), key


def test_g5_assembler_m_mixing_paths(be_workspace):
    """python vs numba identical + H Hermitian under a real CF set."""
    from atomx.physics.csf import sparse_ci as SC
    od, bsp = be_workspace
    # q=0 (m-diagonal, m-dependent) + q=+-2 (m-mixing): exercises both
    # halves of the m-resolved channel; Hermitian set (real, (-1)^q)
    bkq = {(2, 0): 5000.0, (2, 2): 3000.0, (2, -2): 3000.0}
    cf = build_cf_operator(od, bsp, 4, 50, bkq)

    # m-mixing delta over the assembler's own spinor labels: same-kappa
    # pairs; element independent of the m-blind base h
    def m_delta(ia, ic):
        sa, sc = spinors[ia], spinors[ic]
        try:
            ea = cf.index[sa]
            ec = cf.index[sc]
        except KeyError:
            return 0.0
        return float(cf(ea, ec))

    from atomx.config.csf import enumerate_jj_csfs_many
    from atomx.methods.mcdhf.excitations import generate_sd_excitations
    from atomx.physics.csf.sparse_ci import patterns_from_csfs
    cfgs = generate_sd_excitations("1s2 2s2", ["2s", "2p"],
                                   core_orbitals=["1s"], max_excitations=2)
    pats = patterns_from_csfs(enumerate_jj_csfs_many(cfgs))
    masks, spinors = SC.enumerate_determinants(pats, od, 0)
    Hp = SC.assemble_sparse_hamiltonian(masks, spinors, od, bsp, 4, 50,
                                        method="python",
                                        one_body_m_delta=m_delta)
    Hn = SC.assemble_sparse_hamiltonian(masks, spinors, od, bsp, 4, 50,
                                        method="numba",
                                        one_body_m_delta=m_delta)
    D = (Hp - Hn).tocoo()
    assert D.data.size == 0 or np.max(np.abs(D.data)) < 1e-12

    Hd = (Hp - Hp.T).tocoo()
    assert Hd.data.size == 0 or np.max(np.abs(Hd.data)) < 1e-10

    # the CF actually does something: spectrum moves
    from scipy.sparse.linalg import eigsh
    w0 = eigsh(SC.assemble_sparse_hamiltonian(masks, spinors, od, bsp,
                                              4, 50), k=1, which="SA",
              return_eigenvectors=False)[0]
    w1 = eigsh(Hp, k=1, which="SA", return_eigenvectors=False)[0]
    assert w1 != pytest.approx(w0, abs=1e-8)
    assert math.isfinite(w1)


# ---- G6/G7: review-fix gates (kimi critique, 2026-10-01) -----------------

def test_g6_cross_n_elements_enabled(be_workspace):
    """Same-kappa different-n CF elements are real physics (<4f|r^k|5f>);
    the old code silently zeroed them (review issue #2)."""
    od, bsp = be_workspace
    cf = build_cf_operator(od, bsp, 4, 50, {(2, 0): 5000.0})
    up2p = cf.index[(2, -2, 1)]
    up3p = cf.index[(3, -2, 1)]
    fwd = cf(up2p, up3p)
    rev = cf(up3p, up2p)
    assert fwd != 0.0, "cross-n element must not be silently zero"
    assert fwd == pytest.approx(rev, rel=1e-10), "cross-n must be Hermitian"


def test_g7_point_charge_closed_form_and_guards():
    """Axial charge: A_k^0 = -q/R^{k+1} exactly; q!=0 vanishes; the
    r_inside guard refuses charges inside the shell; output feeds
    build_cf_operator with hermiticity='strict'."""
    from atomx.methods.csf.crystal_field import (CM_PER_HARTREE,
                                                 point_charge_bkq)
    R = 3.0
    bkq = point_charge_bkq([(1.0, (0.0, 0.0, R))], (0.0, 0.0, 0.0),
                           kmax=6, r_inside=1.0)
    for k in (2, 4, 6):
        want = -1.0 / R ** (k + 1) * CM_PER_HARTREE
        assert bkq[(k, 0)] == pytest.approx(want, rel=1e-12), k
        for q in range(-k, k + 1):
            if q != 0:
                assert abs(bkq[(k, q)]) < 1e-14 * abs(bkq[(k, 0)])
    # r<R guard
    with pytest.raises(ValueError, match="inside the shell"):
        point_charge_bkq([(1.0, (0.0, 0.0, 0.5))], (0.0, 0.0, 0.0),
                         kmax=6, r_inside=1.0)
    with pytest.raises(ValueError, match="r_inside"):
        point_charge_bkq([(1.0, (0.0, 0.0, 3.0))], (0.0, 0.0, 0.0))
    # general-site output is Hermitian by construction -> strict build
    gen = point_charge_bkq([(1.0, (1.0, 2.0, 2.0)), (-2.0, (-1.5, 0.5, 2.5))],
                           (0.0, 0.0, 0.0), kmax=4, r_inside=1.0)
    keys = [(4, -4), (4, 3)]
    od = {k: (np.array([1.0]), np.array([0.0])) for k in keys}
    build_cf_operator(od, None, 4, 50, gen,
                      radial=lambda a, b, k: 1.0, hermiticity="strict")


def test_g7b_hermiticity_validation_warns():
    keys = [(4, -4), (4, 3)]
    od = {k: (np.array([1.0]), np.array([0.0])) for k in keys}
    with pytest.warns(UserWarning, match="non-Hermitian"):
        build_cf_operator(od, None, 4, 50, {(2, 1): 100.0},
                          radial=lambda a, b, k: 1.0)
    with pytest.raises(ValueError, match="non-Hermitian"):
        build_cf_operator(od, None, 4, 50, {(2, 1): 100.0},
                          radial=lambda a, b, k: 1.0, hermiticity="strict")
    with pytest.raises(ValueError, match=r"\|q\| > k"):
        build_cf_operator(od, None, 4, 50, {(2, 3): 100.0},
                          radial=lambda a, b, k: 1.0)


def test_g7c_Y_vs_sympy():
    """The module spherical harmonic vs sympy Ynm, generic direction."""
    from atomx.methods.csf.crystal_field import _Y_kq
    import math

    from sympy import N, exp, factorial as sfac, pi as spi, sqrt as ssqrt
    from sympy.functions.special.polynomials import assoc_legendre

    th, ph = 0.9, 2.1
    x = math.cos(th)
    for k in (2, 4, 6):
        for q in range(-k, k + 1):
            got = _Y_kq(k, q, th, ph)
            yy = (ssqrt((2 * k + 1) / (4 * spi)
                        * sfac(k - q) / sfac(k + q))
                  * assoc_legendre(k, q, x) * exp(1j * q * ph))
            ref = complex(N(yy))
            assert got == pytest.approx(ref, rel=1e-10, abs=1e-12), (k, q)
