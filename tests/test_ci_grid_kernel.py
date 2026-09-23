"""Grid-flavour CI kernel: shooting-engine numerics inside build_ci_matrix.

The CI kernel must evaluate matrix elements with the SAME numerics as the
inner SCF energy functional (the consistency principle behind the Pb +74 Ha
fix, commit 008cb98).  For the shooting engine that means grid orbitals,
``_slater_integral_slag6`` quadrature (what ``ShootingHFSolver
._two_electron_energy`` uses), and the shooting solver's own nuclear
potential array.  These gates pin exactly that:

(a) closed-shell single-CSF CI total on converged ``ShootingHFSolver``
    orbitals equals ``ShootingResult.total_energy`` to <= 1e-6 relative
    (He and Ne; bare HF).  Measured slack at the time of writing:
    He 1.8e-10, Ne 5.9e-10 (Fermi nucleus), He point 1.8e-10.
(b) for Ne, each ``<a|h_D|a>`` + jj-averaged mean-field assembly
    reproduces the shooting orbital eigenvalue to ~1e-5 relative
    (measured <= 3.1e-8).
(c) grid-flavour F^k/G^k on analytic hydrogenic orbitals sampled on the
    SGRID reproduce the exact Condon-Shortley values to <= 1e-6 relative
    (measured <= 2.1e-11).
"""

import numpy as np
import pytest

from atomx.angular.coefficients import exchange_k_range, exchange_lambda
from atomx.methods.ci import ci_solve
from atomx.methods.dirac_fock import Subshell
from atomx.methods.mcdhf import matrix as ci_matrix
from atomx.methods.mcdhf.config import CSF, get_shared_orbital_basis
from atomx.methods.mcdhf.matrix import GridCIWorkspace, build_ci_matrix
from atomx.numerics.shooting.grid import sgrid
from atomx.numerics.shooting.hf_shooting import ShootingHFSolver

TOTAL_RTOL = 1.0e-6      # gate (a)
EIGEN_RTOL = 1.0e-5      # gate (b)
SLATER_RTOL = 1.0e-6     # gate (c)

CASES = {
    # (Z, [(n, kappa, occ), ...], nuclear)
    "He": (2, [(1, -1, 2)], "fermi"),
    "He_point": (2, [(1, -1, 2)], "point"),
    "Ne": (10, [(1, -1, 2), (2, -1, 2), (2, 1, 2), (2, -2, 4)], "fermi"),
}


@pytest.fixture(scope="module")
def solved():
    """Converged bare-HF shooting solves + their grid CI workspaces."""
    out = {}
    for name, (Z, shells, nuclear) in CASES.items():
        subshells = [Subshell(n, kappa, occ) for n, kappa, occ in shells]
        solver = ShootingHFSolver(Z, subshells, nuclear=nuclear)
        res = solver.solve(tol=1.0e-9)
        assert res.converged, f"{name}: shooting HF did not converge"
        ws = GridCIWorkspace.from_shooting(solver, res)
        orbitals = {(sh.n, sh.kappa): (res._pa[i], res._qa[i])
                    for i, sh in enumerate(subshells)}
        out[name] = (Z, subshells, solver, res, ws, orbitals)
    return out


def _single_csf(subshells, label):
    return CSF(subshells=[Subshell(sh.n, sh.kappa, sh.occupation)
                          for sh in subshells],
               total_J2=0, parity=1, label=label)


# ---------------------------------------------------------------------------
# Gate (a): closed-shell single-CSF CI total == ShootingResult.total_energy
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", list(CASES))
def test_closed_shell_ci_total_matches_shooting(solved, name):
    Z, subshells, _solver, res, ws, orbitals = solved[name]
    csf = _single_csf(subshells, name)
    ci = ci_solve([csf], orbitals, get_shared_orbital_basis([csf]),
                  ws, Z, ws.n)
    rel = abs(ci.ground_energy - res.total_energy) / abs(res.total_energy)
    assert rel <= TOTAL_RTOL, (
        f"{name}: CI {ci.ground_energy!r} vs shooting "
        f"{res.total_energy!r} (rel {rel:.2e})")


def test_two_electron_assembly_matches_solver(solved):
    """CI diagonal minus one-body == the solver's own E_J - E_K.

    Isolates the two-electron assembly from the finite-difference
    one-body error: both sides use ``_slater_integral_slag6`` with the
    identical jj-average coefficients, so they must agree to roundoff.
    """
    Z, subshells, solver, res, ws, orbitals = solved["Ne"]
    csf = _single_csf(subshells, "Ne")
    ci = ci_solve([csf], orbitals, get_shared_orbital_basis([csf]),
                  ws, Z, ws.n)
    h_sum = sum(
        sh.occupation * ci_matrix._h_one_body(
            *orbitals[(sh.n, sh.kappa)], *orbitals[(sh.n, sh.kappa)],
            sh.kappa, Z, ws, ws.n)
        for sh in subshells)
    e_j, e_k = solver._two_electron_energy(res._pa, res._qa,
                                           ws.r, ws.difr, ws.n)
    e_2e_ci = ci.ground_energy - h_sum
    assert abs(e_2e_ci - (e_j - e_k)) <= 1.0e-8 * abs(e_j - e_k)


# ---------------------------------------------------------------------------
# Gate (b): <a|h_D|a> + mean field reproduces the shooting eigenvalues
# ---------------------------------------------------------------------------


def test_ne_eigenvalue_reconstruction(solved):
    """eps_a = <a|h|a> + <a|V_dir - K|a> with the Froese-Fischer jj-averaged
    mean field of ``ShootingHFSolver`` (module docstring there), all radial
    integrals through the grid-flavour kernel helpers."""
    Z, subshells, _solver, res, ws, orbitals = solved["Ne"]
    ci_matrix._SLATER_MEMO.clear()
    occ = [float(sh.occupation) for sh in subshells]
    kap = [sh.kappa for sh in subshells]
    tj = [2 * abs(k) - 1 for k in kap]

    for a, sh in enumerate(subshells):
        Pa, Qa = orbitals[(sh.n, sh.kappa)]
        la = kap[a] if kap[a] > 0 else -kap[a] - 1
        eps_pred = ci_matrix._h_one_body(Pa, Qa, Pa, Qa, kap[a], Z, ws, ws.n)
        # same-shell direct + F^k multipoles ((N-1) convention)
        eps_pred += (occ[a] - 1.0) * ci_matrix._fk(Pa, Qa, Pa, Qa, ws, 0)
        for k in range(2, tj[a] + 1):
            if (k + 2 * la) % 2 != 0:
                continue
            lam = exchange_lambda(kap[a], kap[a], k)
            if lam < 1.0e-15:
                continue
            eps_pred -= ((occ[a] - 1.0) * lam / (tj[a] * (tj[a] + 1.0))
                         * ci_matrix._fk(Pa, Qa, Pa, Qa, ws, k))
        # cross-shell direct + exchange
        for b, shb in enumerate(subshells):
            if b == a:
                continue
            Pb, Qb = orbitals[(shb.n, shb.kappa)]
            eps_pred += occ[b] * ci_matrix._fk(Pa, Qa, Pb, Qb, ws, 0)
            for k in exchange_k_range(kap[a], kap[b]):
                lam = exchange_lambda(kap[a], kap[b], k)
                if lam < 1.0e-15:
                    continue
                eps_pred -= (occ[b] * lam / ((tj[a] + 1.0) * (tj[b] + 1.0))
                             * ci_matrix._gk(Pa, Qa, Pb, Qb, ws, k))

        eps = res.orbital_energies[sh.label]
        rel = abs(eps_pred - eps) / abs(eps)
        assert rel <= EIGEN_RTOL, (
            f"{sh.label}: predicted {eps_pred!r} vs eps {eps!r} "
            f"(rel {rel:.2e})")


# ---------------------------------------------------------------------------
# Gate (c): grid-flavour Slater integrals vs analytic hydrogenic values
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def hydrogenic_grid():
    """Analytic NR hydrogenic P(r) (Q=0) sampled on the solver's SGRID."""
    # Same grid parameters as ShootingSolver.solve (rn, r2, drn, n).
    r, difr, n = sgrid(75.0, 1.0e-7, 0.5, 1000)
    r, difr = r[:n], difr[:n]
    ws = GridCIWorkspace(r, difr, np.full(n, -1.0))
    Z = 1.0
    orb = {
        "1s": 2.0 * Z ** 1.5 * r * np.exp(-Z * r),
        "2s": Z ** 1.5 / (2.0 * np.sqrt(2.0)) * (2.0 - Z * r) * r
              * np.exp(-Z * r / 2.0),
        "2p": Z ** 1.5 / (2.0 * np.sqrt(6.0)) * Z * r ** 2
              * np.exp(-Z * r / 2.0),
    }
    return ws, orb, np.zeros(n)


# Exact Condon-Shortley hydrogenic Slater integrals at Z=1
# (verified symbolically with sympy: F^k/G^k of the analytic 1s/2s/2p).
HYDROGENIC_CASES = [
    ("f", "1s", "1s", 0, 5.0 / 8.0),
    ("f", "1s", "2s", 0, 17.0 / 81.0),
    ("g", "1s", "2s", 0, 16.0 / 729.0),
    ("f", "1s", "2p", 0, 59.0 / 243.0),
    ("g", "1s", "2p", 1, 112.0 / 2187.0),
    ("f", "2p", "2p", 0, 93.0 / 512.0),
    ("f", "2p", "2p", 2, 45.0 / 512.0),
]


@pytest.mark.parametrize("kind,a,b,k,exact", HYDROGENIC_CASES)
def test_hydrogenic_slater_integrals(hydrogenic_grid, kind, a, b, k, exact):
    ws, orb, Q = hydrogenic_grid
    ci_matrix._SLATER_MEMO.clear()
    fn = ci_matrix._fk if kind == "f" else ci_matrix._gk
    got = fn(orb[a], Q, orb[b], Q, ws, k)
    rel = abs(got - exact) / exact
    assert rel <= SLATER_RTOL, (
        f"{kind.upper()}^{k}({a},{b}): got {got!r} want {exact!r} "
        f"(rel {rel:.2e})")


@pytest.mark.parametrize("k,gk_exact,fk_exact", [
    (0, 176.0 / 2187.0, 59.0 / 243.0),
    (1, 112.0 / 2187.0, 8.0 / 81.0),
])
def test_rk_pairs_its_arguments_as_ab_cd(hydrogenic_grid, k, gk_exact, fk_exact):
    """``_rk(a, b, c, d)`` is ``R^k(ab, cd)``, pinned against exact closed forms.

    Callers pass the roles explicitly — ``_rk(Pa, Pc, Pb, Pd)`` asks for
    ``R^k(ac, bd)``.  The previous version of this test asserted
    ``_rk(1s, 2s, 1s, 2s) == _gk(1s, 2s)``, which **could not fail**: on the
    grid path both are the same ``_slater_integral_slag6`` expression over
    the same two pair densities (``matrix.py``).  The discriminating
    statement is against the *wrong* pairing's value:

    * correct reading of ``_rk(1s, 2p, 1s, 2p, k)``: ``R^k(1s1p, 1s2p)``
      with equal pairs, i.e. the exchange integral ``G^k(1s,2p)``;
    * the (a,c)(b,d) reading would give ``R^k(1s1s, 2p2p) = F^k(1s,2p)``.

    Both are exact Condon-Shortley hydrogenic values (independently verified
    with sympy for the analytic 1s/2p functions): ``G^0 = 176/2187``,
    ``G^1 = 112/2187``, ``F^0 = 59/243``, ``F^1 = 8/81``.  They differ by
    60 % / 48 %, so the pairing cannot pass silently.
    """
    ws, orb, Q = hydrogenic_grid
    ci_matrix._SLATER_MEMO.clear()
    got = ci_matrix._rk(orb["1s"], Q, orb["2p"], Q,
                        orb["1s"], Q, orb["2p"], Q, ws, k)
    assert got == pytest.approx(gk_exact, rel=SLATER_RTOL), (
        f"R^{k}(1s 2p, 1s 2p) = {got!r}, expected G^{k}(1s,2p) = {gk_exact!r}")
    ci_matrix._SLATER_MEMO.clear()
    alt = ci_matrix._fk(orb["1s"], Q, orb["2p"], Q, ws, k)
    assert alt == pytest.approx(fk_exact, rel=SLATER_RTOL)
    assert abs(got - alt) > 0.1 * alt, (
        f"the (a,c)(b,d) reading F^{k} = {alt!r} is not distinguishable from "
        f"_rk = {got!r}; the assertion above is then vacuous")


# ---------------------------------------------------------------------------
# Structural: multi-CSF grid build (off-diagonal path) + flavour guards
# ---------------------------------------------------------------------------


@pytest.mark.snapshot
def test_one_diff_grid_matrix(hydrogenic_grid):
    """Two-CSF 1s2 / 1s1 2s1 build on grid orbitals: symmetric, and the
    1-diff element reduces to <1s|h_D|2s> (no spectators, sqrt factor 1).

    The pinned element (0.169929196167459 Ha) is atomx's own recorded
    value, so the test carries the ``snapshot`` marker; the structural
    assertions (symmetry, non-zero) are the part that does not depend on it.
    """
    ws, orb, Q = hydrogenic_grid
    orbitals = {(1, -1): (orb["1s"], Q), (2, -1): (orb["2s"], Q)}
    csfs = [
        CSF(subshells=[Subshell(1, -1, 2)], total_J2=0, parity=1,
            label="1s2"),
        CSF(subshells=[Subshell(1, -1, 1), Subshell(2, -1, 1)],
            total_J2=0, parity=1, label="1s1 2s1"),
    ]
    H = build_ci_matrix(csfs, orbitals, get_shared_orbital_basis(csfs),
                        ws, 1, ws.n)
    assert np.all(np.isfinite(H))
    assert H[0, 1] == H[1, 0]
    h12 = ci_matrix._h_one_body(orb["1s"], Q, orb["2s"], Q, -1, 1, ws, ws.n)
    assert h12 != 0.0
    # The exact element is NOT h12: the 1s spectator contributes a two-body
    # exchange term and the CSF normalisation brings a factor sqrt(2).  The
    # value below is reproduced to 1e-14 by the independent m-scheme oracle
    # (benchmarks/mscheme_ci_reference.py) on the same determinants, and the
    # two eigenvalues of this 2x2 block reproduce two of the three oracle
    # eigenvalues exactly (the third is the J=1 state this CSF list omits).
    # The test used to assert H[0, 1] == h12, which was the old
    # configuration-average one-diff formula.
    assert H[0, 1] == pytest.approx(0.169929196167459, rel=1.0e-12)


def test_grid_flavour_rejects_bspline_only_kwargs(hydrogenic_grid):
    ws, orb, Q = hydrogenic_grid
    orbitals = {(1, -1): (orb["1s"], Q)}
    csfs = [CSF(subshells=[Subshell(1, -1, 2)], total_J2=0, parity=1,
                label="1s2")]
    basis = get_shared_orbital_basis(csfs)
    with pytest.raises(ValueError, match="nuclear_potential"):
        build_ci_matrix(csfs, orbitals, basis, ws, 1, ws.n,
                        nuclear_potential=lambda r: -1.0 / r)
    with pytest.raises(ValueError, match="convol_cache"):
        build_ci_matrix(csfs, orbitals, basis, ws, 1, ws.n,
                        convol_cache=object())
    with pytest.raises(ValueError, match="ns"):
        build_ci_matrix(csfs, orbitals, basis, ws, 1, ws.n + 1)


def test_workspace_shape_validation():
    r = np.linspace(0.0, 1.0, 32)
    with pytest.raises(ValueError):
        GridCIWorkspace(r, r[:-1], r)
    with pytest.raises(ValueError):
        GridCIWorkspace(r.reshape(4, 8), r.reshape(4, 8), r.reshape(4, 8))
