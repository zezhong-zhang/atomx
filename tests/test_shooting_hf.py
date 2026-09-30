"""Non-local HF exchange on the ODE shooting engine vs B-spline DHF.

The B-spline ``DFSolver`` computes the jj-averaged Dirac-Hartree-Fock
equations exactly (Galerkin, DBSR_HF algorithm); the shooting engine
(:mod:`atomx.numerics.shooting.hartree_fock`) solves the SAME equations
— same ``exchange_lambda`` jj-average angular coefficients — with the
Froese-Fischer inhomogeneous-equation method on Salvat's SGRID:
exchange enters as a Y^k source term, and each orbital is the
variation-of-parameters (Green function) solution whose eigenvalue is
pinned by the normalisation-Newton condition.

Cross-engine agreement gates (different radial numerics, so this is a
real validation, not a regression pin):

* total energy:    |ΔE|/|E|   <= 1e-4
* orbital energy:  |Δeps|/|eps| <= 5e-4  (every occupied orbital)

Measured slack at the time of writing (point vs Fermi nucleus included):
He 7e-9, Ne 4e-7, Fe XVII 4e-6 on totals; <= 5e-5 on orbitals.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from atomx import Atom, Exchange  # noqa: E402

TOTAL_RTOL = 1e-4
ORBITAL_RTOL = 5e-4

CASES = {
    "He": dict(Z=2),
    "Ne": dict(Z=10),
    "FeXVII": dict(Z=26, charge=16),
}


@pytest.fixture(scope="module")
def solved():
    """Solve each case once on both engines.

    Correction-free on both sides (breit/qed default True since the
    2026-07 defaults flip): this file validates the bare HF radial
    numerics, and ``test_energy_components_consistent`` requires
    ``total_energy == E_1e + E_2e`` exactly.
    """
    out = {}
    for name, kw in CASES.items():
        shooting = Atom(
                        engine="shooting",
                        exchange="hf",
                        corrections="none",
                        **kw,
                   ).solve()
        bspline = Atom(
                       engine="bspline",
                       exchange="hf",
                       corrections="none",
                       **kw,
                  ).solve()
        out[name] = (shooting, bspline)
    return out


@pytest.mark.parametrize("name", list(CASES))
def test_converged(solved, name):
    shooting, bspline = solved[name]
    assert shooting.converged
    assert bspline.converged


@pytest.mark.parametrize("name", list(CASES))
def test_total_energy_matches_bspline(solved, name):
    shooting, bspline = solved[name]
    assert np.isfinite(shooting.total_energy)
    assert np.isfinite(bspline.total_energy)
    rel = abs(shooting.total_energy - bspline.total_energy) / abs(
        bspline.total_energy)
    assert rel <= TOTAL_RTOL, (
        f"{name}: shooting {shooting.total_energy!r} vs "
        f"bspline {bspline.total_energy!r} (rel {rel:.2e})")


@pytest.mark.parametrize("name", list(CASES))
def test_orbital_energies_match_bspline(solved, name):
    shooting, bspline = solved[name]
    assert set(shooting.orbital_energies) == set(bspline.orbital_energies)
    for label, eps_b in bspline.orbital_energies.items():
        eps_s = shooting.orbital_energies[label]
        rel = abs(eps_s - eps_b) / abs(eps_b)
        assert rel <= ORBITAL_RTOL, (
            f"{name} {label}: shooting {eps_s!r} vs bspline {eps_b!r} "
            f"(rel {rel:.2e})")


def test_provenance_and_method_metadata(solved):
    shooting, _ = solved["Ne"]
    # Hamiltonian averaging is explicit even for the closed-shell limit.
    assert shooting.provenance == {"engine": "shooting",
                                   "exchange": "hf",
                                   "averaging": "hamiltonian"}
    res = shooting.result
    assert res.method_name == "Dirac-Hartree-Fock (shooting)"
    assert res.theory_level.name == "DIRAC_HARTREE_FOCK"


def test_energy_components_consistent(solved):
    """E = E_1e + E_2e and E_2e = E_J - E_K on the shooting result."""
    shooting, _ = solved["Ne"]
    comp = shooting.energy_components
    assert comp["E_2e"] == pytest.approx(comp["E_J"] - comp["E_K"])
    assert shooting.total_energy == pytest.approx(
        comp["E_1e"] + comp["E_2e"], rel=1e-12)


def test_exchange_lowers_valence_orbitals_vs_local():
    """Sanity: non-local HF binds Ne valence more deeply than Gu's
    local exchange on the same engine (the known DFS shortfall)."""
    hf = Atom(Z=10, engine="shooting", exchange="hf").solve()
    gu = Atom(Z=10, engine="shooting", exchange="gu").solve()
    assert hf.orbital_energies["2p3/2"] < gu.orbital_energies["2p3/2"]


def test_alpha_x_still_rejected_for_hf():
    with pytest.raises(ValueError, match="alpha_x"):
        Atom(Z=10, engine="shooting", exchange=Exchange(model="hf", alpha_x=0.7))


@pytest.mark.parametrize("Z", [27, 82])
def test_fermi_nuclear_model_matches_bspline_engine(Z):
    """Both engines must solve the SAME Fermi nuclear model.

    ``ShootingHFSolver._nuclear_rv`` takes its (c, a) shape parameters
    from the library-wide ``fermi_params`` (DBSR_HF t = 2.30 fm +
    empirical rms) — the model ``DFSolver`` gets via
    ``make_nuclear_potential``.  The Salvat literals it used before
    (R1 = 1.07 A^{1/3} fm, a = 0.546 fm) have a ~4% smaller rms radius
    at Z=82; the ~5.5 Ha Pb finite-size shift turned that into a flat
    0.351 Ha cross-engine offset on every HF/ci total (measured
    -20914.1136 vs -20913.7625 on the Pb ci route).  Gate: the
    slag6-built r·V_nuc equals r·V of the shared analytic potential to
    1e-6 relative (measured 3.2e-9 at Z=27, 1.2e-8 at Z=82).
    """
    import numpy as np

    from atomx.methods.dirac_fock import Subshell
    from atomx.numerics.shooting.grid import sgrid
    from atomx.numerics.shooting.hartree_fock import ShootingHFSolver
    from atomx.potentials.nuclear import make_nuclear_potential

    r, difr, n = sgrid(75.0, 1.0e-7, 0.5, 1000)
    r, difr = r[:n], difr[:n]
    solver = ShootingHFSolver(Z, [Subshell(1, -1, 2.0)], nuclear="fermi")
    rv = solver._nuclear_rv(r, difr, n)
    V = make_nuclear_potential(Z, "fermi")
    rv_ref = np.array([x * V(x) for x in r])
    rel = np.max(np.abs(rv - rv_ref) / np.maximum(np.abs(rv_ref), 1e-3))
    assert rel <= 1e-6, f"Z={Z}: nuclear models diverge (rel {rel:.2e})"
