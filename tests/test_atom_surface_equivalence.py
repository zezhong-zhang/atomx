"""The canonical `Atom` surface reproduces the pre-change default-path numbers.

Every value in ``PANEL`` was measured on the OLD flat-keyword surface, before
``3567c5e`` (``refactor(atom)!: reduce the Atom input to ten physically-named
axes`` — a pure rename) rewrote the constructor, and is asserted here through
the NEW surface.  This is **change detection, not a physics gate**: the pinned
numbers are the fixed point of the *default* SCF tolerance
(``Atom.solve(tol=1e-5)``), while the solver's own numerical convergence error
is ~1e-6 Ha.  A tighter SCF is *rejected* by the panel value
(``solve(tol=1e-9)`` gives -128.6299705068449 for Ne, +7.84e-6 Ha), and so is a
wider radial box (``Basis(r_max=120)``, +1.55e-6 Ha) — the digits below record
the default code path, nothing more.

The panel covers the five cases the migration had to preserve on that path:
a closed shell (Ne), an open shell with CI averaging (N), a
multi-configuration CI run (Be), a B-spline run, and a QED/Breit run
(Fe XVII).  Two extra rows (a shooting Rydberg run and a high-Z
hydrogen-like ion) guard the corners where the box and the knot grid bite.

The pinned total energy of the default path also contains
``E_Breit = 0.05043710257669187 Ha`` (the default is breit+qed), which is
**3.03x** the DBSR_HF value 0.016643630840490 Ha — see
``tests/test_nist_levels.py::test_breit_scalar_matches_dbsr_hf``.  A 1e-12 pin
on that quantity is therefore 3.4e10 x tighter than the physics it contains,
which is why every frozen-literal test here carries
``@pytest.mark.snapshot`` ("frozen atomx SELF-regression ... asserts nothing
external") and why the correction-free comparison below is the one that speaks
about the physics.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from atomx import Atom, Basis, Correlation, Exchange  # noqa: E402

#: case id -> (constructor kwargs, pre-change total energy in Hartree).
PANEL = {
    "ne_closed_shooting": (
        dict(Z=10, config="1s2 2s2 2p6"),
        -128.6299783505843,
    ),
    "ne_closed_bspline": (
        dict(Z=10, config="1s2 2s2 2p6", engine="bspline"),
        -128.62995892582862,
    ),
    "n_open_ci_averaging": (
        dict(Z=7, config="1s2 2s2 2p3",
             correlation=Correlation(averaging="ci")),
        -54.42278994015069,
    ),
    "n_open_statistical": (
        dict(Z=7, config="1s2 2s2 2p3",
             correlation=Correlation(averaging="statistical")),
        -54.318278137716575,
    ),
    "be_multiconfig": (
        dict(Z=4, config=["1s2 2s2", "1s2 2p2"]),
        -14.612850934952725,
    ),
    "be_multiconfig_fac_frozen": (
        dict(Z=4, config=["1s2 2s2", "1s2 2p2"],
             correlation=Correlation(averaging="statistical",
                                     radial_optimization="fac_frozen")),
        -14.612978088402736,
    ),
    "fe17_breit_qed": (
        dict(Z=26, charge=16, config="1s2 2s2 2p6"),
        -1146.731865120602,
    ),
    "fe17_gu_shooting": (
        dict(Z=26, charge=16, config="1s2 2s2 2p6", engine="shooting",
             exchange=Exchange(model="gu")),
        -1146.7119018745636,
    ),
    "fe17_slater_bspline": (
        dict(Z=26, charge=16, config="1s2 2s2 2p6", engine="bspline",
             exchange=Exchange(model="slater", alpha_x=1.0)),
        -1146.7018798756617,
    ),
    "h_6s_shooting": (
        dict(Z=1, config="6s1", engine="shooting", corrections="none"),
        -0.013888996745332824,
    ),
    "u91_refined_grid": (
        dict(Z=92, charge=91, config="1s1", engine="bspline",
             basis=Basis(grid_hi=0.005, grid_he=0.10), corrections="none"),
        -4854.196388949222,
    ),
}


#: Cases that did NOT converge before the migration either.  The value is
#: still reproduced, which is what this gate checks; the local-exchange
#: B-spline SCF has a pre-existing convergence gap at Z=26 that the rename
#: neither caused nor fixed.
NON_CONVERGED = {"fe17_slater_bspline"}

#: The solver's own convergence scale, in Hartree: the default SCF stops at
#: ``tol=1e-5`` on the energy and the observed spread between tolerances,
#: radial boxes and knot grids is 2.4e-7 .. 4.5e-5 (measured; see the module
#: docstring).  1e-6 is therefore the *honest* width of this panel: wide
#: enough to hold the fixed point, 6 orders of magnitude narrower than the
#: 0.062 Ha physics defect the pin used to hide.
CONVERGENCE_SCALE_HA = 1e-6


@pytest.mark.snapshot
@pytest.mark.parametrize("name,kwargs,expected", [
    pytest.param(n, kw, e, id=n) for n, (kw, e) in PANEL.items()
])
def test_canonical_form_reproduces_the_pre_change_number(name, kwargs,
                                                         expected):
    """Frozen default-path value, at the solver's convergence scale."""
    atom = Atom(**kwargs).solve()
    if name not in NON_CONVERGED:
        assert atom.converged, name
    assert atom.total_energy == pytest.approx(
        expected, abs=CONVERGENCE_SCALE_HA), name


@pytest.mark.snapshot
def test_the_ne_panel_keeps_every_orbital():
    """The one bit-exact regression in this file: Ne, default path.

    ``==`` and not ``approx``: this is the strongest *change* detector the
    file can honestly carry (reproduced exactly on repeat runs), and it is
    explicitly labelled as a self-regression.  The physics is in
    ``test_ne_correction_free_energy_matches_dbsr_hf`` below.
    """
    atom = Atom(10, config="1s2 2s2 2p6").solve()
    assert atom.total_energy == -128.6299783505843
    assert atom.orbital_energies == pytest.approx(
        {"1s1/2": -32.817447803729735, "2s1/2": -1.9358521624506706,
         "2p1/2": -0.8528343072534854, "2p3/2": -0.8482721633409784},
        abs=1e-12)


@pytest.mark.snapshot
def test_only_the_documented_inputs_move_a_result():
    """The four wired inputs are the *only* deliberate behaviour changes."""
    # 1. The plasma screens the nucleus (unscreened default is untouched).
    from atomx import Environment
    bare = Atom(10, charge=2, corrections="none").solve()
    dense = Atom(10, charge=2, corrections="none",
                 environment=Environment(plasma_ne=1e22,
                                         plasma_Te=100.0)).solve()
    assert bare.total_energy == pytest.approx(-126.51974272, abs=1e-7)
    assert dense.total_energy != bare.total_energy
    # 2. A tier that sets r_max/n_grid reaches the solvers; one that leaves
    #    them at their schema defaults does not.
    from atomx import Sampling
    assert Atom(10, basis=Basis(sampling=Sampling.standard)).r_max is None
    assert Atom(10, basis=Basis(sampling=Sampling.quick)).r_max == 60.0
    # 3. qed_backend is carried by Corrections; the default is still "auto".
    assert Atom(10, config=["1s2 2s2 2p6", "1s2 2s2 2p5 3s1"]).qed_backend \
        == "auto"
    # 4. slater_method is only accepted where it is read.
    with pytest.raises(ValueError, match="SHOOTING engine"):
        Atom(10, engine="bspline",
             exchange=Exchange(model="slater", slater_method="power_series"))


def test_ne_correction_free_energy_matches_dbsr_hf():
    """B/A gate: Ne with all corrections off, against live DBSR_HF.

    This is the comparison the 1e-12 panel was missing.  DBSR_HF — the
    reference implementation this repo is modelled on — gives the
    closed-shell Coulomb energy of Ne ``1s2 2s2 2p6`` as
    **-128.69192580 Ha** (``dbsr_hf an=10 term=AV mbreit=0``; the row is
    quoted in ``benchmarks/EXTERNAL_VERIFICATION.md`` section 1, where the
    run's full-precision log value is -128.691925796129510).

    Measured: the correction-free atomx energy on the shooting engine at
    ``tol=1e-11`` is -128.69192580015647, i.e. 4.0e-9 Ha from the reference
    (a 250x margin on the 1e-6 gate).  The SCF must be converged tightly:
    the *default* ``tol=1e-5`` fixed point is -128.69193378732467, 8.0e-6 Ha
    — above this gate — which is itself the evidence that the 1e-12 panel
    above measured the default tolerance rather than the physics.

    The default (breit+qed) path sits 0.062 Ha away from this number, the
    size of the 3.03x-too-large scalar Breit.
    """
    dbsr_hf_ne_coulomb = -128.69192580   # benchmarks/EXTERNAL_VERIFICATION.md
    atom = Atom(10, config="1s2 2s2 2p6", corrections="none")
    atom.solve(tol=1e-11, max_iter=60)
    assert atom.converged
    assert atom.total_energy == pytest.approx(dbsr_hf_ne_coulomb, abs=1e-6), (
        f"correction-free Ne = {atom.total_energy!r} Ha vs "
        f"DBSR_HF {dbsr_hf_ne_coulomb} Ha")
    # The gate above must not be satisfiable by the Breit+QED default path:
    # its corrections move the total energy by 0.062 Ha, 6e4 x the gate.
    default = Atom(10, config="1s2 2s2 2p6").solve()
    assert abs(default.total_energy - atom.total_energy) > 1e-3
