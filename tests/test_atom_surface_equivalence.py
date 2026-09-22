"""The canonical `Atom` surface reproduces the pre-change numbers.

Every value below was measured on the OLD flat-keyword surface, before the
constructor was rewritten, and is asserted here through the NEW surface.  A
failure means the rename changed physics — which only four inputs are allowed
to do (see `tests/test_atom_inputs_wired.py`): the plasma, the accuracy
tier's ``r_max``/``n_grid``, and the multi-configuration ``qed_backend``.

The panel covers the five cases the migration had to preserve bit-for-bit:
a closed shell (Ne), an open shell with CI averaging (N), a
multi-configuration CI run (Be), a B-spline run, and a QED/Breit run
(Fe XVII).  Two extra rows (a shooting Rydberg run and a high-Z
hydrogen-like ion) guard the corners where the box and the knot grid bite.
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
#: still bit-identical, which is what this gate checks; the local-exchange
#: B-spline SCF has a pre-existing convergence gap at Z=26 that the rename
#: neither caused nor fixed.
NON_CONVERGED = {"fe17_slater_bspline"}


@pytest.mark.parametrize("name,kwargs,expected", [
    pytest.param(n, kw, e, id=n) for n, (kw, e) in PANEL.items()
])
def test_canonical_form_reproduces_the_pre_change_number(name, kwargs,
                                                         expected):
    atom = Atom(**kwargs).solve()
    if name not in NON_CONVERGED:
        assert atom.converged, name
    assert atom.total_energy == pytest.approx(expected, abs=1e-12), name


def test_the_ne_panel_keeps_every_orbital():
    """Bit-identical orbitals, not just the total energy."""
    atom = Atom(10, config="1s2 2s2 2p6").solve()
    assert atom.total_energy == pytest.approx(-128.6299783505843, abs=1e-12)
    assert atom.orbital_energies == pytest.approx(
        {"1s1/2": -32.817447803729735, "2s1/2": -1.9358521624506706,
         "2p1/2": -0.8528343072534854, "2p3/2": -0.8482721633409784},
        abs=1e-12)


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
