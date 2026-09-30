"""Validate ``_slater_integral_slag6`` vs the power-series reference.

The shooting solver carries two quadrature backends for the Slater
integral F^L(f1, f2):

* ``_slater_integral`` (``slater_method='power_series'``) — Salvat/DHFS.f
  power-series ODE.  Bit-accurate for smooth densities; one recent
  concern was that the inner series might stop converging and the
  ``MAX_NSTEPS`` cap kick in for very sharp heavy-Z 1s densities.

* ``_slater_integral_slag6`` (``slater_method='slag6'``, default) —
  direct 6-point-Lagrange forward and backward integrals for the
  Hartree Y^L(r), matching the same SLAG6 routine that the outer
  SCF uses everywhere else (``_sfield``).

These tests lock in that both backends agree across the full Z range
on realistic bound-state densities, so either one is a valid choice
for downstream use.  Heavy-Z total-energy accuracy vs FAC/sfac is a
separate concern (see ``tests/test_fac_dfs_shooting.py``) — this
file is scoped to the Slater-integral quadrature itself.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from atomx import Atom, Exchange
from atomx.constants import HARTREE_EV
from atomx.numerics.shooting.grid import sgrid
from atomx.numerics.shooting.coulomb import (
    _slater_integral,
    _slater_integral_slag6,
)


def _hydrogenic_density_on_grid(z_eff: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a hydrogenic-1s radial density on the default shooting grid.

    Returns the (r, difr, f) triple with f normalised so that
    ``∫ f dr = 1`` on the quadrature grid.  Shape matches a real Dirac
    density well enough that the Slater-integral quadratures see the
    same kinds of small-r and large-r behaviour they'd see in an SCF.
    """
    n_grid = 1000
    r_max = 75.0
    r2 = 1.0e-7
    drn = max(2.0 * r_max / n_grid, 0.5)
    r, difr, np_ = sgrid(r_max, r2, drn, n_grid)

    f = np.zeros(np_)
    for i in range(np_):
        p = 2.0 * z_eff ** 1.5 * r[i] * math.exp(-z_eff * r[i])
        f[i] = p * p
    norm = 0.0
    for i in range(np_):
        norm += f[i] * difr[i]
    f *= 1.0 / norm
    return r, difr, f


@pytest.mark.parametrize("z_eff", [9.0, 30.0, 70.0])
@pytest.mark.parametrize("L", [0, 1, 2])
def test_slag6_matches_power_series_for_hydrogenic_density(
    z_eff: float, L: int
) -> None:
    """F^L(f, f) must agree between backends to 10⁻⁷ relative for a
    hydrogenic-1s density up to Z_eff = 70.

    If either backend develops a loss of precision (e.g. the
    power-series hits its ``MAX_NSTEPS`` cap for very sharp inner
    peaks, or SLAG6 mishandles the boundary), that will show up
    here first.
    """
    r, difr, f = _hydrogenic_density_on_grid(z_eff)
    np_ = r.size

    f_ps = _slater_integral(f, f, L, r, difr, np_)
    f_sg = _slater_integral_slag6(f, f, L, r, difr, np_)

    rel = abs(f_sg - f_ps) / max(abs(f_ps), 1.0e-30)
    assert rel < 1e-7, (
        f"Z_eff={z_eff}, L={L}: slag6={f_sg:.12e}, "
        f"power-series={f_ps:.12e}, rel diff {rel:.2e}"
    )


def test_slater_method_flag_changes_total_energy_path() -> None:
    """Both ``slater_method`` values must produce a solvable SCF at low Z
    and give closely-agreeing total energies — SLAG6 is the default
    but ``power_series`` must remain a valid user choice.

    We use Ne (Z=10 closed-shell), where the two backends agree to
    better than 10⁻⁷ relative on the total binding energy (both
    recover the 3501.1 eV sfac reference to ≤ 2 × 10⁻⁵).
    """
    results = {}
    for method in ("slag6", "power_series"):
        atom = Atom(
                    Z=10,
                    charge=0,
                    config="1s2 2s2 2p6",
                    engine="shooting",
                    exchange=Exchange(model="gu", slater_method=method),
               )
        atom.solve()
        assert atom.total_energy is not None
        results[method] = atom.total_energy * HARTREE_EV

    rel = (abs(results["slag6"] - results["power_series"])
           / abs(results["power_series"]))
    assert rel < 1e-6, (
        f"Backends disagree at Z=10: slag6={results['slag6']:.6f} eV, "
        f"power_series={results['power_series']:.6f} eV, rel={rel:.2e}"
    )


def test_slater_method_rejects_unknown_value() -> None:
    """Unknown ``slater_method`` values raise ``ValueError`` at
    ``Atom`` construction time — fail fast rather than silently."""
    with pytest.raises(ValueError, match="slater_method"):
        Atom(
             Z=10,
             config="1s2 2s2 2p6",
             engine="shooting",
             exchange=Exchange(model="gu", slater_method="nope"),
        )


def test_slag6_is_default() -> None:
    """``slater_method='slag6'`` is the harness default.  This test
    pins that choice so a silent flip back to power-series (which is
    available but no longer the recommended default) would be caught.
    """
    atom = Atom(Z=10, config="1s2 2s2 2p6", engine="shooting", exchange="gu")
    assert atom.slater_method == "slag6"


@pytest.mark.parametrize(
    "Z, config, ref_Eh",
    [
        # Neutral-atom DHFS.f reference totals (point nucleus, Slater α=1.5).
        # Source: benchmarks/results_shooting.csv — Salvat's independent
        # Fortran radial/DHFS (Z010.in, α=1.5, 1000-pt grid, R_max=75 a₀;
        # re-run 2026-09-22 → "Total binding energy = -1.2862339E+02 Eh").
        # This is one of FOUR pins of the Ne ground state in this suite, each
        # a different physics question:
        #   -128.69197   point nucleus, HF exchange,     B-spline DFSolver
        #                (tests/test_scf.py:142; DBSR_HF -128.69192580)
        #   -128.6299783 Fermi nucleus, HF exchange,     B-spline
        #                (tests/test_atom_surface_equivalence.py:32/105)
        #   -128.623390  point nucleus, Slater α=1.5,    shooting   <- here
        #   -128.623437  Fermi nucleus, Slater α=1.5,    B-spline
        #                (tests/test_exchange_model_is_honoured.py:9)
        (10, "1s2 2s2 2p6", -128.623390000),
        (26, "1s2 2s2 2p6 3s2 3p6 3d6 4s2", -1271.199400000),
        (82, "1s2 2s2 2p6 3s2 3p6 3d10 4s2 4p6 4d10 5s2 5p6 4f14 5d10 6s2 6p2",
         -20918.805),
    ],
)
@pytest.mark.parametrize("method", ["power_series", "slag6"])
def test_both_backends_match_dhfs_fortran_for_neutral_atoms(
    Z: int, config: str, ref_Eh: float, method: str,
) -> None:
    """Neither backend is allowed to regress the bit-for-bit DHFS.f
    agreement that the shooting SCF delivers for neutral atoms.

    Gate 5 × 10⁻⁸ relative: tighter than both backends' actual
    agreement on the current panel (~2 × 10⁻⁸), loose enough to absorb
    platform FP drift.
    """
    from atomx.config.csf import _parse_config_string
    from atomx.numerics.shooting.local import ShootingSolver

    subs = _parse_config_string(config)
    solver = ShootingSolver(
        Z=Z, subshells=subs,
        n_grid=1000, r_max=75.0, alpha_x=1.5,
        slater_method=method,
    )
    result = solver.solve()
    rel = abs(result.total_energy - ref_Eh) / abs(ref_Eh)
    assert rel < 5e-8, (
        f"Z={Z}, {method}: atomx={result.total_energy:.9f} Ha, "
        f"DHFS.f={ref_Eh:.6f} Ha, rel={rel:.2e}"
    )
