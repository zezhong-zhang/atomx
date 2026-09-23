"""Within-(J, parity)-block CI tests for the J-resolved CSF basis.

These exercise the J-resolved CI path added for the Fe XVII acceptance
target (1s² 2s² 2p⁶ ground + 2p⁵ 3l and 2s 2p⁶ 3l L-shell excitations,
37 J-resolved CSFs).

**What the headline gates are, after the 2026-09 audit.**  The three
original "acceptance" gates were structural counts (≥ 30 distinct levels,
``distinct == len(csfs)``) plus a 200 eV FAC band.  All three pass unchanged
on ``matrix._build_ci_matrix_approximate`` — the kernel the repo removed for
being physically wrong: it returns -1149.2972523 Ha for this very CSF list
against the exact -1148.5051465 Ha, i.e. **0.79 Ha below the exact minimum
of its own space** (non-variational), while still producing 37 distinct
levels inside the FAC band.  Counts and wide bands cannot see that.

The file now gates the two properties that can:

* ``test_fe17_spectrum_matches_the_mscheme_reference`` — the whole 37-level
  spectrum against the m-scheme oracle
  (``benchmarks/mscheme_ci_reference``) on the identical orbitals and the
  identical nuclear potential: agreement is 9e-13 Ha (exact kernel) versus
  0.79 Ha (removed kernel).  The oracle shares the *radial* integrals but
  assembles the determinant Hamiltonian independently (see the honesty note
  in ``tests/test_ci_matrix_vs_mscheme_reference.py``).
* ``test_fe17_ground_state_is_variational`` — the Rayleigh-Ritz floor: a
  variational CI cannot lie below the exact lowest eigenvalue of the space
  it spans, which the oracle supplies.  The removed kernel violates it by
  0.79 Ha; the exact kernel satisfies it to 9e-13 Ha.

The FAC comparison is tightened from 200 eV to 10 eV: the measured residual
of the exact kernel is 4.7-5.2 eV per level (this machine, FAC 1.1.5, the
in-file script), while the removed kernel sits 28.4-29.1 eV out, so the band
now separates the two.

Availability of ``sfac`` is probed by *running* it (one trivial deck): a
binary that exists but cannot load its gcc runtime used to disable the only
external comparison in this file silently.  The probe repairs the documented
``FAC_DYLD_LIB`` case itself (via ``gfortran -print-file-name``) before
giving up.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

from atomx import Atom, Correlation
from atomx.constants import ALPHA, HARTREE_EV
from atomx.methods.mcdhf import matrix
from atomx.methods.mcdhf.config import enumerate_jj_csfs
from atomx.methods.mcdhf.solver import MCDHFSolver

BENCH_DIR = Path(__file__).resolve().parent.parent / "benchmarks"
sys.path.insert(0, str(BENCH_DIR))

from mscheme_ci_reference import (  # noqa: E402
    mscheme_levels,
    patterns_from_configs,
)

FE17_CONFIGS = [
    "1s2 2s2 2p6",
    "1s2 2s2 2p5 3s1",
    "1s2 2s2 2p5 3p1",
    "1s2 2s2 2p5 3d1",
    "1s2 2s1 2p6 3s1",
    "1s2 2s1 2p6 3p1",
    "1s2 2s1 2p6 3d1",
]

#: Agreement of ``build_ci_matrix`` with the m-scheme oracle on the Fe XVII
#: space and orbitals (measured 9.1e-13 Ha); the removed kernel is 0.79 Ha
#: out, so the gate discriminates by twelve orders of magnitude.
SPECTRUM_TOL_HA = 1e-8

#: Per-level FAC band.  Measured residual of the exact kernel: 4.7-5.2 eV
#: (max 5.21 eV); the removed legacy kernel: 28.4-29.1 eV (max 29.1 eV).
FAC_ABS_TOL_EV = 10.0


def _fe17_solve():
    """Solve Fe XVII and capture the exact CI inputs the solver used.

    ``build_ci_matrix`` is spied on rather than re-derived: the orbitals,
    B-spline workspace, ``ns`` and — critically — the solver's own
    ``nuclear_potential`` are needed for an apples-to-apples oracle call.
    At Z = 26 the finite-nucleus/point-nucleus difference is 4.4 mHa, four
    orders larger than the 1e-8 gate, so it must be carried over.
    """
    captured: dict = {}
    original = matrix.build_ci_matrix

    def spy(*args, **kwargs):
        H = original(*args, **kwargs)
        captured.update(
            csfs=args[0],
            orbitals_dict=dict(args[1]),
            bsp_data=args[3],
            Z=args[4],
            ns=args[5],
            nuclear_potential=kwargs.get("nuclear_potential"),
        )
        return H

    matrix.build_ci_matrix = spy
    try:
        atom = Atom(
                    Z=26,
                    charge=16,
                    config=FE17_CONFIGS,
                    corrections="none",
                    correlation=Correlation(j_resolved_csfs=True),
               )
        atom.solve()
    finally:
        matrix.build_ci_matrix = original
    return atom, captured


def _mscheme_spectrum(captured):
    """Exact spectrum of the same CSF space on the same orbitals."""
    matrix._SLATER_MEMO.clear()
    if captured.get("nuclear_potential") is not None:
        matrix._SLATER_MEMO["_nuclear_potential"] = captured["nuclear_potential"]
    return mscheme_levels(
        captured["orbitals_dict"], captured["bsp_data"], captured["Z"],
        captured["ns"], patterns_from_configs(FE17_CONFIGS), M2=0,
    )


@pytest.fixture(scope="module")
def fe17() -> dict:
    """One Fe XVII solve plus its m-scheme reference spectrum."""
    atom, captured = _fe17_solve()
    energies = np.asarray(atom._result.state_energies)
    oracle, n_det = _mscheme_spectrum(captured)
    return {
        "atom": atom,
        "captured": captured,
        "energies": energies,
        "oracle": np.asarray(oracle),
        "n_det": n_det,
        "levels_ev": (energies - energies[0]) * HARTREE_EV,
    }


@pytest.mark.snapshot
def test_fe17_csf_count_is_37():
    """The 7-configuration Fe XVII basis enumerates exactly 37 CSFs.

    A generator count (HYP-165 M3): frozen self-regression, no physics.
    """
    csfs = []
    for cfg in FE17_CONFIGS:
        csfs.extend(enumerate_jj_csfs(cfg))
    assert len(csfs) == 37


def test_fe17_spectrum_matches_the_mscheme_reference(fe17):
    """EVERY Fe XVII level, not just the ground state, from the oracle.

    ``build_ci_matrix`` spans the same determinants as the oracle's
    m-scheme construction, so the sorted spectra must coincide.  This
    replaces the old ``distinct == len(csfs)`` count, which was an
    assertion about accidental degeneracies in a 37x37 matrix: it passed
    unchanged for the removed kernel (37 distinct levels, 0.79 Ha below
    the exact minimum), while this gate is out by 0.79 Ha for that kernel.
    """
    energies = fe17["energies"]
    oracle = fe17["oracle"]
    assert fe17["n_det"] == len(fe17["captured"]["csfs"]) == len(energies)
    worst = float(np.abs(np.sort(energies) - oracle).max())
    assert worst < SPECTRUM_TOL_HA, (
        f"Fe XVII spectrum differs from the m-scheme oracle by {worst:.3e} Ha "
        f"(gate {SPECTRUM_TOL_HA:g}); the removed legacy kernel is 0.79 Ha out")


def test_fe17_ground_state_is_variational(fe17):
    """Rayleigh-Ritz floor and the reference-determinant ceiling.

    Floor: the lowest eigenvalue of the *exact* Hamiltonian restricted to
    the space the CSF list spans is the oracle's minimum.  A variational
    method cannot return anything below it; the removed legacy kernel
    returned -1149.2972523 Ha against the oracle's -1148.5061392 Ha at the
    very same orbitals, i.e. 0.7911 Ha below the floor.

    Ceiling: the space contains 1s² 2s² 2p⁶, so the CI minimum must also be
    at or below that determinant's expectation value on the same orbitals
    (which is exactly the closed-shell 1-CSF energy — the even-parity J=0
    block of this list is one-dimensional).  Measured: -1148.5051465 Ha
    against the 1-CSF -1148.4263440 Ha, 78.8 mHa of genuine correlation.
    """
    captured = fe17["captured"]
    e_ci = float(np.min(fe17["energies"]))
    e_floor = float(np.min(fe17["oracle"]))
    assert e_ci >= e_floor - SPECTRUM_TOL_HA, (
        f"Fe XVII CI ground state {e_ci!r} Ha lies "
        f"{e_floor - e_ci:.4f} Ha BELOW the exact minimum of its own space "
        f"({e_floor!r} Ha) — non-variational")

    closed = enumerate_jj_csfs("1s2 2s2 2p6")
    assert len(closed) == 1
    H_ref = matrix.build_ci_matrix(
        closed, captured["orbitals_dict"], None, captured["bsp_data"],
        captured["Z"], captured["ns"],
        nuclear_potential=captured["nuclear_potential"])
    e_ref = float(H_ref[0, 0])
    assert e_ci <= e_ref + SPECTRUM_TOL_HA, (
        f"Fe XVII CI ground state {e_ci!r} Ha is above the "
        f"1s2 2s2 2p6 reference determinant {e_ref!r} Ha contained in the "
        f"same space")
    # The ground state of a space whose only even-parity J = 0 CSF is the
    # closed shell must be that CSF: J = 0, even parity.
    assert int(fe17["atom"]._result.state_J2[int(np.argmin(fe17["energies"]))]) == 0


def _gfortran_runtime_dir() -> str | None:
    """Directory of ``libgfortran.5.dylib``, or None if it cannot be found.

    The exact remediation ``run_sfac``'s own error message prescribes
    (``FAC_DYLD_LIB=$(dirname $(gfortran -print-file-name=...))``), done
    programmatically so the probe does not have to skip on a repairable
    environment problem.
    """
    if os.environ.get("FAC_DYLD_LIB"):
        return None  # configured and still failing: the skip is authoritative
    gfortran = shutil.which("gfortran")
    if gfortran is None:
        return None
    try:
        lib = subprocess.run(
            [gfortran, "-print-file-name=libgfortran.5.dylib"],
            capture_output=True, text=True, timeout=60,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    if not lib or not Path(lib).exists():
        return None
    return str(Path(lib).parent)


def _sfac_available() -> bool:
    """True when an sfac *runs*, not merely when a file of that name exists.

    Path existence was not enough: a relocated copy of the binary that cannot
    load its gcc runtime (``dyld: Library not loaded: @rpath/libgfortran.5.dylib``)
    satisfied the old check and then failed the test as if the physics
    disagreed.  ``run_sfac`` raises :class:`SfacUnavailable` for that case,
    and this probe repairs it once — locating the runtime with ``gfortran``
    and re-probing — before reporting the file's only external comparison as
    unavailable.  The repair is written to ``fac_runner.DYLD_LIB`` so the
    FAC test itself inherits it (measured on this machine: without the
    repair the file reported ``4 passed, 1 skipped`` although
    ``/usr/local/bin/sfac`` is present, executable and fully functional).
    """
    try:
        import fac_runner
        from fac_runner import SfacUnavailable, resolve_sfac_bin, run_sfac
    except ImportError:
        return False
    binary = resolve_sfac_bin()
    if binary is None or not (Path(binary).exists() and os.access(binary, os.X_OK)):
        return False

    def probe() -> bool:
        with tempfile.TemporaryDirectory() as tmp:
            deck = Path(tmp) / "probe.sf"
            deck.write_text("SetAtom('He')\n")
            try:
                run_sfac(deck, workdir=Path(tmp), timeout=120)
            except SfacUnavailable:
                return False
            except Exception:
                # The probe is not a valid FAC deck for every build; only a
                # load failure is disqualifying here.
                return True
        return True

    if probe():
        return True
    lib_dir = _gfortran_runtime_dir()
    if lib_dir is None:
        return False
    fac_runner.DYLD_LIB = lib_dir
    return probe()


@pytest.mark.skipif(not _sfac_available(),
                    reason="sfac binary not available")
def test_fe17_first_10_levels_vs_fac(fe17):
    """Compare atomx Fe XVII first 10 levels to FAC sfac (LS-shell band).

    **Tolerance.**  We compare ``E_n - E_0`` for n = 1..9 to the FAC
    ``2*7 3*1`` reference (Ne-like Fe with one L-shell hole and one
    M-shell electron).  The residual is the orbital protocol: atomx uses
    MCDHF Guerra averaging (re-optimises all orbitals in the J-resolved CSF
    mixture), FAC's ``OptimizeRadial`` only optimises the ``n2``
    configuration and keeps ``n3`` frozen.  Measured on this machine:

    * exact kernel: 4.70-5.21 eV per level;
    * removed legacy kernel: 28.4-29.1 eV per level (it also lands 0.79 Ha
      below the variational floor, which the two gates above catch).

    10 eV therefore separates the correct matrix from the removed one; the
    old 200 eV band did not (it was 6.9x above the legacy kernel's error).
    """
    sys.path.insert(0, str(BENCH_DIR))
    from fac_runner import (  # type: ignore[import-not-found]
        SfacUnavailable,
        parse_levels,
        run_sfac,
    )

    FAC_SCRIPT = """SetAtom('Fe')
Closed('1s')
Config('2*8', group='n2')
Config('n3', '2*7 3*1')
ConfigEnergy(0)
OptimizeRadial(['n2'])
ConfigEnergy(1)
Structure('fe17.lev.b', ['n2', 'n3'])
MemENTable('fe17.lev.b')
PrintTable('fe17.lev.b', 'fe17.lev', 1)
"""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        sf = tmp_path / "fe17.sf"
        sf.write_text(FAC_SCRIPT)
        try:
            run_sfac(sf, workdir=tmp_path, timeout=300)
        except SfacUnavailable as exc:
            pytest.skip(f"sfac cannot run here: {exc}")
        fac_levels = parse_levels(tmp_path / "fe17.lev")

    fac_levels.sort(key=lambda r: r["ENERGY_eV"])
    fac_e = np.array([r["ENERGY_eV"] for r in fac_levels[:10]])

    assert len(fe17["levels_ev"]) >= 10, "atomx returned < 10 levels"
    diffs = fe17["levels_ev"][:10] - fac_e
    assert np.all(np.abs(diffs) <= FAC_ABS_TOL_EV), (
        f"atomx Fe XVII first 10 levels differ from FAC by more than "
        f"{FAC_ABS_TOL_EV} eV: diffs = {diffs}"
    )


def _dirac_hydrogenic(n: int, kappa: int, Z: float = 1.0) -> float:
    """Closed-form point-nucleus Dirac energy, Hartree (``c^2`` from alpha)."""
    c2 = 1.0 / ALPHA ** 2
    gamma = float(np.sqrt(kappa * kappa - (Z * ALPHA) ** 2))
    ratio = Z * ALPHA / (n - abs(kappa) + gamma)
    return c2 * ((1.0 + ratio * ratio) ** -0.5 - 1.0)


def _hlike_solved():
    csfs = []
    for cfg in ["1s1", "2s1", "2p1"]:
        csfs.extend(enumerate_jj_csfs(cfg))
    solver = MCDHFSolver(Z=1, csfs=csfs, ns=30, ks=7, use_physics_grid=False)
    return solver.solve(max_iter=3, scf_tol=1e-4, ci_tol=1e-6)


def _by_dominant_label(res):
    """Map each CSF label to its state's energy via dominant eigenvector
    component — the same pairing ``MCDHFResult.state_dominant_J2`` uses.

    Position-wise zipping of ``csfs`` with sorted ``state_energies`` is
    WRONG here: ``enumerate_jj_csfs`` orders the 2p3/2 CSF before 2p1/2,
    while the sorted eigenvalues place the more-bound 2p1/2 first, and
    2s1/2 sits 5e-10 Ha from 2p1/2 (Dirac degeneracy at discretisation
    accuracy).  Only the dominant-component pairing is invariant.
    """
    out = {}
    for k in range(res.n_states):
        dom = int(np.argmax(np.abs(res.mixing_coefficients[k])))
        out[res.csfs[dom].label.split()[0]] = float(res.state_energies[k])
    return out


def test_J_block_lift_h2plus_n2_p_states():
    """H-like Z=1: the J-block lift is identically zero for one-electron CSFs.

    No CSF in this basis has two open subshells, so the J-coupled Coulomb
    correction the file is about cannot contribute at all: each level must
    be the *exact* point-nucleus Dirac energy of its spinor.  Measured
    against the closed form (``alpha`` from ``atomx.constants``):
    1s -0.500006656596 (1.2e-12 Ha), 2s -0.125002080739 (5.5e-10 Ha).

    History: the 2p levels of this solve used to miss the closed form by
    ~0.017 Ha with the fine-structure ordering inverted — a spectator
    mean-field defect (R7 self weight −J_i), fixed 2026-09-23 and gated
    by ``test_one_electron_2p_levels_match_the_dirac_closed_form`` below.
    """
    res = _hlike_solved()
    assert res.n_states == 4
    assert len(res.csfs) == 4
    by_label = _by_dominant_label(res)
    assert by_label["1s1/2^1"] == pytest.approx(
        _dirac_hydrogenic(1, -1), abs=1e-8)
    assert by_label["2s1/2^1"] == pytest.approx(
        _dirac_hydrogenic(2, -1), abs=1e-8)


def test_one_electron_2p_levels_match_the_dirac_closed_form():
    """H-like 2p must reproduce the closed-form Dirac energy.

    Fixed 2026-09-23: the R7 spectator mean field carried a self weight
    of ``q_ref − 1`` (−1 for a virtual absent from the reference), which
    under this codebase's angular conventions is a pure ``−J_i``
    anti-screening term (self-exchange is structurally zero for i == j,
    so nothing cancels it).  H-like 2p was over-bound to −0.3616 Ha with
    the ordering inverted.  The row is now the parent configurations'
    own core: for H-like the 2p parent's core is empty (hole at 1s), so
    the spectator sees the bare-nucleus field and all four n ≤ 2 levels
    match the closed form to ≤ 5.5e-10 Ha
    (``docs/test_audit/PHYSICS_hlike_2p_spectator.md``).
    """
    res = _hlike_solved()
    by_label = _by_dominant_label(res)
    assert by_label["2p3/2^1"] == pytest.approx(
        _dirac_hydrogenic(2, -2), abs=1e-6)
    assert by_label["2p1/2^1"] == pytest.approx(
        _dirac_hydrogenic(2, 1), abs=1e-6)
    # The physical ordering: 2p3/2 sits ABOVE 2p1/2 by the fine-structure
    # splitting (1.66e-6 Ha at Z=1).
    assert by_label["2p3/2^1"] > by_label["2p1/2^1"]
