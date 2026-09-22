"""Gate ``build_ci_matrix`` against an independent m-scheme CI oracle.

``benchmarks/mscheme_ci_reference.py`` rebuilds the CI from the
Wigner-Eckart theorem with sympy 3j symbols and explicit Slater-Condon
rules in the m-scheme determinant basis.  It shares no angular code with
``atomx.methods.mcdhf.matrix`` — only the radial integrals and the
orbitals — so agreement between the two is a genuine cross-check.

Every test here is now a plain assertion: the N1 defect described in
``docs/CORRELATION_SD_RESEARCH_2026-09-20.md`` is fixed.  ``build_ci_matrix``
expands each CSF over the Slater determinants of its configuration and
projects the exact determinant Hamiltonian, so the CSF matrix *is*
``<CSF_r|H|CSF_s>``.

History note, because it changes the numbers: measuring these gates
exposed a bug in the oracle itself.  ``mscheme_levels`` skipped the
exchange integral whenever the *direct* one vanished by angular selection
rules (``if v == 0.0: continue``), which broke rotational invariance — the
Be ``1s2 2s1 2p1`` J=1 level moved by 40 mHa between the ``2M = 0`` and
``2M = 2`` blocks.  The bug is fixed in
``benchmarks/mscheme_ci_reference.py`` (see the comment there and
:func:`test_oracle_is_rotationally_invariant` below); the pre-fix oracle
numbers quoted in the N1 document for open-shell off-diagonal elements
(Be singles ``-14.595092``, Be SD ``fac_frozen`` ``-14.654637``) were
22 mHa too low and are superseded by the values asserted here.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

BENCH_DIR = Path(__file__).resolve().parent.parent / "benchmarks"
sys.path.insert(0, str(BENCH_DIR))

from mscheme_ci_reference import (  # noqa: E402
    mscheme_levels,
    patterns_from_configs,
)

import atomx.methods.mcdhf.solver as _solver_mod  # noqa: E402
from atomx.methods.mcdhf.config import enumerate_jj_csfs  # noqa: E402
from atomx.methods.mcdhf.solver import MCDHFSolver  # noqa: E402

# Exact non-relativistic ground state of Be (Drake).  The Dirac-Coulomb
# value with a finite Fermi nucleus is ~2.5 mHa lower.
BE_EXACT_NR = -14.66736
BE_DC_LOWER_BOUND = BE_EXACT_NR - 0.005


def _run_capturing_ci(config_strings, Z=4, **kwargs):
    """Solve and return (result, csfs, ci_solve arguments).

    The CI call is captured by wrapping
    ``atomx.methods.mcdhf.solver.ci_solve`` — the solver imports the name
    into its own module, so patching the module attribute is enough.
    """
    captured = {}
    original = _solver_mod.ci_solve

    def spy(csfs, orbitals_dict, orbital_basis, bsp_data, Z_, ns, **kw):
        res = original(csfs, orbitals_dict, orbital_basis, bsp_data, Z_, ns, **kw)
        captured.update(csfs=csfs, orbitals_dict=dict(orbitals_dict),
                        bsp_data=bsp_data, Z=Z_, ns=ns, ci_result=res,
                        nuclear_potential=kw.get("nuclear_potential"),
                        convol_cache=kw.get("convol_cache"))
        return res

    _solver_mod.ci_solve = spy
    try:
        csfs = []
        for cfg in config_strings:
            csfs.extend(enumerate_jj_csfs(cfg))
        solver = MCDHFSolver(Z=Z, csfs=csfs, ns=113, ks=9, **kwargs)
        result = solver.solve(max_iter=60, scf_tol=1e-6)
    finally:
        _solver_mod.ci_solve = original
    return result, csfs, captured


def _build(captured):
    """Rebuild H exactly as ``ci_solve`` did, same workspace and nucleus.

    Passing the solver's own ``nuclear_potential`` matters: a point nucleus
    reproduces the finite-nucleus CI to ~1e-8 Ha at Z = 4, which is the
    same order as the tolerances used here.
    """
    from atomx.methods.mcdhf.matrix import build_ci_matrix

    return build_ci_matrix(
        captured["csfs"], captured["orbitals_dict"], None, captured["bsp_data"],
        captured["Z"], captured["ns"],
        nuclear_potential=captured.get("nuclear_potential"),
        convol_cache=captured.get("convol_cache"))


def _reference_levels(captured, config_strings, M2=0):
    return mscheme_levels(
        captured["orbitals_dict"], captured["bsp_data"], captured["Z"],
        captured["ns"], patterns_from_configs(config_strings), M2=M2,
    )


# ---------------------------------------------------------------------------
# Passing gates
# ---------------------------------------------------------------------------

def test_closed_shell_ci_equals_mscheme_reference():
    """1s2 2s2 is one determinant: the two constructions must agree.

    This is the oracle's own calibration against the repo (and through
    ``_diagonal_element`` against ``DFSolver._compute_total_energy``).
    """
    cfgs = ["1s2 2s2"]
    result, _csfs, cap = _run_capturing_ci(cfgs, radial_optimization="fac_frozen")
    evals, n_det = _reference_levels(cap, cfgs)
    assert n_det == 1
    assert result.total_energy == pytest.approx(float(evals[0]), abs=1e-9)


def test_csf_list_is_a_basis_of_the_intended_space():
    """The SD CSF count equals the determinant-space dimension, per config.

    Guards against the tempting-but-wrong diagnosis that the non-variational
    SD energy is caused by a linearly dependent / duplicated CSF list
    ("399 CSFs from 99 occupation vectors").  It is not: the multiplicities
    are exactly the m-scheme dimensions.
    """
    import itertools

    from atomx.methods.mcdhf.excitations import (
        generate_sd_excitations,
        suggest_virtual_orbitals,
    )

    def subshell_m2_dist(twoj, q):
        ms = list(range(-twoj, twoj + 1, 2))
        dist: dict[int, int] = {}
        for combo in itertools.combinations(ms, q):
            dist[sum(combo)] = dist.get(sum(combo), 0) + 1
        return dist

    def space_dim(patterns):
        total = 0
        for pattern in patterns:
            dist = {0: 1}
            for (_n, kappa, q) in pattern:
                nd = subshell_m2_dist(2 * abs(kappa) - 1, q)
                new: dict[int, int] = {}
                for a, ca in dist.items():
                    for b, cb in nd.items():
                        new[a + b] = new.get(a + b, 0) + ca * cb
                dist = new
            total += dist.get(0, 0)
        return total

    configs = generate_sd_excitations(
        "1s2 2s2", suggest_virtual_orbitals(4, 1), core_orbitals=None,
        max_excitations=2)
    n_csf = 0
    for cfg in configs:
        csfs = enumerate_jj_csfs(cfg)
        patterns = []
        for csf in csfs:
            p = tuple(sorted((sh.n, sh.kappa, int(round(sh.occupation)))
                             for sh in csf.subshells))
            if p not in patterns:
                patterns.append(p)
        assert len(csfs) == space_dim(patterns), (
            f"{cfg}: {len(csfs)} CSFs but determinant space has "
            f"dimension {space_dim(patterns)}")
        n_csf += len(csfs)
    assert n_csf == 399


def test_ci_matrix_is_symmetric():
    """``build_ci_matrix`` must return a symmetric matrix."""
    cfgs = ["1s2 2s2", "1s2 2s1 2p1"]
    _result, _csfs, cap = _run_capturing_ci(cfgs,
                                            radial_optimization="fac_frozen")
    H = _build(cap)
    asymmetric = np.abs(H - H.T).max()
    assert asymmetric < 2e-11, f"H - H.T max = {asymmetric}"


# ---------------------------------------------------------------------------
# The N1 gates that used to be strict xfails
# ---------------------------------------------------------------------------

def test_two_open_subshell_block_matches_mscheme_reference():
    """Open-shell J-blocks must reproduce the exact CSF matrix elements.

    Was the N1 xfail: atomx gave -14.708567 Ha (0.13 Ha below the m-scheme
    oracle on the SAME orbitals); the exact projection gives the reference
    energy itself, as Brillouin's theorem requires
    (``<1s2 2s2|H|1s2 2s1 2p1> ~ 0``).
    """
    cfgs = ["1s2 2s2", "1s2 2s1 2p1"]
    result, _csfs, cap = _run_capturing_ci(cfgs,
                                           radial_optimization="fac_frozen")
    evals, _ = _reference_levels(cap, cfgs)
    assert result.total_energy == pytest.approx(float(evals[0]), abs=1e-6)


def test_single_config_open_shell_block_matches_mscheme_reference():
    """Was the N1 xfail: the J=1 block of ``1s2 2s2p`` came out at -14.711459
    Ha against the m-scheme -14.490933 / -14.468254 Ha on identical
    orbitals, with ``_off_diagonal_J_correction`` (+0.431282) contributing
    more than twice its own base element (-0.193857).
    """
    cfgs = ["1s2 2s1 2p1"]
    result, _csfs, cap = _run_capturing_ci(cfgs,
                                           radial_optimization="fac_frozen")
    evals, _ = _reference_levels(cap, cfgs)
    assert result.total_energy == pytest.approx(float(evals[0]), abs=1e-6)


@pytest.mark.slow
def test_singles_correlation_is_variational():
    """Was the N1 xfail: Be singles (29 CSF, fac_frozen) returned -15.400463
    Ha, 0.8 Ha below the exact ground state; the exact value is the
    reference energy itself, which is Brillouin's theorem for a singles
    expansion on (near-)HF orbitals.
    """
    configs = [
        "1s2 2s2", "1s1 2s2 2p1", "1s1 2s2 3s1", "1s1 2s2 3p1",
        "1s1 2s2 3d1", "1s2 2s1 2p1", "1s2 2s1 3s1", "1s2 2s1 3p1",
        "1s2 2s1 3d1",
    ]
    result, _csfs, cap = _run_capturing_ci(configs,
                                           radial_optimization="fac_frozen")
    assert result.total_energy >= BE_DC_LOWER_BOUND
    evals, _ = _reference_levels(cap, configs)
    assert result.total_energy == pytest.approx(float(evals[0]), abs=1e-6)


def test_no_coupling_between_different_total_j():
    """Was the N1 xfail: the builder returned non-zero elements between CSFs
    of different total J (e.g. -14.479364 between ``1s2 2s1 2p1`` J=0 and
    J=1), masked only by the (J, parity) filter in ``ci._solve_blocked``.
    The exact projection writes nothing outside the block, so the elements
    are exactly ``0.0``.
    """
    cfgs = ["1s2 2s1 2p1"]
    _result, _csfs, cap = _run_capturing_ci(cfgs,
                                            radial_optimization="fac_frozen")
    csfs = cap["csfs"]
    H = _build(cap)
    worst = 0.0
    for i, a in enumerate(csfs):
        for j, b in enumerate(csfs):
            if (a.total_J2, a.parity) != (b.total_J2, b.parity):
                worst = max(worst, abs(H[i, j]))
    assert worst < 1e-10


@pytest.mark.slow
def test_sd_correlation_is_variational():
    """Was the N1 xfail: Be SD (399 CSF, fac_frozen) returned -16.358105 Ha,
    1.7 Ha below the exact ground state.  The exact projection lands at
    -14.632497 Ha on the same orbitals in the same 399-dimensional space.
    """
    from atomx.methods.mcdhf.excitations import (
        generate_sd_excitations,
        suggest_virtual_orbitals,
    )

    configs = generate_sd_excitations(
        "1s2 2s2", suggest_virtual_orbitals(4, 1), core_orbitals=None,
        max_excitations=2)
    result, _csfs, cap = _run_capturing_ci(configs,
                                           radial_optimization="fac_frozen")
    assert result.total_energy >= BE_DC_LOWER_BOUND
    evals, n_det = _reference_levels(cap, configs)
    assert n_det == len(cap["csfs"])
    assert result.total_energy == pytest.approx(float(evals[0]), abs=1e-6)


# ---------------------------------------------------------------------------
# Stronger oracle gates added with the fix
# ---------------------------------------------------------------------------

def test_full_spectrum_matches_mscheme_reference():
    """EVERY eigenvalue of H, not just the ground state, must match.

    ``build_ci_matrix`` is block diagonal in ``(2J, parity)``, and at
    ``2M = 0`` each CSF contributes one state, so the sorted spectrum of H
    must equal the sorted spectrum of the oracle's truncated determinant
    Hamiltonian.  This is the strongest form of the cross-check: a single
    wrong element anywhere shows up here even when the ground state happens
    to be dominated by a correct diagonal.
    """
    for cfgs in (["1s2 2s2", "1s2 2s1 2p1"],
                 ["1s2 2s1 2p1"],
                 ["1s2 2s2", "1s2 2p2"],
                 ["1s1 2s2 2p1", "1s2 2s1 2p1"]):
        _result, _csfs, cap = _run_capturing_ci(
            cfgs, radial_optimization="fac_frozen")
        H = _build(cap)
        evals, n_det = _reference_levels(cap, cfgs)
        assert n_det == len(cap["csfs"])
        assert np.abs(np.linalg.eigvalsh(H) - evals).max() < 1e-10, (
            f"{cfgs}: spectrum differs from the m-scheme oracle")


def test_oracle_is_rotationally_invariant():
    """A scalar Hamiltonian's level energies cannot depend on the 2M sector.

    This is the invariant that exposed the dropped-exchange-integral bug in
    the oracle (``if v == 0.0: continue`` skipped ``v2`` whenever the direct
    integral vanished, which is not a symmetry).  It is kept as a permanent
    gate because the oracle is the arbiter for every other number here.
    """
    cfgs = ["1s2 2s1 2p1"]
    _result, _csfs, cap = _run_capturing_ci(cfgs,
                                            radial_optimization="fac_frozen")
    levels = {}
    for M2 in (0, 2, 4):
        evals, _nd = _reference_levels(cap, cfgs, M2=M2)
        levels[M2] = set(np.round(evals, 10).tolist())
    # Every level seen at a higher M must also be seen at M = 0, at the
    # same energy.
    for M2 in (2, 4):
        missing = {e for e in levels[M2]
                   if not any(abs(e - e0) < 1e-9 for e0 in levels[0])}
        assert not missing, (
            f"levels at 2M = {M2} are absent from the 2M = 0 spectrum: "
            f"{sorted(missing)} — the oracle is not rotationally invariant")


def test_singles_expansion_gains_nothing_on_hf_orbitals():
    """Brillouin's theorem, as a physics gate on the whole CI path.

    For orbitals self-consistent for the reference determinant,
    ``<ref|H|single> = 0`` exactly, so a singles-only expansion buys
    nothing.  The old kernel "gained" 0.8 Ha here — an unmissable sign that
    the matrix was not the Hamiltonian.
    """
    configs = [
        "1s2 2s2", "1s1 2s2 2p1", "1s1 2s2 3s1", "1s1 2s2 3p1",
        "1s1 2s2 3d1", "1s2 2s1 2p1", "1s2 2s1 3s1", "1s2 2s1 3p1",
        "1s2 2s1 3d1",
    ]
    reference, _csfs, _cap = _run_capturing_ci(
        ["1s2 2s2"], radial_optimization="fac_frozen")
    singles, _csfs, _cap = _run_capturing_ci(
        configs, radial_optimization="fac_frozen")
    assert singles.total_energy <= reference.total_energy
    # The gain must be correlation-free to within the (small) orbital
    # relaxation of the frozen-virtual protocol.
    assert singles.total_energy - reference.total_energy > -5e-4, (
        f"singles gained {singles.total_energy - reference.total_energy:.6f} "
        f"Ha over the reference — Brillouin's theorem is violated")


def test_ci_matrix_is_deterministic_and_has_no_cross_block_noise():
    """Same inputs, bit-identical output; zero off-block elements."""
    cfgs = ["1s2 2s2", "1s2 2s1 2p1"]
    _result, _csfs, cap = _run_capturing_ci(cfgs,
                                            radial_optimization="fac_frozen")
    H1 = _build(cap)
    H2 = _build(cap)
    assert np.array_equal(H1, H2)
    csfs = cap["csfs"]
    for i, a in enumerate(csfs):
        for j, b in enumerate(csfs):
            if (a.total_J2, a.parity) != (b.total_J2, b.parity):
                assert H1[i, j] == 0.0


@pytest.mark.parametrize("protocol", ["fac_frozen", "mcdhf"])
def test_he_like_1s2p_levels_match_oracle(protocol):
    """He-like ``1s2p``: all five levels, both radial protocols.

    The two J=1 levels are the ``3P1`` / ``1P1`` pair, separated by
    ``2 G^1(1s,2p)`` in the LS limit; the oracle reproduces the same five
    levels, so the LS-limit structure of the angular algebra is pinned
    here rather than through a hand-derived Clebsch-Gordan factor.
    """
    cfgs = ["1s2", "1s1 2p1"]
    result, _csfs, cap = _run_capturing_ci(
        cfgs, Z=2, radial_optimization=protocol)
    evals, n_det = _reference_levels(cap, cfgs)
    assert n_det == len(cap["csfs"]) == 5
    H = _build(cap)
    assert np.abs(np.sort(np.linalg.eigvalsh(H)) - evals).max() < 1e-10
    two_j1 = [i for i, csf in enumerate(cap["csfs"]) if csf.total_J2 == 2]
    assert len(two_j1) == 2
    splitting = abs(float(H[two_j1[0], two_j1[0]] - H[two_j1[1], two_j1[1]]))
    # The diagonal J=1 elements already differ; the exact eigenvalues differ
    # by the same 2 G^1 scale (both are checked against the oracle above).
    assert splitting > 0.0
    assert result.total_energy == pytest.approx(float(evals[0]), abs=1e-9)
