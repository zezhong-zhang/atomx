"""M5 tests: MCDHF (Multi-Configuration Dirac-Hartree-Fock).

Tests CSF parsing, CI matrix construction, and MCDHF solver.
"""

import numpy as np
import pytest
from _be_reference import BE_DC_SANITY_FLOOR

from atomx.config.csf import (
    _parse_config_string,
    enumerate_jj_csfs,
    get_shared_orbital_basis,
    make_csf_list,
)
from atomx.constants import HARTREE_CM1
from atomx.methods.dirac_fock import Subshell
from atomx.methods.mcdhf.solver import MCDHFResult, MCDHFSolver


class TestCSFParsing:
    def test_parse_helium(self):
        """Parse '1s2' → one subshell."""
        shells = _parse_config_string("1s2")
        assert len(shells) == 1
        assert shells[0].n == 1
        assert shells[0].kappa == -1
        assert shells[0].occupation == 2

    def test_parse_neon(self):
        """Parse '1s2 2s2 2p6' → 1s, 2s, 2p1/2, 2p3/2."""
        shells = _parse_config_string("1s2 2s2 2p6")
        assert sum(sh.occupation for sh in shells) == 10

    def test_parse_excited(self):
        """Parse excited config '1s2 2s1 2p6 3s1'."""
        shells = _parse_config_string("1s2 2s1 2p6 3s1")
        assert sum(sh.occupation for sh in shells) == 10

    def test_make_csf_list(self):
        csfs = make_csf_list(10, ["1s2 2s2 2p6", "1s2 2s1 2p6 3s1"])
        assert len(csfs) == 2
        assert csfs[0].n_electrons == 10
        assert csfs[1].n_electrons == 10

    def test_config_diff(self):
        """Two configs that differ by 1 electron (2s→3s)."""
        csfs = make_csf_list(10, ["1s2 2s2 2p6", "1s2 2s1 2p6 3s1"])
        diffs = csfs[0].differs_from(csfs[1])
        assert len(diffs) >= 2  # 2s occupation differs, 3s appears

    def test_n_electron_diff(self):
        csfs = make_csf_list(10, ["1s2 2s2 2p6", "1s2 2s1 2p6 3s1"])
        assert csfs[0].n_electron_diff(csfs[1]) == 1

    def test_shared_orbital_basis(self):
        csfs = make_csf_list(4, ["1s2 2s2", "1s2 2s1 2p1"])
        basis = get_shared_orbital_basis(csfs)
        # Should have 1s, 2s, and at least one 2p orbital
        n_kappa_set = {(n, k) for n, k in basis}
        assert (1, -1) in n_kappa_set  # 1s
        assert (2, -1) in n_kappa_set  # 2s


class TestMCDHFSolver:
    """Test MCDHF solver with simple cases."""

    def test_single_config_matches_dhf(self):
        """MCDHF with 1 CSF should match DHF."""
        csfs = make_csf_list(2, ["1s2"])
        solver = MCDHFSolver(Z=2, csfs=csfs, ns=40, ks=9)
        result = solver.solve(max_iter=5, scf_tol=1e-4, ci_tol=1e-6)
        # Single CSF → ground energy = DHF energy.  External anchor:
        # DBSR_HF ``an=2 term=AV mbreit=0`` gives He Coulomb -2.86181332 Ha
        # (benchmarks/EXTERNAL_VERIFICATION.md §1; the non-relativistic HF
        # limit is -2.8616799956, Froese Fischer).  The old gate here was
        # -2.86 +- 0.1 Ha (3.5 %), which accepts the uncorrelated
        # hydrogenic value -2.8477 as readily as the DF solution; the
        # measured deviation is 1e-12 Ha.
        assert result.n_states == 1
        assert result.ground_energy == pytest.approx(-2.86181332, abs=1e-6)

    def test_two_config_be(self):
        """Be with 2 CSFs: 1s2 2s2 + 1s2 2p2 (2s-2p near-degeneracy).

        Check correlation gain and a loose energy scale. Nested-space
        variational bounds are tested separately on one common Hamiltonian.
        """
        csfs = make_csf_list(4, ["1s2 2s2", "1s2 2p2"])
        solver = MCDHFSolver(Z=4, csfs=csfs, ns=40, ks=9)
        result = solver.solve(max_iter=10, scf_tol=1e-4, ci_tol=1e-6)
        assert result.n_states == 2
        # Ground state should be lower than the excited state
        assert result.state_energies[0] < result.state_energies[1]
        # A gross-collapse diagnostic, not a rigorous relativistic bound.
        assert np.all(result.state_energies > BE_DC_SANITY_FLOOR), (
            f"levels {result.state_energies!r} are below the diagnostic Be "
            f"energy floor {BE_DC_SANITY_FLOOR} Ha")

        ref_csfs = make_csf_list(4, ["1s2 2s2"])
        ref = MCDHFSolver(Z=4, csfs=ref_csfs, ns=40, ks=9)
        e_ref = ref.solve(max_iter=10, scf_tol=1e-4, ci_tol=1e-6).ground_energy
        assert result.ground_energy < e_ref, (
            f"2-CSF CI {result.ground_energy!r} Ha is not below its own "
            f"1-CSF reference {e_ref!r} Ha — the 2p² admixture gained nothing")

    def test_variational_projector_fock_matches_nuclear_on_light_atom(self):
        """F3 gate: the Fock-built no-pair space reproduces the nuclear one
        where screening is negligible, and diagnostics report leakage.

        The no-pair projector used to be nuclear-spectrum-only, with the
        screened-Fock idea patched into the *preconditioner* alone
        (critical review F3, 2026-09-27).  For Be the valence shell sees
        an almost-unscreened core, so both boundaries must agree tightly;
        a large disagreement here would mean the two projectors disagree
        even in the light-atom limit and the default could not be trusted.
        """
        energies = {}
        leaks = {}
        for projector in ("nuclear", "fock"):
            csfs = make_csf_list(4, ["1s2 2s2", "1s2 2p2"])
            solver = MCDHFSolver(Z=4, csfs=csfs, ns=30, ks=9,
                                 variational_projector=projector)
            result = solver.solve(max_iter=8, scf_tol=1e-4, ci_tol=1e-6)
            energies[projector] = result.ground_energy
            diag = dict(result.solver_diagnostics["_variational_diagnostics"])
            leaks[projector] = diag["no_pair_projector_leakage"]
        assert abs(energies["nuclear"] - energies["fock"]) < 1e-6, (
            f"no-pair projector choice moved the Be energy by "
            f"{abs(energies['nuclear'] - energies['fock']):.3e} Ha — the "
            "nuclear and Fock boundaries should coincide for a light atom")
        for projector, leak in leaks.items():
            assert set(leak) == {-1, 1}, (
                f"projector={projector}: leakage diagnostic missing kappa "
                f"blocks, got {leak!r}")
            assert all(0.0 <= v <= 1.0 for v in leak.values())

    def test_per_block_orbital_sets_are_populated_when_enabled(self):
        """Opt-in DODS/EOL scaffold produces block-keyed orbital sets."""
        csfs = []
        for cfg in ["1s2", "1s1 2p1"]:
            csfs.extend(enumerate_jj_csfs(cfg))
        solver = MCDHFSolver(
            Z=6,
            csfs=csfs,
            ns=24,
            ks=7,
            use_physics_grid=False,
            per_block_orbital_sets=True,
        )
        result = solver.solve(max_iter=3, scf_tol=1e-3, ci_tol=1e-5, scf_max_iter=8)
        assert result.per_block_orbitals
        assert all(isinstance(block, tuple) and len(block) == 2 for block in result.per_block_orbitals)
        assert all(result.per_block_orbitals[block] for block in result.per_block_orbitals)

    def test_excitation_energies(self):
        """Excitation energies: unit identity and a positive first excitation.

        ``de[0] == 0`` is definitional; what this now gates is the SI to
        cm-1 conversion (a real silent-failure mode: the wrong constant is
        invisible without a cross-unit comparison) and the sign of the
        first excitation.
        """
        csfs = make_csf_list(4, ["1s2 2s2", "1s2 2p2"])
        solver = MCDHFSolver(Z=4, csfs=csfs, ns=40, ks=9)
        result = solver.solve(max_iter=10, scf_tol=1e-4, ci_tol=1e-6)
        de = result.excitation_energies(unit="cm-1")
        de_ha = result.excitation_energies()
        assert de[0] == 0.0  # ground state
        for cm1, ha in zip(de, de_ha, strict=True):
            assert cm1 == pytest.approx(float(ha) * HARTREE_CM1, rel=1e-12)
        if len(de) > 1:
            assert de[1] > 0  # excited states above ground

    def test_mixing_coefficients(self):
        """Mixing coefficients should be normalized."""
        csfs = make_csf_list(4, ["1s2 2s2", "1s2 2p2"])
        solver = MCDHFSolver(Z=4, csfs=csfs, ns=40, ks=9)
        result = solver.solve(max_iter=10, scf_tol=1e-4, ci_tol=1e-6)
        for state in range(result.n_states):
            norm = np.sum(result.mixing_coefficients[:, state] ** 2)
            assert norm == pytest.approx(1.0, abs=1e-10)


def test_fac_frozen_radial_optimization_keeps_virtual_occupations_tiny():
    from atomx.config.csf import enumerate_jj_csfs

    csfs = []
    for cfg in ["1s2 2s2 2p6", "1s2 2s2 2p5 3d1"]:
        csfs.extend(enumerate_jj_csfs(cfg))

    solver = MCDHFSolver(
        Z=26,
        csfs=csfs,
        ns=35,
        ks=7,
        use_physics_grid=False,
        radial_optimization="fac_frozen",
    )
    subshells = solver._fac_frozen_subshells()
    occ = {(sh.n, sh.kappa): sh.occupation for sh in subshells}

    assert occ[(2, 1)] == 2
    assert occ[(2, -2)] == 4
    assert occ[(3, 2)] == 0.001
    assert occ[(3, -3)] == 0.001


def test_fac_frozen_local_radial_optimization_solves_with_frozen_occupations():
    from atomx.config.csf import enumerate_jj_csfs

    csfs = []
    for cfg in ["1s2", "1s1 2p1"]:
        csfs.extend(enumerate_jj_csfs(cfg))

    solver = MCDHFSolver(
        Z=6,
        csfs=csfs,
        ns=28,
        ks=7,
        use_physics_grid=False,
        radial_optimization="fac_frozen_local",
    )
    result = solver.solve(max_iter=4, scf_tol=1e-3, ci_tol=1e-5, scf_max_iter=8)

    assert result.converged
    assert result.orbitals_dict


class TestManySameKappaBookkeeping:
    """Regression tests for the Pb 6p² under-binding bug.

    Root cause was twofold: (a) the CI kernel evaluated Slater integrals
    with the Y^k-quadrature fallback and a point-nucleus one-electron
    Hamiltonian, drifting ~74 Ha from the (cell-integral + Fermi-nucleus)
    SCF functional at Z=82; (b) orbital extraction used positional
    per-kappa counters, fragile with many same-kappa shells (Pb has six
    ns and five np shells).  See physics/csf/matrix.py and
    SCFResult.orbitals_by_subshell.
    """

    def test_orbitals_by_subshell_many_same_kappa(self):
        """(n, kappa)-keyed extraction must be exact for 6 same-kappa shells.

        Builds a synthetic SCFResult whose per-kappa orbital lists carry
        recognizable payloads and whose global subshell order interleaves
        kappas (as MCDHF's (n, |kappa|)-sorted avg_subshells does), then
        checks every orbital lands on the right (n, kappa) key.
        """
        from atomx.methods.dirac_fock import SCFResult

        # Pb-like valence bookkeeping: 6 s-shells (kappa=-1),
        # 5 p1/2 (kappa=+1), 5 p3/2 (kappa=-2), interleaved by n.
        shells = []
        for n in range(1, 7):
            shells.append(Subshell(n=n, kappa=-1, occupation=2.0))
            if n >= 2:
                shells.append(Subshell(n=n, kappa=1, occupation=2.0))
                shells.append(Subshell(n=n, kappa=-2, occupation=4.0))

        # Payload: P[0] encodes 10*n + a small kappa tag so a mixed-up
        # index is detected immediately.
        tag = {-1: 1, 1: 2, -2: 3}
        orbitals_raw = {}
        counters = {}
        for sh in shells:
            idx = counters.get(sh.kappa, 0)
            counters[sh.kappa] = idx + 1
            P = np.full(4, 10.0 * sh.n + tag[sh.kappa])
            Q = np.full(4, -(10.0 * sh.n + tag[sh.kappa]))
            orbitals_raw.setdefault(sh.kappa, []).append((P, Q, sh.occupation))

        res = SCFResult(
            converged=True, iterations=1, residual=0.0, total_energy=0.0,
            orbital_energies={}, subshells=shells,
            orbitals_raw=orbitals_raw, Z=82,
        )
        keyed = res.orbitals_by_subshell()
        assert len(keyed) == len(shells)
        for sh in shells:
            P, Q, occ = keyed[(sh.n, sh.kappa)]
            expected = 10.0 * sh.n + tag[sh.kappa]
            assert P[0] == expected
            assert Q[0] == -expected
            assert occ == sh.occupation

    def test_jj_configs_floor_prevents_phantom_collapse(self):
        """CSFs missing a jj PARTNER must carry the 0.001 floor, not 0.

        With a hard zero, a CI-weight collapse onto such a CSF zeroes
        every WW(i, ·) product for the missing jj component, its Fock
        operator loses the entire electron-repulsion term, and the
        orbital falls to a hydrogenic bare-nucleus state (Pb 6p3/2 ->
        -95.6 Ha).  The floor is scoped to same-(n, l) jj partners only:
        orbitals from OTHER NR shells (pure correlation orbitals) keep
        hard zeros so their converged shapes are unchanged.
        """
        # (a) jj partner floor: the (2p1/2)^2 CSF must carry 2p3/2=0.001
        csfs = enumerate_jj_csfs("1s2 2s2 2p2")
        solver = MCDHFSolver(Z=6, csfs=csfs, use_physics_grid=False)
        avg = solver._averaged_subshells()
        jj = solver._jj_configs(avg, np.ones(len(csfs)) / len(csfs))
        for cfg in jj:
            for sh in cfg.subshells:
                assert sh.occupation >= 0.001, (
                    f"jj-partner floor missing for {sh.n}, kappa={sh.kappa}")

        # (b) correlation orbitals stay hard-zero: He-like 1s2 + 2s2 + 2p2
        # expansion — the 1s2 reference CSF must NOT acquire phantom 2s/2p
        # occupation (their mean field defines the correlation-orbital
        # shapes that observables like the He-like M1 rate are tuned to).
        csfs = []
        for cfg_str in ["1s2", "2s2", "2p2"]:
            csfs.extend(enumerate_jj_csfs(cfg_str))
        solver = MCDHFSolver(Z=10, csfs=csfs, use_physics_grid=False)
        avg = solver._averaged_subshells()
        jj = solver._jj_configs(avg, np.ones(len(csfs)) / len(csfs))
        ref = jj[0]  # the 1s2 CSF
        occ = {(sh.n, sh.kappa): sh.occupation for sh in ref.subshells}
        assert occ[(1, -1)] == 2.0
        assert occ[(2, -1)] == 0.0   # correlation 2s: no floor
        assert occ[(2, 1)] == 0.0    # correlation 2p1/2: no floor
        assert occ[(2, -2)] == 0.0   # correlation 2p3/2: no floor

    def test_mcdhf_result_energy_components_legacy_keys(self):
        """MCDHFResult.energy_components mirrors the SCFResult legacy keys."""
        res = MCDHFResult(
            converged=True, iterations=1,
            state_energies=np.array([-100.0]),
            mixing_coefficients=np.eye(1),
            ground_energy=-100.0, orbital_energies={}, csfs=[],
        )
        assert res.energy_components == {"E_total": -100.0}
        assert res.energy_breakdown == res.energy_components

        res.solver_diagnostics["_E_Breit"] = 0.5
        res.solver_diagnostics["_E_QED"] = {
            "E_SE": 0.3, "E_VP": -0.1, "E_QED_total": 0.2,
            "se_method": "mohr", "qed_backend": "auto",
        }
        comps = res.energy_components
        assert comps["E_Breit"] == 0.5
        assert comps["E_SE"] == 0.3
        assert comps["E_VP"] == -0.1
        assert comps["E_QED_total"] == 0.2
        assert "se_method" not in comps
        assert all(isinstance(value, (int, float)) for value in comps.values())
        # Non-legacy provenance keys stay off the numeric dict
        assert "qed_backend" not in comps


@pytest.mark.slow
def test_pb_bspline_matches_shooting_bare_hf():
    """Pb (Z=82, 6p² open shell): B-spline MCDHF vs shooting bare-HF totals.

    The two engines share nothing numerically (B-spline Galerkin +
    cell-algorithm Slater integrals vs ODE shooting + Y^k sources), so
    agreement pins down both.  Before the CI-kernel fix the B-spline
    route came out 74 Ha under-bound (-20840 vs -20914) with 6p3/2
    mislabeled at 4p-scale energy (-95.6 Ha).
    """
    from atomx.atom import Atom

    ref = Atom(Z=82, exchange="hf", engine="shooting", corrections="none")
    ref.solve()
    E_shoot = ref.total_energy

    atom = Atom(Z=82, exchange="hf", engine="bspline", corrections="none")
    atom.solve()
    E_bsp = atom.total_energy

    # Desclaux (1973) DHF gives ~ -20913.9; both engines must sit there.
    assert E_bsp == pytest.approx(E_shoot, rel=1e-3)
    assert E_bsp == pytest.approx(-20913.7, abs=2.0)

    # 6p3/2 must be a valence-scale energy on BOTH engines (the bug put
    # it at -95.6 Ha), and every same-(l, j) series must bind less with
    # increasing n.
    for oe in (atom._result.orbital_energies, ref._result.orbital_energies):
        assert -0.5 < oe["6p3/2"] < -0.05
        assert -0.5 < oe["6p1/2"] < -0.1
        assert -1.0 < oe["6s1/2"] < -0.3
        p32 = [oe[f"{n}p3/2"] for n in range(2, 7)]
        assert p32 == sorted(p32), "p3/2 series not monotone in n"
        s12 = [oe[f"{n}s1/2"] for n in range(1, 7)]
        assert s12 == sorted(s12), "s1/2 series not monotone in n"
