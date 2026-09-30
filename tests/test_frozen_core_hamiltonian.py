"""Frozen-core normal ordering preserves the independently audited Hamiltonian."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from atomx.physics.csf.frozen_core import closed_core_one_body
from atomx.physics.csf.matrix import _radial_evaluator
from atomx.scattering.autoionization.mscheme_ci import ResonanceManifold
from benchmarks.bench_reference_ci_operator import common_grid, freeze

DATA = Path(__file__).parent/'data'
CASES = [c for c in json.loads((DATA/'fac_ci_operator.json').read_text())['cases']
         if c['sequence'] == 'neon_like']
GROUPS = [([(2, -1)], 2), ([(2, 1), (2, -2)], 6), ([(3, -1)], 0),
          ([(3, 1), (3, -2)], 0), ([(3, 2), (3, -3)], 0)]
CONFIGS = [(2, 6, 0, 0, 0), (2, 5, 1, 0, 0), (2, 5, 0, 1, 0),
           (2, 5, 0, 0, 1), (1, 6, 1, 0, 0), (1, 6, 0, 1, 0), (1, 6, 0, 0, 1)]


@pytest.mark.external_fac
@pytest.mark.parametrize('case', CASES, ids=lambda c: f"Z{c['z']}")
def test_native_frozen_core_reproduces_all_fac_levels(case):
    """37 levels, including n=2 radial coherence and both hole shells.

    FAC supplies bare Dirac elements (holding c fixed); Atomx recomputes
    every core and active Coulomb integral on the same FAC spinors. The
    bound Hamiltonian must be independent of the continuum sampling.
    """
    with np.load(DATA/'fac_ci_spinors.npz') as arrays:
        sources = {(o['n'], o['kappa']): arrays[o['array']] for o in case['orbitals']}
    waves, ws = common_grid(sources, case['z'], 8000, alpha=case['fac_alpha'])
    native = _radial_evaluator(waves, ws, case['z'], ws.n)
    fac = {freeze(row['key']): row['value_Ha'] for row in case['radial']}
    radial = lambda key: fac[key] if key[0] == 'h' else native(key)
    active = set(waves)-{(1, -1)}
    one_body = closed_core_one_body(active, {(1, -1): 2}, radial)
    core_energy = 2*fac['h', 1, -1, 1]+radial(('r', 0, *((1, -1),)*4))
    expected = np.sort(np.concatenate([np.linalg.eigvalsh(b['matrix_Ha']) for b in case['blocks']]))
    spectra = []
    for stride in (1, 17):
        r = ws.r[1::stride]
        fine_waves = {key: (p[1::stride], q[1::stride]) for key, (p, q) in waves.items()}
        man = ResonanceManifold(SimpleNamespace(Z=case['z']), GROUPS,
            r_fine=r, dr_fine=np.gradient(r), orbital_map=fine_waves,
            orbital_energies={key: 0. for key in active}, configs=CONFIGS,
            one_body=one_body, bound_radial=radial)
        dets = man.determinants(0)
        h = man.hamiltonian(dets)
        np.testing.assert_allclose(h, h.T, atol=1e-12, rtol=0)
        energies, vectors = np.linalg.eigh(h)
        jsq = np.diag(vectors.T @ man.j2_matrix(dets) @ vectors)
        twoj = np.rint(np.sqrt(1+4*jsq)-1).astype(int)
        reference_js = sorted(b['two_J'] for b in case['blocks'] for _ in b['matrix_Ha'])
        assert sorted(twoj) == reference_js
        assert len(energies) == len(expected) == 37
        spectra.append(energies+core_energy)
        np.testing.assert_allclose(spectra[-1], expected, atol=2e-9, rtol=0)
    np.testing.assert_array_equal(*spectra)


def test_closed_core_requires_full_disjoint_subshells():
    with pytest.raises(ValueError, match='closed subshell'):
        closed_core_one_body([(3, -1)], {(2, -2): 3}, lambda key: 1.)
    with pytest.raises(ValueError, match='disjoint'):
        closed_core_one_body([(2, -2)], {(2, -2): 4}, lambda key: 1.)


@pytest.mark.parametrize('core_kappa', [-1, 1, -2, 2, -3])
@pytest.mark.parametrize('active_kappa', [-1, 1, -2, 2, -3])
def test_closed_shell_exchange_matches_explicit_m_sum(core_kappa, active_kappa):
    """Check the analytic core contraction for every m and every allowed rank."""
    from atomx.angular.coefficients import ckm
    from atomx.scattering.distorted_wave.solver import _allowed_k
    a, b, c = (4, active_kappa), (5, active_kappa), (3, core_kappa)
    for rank in _allowed_k(active_kappa, core_kappa):
        # A pure exchange radial integral isolates the angular coefficient;
        # n_a != n_b tests the same coefficient in off-diagonal h_eff.
        def radial(key):
            return float(key == ('r', rank, a, c, c, b))
        reduced = -closed_core_one_body([a, b], {c: 2*abs(core_kappa)}, radial)[a, b]
        ja, jc = 2*abs(active_kappa)-1, 2*abs(core_kappa)-1
        for ma in range(-ja, ja+1, 2):
            explicit = 0.
            for mc in range(-jc, jc+1, 2):
                q = ma-mc
                if abs(q) <= 2*rank:
                    explicit += ((-1.)**(q//2)
                        * ckm(active_kappa, ma, core_kappa, mc, rank, q)
                        * ckm(core_kappa, mc, active_kappa, ma, rank, -q))
            assert reduced == pytest.approx(explicit, abs=1e-14)


@pytest.mark.parametrize('nuclear', ['point', 'fermi'])
def test_local_core_uses_solved_nucleus_environment_and_native_metric(nuclear, tmp_path):
    """Independent differential h_D agrees with the local eigenstate identity.

    Use a heavy nucleus and an external field so reconstructing -Z/r or
    dropping screening fails. Changing continuum sampling cannot change
    the bound energies or finite-frequency photon matrix elements.
    """
    from atomx.config.subshell import Subshell
    from atomx.numerics.shooting.local import ShootingSolver
    from atomx.results import ShootingResult
    from atomx.scattering.autoionization.mscheme_ci import manifold_from_configs

    result = ShootingSolver(82, [Subshell(1, -1, 2), Subshell(2, -1, 2),
        Subshell(2, 1, 2), Subshell(2, -2, 4)], n_grid=4000, nuclear=nuclear,
        exchange='gu', screening_potential=lambda r: .2*np.exp(-r)).solve(
            tol=1e-10, max_iter=150)
    assert result.converged
    restored = ShootingResult.load(result.save(tmp_path/'native'))
    np.testing.assert_array_equal(restored._rv_nuc, result._rv_nuc)
    manifolds = [manifold_from_configs(restored, ['2s1 3p1'], e_max_eV=e)[0]
                 for e in (1., 10000.)]
    assert len(manifolds[0].r_fine) != len(manifolds[1].r_fine)
    values = []
    for man in manifolds:
        active = {key for keys, _ in man.groups for key in keys}
        core = {(sh.n, sh.kappa): sh.occupation for sh in result.subshells
                if (sh.n, sh.kappa) not in active}
        differential = closed_core_one_body(active, core, man.bound_radial)
        for key, expected in differential.items():
            assert man.one_body[key] == pytest.approx(expected, abs=2e-7, rel=0)
        h = man.hamiltonian(man.determinants(0))
        hole = next(o for o in man.orbitals if o.n == 2)
        electron = next(o for o in man.orbitals if o.n == 3 and o.kappa == 1)
        values.append((h, man.electric_radial(hole, electron, 1, -500.)))
    np.testing.assert_array_equal(values[0][0], values[1][0])
    assert values[0][1] == values[1][1] != 0.
    restored._rv_nuc = None
    with pytest.raises(ValueError, match='nuclear/environment field'):
        manifold_from_configs(restored, ['2s1 3p1'])


def test_bspline_frozen_core_matches_full_ci_with_finite_nucleus():
    """Virtuals must share the occupied Fock operator and radial metric."""
    from atomx import Atom
    from atomx.config.csf import enumerate_jj_csfs_many, get_shared_orbital_basis
    from atomx.numerics.bspline.dirac_assembly import build_dirac_matrices
    from atomx.physics.csf.matrix import build_ci_matrix
    from atomx.scattering.autoionization.mscheme_ci import manifold_from_configs

    atom = Atom(26, charge=16, config='1s2 2s2 2p6', engine='bspline',
                corrections='none', nuclear='fermi')
    atom.solve(tol=1e-10, max_iter=150)
    assert atom.converged
    result = atom.result
    man, bsp = manifold_from_configs(result, ['2p5 3s1'])
    ns = bsp['ns']
    for kappa in (-1, 1, -2):
        _, metric = build_dirac_matrices(ns, kappa, result.Z, bsp, result._nuclear_potential)
        keys = sorted(key for key in man.bound_orbitals if key[1] == kappa)
        coefficients = np.column_stack([np.concatenate(man.bound_orbitals[key]) for key in keys])
        np.testing.assert_allclose(coefficients.T @ metric @ coefficients,
                                   np.eye(len(keys)), atol=1e-7, rtol=0)
    csfs = enumerate_jj_csfs_many(['1s2 2s2 2p6', '1s2 2s2 2p5 3s1'])
    full = build_ci_matrix(csfs, man.bound_orbitals, get_shared_orbital_basis(csfs),
                           bsp, result.Z, ns, nuclear_potential=result._nuclear_potential)
    energies = np.linalg.eigvalsh(full)
    frozen = np.linalg.eigvalsh(man.hamiltonian(man.determinants(0)))-man.ion_energy()
    assert len(frozen) == 4 and len(energies) == 5
    np.testing.assert_allclose(frozen, energies[1:]-energies[0], atol=2e-10, rtol=0)
