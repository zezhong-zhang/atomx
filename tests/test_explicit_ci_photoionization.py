"""Signed removal, closed spectators, native continua and Milne controls."""

from dataclasses import replace
from itertools import combinations

import numpy as np
import pytest

from atomx import Atom, Basis
from atomx.angular.dyson import dyson_annihilation, scalar_radial_density
from atomx.config.csf import enumerate_jj_csfs
from atomx.methods.ci_blocks import solve_ci_blocks
from atomx.scattering.ci_continuum import TwoElectronCIChannels
from atomx.scattering.models import ContinuumModel
from atomx.scattering.photoionization.ci import ExplicitCIPhotoionization


def continuum(nodes=4000):
    return ContinuumModel.shooting(n_grid=nodes, r_max=80., orbital_removal='none')


@pytest.mark.parametrize('number', [1, 2, 3, 4])
def test_dyson_independent_sorted_list_algebra_and_particle_sum(number):
    rng = np.random.default_rng(1984+number)
    masks = [sum(1 << i for i in inds) for inds in combinations(range(5), number)]
    coeff = rng.normal(size=len(masks))
    coeff /= np.linalg.norm(coeff)
    ket = dict(zip(masks, coeff, strict=True))
    total = 0.
    for occupied in combinations(range(5), number-1):
        bra_mask = sum(1 << i for i in occupied)
        actual = dyson_annihilation({bra_mask: 1.}, ket, 5)
        expected = np.zeros(5)
        for index in range(5):
            for mask, c in ket.items():
                initial = [i for i in range(5) if mask & (1 << i)]
                if index in initial:
                    position = initial.index(index)
                    remaining = tuple(i for i in initial if i != index)
                    if remaining == occupied:
                        expected[index] += (-1)**position*c
        np.testing.assert_allclose(actual, expected, atol=1e-16)
        total += actual@actual
    assert total == pytest.approx(number, abs=2e-15)


def test_dyson_coherent_phase_and_small_amplitudes_not_screened():
    eps = 1e-16
    ket = {0b0011: 1/np.sqrt(2), 0b0101: 1/np.sqrt(2), 0b1001: eps}
    d = dyson_annihilation({1: 1.}, ket, 4)
    assert d[1]+d[2] == pytest.approx(-np.sqrt(2))
    ket[0b0101] *= -1
    dark = dyson_annihilation({1: 1.}, ket, 4)
    assert dark[1]+dark[2] == 0
    assert dark[3] == -eps
    with pytest.raises(ValueError, match='layout'):
        dyson_annihilation({32: 1.}, ket, 4)
    with pytest.raises(ValueError, match='real'):
        dyson_annihilation({1: 1.+0j}, ket, 4)


def test_sparse_radial_density_matches_independent_list_operator():
    spinors = ((1, -1, -1), (1, -1, 1), (2, -1, -1), (2, -1, 1))
    rng = np.random.default_rng(3187)
    configurations = list(combinations(range(4), 2))
    amplitudes = rng.normal(size=len(configurations))
    amplitudes /= np.linalg.norm(amplitudes)
    vector = {sum(1 << i for i in c): a for c, a in zip(configurations, amplitudes, strict=True)}
    keys, actual = scalar_radial_density(vector, spinors)
    expected = np.zeros((2, 2))
    for initial, ci in zip(configurations, amplitudes, strict=True):
        for q in initial:
            rest = list(initial)
            sq = (-1)**rest.index(q)
            rest.remove(q)
            for p, (n, k, m) in enumerate(spinors):
                if p in rest or spinors[q][1:] != (k, m):
                    continue
                sp = (-1)**sum(x < p for x in rest)
                final = tuple(sorted(rest+[p]))
                cf = amplitudes[configurations.index(final)]
                expected[n-1, spinors[q][0]-1] += cf*ci*sq*sp
    assert keys == ((1, -1), (2, -1))
    np.testing.assert_allclose(actual, expected, atol=3e-16)
    assert np.trace(actual) == pytest.approx(2., abs=1e-15)
    assert abs(actual[0, 1]) > .01


def solve_sectors(result, configurations, Z):
    orbitals = result.orbitals_dict
    sectors = {}
    for config in configurations:
        csfs = tuple(enumerate_jj_csfs(config))
        spec = solve_ci_blocks(csfs, orbitals, result.bsp_data, Z, result.bsp_data['ns'],
                                nuclear_potential=result._nuclear_potential)
        sectors[csfs[0].n_electrons] = csfs, spec
    return sectors, orbitals


@pytest.fixture(scope='module')
def helium():
    atom = Atom(Z=2, config=['1s2'], basis=Basis(ns=40, ks=9, r_max=50), corrections='none')
    atom.solve(max_iter=100, tol=1e-9)
    assert atom.converged
    sectors, orbitals = solve_sectors(atom.result, ('1s2', '1s1'), 2)
    provider = ExplicitCIPhotoionization(sectors, orbitals, atom.result.bsp_data, 2,
             nuclear_potential=atom.result._nuclear_potential, continuum_model=continuum(),
             coverage='complete retained1s sectors only')
    return atom, sectors, orbitals, provider


def test_native_two_electron_cross_section_matches_independent_existing_channel(helium):
    atom, _, _, provider = helium
    final, initial = 0, 1
    threshold = provider.threshold_eV(initial, final)
    old = TwoElectronCIChannels(atom, continuum_model=continuum())
    for excess in (5., 30.):
        omega = threshold+excess
        expected = sum(value*value for ch in old.photoionization(omega) if ch.initial_electrons == 2
                       for _, value in ch.components)
        assert provider.cross_section_cm2(initial, final, omega) == pytest.approx(expected, rel=2e-10)
    np.testing.assert_array_equal(provider.cross_section_cm2(initial, final, [threshold-1, threshold]), [0, 0])
    rho = provider._densities[final]
    assert np.trace(rho) == pytest.approx(1.)
    field = provider._field(final)
    assert float(field(10000.))*10000 == pytest.approx(-1., abs=1e-10)


def test_milne_quadrature_has_independent_analytic_photon_moment(helium, monkeypatch):
    *_, provider = helium
    I, T = provider.threshold_eV(1, 0), 20.
    def analytic(i, f, photons, kinetic):
        return 1e-18*(I/photons)**2
    monkeypatch.setattr(provider, '_cross_section', analytic)
    actual = provider.recombination_moments(1, 0, T, order=8)
    from atomx.plasma.crm import radiative_recombination_rate_cm3_s
    expected = radiative_recombination_rate_cm3_s(I, T, g_bound=1, g_continuum=2,
                           cross_section_cm2=lambda e: 1e-18*(I/e)**2)
    assert actual['alpha_cm3_s'] == pytest.approx(expected, rel=2e-12)
    assert actual['photon_eV_cm3_s']/actual['alpha_cm3_s'] == pytest.approx(I+T, rel=2e-14)


def test_phase_identity_ownership_and_resource_rejection(helium):
    atom, sectors, orbitals, provider = helium
    kwargs = dict(nuclear_potential=atom.result._nuclear_potential, continuum_model=continuum(), coverage='test')
    for limits in (dict(max_levels=1), dict(max_magnetic_entries=1)):
        with pytest.raises(MemoryError):
            ExplicitCIPhotoionization(sectors, orbitals, atom.result.bsp_data, 2, **kwargs, **limits)
    wrong = {k: tuple(-a for a in pair) for k, pair in orbitals.items()}
    with pytest.raises(ValueError, match='reference mismatch'):
        ExplicitCIPhotoionization(sectors, wrong, atom.result.bsp_data, 2, **kwargs)
    csfs, spectrum = sectors[2]
    block = spectrum.blocks[0]
    corrupt = replace(spectrum, blocks=(replace(block, energies_Ha=block.energies_Ha+1.),))
    with pytest.raises(ValueError, match='residual'):
        ExplicitCIPhotoionization({1: sectors[1], 2: (csfs, corrupt)}, orbitals, atom.result.bsp_data, 2, **kwargs)
    with pytest.raises(ValueError):
        provider.bound[1, -1][0].setflags(write=True)
    limited = ExplicitCIPhotoionization(sectors, orbitals, atom.result.bsp_data, 2, max_waves=1, **kwargs)
    with pytest.raises(MemoryError, match='wave cap'):
        limited.cross_section_cm2(1, 0, limited.threshold_eV(1, 0)+20)
    assert not limited._waves
    with pytest.raises(ValueError, match='in-range'):
        provider.threshold_eV(-1, 0)


def test_closed_core_spectators_preserve_exact_removal_strength():
    from atomx.angular.csf_hamiltonian import csf_magnetic_vector, csf_signature
    # Complete d2 fine structure over a36-electron closed core. Compare
    # all daughter magnetic states with the valence electron-number sum2.
    core = '1s2 2s2 2p6 3s2 3p6 3d10 4s2 4p6'
    parents, daughters = enumerate_jj_csfs(core+' 4d2'), enumerate_jj_csfs(core+' 4d1')
    keys = tuple(sorted({(s.n, s.kappa) for c in parents+daughters for s in c.subshells}))
    spinors = [(n, k, m) for n, k in keys for m in range(-2*abs(k)+1, 2*abs(k), 2)]
    assert len(parents) == 9 and len(daughters) == 2 and len(spinors) == 46
    for parent in parents:
        for mi in range(-parent.total_J2, parent.total_J2+1, 2):
            ket = dict(csf_magnetic_vector(csf_signature(parent), mi, keys))
            total = 0.
            for final in daughters:
                for mf in range(-final.total_J2, final.total_J2+1, 2):
                    bra = dict(csf_magnetic_vector(csf_signature(final), mf, keys))
                    removal = dyson_annihilation(bra, ket, len(spinors))
                    assert all(v == 0 for v, (n, k, _) in zip(removal, spinors, strict=True) if (n, k) not in ((4, 2), (4, -3)))
                    total += removal@removal
            assert total == pytest.approx(2., abs=5e-14)


def test_ci_radial_interference_matches_independent_spin_spherical_harmonics():
    # (|1s^2>+|2s^2>)/sqrt2, with a coherent final1s/2s spinor.
    # Integrating Y00*C10 against the physical p1/2 and p3/2 spinors
    # gives magnitudes1/3 and sqrt2/3, independently of ckm.
    from types import SimpleNamespace
    provider = ExplicitCIPhotoionization.__new__(ExplicitCIPhotoionization)
    provider.dipole_mode = 'complete'
    provider.dyson_engine = 'python'
    provider.keys = ((1, -1), (2, -1))
    provider.spinors = ((1, -1, -1), (1, -1, 1), (2, -1, -1), (2, -1, 1))
    provider._angular = {}
    provider._vectors = [{0: {3: 1/np.sqrt(2), 12: 1/np.sqrt(2)}},
                         {-1: {1: 1/np.sqrt(2), 4: 1/np.sqrt(2)},
                           1: {2: 1/np.sqrt(2), 8: 1/np.sqrt(2)}}]
    provider.levels = (SimpleNamespace(g=1), SimpleNamespace(g=2))
    for k, denominator in ((1, 18), (-2, 9)):
        tensor = provider._angular_tensor(0, 1, k)
        for radial in (np.array([1., 2.]), np.array([1., -1.])):
            assert np.sum((tensor@radial)**2) == pytest.approx(radial.sum()**2/denominator, abs=2e-15)
        incoherent = np.sum((tensor*np.array([1., -1.]))**2)
        assert incoherent > .1  # A diagonal-only approximation misses exact darkness.


def angular_snapshot(parents, daughters, parent_coeff, daughter_coeff, *, reduced, engine='python'):
    from types import SimpleNamespace

    from atomx.angular.csf_hamiltonian import csf_magnetic_vector, csf_signature
    provider = ExplicitCIPhotoionization.__new__(ExplicitCIPhotoionization)
    provider.dipole_mode = 'reduced-dyson' if reduced else 'complete'
    provider.dyson_engine = engine
    provider.keys = tuple(sorted({(s.n, s.kappa) for c in parents+daughters for s in c.subshells}))
    provider.spinors = tuple((n, k, m) for n, k in provider.keys
                            for m in range(-2*abs(k)+1, 2*abs(k), 2))
    provider._angular, provider._removal = {}, {}
    provider._vectors, provider.levels = [], []
    for basis, coeff in ((parents, parent_coeff), (daughters, daughter_coeff)):
        j = basis[0].total_J2
        assert all(c.total_J2 == j for c in basis)
        vectors = {}
        for m in ((j,) if reduced else range(-j, j+1, 2)):
            vector = {}
            for csf, amplitude in zip(basis, coeff, strict=True):
                for mask, value in csf_magnetic_vector(csf_signature(csf), m, provider.keys):
                    vector[mask] = vector.get(mask, 0.)+amplitude*value
            vectors[m] = vector
        provider._vectors.append(vectors)
        provider.levels.append(SimpleNamespace(two_J=j, g=j+1))
    return provider


@pytest.mark.parametrize('engine', ['python', 'compiled'])
def test_highest_dyson_reconstructs_signed_complete_mixed_radial_trace(engine):
    rng = np.random.default_rng(9321)
    all_parents = [c for config in ('1s2', '1s1 2s1', '2s2', '1s1 2p1', '2s1 2p1') for c in enumerate_jj_csfs(config)]
    all_daughters = [c for config in ('1s1', '2s1', '2p1') for c in enumerate_jj_csfs(config)]
    for j, parity in sorted({(c.total_J2, c.parity) for c in all_parents}):
        parents = [c for c in all_parents if (c.total_J2, c.parity) == (j, parity)]
        ci = rng.normal(size=len(parents))
        ci /= np.linalg.norm(ci)
        for jf, pf in sorted({(c.total_J2, c.parity) for c in all_daughters}):
            daughters = [c for c in all_daughters if (c.total_J2, c.parity) == (jf, pf)]
            cf = rng.normal(size=len(daughters))
            cf /= np.linalg.norm(cf)
            complete = angular_snapshot(parents, daughters, ci, cf, reduced=False)
            reduced = angular_snapshot(parents, daughters, ci, cf, reduced=True, engine=engine)
            negative = angular_snapshot(parents, daughters, ci, -cf, reduced=True, engine=engine)
            for k in (-3, -2, -1, 1, 2):
                expected = complete._angular_tensor(0, 1, k)
                actual = reduced._angular_tensor(0, 1, k)
                np.testing.assert_allclose(actual, expected, atol=2e-15, rtol=2e-14)
                np.testing.assert_allclose(negative._angular_tensor(0, 1, k), -actual,
                                           atol=2e-15, rtol=2e-14)
                radial = rng.normal(size=len(complete.keys))
                assert np.sum((actual@radial)**2) == pytest.approx(np.sum((expected@radial)**2), abs=2e-15)


def test_highest_dyson_closed_core_d2_complete_magnetic_control():
    core = '1s2 2s2 2p6 3s2 3p6 3d10 4s2 4p6'
    parents, daughters = enumerate_jj_csfs(core+' 4d2'), enumerate_jj_csfs(core+' 4d1')
    for parent in parents:
        for daughter in daughters:
            complete = angular_snapshot([parent], [daughter], [1.], [1.], reduced=False)
            reduced = angular_snapshot([parent], [daughter], [1.], [1.], reduced=True)
            for k in (-4, -2, 1, 3):
                np.testing.assert_allclose(reduced._angular_tensor(0, 1, k),
                                           complete._angular_tensor(0, 1, k), atol=1e-14, rtol=1e-13)


def test_highest_dyson_radial_darkness_and_support_guards():
    from atomx.angular.dyson import reduced_dyson_annihilation
    parents = enumerate_jj_csfs('1s2')+enumerate_jj_csfs('2s2')
    daughters = enumerate_jj_csfs('1s1')+enumerate_jj_csfs('2s1')
    reduced = angular_snapshot(parents, daughters, np.ones(2)/np.sqrt(2),
                               np.ones(2)/np.sqrt(2), reduced=True)
    tensor = reduced._angular_tensor(0, 1, -2)
    assert np.sum((tensor@np.array([1., -1.]))**2) < 1e-30
    assert np.sum((tensor*np.array([1., -1.]))**2) > .1
    with pytest.raises(ValueError, match='highest magnetic'):
        reduced_dyson_annihilation({1: 1.}, {3: 1.}, ((1, -1, -1), (1, -1, 1)), 0, 1)
    # J=0 -> Jf=3/2 plus an s electron is exactly forbidden.
    keys, coeff = reduced_dyson_annihilation({}, {}, ((1, -1, -1), (1, -1, 1)), 0, 3)
    assert keys == ((1, -1),) and np.array_equal(coeff, [0.])


@pytest.mark.parametrize('engine', ['python', 'compiled'])
def test_native_highest_dyson_pi_and_milne_match_complete_magnetic_trace(helium, engine):
    atom, sectors, orbitals, complete = helium
    kwargs = dict(nuclear_potential=atom.result._nuclear_potential, continuum_model=continuum(),
                  coverage='complete retained1s sectors only', magnetic_mode='highest-all')
    reduced = ExplicitCIPhotoionization(sectors, orbitals, atom.result.bsp_data, 2,
                                       **kwargs, dipole_mode='reduced-dyson', dyson_engine=engine)
    threshold = complete.threshold_eV(1, 0)
    np.testing.assert_allclose(reduced.cross_section_cm2(1, 0, threshold+np.array([5., 30.])),
                               complete.cross_section_cm2(1, 0, threshold+np.array([5., 30.])), rtol=1e-13)
    expected = complete.recombination_moments(1, 0, 20., order=8)
    actual = reduced.recombination_moments(1, 0, 20., order=8)
    assert actual == pytest.approx(expected, rel=1e-13)
    assert reduced.source_id != complete.source_id
    unsupported = ExplicitCIPhotoionization(sectors, orbitals, atom.result.bsp_data, 2, **kwargs)
    with pytest.raises(ValueError, match='complete initial magnetic'):
        unsupported.cross_section_cm2(1, 0, threshold+5.)
    with pytest.raises(ValueError, match='dipole mode'):
        ExplicitCIPhotoionization(sectors, orbitals, atom.result.bsp_data, 2, **kwargs, dipole_mode='guess')
