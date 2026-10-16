"""Independent linked commutator, extensivity and native scalar SD anchors."""
import itertools
from unittest.mock import patch

import numpy as np
import pytest
from test_spinor_integrals_fscc import native as native_fixture

from atomx.methods.linearized_sd import compute, plan_scalar_sd, solve

native = native_fixture


def _apply(mask, operations):
    """Independent elementary Fock operation, rightmost acts first."""
    coefficient = 1
    for bit, create in reversed(operations):
        if bool(mask & (1 << bit)) == create:
            return None, 0
        coefficient *= (-1)**((mask & ((1 << bit)-1)).bit_count())
        mask ^= 1 << bit
    return mask, coefficient


def _full(n, electrons, onebody, pairs=()):
    masks = tuple(sum(1 << a for a in occupied) for occupied in itertools.combinations(range(n), electrons))
    index = {mask: i for i, mask in enumerate(masks)}
    h = np.zeros((len(masks), len(masks)))
    for col, mask in enumerate(masks):
        for a, b in itertools.product(range(n), repeat=2):
            target, sign = _apply(mask, ((a, True), (b, False)))
            if sign:
                h[index[target], col] += sign*onebody[a, b]
        for holes, particles, value in pairs:
            ops = tuple((a, True) for a in particles)+tuple((b, False) for b in reversed(holes))
            target, sign = _apply(mask, ops)
            if sign:
                h[index[target], col] += value*sign
                h[col, index[target]] += value*sign
    return masks, .5*(h+h.T)


def _sd(masks, ref):
    selected = [ref]+[m for rank in (1, 2) for m in masks if (ref & ~m).bit_count() == rank]
    return tuple(((mask, 1.),) for mask in selected)


@pytest.mark.parametrize('electrons,n', [(3, 6), (4, 8)])
def test_full_linked_commutator_noncanonical_and_mixed_signed_vectors(electrons, n):
    rng = np.random.default_rng(20261009+n)
    onebody = np.diag(np.r_[np.linspace(-2, -1, electrons), np.linspace(.2, 1, n-electrons)])
    perturbation = rng.normal(size=(n, n))*.015
    onebody += perturbation+perturbation.T
    masks, h = _full(n, electrons, onebody,
                    [((0, 1), (electrons, electrons+1), .06),
                     ((1, 2), (electrons+1, electrons+2), -.04)])
    ref = (1 << electrons)-1
    original = _sd(masks, ref)
    c = np.array([[dict(row).get(mask, 0) for row in original] for mask in masks])
    ranks = [(ref & ~next(iter(dict(row)))).bit_count() for row in original]
    for rank in (1, 2):
        indices = np.flatnonzero(np.array(ranks) == rank)
        rotation, _ = np.linalg.qr(rng.normal(size=(len(indices), len(indices))))
        c[:, indices] = c[:, indices]@rotation
    # Flip the reference phase too: generators must still map Phi to C_mu.
    c[:, 0] *= -1
    vectors = tuple(tuple((mask, float(c[i, j])) for i, mask in enumerate(masks) if c[i, j] != 0)
                    for j in range(c.shape[1]))
    projected = c.T@h@c
    projected = .5*(projected+projected.T)
    result = solve(projected, vectors, ref, reference_identity='independent full Fock space')
    generators = []
    index = {mask: i for i, mask in enumerate(masks)}
    for j in range(1, c.shape[1]):
        generator = np.zeros_like(h)
        for destination, coefficient in vectors[j]:
            holes = [i for i in range(n) if ref & ~destination & (1 << i)]
            particles = [i for i in range(n) if destination & ~ref & (1 << i)]
            ops = tuple((a, True) for a in particles)+tuple((b, False) for b in reversed(holes))
            _, ref_sign = _apply(ref, ops)
            for column, mask in enumerate(masks):
                target, sign = _apply(mask, ops)
                if sign:
                    generator[index[target], column] += coefficient*sign/(c[index[ref], 0]*ref_sign)
        generators.append(generator)
    independent = np.column_stack([c[:, 1:].T@(h@g-g@h)@c[:, 0] for g in generators])
    np.testing.assert_allclose(result.equation_matrix_Ha, independent, atol=1e-11)
    t = sum(amplitude*g for amplitude, g in zip(result.amplitudes, generators, strict=True))
    residual = c[:, 1:].T@(h+h@t-t@h)@c[:, 0]
    assert np.max(abs(residual)) < 1e-11
    assert np.max(abs(result.singles_product_subtraction_Ha)) > .005
    # Simply solving (H_QQ-E_ref)t=-H_Q0 omits the noncanonical linked term.
    unlinked = np.linalg.solve(projected[1:, 1:]-projected[0, 0]*np.eye(len(result.amplitudes)), -projected[1:, 0])
    assert np.max(abs(unlinked-result.amplitudes)) > 1e-6
    assert result.equation_residual_Ha < 1e-11
    with pytest.raises(ValueError):
        result.amplitudes.setflags(write=True)


def test_separated_pairs_linked_size_extensivity_not_cisd_or_ccsd():
    eps = np.array([-.5, -.45, .2, .3, -.6, -.55, .4, .45])
    pairs = [((0, 1), (2, 3), .12), ((4, 5), (6, 7), -.09)]
    masks, h = _full(8, 4, np.diag(eps), pairs)
    ref = sum(1 << i for i in (0, 1, 4, 5))
    vectors = _sd(masks, ref)
    index = [masks.index(row[0][0]) for row in vectors]
    result = solve(h[np.ix_(index, index)], vectors, ref, reference_identity='separated pairs')
    expected = sum(c*c/(sum(eps[list(a)])-sum(eps[list(b)])) for a, b, c in pairs)
    assert result.correlation_energy_Ha == pytest.approx(expected, abs=1e-11)
    exact = np.linalg.eigvalsh(h)[0]
    cisd = np.linalg.eigvalsh(h[np.ix_(index, index)])[0]
    assert abs(result.total_energy_Ha-exact) > 1e-6
    assert abs(result.total_energy_Ha-cisd) > 1e-6


def test_weak_coupling_has_independent_mp2_energy_and_first_order_amplitudes():
    eps = np.array([-1.3, -1.1, .2, .4])
    masks, bare = _full(4, 2, np.diag(eps))
    ref = 3
    vectors = _sd(masks, ref)
    indices = [masks.index(row[0][0]) for row in vectors]
    for coupling in (.02, 1e-4):
        _, h = _full(4, 2, np.diag(eps), [((0, 1), (2, 3), coupling)])
        result = solve(h[np.ix_(indices, indices)], vectors, ref, reference_identity='weak-coupling MP2')
        denominator = eps[0]+eps[1]-eps[2]-eps[3]
        assert result.correlation_energy_Ha == pytest.approx(coupling**2/denominator, rel=1e-12)
        assert max(abs(result.amplitudes)) == pytest.approx(abs(coupling/denominator), rel=1e-12)


@pytest.mark.parametrize('configuration,occupied,targets,virtual', [
    ('1s2', [(1, -1)], ['2s', '2p'], [(2, -1), (2, 1), (2, -2)]),
    ('1s2 2s2 2p6', [(1, -1), (2, -1), (2, 1), (2, -2)],
     ['3s', '3p', '3d'], [(3, -1), (3, 1), (3, -2), (3, 2), (3, -3)]),
    ('1s2 2s2', [(1, -1), (2, -1)], ['3s', '3p'], [(3, -1), (3, 1), (3, -2)]),
])
def test_independent_scalar_count_equals_full_jj_csf_enumeration(configuration, occupied, targets, virtual):
    from atomx.methods.ci_sd import plan_sd_ci_space
    reduced = plan_scalar_sd(occupied, virtual)
    explicit = plan_sd_ci_space((configuration,), targets, J2=0, parity=1,
                               dimension_cap=1000, max_determinants=10000)
    assert reduced.amplitudes+1 == explicit.n_csfs
    assert reduced.amplitudes < reduced.magnetic_symmetry_coordinates < reduced.unrestricted_coordinates
    assert reduced.resources()['real_amplitude_vector_bytes'] == 8*reduced.amplitudes


def test_grouped_large_pool_count_matches_independent_radial_pair_enumeration():
    from collections import Counter

    from atomx.angular.recoupling import l_of_kappa, twoj_of_kappa
    occupied = [(1, -1), (2, -1), (2, 1), (2, -2)]
    virtual = [(n, k) for n in range(3, 9) for k in (-1, 1, -2, 2, -3)]
    def pairs(keys):
        counter = Counter()
        for a, b in itertools.combinations_with_replacement(keys, 2):
            ja, jb = twoj_of_kappa(a[1]), twoj_of_kappa(b[1])
            for j in range(abs(ja-jb), ja+jb+1, 2):
                if a != b or (-1)**((ja+jb-j)//2) == -1:
                    counter[j, (-1)**(l_of_kappa(a[1])+l_of_kappa(b[1]))] += 1
        return counter
    op, vp = pairs(occupied), pairs(virtual)
    result = plan_scalar_sd(occupied, virtual)
    assert result.doubles == sum(count*vp[k] for k, count in op.items())
    assert dict(result.pair_channels) == {key: (count, vp[key]) for key, count in op.items() if vp[key]}


@pytest.mark.parametrize('breit', [False, True])
def test_native_be_full_j0_projection_with_static_operator_coverage(native, breit):
    from atomx.angular.csf_hamiltonian import csf_magnetic_vector, csf_signature
    from atomx.config.csf import get_shared_orbital_basis
    from atomx.methods.ci_sd import plan_sd_ci_space
    from atomx.physics.csf.matrix import build_ci_matrix
    core, od = native
    result = compute('1s2', ('2s', '2p'), od, core._bsp_data, 4, 30,
                     reference_identity='Be2+ full J0 SD', include_breit=breit,
                     nuclear_potential=core._nuclear_potential)
    plan = plan_sd_ci_space(('1s2',), ('2s', '2p'), J2=0, parity=1)
    csfs, p = plan.materialize()
    csfs = (csfs[p[0]],)+tuple(c for i, c in enumerate(csfs) if i != p[0])
    keys = get_shared_orbital_basis(csfs)
    h = build_ci_matrix(csfs, od, keys, core._bsp_data, 4, 30,
                        include_breit=breit, nuclear_potential=core._nuclear_potential)
    vectors = tuple(csf_magnetic_vector(csf_signature(c), 0, tuple(keys)) for c in csfs)
    replay = solve(h, vectors, vectors[0][0][0], reference_identity='same native snapshot')
    assert result.total_energy_Ha == replay.total_energy_Ha
    assert result.equation_residual_Ha < 1e-12
    assert result.provenance['scalar_plan']['scalar_amplitudes'] == 4
    assert ('static-breit' in result.provenance['hamiltonian_terms']) == breit
    # The two methods are very close in this weakly coupled native pool.
    # Their difference is not an independently meaningful accuracy gate;
    # the separated-pair and full-commutator controls distinguish them.


def test_complete_model_rejections_before_physical_assembly(native):
    core, od = native
    options = dict(reference_identity='guard', nuclear_potential=core._nuclear_potential)
    with patch('atomx.physics.csf.matrix.build_ci_matrix') as assembly:
        with pytest.raises(ValueError, match='CSFs'):
            compute('1s2', ('2s', '2p'), od, core._bsp_data, 4, 30, dimension_cap=1, **options)
        with pytest.raises(MemoryError):
            compute('1s2', ('2s', '2p'), od, core._bsp_data, 4, 30, memory_bytes=1, **options)
        with pytest.raises(ValueError, match='closed-shell'):
            compute('1s1', ('2s',), od, core._bsp_data, 4, 30, **options)
        assembly.assert_not_called()
    with pytest.raises(ValueError, match='disjoint'):
        plan_scalar_sd([(1, -1)], [(1, -1)])
    with pytest.raises(ValueError, match='keys'):
        plan_scalar_sd([(1, True)], [(2, -1)])
    with pytest.raises(ValueError, match='orthonormal'):
        solve(np.diag([0., 1.]), (((3, 1.),), ((5, 2.),)), 3, reference_identity='guard')
    with pytest.raises(ValueError, match='SD rank'):
        solve(np.diag([0., 1.]), (((7, 1.),), ((56, 1.),)), 7, reference_identity='guard')
    with pytest.raises(np.linalg.LinAlgError):
        solve(np.zeros((2, 2)), (((3, 1.),), ((5, 1.),)), 3, reference_identity='guard')
