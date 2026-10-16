"""Retained complete H preserves signed CI and rejects stale physical inputs."""
import copy
import hashlib
import json
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from atomx.config.csf import enumerate_jj_csfs
from atomx.methods.ci import CIResult, ci_solve
from atomx.methods.csf.variational import CSFOrbitalFunctional
from atomx.numerics.bspline.basis import prepare_bspline_data
from atomx.numerics.bspline.dirac_assembly import solve_dirac
from atomx.numerics.bspline.knots import make_knots
from atomx.physics.csf.hamiltonian_snapshot import CIHamiltonianSnapshot
from atomx.physics.csf.matrix import GridCIWorkspace
from atomx.physics.csf.radial_snapshot import CIRadialSnapshot
from atomx.physics.csf.state_average import CIStateAverage


@pytest.fixture(scope="module")
def problem():
    ns, ks = 35, 9
    data = prepare_bspline_data(make_knots(ns, ks, .005, 30), ns, ks)
    configs = ("1s2", "1s1 2s1", "1s1 3s1", "2s2", "2s1 3s1", "3s2",
               "2p2", "2p1 3p1", "3p2")
    csfs = [c for cfg in configs for c in enumerate_jj_csfs(cfg)]
    target = CIStateAverage(((0, 1, 0, 1), (0, 1, 1, 3)))

    def nuclear(radius):
        return -2 / radius

    functional = CSFOrbitalFunctional(csfs, data, 2, nuclear, state_average=target)
    orbitals = {}
    for kappa in functional.metric:
        spectrum = solve_dirac(functional.one_body[kappa], functional.metric[kappa], ns, kappa)
        ell = kappa if kappa > 0 else -kappa - 1
        for n, k in functional.keys:
            if k == kappa:
                p, q, _ = spectrum.get_orbital(n - ell - 1 + int(kappa > 0))
                orbitals[n, k] = p, q
    return functional, orbitals, nuclear


def inputs(problem):
    f, orbitals, nuclear = problem
    return (f.csfs, orbitals, f.keys, f.data, f.Z, f.ns), {"nuclear_potential": nuclear}


def retained(problem, **options):
    args, kwargs = inputs(problem)
    return ci_solve(*args, **kwargs, retain_hamiltonian=True, **options)


def test_complete_signed_matrix_spectrum_and_gradient_are_unchanged(problem):
    from atomx.physics.csf.matrix import build_ci_matrix
    args, kwargs = inputs(problem)
    f, orbitals, _ = problem
    ordinary = ci_solve(*args, **kwargs)
    result = retained(problem)
    h = result.hamiltonian_for(*args, **kwargs)
    independent_h = build_ci_matrix(*args, **kwargs)
    np.testing.assert_array_equal(h, independent_h)
    np.testing.assert_array_equal(result.state_energies, ordinary.state_energies)
    assert h.shape == (len(f.csfs), len(f.csfs))
    assert np.max(np.abs(h - np.diag(np.diag(h)))) > 1e-3
    assert np.max(np.abs(h @ result.mixing_coefficients -
                         result.mixing_coefficients * result.state_energies)) < 1e-10
    np.testing.assert_allclose(result.mixing_coefficients.T @ result.mixing_coefficients,
                               np.eye(len(f.csfs)), atol=1e-12, rtol=0)
    # Test every signed radial component; no saved-column comparison at degeneracy.
    e0, g0, _ = f.evaluate(orbitals, ci_result=ordinary)
    e1, g1, _ = f.evaluate(orbitals, ci_result=result)
    assert e0 == e1
    for key in f.keys:
        np.testing.assert_array_equal(g1[key], g0[key])


def test_actual_one_build_trial_gradient_and_checkpoint_reuse(problem, monkeypatch):
    import atomx.physics.csf.matrix as matrix
    f, orbitals, nuclear = problem
    counted = []
    original = matrix.build_ci_matrix

    def build(*args, **kwargs):
        counted.append(len(args[0]))
        return original(*args, **kwargs)

    monkeypatch.setattr(matrix, "build_ci_matrix", build)
    opted = CSFOrbitalFunctional(f.csfs, f.data, f.Z, nuclear,
        state_average=f.state_average, retain_hamiltonian=True)
    _, trial_ci = opted.evaluate(orbitals, gradient=False)
    assert counted == [len(f.csfs)]
    _, gradient, accepted_ci = opted.evaluate(orbitals, ci_result=trial_ci)
    assert accepted_ci is trial_ci
    args, kwargs = inputs(problem)
    checkpoint_h = accepted_ci.hamiltonian_for(*args, **kwargs)
    assert counted == [len(f.csfs)]
    replay_h = matrix.build_ci_matrix(*args, **kwargs)
    assert counted == [len(f.csfs), len(f.csfs)]
    np.testing.assert_array_equal(checkpoint_h, replay_h)
    assert any(np.max(np.abs(v)) > 1e-3 for v in gradient.values())


def test_owned_storage_has_no_builder_alias_and_no_mutable_header(problem, monkeypatch):
    import atomx.physics.csf.matrix as matrix
    original = matrix.build_ci_matrix
    raw = []

    def build(*args, **kwargs):
        value = original(*args, **kwargs)
        raw.append(value)
        return value

    monkeypatch.setattr(matrix, "build_ci_matrix", build)
    owner = retained(problem).hamiltonian_snapshot
    expected = owner.matrix_Ha.copy()
    raw[0][:] = 0
    np.testing.assert_array_equal(owner.matrix_Ha, expected)
    with pytest.raises(ValueError):
        owner.matrix_Ha[0, 0] = 0
    with pytest.raises(ValueError):
        owner.matrix_Ha.setflags(write=True)
    with pytest.raises(FrozenInstanceError):
        owner.units = "eV"
    header = owner.identity_payload()
    header["radial_binding"]["geometry"] = "stale"
    assert owner.identity_payload()["radial_binding"]["geometry"] != "stale"
    with pytest.raises(TypeError, match="physical inputs"):
        CIHamiltonianSnapshot(matrix_Ha=expected)
    with pytest.raises(TypeError, match="bound CIHamiltonianSnapshot"):
        CIResult(np.ones(1), np.ones((1, 1)), hamiltonian_snapshot=expected)


@pytest.mark.parametrize("change", ("P", "Q", "phase", "orbital_keys", "basis_order", "CSF_order",
                                    "parity", "workspace", "nuclear", "Z", "ns", "cap"))
def test_same_input_binding_rejects_real_stale_changes(problem, change):
    args, kwargs = inputs(problem)
    result = retained(problem)
    args, kwargs = list(args), dict(kwargs)
    args[0], args[1], args[3] = copy.deepcopy(args[0]), copy.deepcopy(args[1]), copy.deepcopy(args[3])
    key = args[2][0]
    if change in ("P", "Q"):
        args[1][key][int(change == "Q")][7] += 1e-9
    elif change == "phase":
        args[1][key] = tuple(-wave for wave in args[1][key])
    elif change == "orbital_keys":
        args[1][(99, key[1])] = args[1].pop(key)
    elif change == "basis_order":
        args[2] = tuple(reversed(args[2]))
    elif change == "CSF_order":
        args[0].reverse()
    elif change == "parity":
        args[0][0].parity *= -1
    elif change == "workspace":
        args[3]["bsp_p"][0, 0, 0] += 1e-9
    elif change == "nuclear":
        kwargs["nuclear_potential"] = lambda radius: -2.00001 / radius
    elif change in ("Z", "ns"):
        args[4 + int(change == "ns")] += 1
    elif change == "cap":
        kwargs["max_determinants"] = result.hamiltonian_snapshot.max_determinants + 1
    with pytest.raises(ValueError):
        result.hamiltonian_for(*args, **kwargs)


def test_changed_nuclear_closure_is_rejected_by_content(problem):
    args, kwargs = inputs(problem)
    scale = [1.0]

    def nuclear(radius):
        return -2 * scale[0] / radius

    kwargs["nuclear_potential"] = nuclear
    result = ci_solve(*args, **kwargs, retain_hamiltonian=True)
    scale[0] = 1.001
    with pytest.raises(ValueError, match="nuclear reference"):
        result.hamiltonian_for(*args, **kwargs)


def test_stale_accepted_result_rejects_before_gradient_or_rebuild(problem, monkeypatch):
    import atomx.physics.csf.matrix as matrix
    f, orbitals, _ = problem
    result = retained(problem)
    changed = copy.deepcopy(orbitals)
    changed[f.keys[0]][0][7] += 1e-9

    def forbidden(*args, **kwargs):
        pytest.fail("stale accepted CI must reject before another matrix or radial gradient")

    monkeypatch.setattr(matrix, "build_ci_matrix", forbidden)
    monkeypatch.setattr(f, "gradient_from_weights", forbidden)
    with pytest.raises(ValueError, match="reference mismatch"):
        f.evaluate(changed, ci_result=result)


@pytest.mark.parametrize("flag", (0, 1, None, "true", np.bool_(True)))
def test_retention_requires_explicit_python_bool_before_build(problem, flag, monkeypatch):
    import atomx.physics.csf.matrix as matrix
    args, kwargs = inputs(problem)
    monkeypatch.setattr(matrix, "build_ci_matrix", lambda *a, **kw: pytest.fail("must reject before build"))
    with pytest.raises(TypeError, match="must be bool"):
        ci_solve(*args, **kwargs, retain_hamiltonian=flag)
    f, _, nuclear = problem
    with pytest.raises(TypeError, match="must be bool"):
        CSFOrbitalFunctional(f.csfs, f.data, f.Z, nuclear, retain_hamiltonian=flag)


@pytest.mark.parametrize("option", ({"n_states": 1}, {"solver": "davidson"}, {"one_body_shifts": {}},
                                   {"one_body_operator": object()}, {"n_states": "all"},
                                   {"include_breit": True, "one_body_shifts": {}}))
def test_unsupported_retained_operator_or_spectrum_rejects_before_build(problem, option, monkeypatch):
    import atomx.physics.csf.matrix as matrix
    args, kwargs = inputs(problem)
    if option.get("n_states") == "all":
        option = {"n_states": len(args[0])}
    monkeypatch.setattr(matrix, "build_ci_matrix", lambda *a, **kw: pytest.fail("must reject before build"))
    with pytest.raises(ValueError, match="complete dense unshifted"):
        ci_solve(*args, **kwargs, retain_hamiltonian=True, **option)


def test_grid_retention_rejects_before_build(problem, monkeypatch):
    import atomx.physics.csf.matrix as matrix
    args, kwargs = inputs(problem)
    grid = np.linspace(.01, 30, 50)
    args = list(args)
    args[3], args[5] = GridCIWorkspace(grid, np.gradient(grid), -2 * np.ones_like(grid)), len(grid)
    monkeypatch.setattr(matrix, "build_ci_matrix", lambda *a, **kw: pytest.fail("must reject before build"))
    with pytest.raises(ValueError, match="B-spline workspace"):
        ci_solve(*args, **kwargs, retain_hamiltonian=True)


def test_default_and_legacy_result_contract_and_owned_radial_consumer(problem):
    args, kwargs = inputs(problem)
    f, orbitals, nuclear = problem
    default = ci_solve(*args, **kwargs)
    assert default.hamiltonian_snapshot is None
    with pytest.raises(ValueError, match="no retained"):
        default.hamiltonian_for(*args, **kwargs)
    # Existing positional construction and a default empty shift map remain valid.
    legacy = CIResult(default.state_energies, default.mixing_coefficients, 10., {"legacy": True})
    assert legacy.ground_energy == default.ground_energy
    e0, g0, _ = f.evaluate(orbitals, ci_result=default)
    e1, g1, _ = f.evaluate(orbitals, ci_result=legacy)
    assert e0 == e1
    for key in f.keys:
        np.testing.assert_array_equal(g1[key], g0[key])
    empty_shifts = ci_solve(*args, **kwargs, one_body_shifts={})
    np.testing.assert_array_equal(empty_shifts.state_energies, default.state_energies)
    radial = CIRadialSnapshot(orbitals, f.data, f.Z, f.ns, nuclear_potential=nuclear)
    result = radial.solve(f.csfs, f.keys, retain_hamiltonian=True)
    h = result.hamiltonian_for(*args, **kwargs)
    ordinary = retained(problem).hamiltonian_for(*args, **kwargs)
    np.testing.assert_array_equal(h, ordinary)


def test_normalized_default_cap_and_all_provided_unused_spinors_are_bound(problem):
    args, kwargs = inputs(problem)
    args = list(args)
    args[1] = copy.deepcopy(args[1])
    unused = (99, -1)
    args[1][unused] = tuple(wave.copy() for wave in args[1][args[2][0]])
    result = ci_solve(*args, **kwargs, retain_hamiltonian=True)
    owner = result.hamiltonian_snapshot
    assert unused in owner.orbital_keys and unused not in owner.orbital_basis_keys
    assert result.hamiltonian_for(*args, **kwargs, max_determinants=owner.max_determinants) is owner.matrix_Ha
    assert result.hamiltonian_for(*args, **kwargs, max_determinants=None) is owner.matrix_Ha
    args[1][unused][0][7] += 1e-9
    with pytest.raises(ValueError, match="reference mismatch"):
        result.hamiltonian_for(*args, **kwargs)


@pytest.mark.parametrize("change", ("P", "workspace", "CSF_order", "keys", "nuclear"))
def test_public_producer_rejects_inputs_changed_during_build(problem, monkeypatch, change):
    import atomx.physics.csf.matrix as matrix
    args, kwargs = inputs(problem)
    args = list(args)
    args[0], args[1], args[3] = copy.deepcopy(args[0]), copy.deepcopy(args[1]), copy.deepcopy(args[3])
    scale = [1.0]

    def nuclear(radius):
        return -2 * scale[0] / radius

    kwargs["nuclear_potential"] = nuclear
    original = matrix.build_ci_matrix
    calls = []

    def build(*positional, **options):
        h = original(*positional, **options)
        calls.append(1)
        if change == "P":
            args[1][args[2][0]][0][7] += 1e-9
        elif change == "workspace":
            args[3]["bsp_p"][0, 0, 0] += 1e-9
        elif change == "CSF_order":
            args[0].reverse()
        elif change == "keys":
            key = args[2][0]
            args[1][(99, key[1])] = args[1].pop(key)
        elif change == "nuclear":
            scale[0] = 1.001
        return h

    monkeypatch.setattr(matrix, "build_ci_matrix", build)
    with pytest.raises(ValueError, match="changed during build|reference mismatch"):
        ci_solve(*args, **kwargs, retain_hamiltonian=True)
    assert calls == [1]


def transverse_options(problem, mode):
    energies = None if mode == 'static' else {
        key: 0. if mode == 'zero' else -2. / key[0] ** 2 + .01 * key[1]
        for key in problem[1]}
    return dict(include_breit=True, breit_orbital_energies=energies)


@pytest.fixture(scope='module')
def transverse_cases(problem):
    """One retained and one ordinary full public CI build per operator mode."""
    import atomx.physics.csf.matrix as matrix
    args, kwargs = inputs(problem)
    ordinary_builder = matrix.build_ci_matrix
    outputs = []

    def build(*args, **kwargs):
        h = ordinary_builder(*args, **kwargs)
        outputs.append(h.copy())
        return h

    cases = {}
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(matrix, 'build_ci_matrix', build)
        for mode in ('static', 'zero', 'finite'):
            options = transverse_options(problem, mode)
            start = len(outputs)
            owned = retained(problem, **options)
            assert len(outputs) == start + 1
            ordinary = ci_solve(*args, **kwargs, **options)
            assert len(outputs) == start + 2
            cases[mode] = owned, ordinary, options, outputs[-1]
    return cases


@pytest.mark.parametrize('mode', ('static', 'zero', 'finite'))
def test_retained_transverse_matches_ordinary_whole_matrix_and_spectrum(problem, transverse_cases, mode):
    args, kwargs = inputs(problem)
    result, ordinary, options, expected = transverse_cases[mode]
    h = result.hamiltonian_for(*args, **kwargs, **options)
    np.testing.assert_array_equal(h, expected)
    np.testing.assert_array_equal(result.state_energies, ordinary.state_energies)
    assert len(result.state_energies) == len(args[0])
    np.testing.assert_array_equal(h, h.T)
    assert np.max(np.abs(h - np.diag(np.diag(h)))) > 1e-3
    assert np.max(np.linalg.norm(h @ result.mixing_coefficients -
                                result.mixing_coefficients * result.state_energies, axis=0)) < 1e-10
    np.testing.assert_allclose(result.mixing_coefficients.T @ result.mixing_coefficients,
                               np.eye(len(args[0])), atol=1e-12, rtol=0)
    owner = result.hamiltonian_snapshot
    assert owner.schema_version == 2
    assert owner.hamiltonian_terms == ('dirac-coulomb',
        'static-breit' if mode == 'static' else 'finite-frequency-transverse')
    with pytest.raises(ValueError, match='operator/frequency mismatch'):
        result.hamiltonian_for(*args, **kwargs)
    with pytest.raises(ValueError):
        h.setflags(write=True)
    # The ordinary full matrix includes signed Breit off-diagonals, not a
    # scalar correction applied after diagonalization.
    from atomx.physics.csf.matrix import build_ci_matrix
    if mode == 'static':
        correction = h - build_ci_matrix(*args, **kwargs)
        assert np.max(np.abs(correction - np.diag(np.diag(correction)))) > 1e-12


def test_static_and_zero_have_distinct_owner_identity_at_the_static_limit(problem, transverse_cases):
    static = transverse_cases['static'][0].hamiltonian_snapshot
    zero = transverse_cases['zero'][0].hamiltonian_snapshot
    np.testing.assert_allclose(static.matrix_Ha, zero.matrix_Ha, atol=1e-9, rtol=0)
    assert static.input_sha256 != zero.input_sha256
    assert not static.breit_orbital_energies_bytes
    assert zero.breit_orbital_energies_bytes
    args, kwargs = inputs(problem)
    with pytest.raises(ValueError, match='operator/frequency mismatch'):
        static.matrix_for(*args, **kwargs, **transverse_cases['zero'][2])
    with pytest.raises(ValueError, match='operator/frequency mismatch'):
        zero.matrix_for(*args, **kwargs, include_breit=True)


def test_finite_owner_has_immutable_exact_frequency_and_shared_provenance(problem, transverse_cases):
    result, _, options, _ = transverse_cases['finite']
    owner = result.hamiltonian_snapshot
    expected = options['breit_orbital_energies']
    keys = tuple(sorted(expected))
    assert owner.breit_orbital_energies_hex == tuple((key, float(expected[key]).hex()) for key in keys)
    encoded = np.asarray([expected[key] for key in keys], dtype='<f8').tobytes()
    assert owner.breit_orbital_energies_bytes == encoded
    header = owner.identity_payload()
    frequency = header['transverse_operator']
    assert frequency['orbital_energies_binary64_le_hex'] == encoded.hex()
    assert frequency['orbital_energies_binary64_le_sha256'] == hashlib.sha256(encoded).hexdigest()
    assert frequency['orbital_energies_Ha'] == result.provenance['breit_orbital_energies_Ha']
    assert frequency['finite_replaces_static_breit'] is True
    assert frequency['frequency_prescription'] == 'symmetric-two-vertex-orbital-differences'
    frequency['orbital_energies_Ha'][0]['energy'] = 999.
    assert owner.identity_payload()['transverse_operator']['orbital_energies_Ha'][0]['energy'] != 999.
    with pytest.raises(FrozenInstanceError):
        owner.breit_orbital_energies_hex = ()
    args, kwargs = inputs(problem)
    changed = dict(expected)
    changed[keys[0]] = np.nextafter(changed[keys[0]], np.inf)
    with pytest.raises(ValueError, match='operator/frequency mismatch'):
        result.hamiltonian_for(*args, **kwargs, include_breit=True, breit_orbital_energies=changed)
    equivalent = dict(reversed(tuple(expected.items())))
    assert result.hamiltonian_for(*args, **kwargs, include_breit=True,
                                  breit_orbital_energies=equivalent) is owner.matrix_Ha


def test_original_dc_header_schema_is_preserved(problem):
    owner = retained(problem).hamiltonian_snapshot
    header = owner.identity_payload()
    assert owner.schema_version == 1
    assert set(header) == {'schema_version', 'representation', 'units', 'hamiltonian_terms',
                           'orbital_keys', 'orbital_basis_keys', 'CSFs', 'max_determinants', 'radial_binding'}
    payload = json.dumps(header, sort_keys=True, separators=(',', ':')).encode()
    assert owner.input_sha256 == hashlib.sha256(payload).hexdigest()


@pytest.mark.parametrize('flag', (0, 1, None, 'true', np.bool_(True), np.bool_(False)))
def test_retained_breit_flag_requires_explicit_bool_before_build(problem, monkeypatch, flag):
    import atomx.physics.csf.matrix as matrix
    monkeypatch.setattr(matrix, 'build_ci_matrix', lambda *a, **kw: pytest.fail('must reject before build'))
    with pytest.raises(TypeError, match='include_breit must be bool'):
        retained(problem, include_breit=flag)


@pytest.mark.parametrize('change', ('missing', 'extra', 'nan', 'inf', 'complex', 'bool', 'string', 'no_breit'))
def test_retained_frequency_coverage_rejects_before_build(problem, monkeypatch, change):
    import atomx.physics.csf.matrix as matrix
    options = transverse_options(problem, 'finite')
    energies = options['breit_orbital_energies']
    key = next(iter(energies))
    if change == 'missing':
        energies.pop(key)
    elif change == 'extra':
        energies[99, -1] = 0.
    elif change == 'no_breit':
        options['include_breit'] = False
    else:
        energies[key] = dict(nan=np.nan, inf=np.inf, complex=1j, bool=True, string='-2.')[change]
    monkeypatch.setattr(matrix, 'build_ci_matrix', lambda *a, **kw: pytest.fail('must reject before build'))
    with pytest.raises(ValueError):
        retained(problem, **options)


@pytest.mark.parametrize('change', ('caller_map', 'builder_copy'))
def test_frequency_mutation_during_build_rejects(problem, monkeypatch, change):
    import atomx.physics.csf.matrix as matrix
    args, kwargs = inputs(problem)
    options = transverse_options(problem, 'finite')
    original = matrix.build_ci_matrix
    calls = []

    def build(*args, **kwargs):
        h = original(*args, **kwargs)
        calls.append(1)
        energies = (options['breit_orbital_energies'] if change == 'caller_map'
                    else kwargs['breit_orbital_energies'])
        key = next(iter(energies))
        energies[key] = np.nextafter(energies[key], np.inf)
        return h

    monkeypatch.setattr(matrix, 'build_ci_matrix', build)
    with pytest.raises(ValueError, match='operator/frequency inputs changed during build'):
        ci_solve(*args, **kwargs, retain_hamiltonian=True, **options)
    assert calls == [1]


@pytest.mark.parametrize('mode', ('static', 'zero', 'finite'))
def test_breit_owner_rejects_stale_radial_inputs_and_dc_functional_before_gradient(
        problem, transverse_cases, monkeypatch, mode):
    f, orbitals, _ = problem
    result, _, options, _ = transverse_cases[mode]
    args, kwargs = inputs(problem)
    changed = copy.deepcopy(orbitals)
    changed[f.keys[0]][0][7] += 1e-9
    with pytest.raises(ValueError, match='reference mismatch'):
        result.hamiltonian_for(args[0], changed, *args[2:], **kwargs, **options)
    monkeypatch.setattr(f, 'gradient_from_weights', lambda *a, **kw: pytest.fail('non-DC gradient'))
    with pytest.raises(ValueError, match='operator/frequency mismatch'):
        f.evaluate(orbitals, ci_result=result)


def test_genuine_breit_owner_rejects_mixed_dc_consumer_before_physical_work(problem, transverse_cases, monkeypatch):
    from atomx.physics.csf import radial_increment
    from atomx.physics.csf.tracking import CIBasisTracking
    f, orbitals, _ = problem
    consumer = copy.copy(f)
    dc = retained(problem)
    consumer.level_tracking = CIBasisTracking(f.state_average, f.csfs, dc, orbitals, f.metric)
    breit = transverse_cases['finite'][0]
    zero = {key: tuple(np.zeros_like(v) for v in pair) for key, pair in orbitals.items()}
    monkeypatch.setattr(radial_increment, 'bspline_dc_radial_increment',
                        lambda *a, **kw: pytest.fail('non-DC radial difference'))
    with pytest.raises(ValueError, match='only complete native DC owners'):
        consumer.mixed_ci_energy_difference(orbitals, orbitals, dc, breit, increments=zero)


def test_unused_provided_spinor_requires_and_binds_its_finite_energy(problem, monkeypatch):
    import atomx.physics.csf.matrix as matrix
    args, kwargs = inputs(problem)
    args = list(args)
    args[1] = copy.deepcopy(args[1])
    unused = (99, -1)
    args[1][unused] = tuple(v.copy() for v in args[1][args[2][0]])
    options = transverse_options(problem, 'finite')
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(matrix, 'build_ci_matrix', lambda *a, **kw: pytest.fail('missing unused energy'))
        with pytest.raises(ValueError, match='exactly every provided orbital'):
            ci_solve(*args, **kwargs, retain_hamiltonian=True, **options)
    options['breit_orbital_energies'][unused] = .5
    result = ci_solve(*args, **kwargs, retain_hamiltonian=True, **options)
    assert unused in result.hamiltonian_snapshot.orbital_keys
    changed = copy.deepcopy(options)
    changed['breit_orbital_energies'][unused] += 1e-9
    with pytest.raises(ValueError, match='operator/frequency mismatch'):
        result.hamiltonian_for(*args, **kwargs, **changed)
