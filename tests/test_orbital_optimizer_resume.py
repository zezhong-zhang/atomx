"""A saved metric resumes an independently soluble constrained eigenproblem."""
import hashlib
import importlib.util
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.linalg import eigh

from atomx.config.csf import enumerate_jj_csfs
from atomx.methods.csf.variational import CSFOrbitalFunctional
from atomx.numerics.bspline.basis import prepare_bspline_data
from atomx.numerics.bspline.dirac_assembly import solve_dirac
from atomx.numerics.bspline.knots import make_knots
from atomx.numerics.bspline.orbital_optimization import (
    OrbitalOptimizationState,
    _hash_value,
    optimize_orbitals,
)


def problem(*, kinetic_balance=False, same_kappa_pair=False):
    data = prepare_bspline_data(make_knots(35, 9, .005, 30), 35, 9,
                               kinetic_balance=kinetic_balance)

    class IndependentParticles(CSFOrbitalFunctional):
        """No repulsion: independent Rayleigh energy and analytic derivative."""
        def evaluate(self, waves, *, gradient=True, ci_result=None):
            self.calls += 1
            weights = ({key: 1. for key in self.keys} if same_kappa_pair
                       else {(1, -1): 2.})
            gradients = {}
            energy = 0.
            for key, weight in weights.items():
                coefficients = np.concatenate(waves[key])
                hc = self.energy_operator[key[1]] @ coefficients
                energy += weight * float(coefficients @ hc)
                gradients[key] = 2 * weight * hc
            ci = SimpleNamespace(ground_energy=energy, recomputation=self.calls)
            if not gradient:
                return energy, ci
            self.last_occupations = {key: self.population for key in weights}
            return energy, gradients, ci

    csfs = enumerate_jj_csfs('1s1 2s1' if same_kappa_pair else '1s2')
    functional = IndependentParticles(csfs, data, 1)
    functional.energy_operator = functional.one_body
    functional.population, functional.calls = 2., 0
    ns = functional.ns
    spectrum = solve_dirac(functional.one_body[-1], functional.metric[-1], ns, -1,
                           nsp=data['nsp'], nsq=data['nsq'])
    combination = np.array([[.9, -.3], [.3, .9], [.1, .2]])
    combination = np.linalg.qr(combination)[0]
    waves = {}
    for column, key in enumerate(functional.keys):
        coefficient = spectrum.coefficients[:, :3] @ combination[:, column]
        waves[key] = coefficient[:ns], coefficient[ns:]
    return functional, waves, spectrum


def saved_problem(*, blocks=False, kinetic_balance=False, same_kappa_pair=False):
    functional, waves, spectrum = problem(kinetic_balance=kinetic_balance,
                                          same_kappa_pair=same_kappa_pair)
    matrices = ({key: functional.one_body[key[1]] for key in functional.keys}
                if blocks else None)
    final, _, report = optimize_orbitals(functional, waves, max_iter=3,
                                         preconditioner_matrices=matrices,
                                         objective_identity='original-target', return_state=True)
    return functional, final, spectrum, report, report['optimizer_state']


def import_state(functional, state, **changes):
    fields = dict(layout=state.layout, coordinates=state.coordinates,
                  inverse_diagonal=state.inverse_diagonal, inverse_blocks=state.inverse_blocks,
                  projector_source=state.projector_source, objective_identity=state.objective_identity,
                  occupation_floor=state.occupation_floor)
    fields.update(changes)
    return OrbitalOptimizationState.from_arrays(functional, **fields)


def independent_metric(functional, state):
    """Apply actual saved inverses, project constraints, and derive g.P.g."""
    g_all, pg_all, energy = [], [], 0.
    for group, (keys, projector, shape, where) in enumerate(state.layout):
        y = state.coordinates[where].reshape(shape)
        c = projector @ y
        gradients = []
        for column, key in enumerate(keys):
            weight = 2. if len(functional.keys) == 1 else 1.
            h = functional.energy_operator[key[1]]
            energy += weight * float(c[:, column] @ h @ c[:, column])
            gradients.append(2 * weight * h @ c[:, column])
        raw = projector.T @ np.column_stack(gradients)
        g = raw - y @ ((y.T @ raw + raw.T @ y) / 2)
        if state.inverse_blocks:
            inverse_action = np.column_stack([state.inverse_blocks[group][column] @ g[:, column]
                                             for column in range(shape[1])])
        else:
            inverse_action = state.inverse_diagonal[where].reshape(shape) * g
        pg = inverse_action - y @ ((y.T @ inverse_action + inverse_action.T @ y) / 2)
        g_all.extend(g.ravel())
        pg_all.extend(pg.ravel())
        np.testing.assert_allclose(c.T @ functional.metric[keys[0][1]] @ c, np.eye(shape[1]), atol=2e-12)
    return energy, np.linalg.norm(pg_all), .5 * abs(np.dot(g_all, pg_all))


@pytest.mark.parametrize('blocks', [False, True])
def test_original_metric_roundtrip_and_full_recomputation(blocks, monkeypatch):
    functional, waves, _, previous, state = saved_problem(blocks=blocks)
    imported = import_state(functional, state)
    assert imported.state_sha256 == state.state_sha256
    assert imported.residual_metric_sha256 == previous['residual_metric_sha256']
    expected_energy, expected_residual, expected_indicator = independent_metric(functional, imported)
    functional.population = 999.  # Must not rescale the restored inverse.
    before = functional.calls
    monkeypatch.setattr('atomx.numerics.bspline.dirac_assembly.solve_dirac',
                        lambda *a, **k: pytest.fail('resume rebuilt its spectrum'))
    final, ci, report = optimize_orbitals(functional, waves, resume_state=imported,
                                         objective_identity='original-target', max_iter=0,
                                         tolerance=1e-7, energy_tolerance=1e-8, return_state=True)
    assert functional.calls == before + 1 and ci.recomputation == functional.calls
    assert report['residual_metric_sha256'] == state.residual_metric_sha256
    assert report['optimizer_state'].state_sha256 == state.state_sha256
    assert report['restored_original_metric'] and report['initial_history_size'] == 0
    assert ci.ground_energy == pytest.approx(expected_energy, abs=2e-15)
    assert report['orbital_residual'] == pytest.approx(expected_residual, rel=1e-13)
    assert report['estimated_energy_error'] == pytest.approx(expected_indicator, rel=1e-13)
    assert report['orbital_tolerance'] == 1e-7 and report['energy_tolerance'] == 1e-8
    for key in waves:
        for left, right in zip(final[key], waves[key], strict=True):
            np.testing.assert_array_equal(left, right)


def test_resume_and_endpoint_restart_have_distinct_yardsticks():
    functional, waves, _, old, state = saved_problem()
    _, _, fresh = optimize_orbitals(functional, waves, max_iter=0)
    _, _, resumed = optimize_orbitals(functional, resume_state=state,
                                      objective_identity=state.objective_identity, max_iter=0)
    assert old['residual_metric_sha256'] != fresh['residual_metric_sha256']
    assert old['residual_metric_sha256'] == resumed['residual_metric_sha256']
    assert old['orbital_residual'] == resumed['orbital_residual']
    assert fresh['orbital_residual'] != resumed['orbital_residual']


def test_changed_objective_needs_opt_in_and_recomputes_ci_gradient():
    functional, _, _, previous, state = saved_problem()
    # Change only the objective; physical basis, nucleus and no-pair S stay fixed.
    functional.energy_operator = {-1: 1.7 * functional.one_body[-1]}
    before = functional.calls
    with pytest.raises(ValueError, match='allow_objective_change'):
        optimize_orbitals(functional, resume_state=state, objective_identity='new-target')
    assert functional.calls == before
    _, ci, report = optimize_orbitals(functional, resume_state=state, objective_identity='new-target',
                                      allow_objective_change=True, max_iter=0, return_state=True)
    energy, residual, indicator = independent_metric(functional, state)
    assert report['objective_changed'] and report['initial_history_size'] == 0
    assert ci.ground_energy == energy
    assert report['orbital_residual'] == pytest.approx(residual, rel=1e-13)
    assert report['estimated_energy_error'] == pytest.approx(indicator, rel=1e-13)
    assert report['orbital_residual'] == pytest.approx(1.7 * previous['orbital_residual'], rel=2e-13)
    assert report['optimizer_state'].objective_identity == 'new-target'
    assert report['optimizer_state'].geometry_sha256 == state.geometry_sha256


def test_accepted_resume_step_has_independent_rayleigh_gram_and_shared_snapshots():
    functional, _, _, _, state = saved_problem(blocks=True)
    checkpoints, progress = [], []
    final, _, report = optimize_orbitals(functional, resume_state=state,
                                         objective_identity=state.objective_identity, max_iter=2,
                                         return_state=True, checkpoint_callback=lambda *args: checkpoints.append(args),
                                         progress_callback=lambda *args: progress.append(args))
    assert len(checkpoints) == len(progress) == 3 and report['iterations'] == 2
    assert all(len(args) == 4 for args in progress)
    for expected_iteration, (iteration, snapshot, waves, ci, diagnostic) in enumerate(checkpoints):
        energy, residual, indicator = independent_metric(functional, snapshot)
        assert diagnostic['accepted_point'] and diagnostic['initial_history_size'] == 0
        assert energy == pytest.approx(ci.ground_energy, abs=2e-15)
        assert ci.ground_energy == diagnostic['energy_ha']
        assert diagnostic['orbital_residual'] == pytest.approx(residual, rel=1e-13)
        assert diagnostic['estimated_energy_error'] == pytest.approx(indicator, rel=1e-13)
        assert diagnostic['geometry_sha256'] == state.geometry_sha256
        assert snapshot._geometry is state._geometry
        assert snapshot.layout[0][1] is state.layout[0][1]
        c = snapshot.layout[0][1] @ snapshot.coordinates.reshape(snapshot.layout[0][2])
        np.testing.assert_array_equal(np.concatenate(waves[1, -1]), c[:, 0])
        assert iteration == expected_iteration
    assert np.all(np.diff(report['energy_history']) < 0)
    assert report['optimizer_state']._geometry is state._geometry
    np.testing.assert_array_equal(final[1, -1][0], checkpoints[-1][2][1, -1][0])


def test_default_path_matches_frozen_prechange_optimizer_bitwise():
    archived = Path(__file__).resolve().parents[1] / 'docs/yb_mcdhf_rci_2026_10_11/public_optimizer_resume/source_before/orbital_optimization.py'
    spec = importlib.util.spec_from_file_location('optimizer_resume_prechange', archived)
    old_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old_module)
    functional, waves, _ = problem()
    old_final, _, old = old_module.optimize_orbitals(functional, waves, max_iter=12)
    final, _, current = optimize_orbitals(functional, waves, max_iter=12)
    assert current == old
    for key in waves:
        for left, right in zip(final[key], old_final[key], strict=True):
            np.testing.assert_array_equal(left, right)


def test_state_owns_sealed_bytes_and_rejects_payload_tampering():
    functional, _, _, _, state = saved_problem(blocks=True)
    caller_point = state.coordinates.copy()
    caller_projector = state.layout[0][1].copy()
    layout = [(state.layout[0][0], caller_projector, state.layout[0][2], state.layout[0][3])]
    imported = import_state(functional, state, layout=layout, coordinates=caller_point)
    identity = imported.state_sha256
    caller_point[:] = 0
    caller_projector[:] = 0
    assert imported.state_sha256 == identity
    for array in (imported.coordinates, imported.inverse_diagonal, imported.layout[0][1],
                  imported.inverse_blocks[0][0], imported._geometry.metrics[0][1]):
        with pytest.raises(ValueError):
            array.setflags(write=True)
        with pytest.raises(ValueError):
            array.flat[0] = 0
    with pytest.raises(FrozenInstanceError):
        imported.objective_identity = 'changed'
    forged_geometry = replace(imported._geometry, inverse_diagonal=2 * imported.inverse_diagonal)
    tampered = replace(imported, _geometry=forged_geometry)
    before = functional.calls
    with pytest.raises(ValueError, match='payload mismatch'):
        optimize_orbitals(functional, resume_state=tampered, objective_identity=tampered.objective_identity)
    assert functional.calls == before


@pytest.mark.parametrize('kind', ['key_order', 'duplicate_kappa', 'missing_key', 'overlap', 'shape',
                                  'not_orthonormal', 'negative_branch', 'nan', 'complex', 'point',
                                  'diagonal', 'inverse_asymmetric', 'inverse_negative', 'outer_array'])
def test_invalid_geometry_is_rejected_without_repair(kind):
    functional, _, _, _, state = saved_problem(blocks=True, same_kappa_pair=True)
    keys, projector, shape, where = state.layout[0]
    fields = {}
    if kind == 'key_order':
        fields['layout'] = [(tuple(reversed(keys)), projector, shape, where)]
    elif kind == 'duplicate_kappa':
        fields['layout'] = list(state.layout) * 2
    elif kind == 'missing_key':
        fields['layout'] = [(keys[:1], projector, shape, where)]
    elif kind == 'overlap':
        fields['layout'] = [(keys, projector, shape, slice(1, where.stop + 1))]
    elif kind == 'shape':
        fields['layout'] = [(keys, projector, (shape[0] + 1, shape[1]), where)]
    elif kind == 'not_orthonormal':
        fields['layout'] = [(keys, 2 * projector, shape, where)]
    elif kind == 'negative_branch':
        p = projector.copy()
        values, eigenvectors = eigh(functional.one_body[-1], functional.metric[-1])
        assert values[0] < -10000
        p[:, 0] = eigenvectors[:, 0]
        fields['layout'] = [(keys, p, shape, where)]
    elif kind in ('nan', 'complex'):
        point = state.coordinates.copy().astype(complex if kind == 'complex' else float)
        point[0] = np.nan if kind == 'nan' else point[0] + 1j
        fields['coordinates'] = point
    elif kind == 'point':
        fields['coordinates'] = 2 * state.coordinates
    elif kind == 'diagonal':
        fields['inverse_diagonal'] = -state.inverse_diagonal
    elif kind in ('inverse_asymmetric', 'inverse_negative'):
        matrices = [[b.copy() for b in group] for group in state.inverse_blocks]
        if kind == 'inverse_negative':
            matrices[0][0] *= -1
        else:
            matrices[0][0][0, 1] += 1.
        fields['inverse_blocks'] = matrices
    else:
        fields['inverse_blocks'] = np.array(state.inverse_blocks)
    with pytest.raises(ValueError):
        import_state(functional, state, **fields)


def test_inactive_padding_cannot_hide_small_null_metric_coordinates():
    functional, _, _, _, state = saved_problem(kinetic_balance=True)
    keys, projector, shape, where = state.layout[0]
    inactive = functional.data['nsp']
    assert inactive < functional.ns
    p = projector.copy()
    p[inactive, 0] = 1e-14
    with pytest.raises(ValueError, match='inactive padded'):
        import_state(functional, state, layout=[(keys, p, shape, where)])
    s = functional.metric[-1]
    s[inactive, 0] = s[0, inactive] = 1e-14
    with pytest.raises(ValueError, match='inactive padded'):
        import_state(functional, state)


def test_combined_actual_radial_gram_cannot_exceed_declared_guard():
    functional, _, _, _, state = saved_problem()
    keys, projector, shape, where = state.layout[0]
    # Each separate Gram defect is 8e-11, but their physical composition
    # exceeds 1e-10: validating Y and P independently is insufficient.
    p = projector * (1. + 4e-11)
    point = state.coordinates * (1. + 4e-11)
    assert np.linalg.norm(p.T @ functional.metric[-1] @ p - np.eye(shape[0]), ord=np.inf) < 1e-10
    assert abs(point @ point - 1.) < 1e-10
    with pytest.raises(ValueError, match='physical radial point'):
        import_state(functional, state, layout=[(keys, p, shape, where)], coordinates=point)


def test_malformed_padded_metric_shape_has_deliberate_value_error():
    functional, _, _, _, state = saved_problem(kinetic_balance=True)
    functional.metric[-1] = functional.metric[-1][:1, :1]
    with pytest.raises(ValueError, match='metric shape'):
        import_state(functional, state)


def test_public_state_requires_explicit_objective_declaration():
    functional, waves, _ = problem()
    before = functional.calls
    with pytest.raises(ValueError, match='objective_identity'):
        optimize_orbitals(functional, waves, return_state=True)
    assert functional.calls == before


@pytest.mark.parametrize('mutation', ['workspace', 'S', 'nucleus', 'one_body', 'keys', 'Z'])
def test_resume_physical_binding_rejects_before_objective_evaluation(mutation):
    functional, _, _, _, state = saved_problem()
    if mutation == 'workspace':
        functional.data['gr'] = functional.data['gr'] + 1e-12
    elif mutation == 'S':
        functional.metric[-1] = 1.0000001 * functional.metric[-1]
    elif mutation == 'nucleus':
        functional.nuclear_potential = lambda r: -1.00001 / r
    elif mutation == 'one_body':
        functional.one_body[-1] = 1.000001 * functional.one_body[-1]
    elif mutation == 'keys':
        functional.keys = list(reversed(functional.keys)) + [(2, -1)]
    else:
        functional.Z += 1
    before = functional.calls
    with pytest.raises(ValueError, match='mismatch|keys'):
        optimize_orbitals(functional, resume_state=state, objective_identity=state.objective_identity)
    assert functional.calls == before


def test_strong_hash_frames_sequences_mappings_and_types():
    def identity(value):
        digest = hashlib.sha256()
        _hash_value(digest, value)
        return digest.hexdigest()
    assert identity([1, 23]) != identity([12, 3])
    assert identity({'a': {'b': 1}}) != identity({'a': {}, 'b': 1})
    assert identity((1, 23)) != identity([1, 23])
    assert identity(1) != identity('1')
    functional, _, _, _, state = saved_problem()
    functional.data['framing_control'] = [1, 23]
    imported = import_state(functional, state)
    functional.data['framing_control'] = [12, 3]
    with pytest.raises(ValueError, match='geometry mismatch'):
        imported.validate(functional)


@pytest.mark.parametrize('changes', [dict(preconditioner_matrices={}),
                                    dict(preconditioner_factory=lambda waves: {}),
                                    dict(orbital_preconditioner='shooting'),
                                    dict(occupation_floor=.1), dict(projector_source='fock')])
def test_incompatible_resume_controls_reject_before_evaluating(changes):
    functional, _, _, _, state = saved_problem()
    before = functional.calls
    with pytest.raises(ValueError, match='resume'):
        optimize_orbitals(functional, resume_state=state, objective_identity=state.objective_identity, **changes)
    assert functional.calls == before


def test_optional_radial_seed_cannot_reproject_or_replace_saved_point():
    functional, waves, _, _, state = saved_problem()
    inconsistent = {key: (p.copy(), q.copy()) for key, (p, q) in waves.items()}
    inconsistent[1, -1][0][0] += 1e-15
    before = functional.calls
    with pytest.raises(ValueError, match='authoritative coordinates'):
        optimize_orbitals(functional, inconsistent, resume_state=state, objective_identity=state.objective_identity)
    assert functional.calls == before


def test_wall_budget_returns_last_accepted_checkpoint_not_trial(monkeypatch):
    functional, _, _, _, state = saved_problem()
    snapshots = []
    _, _, zero = optimize_orbitals(functional, resume_state=state,
                                    objective_identity=state.objective_identity, max_wall_time=0,
                                    return_state=True, checkpoint_callback=lambda *args: snapshots.append(args))
    assert not zero['converged'] and zero['termination_reason'] == 'wall_time_limit'
    assert zero['optimizer_state'].state_sha256 == state.state_sha256
    assert len(snapshots) == 1 and snapshots[0][0] == 0
    ticks = iter([0., 0., 0., 2.])
    monkeypatch.setattr('atomx.numerics.bspline.orbital_optimization.time.monotonic', lambda: next(ticks))
    _, _, trial_stop = optimize_orbitals(functional, resume_state=state,
                                         objective_identity=state.objective_identity, max_wall_time=1,
                                         return_state=True)
    assert trial_stop['termination_reason'] == 'wall_time_limit' and trial_stop['iterations'] == 0
    assert trial_stop['optimizer_state'].state_sha256 == state.state_sha256
    assert len(trial_stop['energy_history']) == 1


def test_fock_import_requires_actual_operator_and_does_not_infer_it_from_inverse():
    functional, _, _, _, state = saved_problem(blocks=True)
    with pytest.raises(ValueError, match='explicit original projector'):
        import_state(functional, state, projector_source='fock')
    imported = import_state(functional, state, projector_source='fock',
                            projector_operators=functional.one_body)
    assert imported.projector_source == 'fock'
    _, _, report = optimize_orbitals(functional, resume_state=imported,
                                      objective_identity=state.objective_identity, projector_source='fock', max_iter=0)
    assert report['residual_metric_sha256'] == state.residual_metric_sha256
