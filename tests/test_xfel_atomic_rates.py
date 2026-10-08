"""Population accounting and empty-hole regressions for native XFEL rates."""
import itertools

import pytest

from atomx.xfel.atomic_rates import (
    auger_candidates,
    build_atomic_rate_table,
    ls_pair_partition,
)

BASE = {(1, -1): 2, (2, -1): 2, (2, 1): 2, (2, -2): 4}


@pytest.mark.parametrize('p1,p3', itertools.product(range(3), range(5)))
def test_jj_pair_partitions_conserve_ls_population(p1, p3):
    config = {(2, 1): p1, (2, -2): p3, (2, -1): 2}
    donors = ((2, 1), (2, -2))
    weights = [ls_pair_partition(config, a, b)
               for a, b in itertools.combinations_with_replacement(donors, 2)]
    assert sum(weights) == pytest.approx(1 if p1 + p3 >= 2 else 0)
    mixed = [ls_pair_partition(config, (2, -1), a) for a in donors]
    assert sum(mixed) == pytest.approx(1 if p1 + p3 else 0)


def test_full_p_shell_uses_ordered_pair_counts():
    assert ls_pair_partition(BASE, (2, 1), (2, 1)) == pytest.approx(2 / 30)
    assert ls_pair_partition(BASE, (2, -2), (2, -2)) == pytest.approx(12 / 30)
    assert ls_pair_partition(BASE, (2, 1), (2, -2)) == pytest.approx(16 / 30)


def test_empty_k_hole_refills_and_scales_with_vacancies():
    config = {k: v for k, v in BASE.items() if k != (1, -1)}
    channels = [ch for ch in auger_candidates(config, tuple(BASE)) if ch[0] == (1, -1)]
    assert len(channels) == 6
    assert sum(c[3] for c in channels) == pytest.approx(6)  # 3 LS families x 2 holes
    for hole, _a, _b, weight, daughter in channels:
        assert daughter[hole] == 1
        assert sum(daughter.values()) == 7
        assert weight > 0
        assert all(0 <= q <= 2 * abs(k) for (n, k), q in daughter.items())
    single = dict(config)
    single[(1, -1)] = 1
    single_channels = [ch for ch in auger_candidates(single, tuple(BASE)) if ch[0] == (1, -1)]
    assert [c[3] for c in channels] == pytest.approx([2 * c[3] for c in single_channels])


def test_closed_base_has_no_auger_candidates():
    assert list(auger_candidates(BASE, tuple(BASE))) == []


def test_single_electron_cannot_auger():
    assert list(auger_candidates({(2, -2): 1}, tuple(BASE))) == []


@pytest.mark.parametrize('base', ({(1, -1): 1}, {(1, -1): 2.5}, {}))
def test_native_builder_rejects_invalid_base_before_solving(base, tmp_path):
    with pytest.raises(ValueError):
        build_atomic_rate_table(10, base, 1000, checkpoint_dir=tmp_path)
    assert not list(tmp_path.iterdir())


def test_scf_coefficients_are_adapted_before_decay(monkeypatch, tmp_path):
    """A missing orbitals_dict previously returned six silent zero widths."""
    from types import SimpleNamespace

    import numpy as np

    from atomx.config.subshell import Subshell
    from atomx.methods.dirac_fock import SCFResult
    from atomx.scattering import autoionization
    from atomx.scattering.photoionization import subshell
    from atomx.xfel.atomic_rates import _rate_state
    from atomx.xfel.cascade import _table_key

    config = dict(BASE)
    config[(1, -1)] = 1
    shells = tuple(BASE)
    fake = SimpleNamespace(
        Z=10, _bsp_data={'test': True}, orbital_energies={},
        subshells=[Subshell(n, k, q) for (n, k), q in config.items()],
        orbitals_by_subshell=lambda: {s: (np.ones(3), np.zeros(3), q)
                                     for s, q in config.items()},
    )
    monkeypatch.setattr(SCFResult, 'load', lambda path: fake)
    calls = []
    def decay(result, **kwargs):
        assert set(result.orbitals_dict) == set(BASE)
        assert all(len(pair) == 2 for pair in result.orbitals_dict.values())
        assert result.occupations == config
        calls.append(kwargs)
        return {'Gamma_per_s': 100}
    monkeypatch.setattr(autoionization, 'auger_rate_csf_average', decay)
    monkeypatch.setattr(subshell, 'subshell_photoionization', lambda *a, **kw:
                        SimpleNamespace(sigma_by_orbital_a0_sq={s.label: [1.] for s in fake.subshells},
                                        provenance={}))
    # Independent energy assignment: PI removes a bound electron; only
    # filling the K hole releases energy in this deliberately simple model.
    energies = {_table_key(config): -100.}
    for shell in config:
        d = dict(config)
        d[shell] -= 1
        energies[_table_key(d)] = -99.
    for h, _a, _b, _w, d in auger_candidates(config, shells):
        energies[_table_key(d)] = -110. if h == (1, -1) else -90.
    _, row = _rate_state((config, shells, {'photon_energy_eV': 1000, 'pi_grid': 40,
                                         'r_max': 80, 'auger_grid': 40},
                         str(tmp_path), {'test': True}, energies))
    assert len(calls) == len(row['auger']) == 6
    assert sum(ch['rate'] for ch in row['auger']) == pytest.approx(300)


def test_failed_scf_does_not_publish_checkpoint(monkeypatch, tmp_path):
    from types import SimpleNamespace

    from atomx.methods.dirac_fock import DFSolver
    from atomx.xfel.atomic_rates import _solve_state

    monkeypatch.setattr(DFSolver, 'solve', lambda *args, **kw:
                        SimpleNamespace(converged=False, total_energy=-1.))
    with pytest.raises(RuntimeError, match='SCF failed'):
        _solve_state(({(1, -1): 2}, ((1, -1),),
                      {'Z': 2, 'ns': 40, 'ks': 7, 'r_max': 30},
                      str(tmp_path), {'test': True}))
    assert not list(tmp_path.iterdir())


def test_checkpoint_identity_mismatch_is_not_silently_reused(tmp_path):
    import json

    from atomx.xfel.atomic_rates import _solve_state, _state_path

    config = {(1, -1): 2}
    path = _state_path(tmp_path, config)
    path.with_suffix('.h5').write_bytes(b'not loaded: incompatible identity')
    path.with_suffix('.json').write_text(json.dumps({'identity': {'old': True}}))
    with pytest.raises(ValueError, match='incompatible SCF checkpoint'):
        _solve_state((config, ((1, -1),), {}, str(tmp_path), {'new': True}))
