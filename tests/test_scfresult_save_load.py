"""Regression tests for SCFResult.save()/load() (HYP-1784).

``orbitals_raw`` per-kappa lists are positionally aligned with the per-kappa
order of ``subshells`` (see ``SCFResult.orbitals_by_subshell``).  ``save()``
used to index them by ``n - 1 - l``, silently dropping orbitals of gapped
configs (e.g. 1s2 3s1) whose quantum-number index exceeds the list length,
and ``load()`` sorted ``orb_*`` group keys lexicographically, scrambling
per-kappa order for configs with >9 subshells.
"""

import h5py
import numpy as np
import pytest

from atomx import Atom
from atomx.methods.dirac_fock import DiracFock, SCFResult, Subshell


def _make_result(subshells, orbitals_raw):
    return SCFResult(
        converged=True,
        iterations=12,
        residual=1e-8,
        total_energy=-100.5,
        orbital_energies={sh.label: -10.0 * i
                          for i, sh in enumerate(subshells, start=1)},
        subshells=subshells,
        energy_components={"one_electron": -110.0, "exchange": -5.5},
        orbitals_raw=orbitals_raw,
    )


def _gapped_result():
    """Gapped config 1s2 3s1: two kappa=-1 subshells with a hole at n=2."""
    subshells = [Subshell(1, -1, 2.0), Subshell(3, -1, 1.0)]
    orbitals_raw = {
        -1: [
            (np.full(8, 1.0), np.full(8, 0.1), 2.0),   # 1s
            (np.full(8, 3.0), np.full(8, 0.3), 1.0),   # 3s
        ],
    }
    return _make_result(subshells, orbitals_raw)


def test_save_writes_all_orbitals_for_gapped_config(tmp_path):
    path = str(tmp_path / "gapped")
    _gapped_result().save(path)
    with h5py.File(f"{path}.h5", "r") as f:
        assert set(f["orbitals"].keys()) == {"orb_0", "orb_1"}


def test_gapped_config_save_load_roundtrip(tmp_path):
    result = _gapped_result()
    path = str(tmp_path / "gapped")
    result.save(path)
    loaded = SCFResult.load(path)

    # Subshells round-trip in order.
    assert [(s.n, s.kappa, s.occupation, s.label)
            for s in loaded.subshells] == [
        (1, -1, 2.0, "1s1/2"),
        (3, -1, 1.0, "3s1/2"),
    ]

    # Scalars and energies round-trip.
    assert loaded.converged is True
    assert loaded.iterations == 12
    assert loaded.residual == pytest.approx(1e-8)
    assert loaded.total_energy == pytest.approx(-100.5)
    assert loaded.orbital_energies == result.orbital_energies
    assert dict(loaded.energy_components) == result.energy_components

    # All orbitals present, positionally aligned per kappa, data intact.
    assert set(loaded.orbitals_raw) == {-1}
    assert len(loaded.orbitals_raw[-1]) == 2
    for orig, got in zip(result.orbitals_raw[-1], loaded.orbitals_raw[-1], strict=True):
        np.testing.assert_allclose(got[0], orig[0])
        np.testing.assert_allclose(got[1], orig[1])
        assert got[2] == pytest.approx(orig[2])

    # The documented accessor sees both (n, kappa) entries.
    by_subshell = loaded.orbitals_by_subshell()
    assert set(by_subshell) == {(1, -1), (3, -1)}


def test_gapped_configuration_exposes_both_radial_waves():
    result = Atom(Z=3, config="1s2 3s1").solve(DiracFock())
    assert result.converged
    assert set(result.waves.orbitals) == {"1s1/2", "3s1/2"}
    assert result.waves.orbitals["3s1/2"].P.size == result.waves.grid.size


def test_save_load_roundtrip_many_same_kappa_subshells(tmp_path):
    """12 ns subshells (>9) exercise numeric ordering of ``orb_*`` keys."""
    n_shells = 12
    subshells = [Subshell(n, -1, 2.0) for n in range(1, n_shells + 1)]
    orbitals_raw = {
        -1: [(np.full(4, float(n)), np.full(4, -float(n)), 2.0)
             for n in range(1, n_shells + 1)],
    }
    result = _make_result(subshells, orbitals_raw)

    path = str(tmp_path / "many")
    result.save(path)
    loaded = SCFResult.load(path)

    assert len(loaded.orbitals_raw[-1]) == n_shells
    for i, (P, Q, occ) in enumerate(loaded.orbitals_raw[-1], start=1):
        np.testing.assert_allclose(P, np.full(4, float(i)))
        np.testing.assert_allclose(Q, np.full(4, -float(i)))
        assert occ == pytest.approx(2.0)
    assert [sh.label for sh in loaded.subshells] == \
        [sh.label for sh in subshells]


def test_save_raises_on_incomplete_orbitals(tmp_path):
    """Partial orbitals_raw must fail loudly, not silently drop orbitals."""
    subshells = [Subshell(1, -1, 2.0), Subshell(3, -1, 1.0)]
    orbitals_raw = {-1: [(np.ones(4), np.ones(4), 2.0)]}  # 3s missing
    result = _make_result(subshells, orbitals_raw)
    with pytest.raises(ValueError, match="does not cover subshells"):
        result.save(str(tmp_path / "incomplete"))


def test_save_load_roundtrip_without_orbitals(tmp_path):
    """Energy-only results (empty orbitals_raw) still round-trip."""
    result = _make_result([Subshell(1, -1, 2.0)], {})
    path = str(tmp_path / "energy_only")
    result.save(path)
    loaded = SCFResult.load(path)
    assert loaded.orbitals_raw == {}
    assert loaded.total_energy == pytest.approx(result.total_energy)
    assert [sh.label for sh in loaded.subshells] == ["1s1/2"]


def test_physical_identity_and_correction_accounting_roundtrip(tmp_path):
    from atomx.calculation_spec import CalculationSpec
    from atomx.corrections.protocol import CorrectionApplication, CorrectionResult
    from atomx.corrections.stack import CorrectionStack, finalize_with_corrections

    result = _gapped_result()
    result.Z = 3
    result._equation = 'dirac'
    result.calculation_spec = CalculationSpec(3, 'DHF', 'bspline',
        {'equation': 'dirac', 'nuclear': 'point'}, {'isotope': 'Li-7'},
        {'ns': 100}, {'include_breit': True})
    result.provenance = {'reference': 'test model'}
    result.solver_diagnostics = {'residual': 1e-9}
    result.correction_application = CorrectionApplication(-100.6, (('breit', .1),))
    result._correction_results = {'breit': CorrectionResult('breit', .1,
        provenance={'operator': 'static Breit'})}
    path = tmp_path / 'physical.h5'
    result.save(path)
    loaded = SCFResult.load(path)
    assert loaded.Z == 3
    assert loaded._equation == 'dirac'
    assert loaded.calculation_spec.to_dict() == result.calculation_spec.to_dict()
    assert loaded.provenance == result.provenance
    assert loaded.solver_diagnostics == result.solver_diagnostics
    assert loaded.correction_application == result.correction_application
    assert loaded._correction_results == result._correction_results
    assert loaded.persistence_metadata['missing'] == []
    with pytest.raises(ValueError, match='already finalized'):
        finalize_with_corrections(loaded, CorrectionStack())


def test_old_archive_reports_missing_model_identity(tmp_path):
    path = tmp_path / 'legacy.h5'
    _gapped_result().save(path)
    with h5py.File(path, 'a') as archive:
        del archive.attrs['schema_version']
        del archive['physical_metadata']
    result = SCFResult.load(path)
    assert result.Z is None
    assert result.calculation_spec is None
    assert result.persistence_metadata == {
        'schema_version': 1, 'missing': ['Z', 'calculation_spec']}


def test_unknown_archive_schema_is_rejected(tmp_path):
    path = tmp_path / 'future.h5'
    _gapped_result().save(path)
    with h5py.File(path, 'a') as archive:
        archive.attrs['schema_version'] = 999
    with pytest.raises(ValueError, match='unsupported SCFResult schema'):
        SCFResult.load(path)


def test_structured_energy_provenance_roundtrip(tmp_path):
    result = _gapped_result()
    result.energy_components['_nuclear_profile'] = {
        'model': 'fermi', 'radius': np.float64(.001), 'optional': None}
    result.save(tmp_path/'structured')
    loaded = SCFResult.load(tmp_path/'structured')
    assert loaded.energy_components == result.energy_components
