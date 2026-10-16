"""Tiny-numerator amplitudes and honest residual bounds in the actual engine."""

import hashlib
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import pytest

from atomx.methods import fscc
from atomx.physics.csf.spinor_integrals import SpinorIntegrals

REPO = Path(__file__).resolve().parents[1]


def test_public_patch_reproduces_the_qualified_numerical_source(tmp_path):
    spec = importlib.util.spec_from_file_location(
        'numerator_patch', REPO / 'tools/expt_numerator_division_patch.py')
    patch = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(patch)
    reference = (REPO / 'docs/atomic_pci_sd_native_2026_10_09/'
                 'expt_compact_public_source001/src/rcc/engine/diveps.c')
    original = reference.read_bytes()
    target = tmp_path / 'src/rcc/engine/diveps.c'
    target.parent.mkdir(parents=True)
    target.write_bytes(original)
    difference = patch.patch_exact_nonzero_division(tmp_path)
    assert hashlib.sha256(target.read_bytes()).hexdigest() == patch.QUALIFIED_SHA256
    assert reference.read_bytes() == original
    assert difference.count('+        if (t == 0.0) {') >= 2
    patched = target.read_bytes()
    with pytest.raises(ValueError, match='unknown pinned'):
        patch.patch_exact_nonzero_division(tmp_path)
    assert target.read_bytes() == patched


@pytest.mark.parametrize('source,capability,marker', [
    ('0'*64, None, False),
    (fscc._LEGACY_DIVEPS_SOURCE_SHA256, 'ATOMX_EXACT_NONZERO_DIVEPS_001', True),
    (fscc._EXACT_NONZERO_DIVEPS_SOURCE_SHA256, None, True),
    (fscc._EXACT_NONZERO_DIVEPS_SOURCE_SHA256, 'ATOMX_EXACT_NONZERO_DIVEPS_001', False),
])
def test_unknown_or_mislabeled_division_contract_rejects_before_export(
        tmp_path, monkeypatch, source, capability, marker):
    from test_fscc_backend_physics_capability import request
    integrals, binary = request(tmp_path, 0.)
    if marker:
        binary.write_bytes(binary.read_bytes()+b'ATOMX_EXACT_NONZERO_DIVEPS_001')
    path = tmp_path / 'atomx_backend.json'
    manifest = json.loads(path.read_text())
    manifest['executable_sha256'] = hashlib.sha256(binary.read_bytes()).hexdigest()
    manifest['source_sha256']['src/rcc/engine/diveps.c'] = source
    manifest['numerator_division'] = capability
    path.write_text(json.dumps(manifest))

    def forbidden(*args, **kwargs):
        pytest.fail('unqualified residual convention reached export or execution')

    monkeypatch.setattr(fscc, 'export_integrals', forbidden)
    monkeypatch.setattr(fscc.subprocess, 'run', forbidden)
    with pytest.raises(ValueError, match='source-bound numerator-division'):
        fscc.compute(integrals, executable=binary, directory=tmp_path/'run')
    assert not (tmp_path/'run').exists()


@pytest.mark.parametrize('kind', ['exact', 'legacy'])
@pytest.mark.parametrize('numerator', [8e-15, 2e-14])
def test_live_analytic_amplitude_and_zero_update_residual_floor(tmp_path, kind, numerator):
    variable = ('ATOMX_EXPT_EXECUTABLE' if kind == 'exact'
                else 'ATOMX_EXPT_LEGACY_EXECUTABLE')
    executable = os.environ.get(variable)
    if not executable:
        pytest.skip(f'set {variable} for actual numerator-division replay')
    from test_fscc_equation_audit import audit
    h = np.array([[-.5, numerator], [numerator, -.499]])
    integrals = SpinorIntegrals(((1, -1, -1), (2, -1, -1)), 1, h, h,
        np.zeros(1), 0., -.5, dict(model='analytic one-electron two-spinor Hamiltonian'))
    directory = tmp_path/'run'
    result = fscc.compute(integrals, executable=executable, directory=directory,
        tolerance=1e-16, residual_bound_tolerance_Ha=1e-12, maxiter=30,
        memory_bytes=128*1024**2, tile_size=4, threads=1, timeout_seconds=30)
    tensor, _ = audit.decode_diagram(directory/'scratch/t1c.dg', nspinors=2)
    amplitude = tensor[0, 1].conjugate()
    _, exact = np.linalg.eigh(h)
    exact_amplitude = exact[1, 0]/exact[0, 0]
    wave = np.array([1., amplitude])
    wave /= np.linalg.norm(wave)
    energy = (wave.conj()@h@wave).real
    residual = float(np.max(abs(h@wave-energy*wave)))
    assert residual <= result.equation_residual_bound_Ha+1e-28
    assert result.amplitude_update_max == 0.
    if kind == 'exact':
        assert result.provenance['numerator_skip_threshold_Ha'] == 0.
        assert abs(amplitude-exact_amplitude) <= 1e-16
    else:
        assert result.provenance['numerator_skip_threshold_Ha'] == 1e-14
        assert result.equation_residual_bound_Ha > 0.
        if numerator < 1e-14:
            assert abs(amplitude-exact_amplitude) > 1e-12
            assert residual > 1e-15
