"""Known incorrect attachment equations must fail before exporting or solving."""

import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from atomx.methods import fscc


def request(tmp_path, fov, capability=None, source_sha=None):
    binary = tmp_path / 'expt.x'
    binary.write_bytes(b'ATOMX_INTEGRALS_V1')
    manifest = dict(abi='ATXCC001', executable_sha256=hashlib.sha256(binary.read_bytes()).hexdigest())
    if capability is not None:
        manifest['noncanonical_attachment_equation'] = capability
    manifest['source_sha256'] = dict(fscc._ATTACHMENT_HEFF_SOURCE_SHA256)
    manifest['source_sha256']['src/rcc/engine/diveps.c'] = fscc._LEGACY_DIVEPS_SOURCE_SHA256
    if source_sha is not None:
        manifest['source_sha256']['src/rcc/models/sector01.c'] = source_sha
    (tmp_path / 'atomx_backend.json').write_text(json.dumps(manifest))
    fock = np.diag([-1., -1., .2, .2]).astype(complex)
    fock[0, 2], fock[2, 0] = fov, np.conj(fov)
    integrals = SimpleNamespace(labels=((1, -1, -1), (1, -1, 1), (2, -1, -1), (2, -1, 1)),
                                noccupied=2, fock_Ha=fock, dense_eri_bytes=336)
    return integrals, binary


@pytest.mark.parametrize('fov', [1e-300, .05j])
@pytest.mark.parametrize('capability,source', [
    (None, None),
    ('pinned original equations', None),
    ('ATOMX_FOV_T2_001', None),
    ('ATOMX_FOV_T2_001', '0' * 64),
])
def test_noncanonical_original_missing_or_mislabeled_source_rejects_before_work(
        tmp_path, monkeypatch, fov, capability, source):
    integrals, binary = request(tmp_path, fov, capability, source)
    def forbidden(*args, **kwargs):
        pytest.fail('known incorrect equations reached export or backend launch')
    monkeypatch.setattr(fscc, 'export_integrals', forbidden)
    monkeypatch.setattr(fscc.subprocess, 'run', forbidden)
    output = tmp_path / 'run'
    with pytest.raises(ValueError, match='source-bound ATOMX_FOV_T2_001'):
        fscc.compute(integrals, executable=binary, directory=output,
                     model='CCSDT', sector='0h1p', active_indices=(2, 3))
    assert not output.exists()
    assert integrals.fock_Ha[0, 2] == fov


@pytest.mark.parametrize('sector,fov,corrected', [
    ('0h1p', .05j, True), ('0h1p', 0., False),
    ('0h0p', .05j, False), ('1h0p', .05j, False),
])
def test_verified_attachment_or_other_sector_keeps_its_export_path(
        tmp_path, monkeypatch, sector, fov, corrected):
    integrals, binary = request(tmp_path, fov,
        'ATOMX_FOV_T2_001' if corrected else None,
        fscc._NONCANONICAL_01_SOURCE_SHA256 if corrected else None)
    class ExportReached(Exception):
        pass
    def sentinel(*args, **kwargs):
        raise ExportReached
    monkeypatch.setattr(fscc, 'export_integrals', sentinel)
    active = (2, 3) if sector == '0h1p' else ((0, 1) if sector == '1h0p' else ())
    with pytest.raises(ExportReached):
        fscc.compute(integrals, executable=binary, directory=tmp_path / 'run',
                     sector=sector, active_indices=active)


@pytest.mark.parametrize('cap', [2*1024**3, 4*1024**3])
def test_complete_export_cap_is_explicit_and_reaches_the_same_verified_path(tmp_path, monkeypatch, cap):
    integrals, binary = request(tmp_path, 0.)
    class ExportReached(Exception):
        pass
    def sentinel(*args, **kwargs):
        assert kwargs['max_integral_file_bytes'] == cap
        raise ExportReached
    monkeypatch.setattr(fscc, 'export_integrals', sentinel)
    with pytest.raises(ExportReached):
        fscc.compute(integrals, executable=binary, directory=tmp_path/'run',
                     sector='0h0p', max_integral_file_bytes=cap)


@pytest.mark.parametrize('cap', [True, 0, -1, 1.5])
def test_invalid_export_cap_rejects_before_backend_or_export(tmp_path, cap):
    integrals, binary = request(tmp_path, 0.)
    with pytest.raises(ValueError, match='integral export byte cap'):
        fscc.compute(integrals, executable=binary, directory=tmp_path/'run',
                     max_integral_file_bytes=cap)
    assert not (tmp_path/'run').exists()


@pytest.mark.parametrize('header', [None, '0'*64,
    '616ba79360f52c3a74a332e4df0bf9a1895bc88f0b1af387e4ce627f36a6c2e8'])
def test_expanded_axial_capacity_requires_actual_source_allocation_before_export(
        tmp_path, monkeypatch, header):
    integrals, binary = request(tmp_path, 0.)
    binary.write_bytes(binary.read_bytes()+b'ATOMX_AXIAL_V1')
    path = tmp_path/'atomx_backend.json'
    manifest = json.loads(path.read_text())
    manifest.update(executable_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
                    tensor_symmetries=['C1', 'axial'], axial_max_omega_x2=126)
    if header is not None:
        manifest['source_sha256']['src/rcc/include/comdef.h'] = header
    path.write_text(json.dumps(manifest))
    def forbidden(*args, **kwargs):
        pytest.fail('unbound irrep allocation reached integral export')
    monkeypatch.setattr(fscc, 'export_integrals', forbidden)
    with pytest.raises(ValueError, match='source-bound512'):
        fscc.compute(integrals, executable=binary, directory=tmp_path/'run',
                     sector='0h0p', tensor_symmetry='axial')
    assert not (tmp_path/'run').exists()
