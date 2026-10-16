"""Matched native DC / tensor CC / complete determinant CI anchors."""
import itertools
import json
import os
from pathlib import Path
import time
import numpy as np
from atomx.methods.dirac_fock import DFSolver, Subshell
from atomx.methods.mbpt import build_virtual_basis
from atomx.physics.csf.spinor_integrals import build_spinor_integrals
from atomx.physics.csf.sparse_ci import assemble_sparse_hamiltonian
from atomx.methods.fscc import compute

HERE = Path(__file__).resolve().parent
BASE = Path('/home/zzhang/code/atomx-references/2026-10-09-ci-fscc')
EXE = Path(os.environ.get('ATOMX_EXPT_EXECUTABLE', BASE/'EXP-T-atomx-build/expt.x'))
LIBS = BASE/'toolchain/usr/lib/x86_64-linux-gnu'
ENV = dict(os.environ)
ENV['LD_LIBRARY_PATH'] = f'{LIBS}:{LIBS}/openblas-serial:'+ENV.get('LD_LIBRARY_PATH', '')
ENV['OPENBLAS_NUM_THREADS'] = '1'


def main():
    start = time.monotonic()
    core = DFSolver(Z=4, subshells=[Subshell(1, -1, 2)], ns=30, ks=7).solve()
    od = {(1, -1): core.orbitals_by_subshell()[1, -1][:2]}
    for k in (-1, 1, -2):
        _, v = build_virtual_basis(core, k, e_max=50, potential_mode='VN', orthogonal_to_occupied=True)
        od[2, k] = v[0][:2]
    a = build_spinor_integrals(od, core._bsp_data, 4, 30, occupied_keys=[(1, -1)],
        virtual_keys=[(2, -1), (2, 1), (2, -2)], nuclear_potential=core._nuclear_potential)
    n = len(a.labels)
    energies = {}
    captured = {}
    for electrons in (2, 3):
        masks = [sum(1 << i for i in c) for c in itertools.combinations(range(n), electrons)]
        raw = assemble_sparse_hamiltonian(masks, a.labels, od, core._bsp_data, 4, 30).toarray()
        assert np.max(abs(raw-raw.T)) < 1e-12
        h = .5*(raw+raw.T)
        # Independent second-quantized contraction of the exported integrals.
        from atomx.angular.csf_hamiltonian import _ops_sign
        oracle = np.zeros_like(h, dtype=complex)
        indices = {m: i for i, m in enumerate(masks)}
        for j, mask in enumerate(masks):
            for p in range(n):
                for q in range(n):
                    sign = _ops_sign(mask, ((p, True), (q, False)))
                    final = (mask ^ (1 << q)) | (1 << p)
                    if sign and final in indices:
                        oracle[indices[final], j] += sign*a.one_body_Ha[p, q]
            for p in range(n):
                for q in range(p+1, n):
                    for r in range(n):
                        for s in range(r+1, n):
                            sign = _ops_sign(mask, ((p, True), (q, True), (s, False), (r, False)))
                            final = (mask ^ (1 << r) ^ (1 << s)) | (1 << p) | (1 << q)
                            if sign and final in indices:
                                oracle[indices[final], j] += sign*a.eri(p, q, r, s)
        integral_error = float(np.max(abs(oracle-h)))
        if integral_error > 1e-10:
            raise RuntimeError(f'exported integral phase/orientation mismatch: {integral_error}')
        e, c = np.linalg.eigh(h)
        pweights = np.sum(abs(c[np.asarray([m & 3 == 3 for m in masks])])**2, axis=0)
        energies[electrons] = e
        captured[f'CI_{electrons}_matrix'] = h
        captured[f'CI_{electrons}_energies'] = e
        captured[f'CI_{electrons}_Pweights'] = pweights
        print('independent integral anchor', electrons, integral_error, flush=True)
    records = []
    for model, tile in [('CCSD', 4), ('CCSDT', 4), ('CCSDT', 32)]:
        r = compute(a, executable=EXE, directory=HERE/f'be_{model.lower()}_tile{tile}_final', model=model,
            sector='0h1p', active_indices=range(2, n), tile_size=tile, tolerance=1e-12,
            residual_bound_tolerance_Ha=1e-9, environment=ENV)
        err2 = abs(r.core_energy_Ha-energies[2][0])
        err3 = float(np.max(abs(np.asarray(r.sector_energies_Ha)-energies[3][:8])))
        record = dict(model=model, tile_size=tile, core_energy_Ha=r.core_energy_Ha,
            energies_Ha=r.sector_energies_Ha, core_FCI_error_Ha=err2, sector_FCI_error_Ha=err3,
            amplitude_update=r.amplitude_update_max, equation_residual_bound_Ha=r.equation_residual_bound_Ha)
        records.append(record)
        print(json.dumps(record), flush=True)
        if err2 > 1e-9 or (model == 'CCSDT' and err3 > 1e-8):
            raise RuntimeError('CC/FCI gate failed')
    np.savez_compressed(HERE/'native_anchors.npz', fock=a.fock_Ha, one_body=a.one_body_Ha,
                        packed_eri=a.packed_eri_Ha, labels=a.labels, **captured)
    (HERE/'native_anchors.json').write_text(json.dumps(dict(records=records,
        integral_identity=dict(a.identity), seconds=time.monotonic()-start, passed=True), indent=2)+'\n')


if __name__ == '__main__':
    main()
