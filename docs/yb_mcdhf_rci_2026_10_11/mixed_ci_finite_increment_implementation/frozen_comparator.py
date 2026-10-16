"""Fresh-process unchanged354 comparator: exactly5 ordinaryH,0CI,1gradient.

Imports are resolved only through the sealed old source002 PYTHONPATH. The
complete saved candidate spectrum is independently validated against ordinary
old H before supplying it as a gradient input; no new eigenproblem is solved.
"""
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

from atomx.config.csf import enumerate_jj_csfs
from atomx.methods.ci import CIResult
from atomx.methods.csf import variational
from atomx.methods.csf.variational import CSFOrbitalFunctional
from atomx.numerics.bspline.basis import prepare_bspline_data
from atomx.numerics.bspline.knots import make_knots
from atomx.physics.csf import matrix, tracking
from atomx.physics.csf.state_average import CIStateAverage
from atomx.physics.csf.tracking import CIBasisTracking


def main():
    output = Path(sys.argv[1]).resolve()
    expected = Path(os.environ['PYTHONPATH']).resolve()
    if expected not in Path(variational.__file__).resolve().parents:
        raise RuntimeError('frozen comparator did not import sealed original354')
    started = dict(H_build=0, CI_solve=0, gradient=0, one_body_assembly=0,
                   physical_overlap=0, tracking_SVD=0)
    completed = dict(started)
    limits = dict(H_build=5, CI_solve=0, gradient=1, one_body_assembly=6,
                  physical_overlap=2, tracking_SVD=2)
    def flush():
        (output/'frozen_calls.json').write_text(json.dumps(dict(started=started, completed=completed,
            limits=limits, source_prefix=str(expected)), indent=2)+'\n')
    def wrap(key, original):
        def call(*args, **kwargs):
            if started[key] >= limits[key]:
                raise RuntimeError(f'frozen {key} cap exceeded')
            started[key] += 1
            flush()
            result = original(*args, **kwargs)
            completed[key] += 1
            flush()
            return result
        return call
    matrix.build_ci_matrix = wrap('H_build', matrix.build_ci_matrix)
    variational.ci_solve = wrap('CI_solve', variational.ci_solve)
    CSFOrbitalFunctional.gradient_from_weights = wrap('gradient', CSFOrbitalFunctional.gradient_from_weights)
    ordinary = variational.build_dirac_matrices
    variational.build_dirac_matrices = wrap('one_body_assembly', ordinary)
    matrix.build_dirac_matrices = wrap('one_body_assembly', ordinary)
    tracking.ci_subspace_overlaps = wrap('physical_overlap', tracking.ci_subspace_overlaps)
    np.linalg.svd = wrap('tracking_SVD', np.linalg.svd)
    flush()
    try:
        ns, ks = 35, 9
        data = prepare_bspline_data(make_knots(ns, ks, .005, 30), ns, ks)
        configs = ('1s2', '1s1 2s1', '1s1 3s1', '2s2', '2s1 3s1', '3s2',
                   '2p2', '2p1 3p1', '3p2')
        csfs = [c for cfg in configs for c in enumerate_jj_csfs(cfg)]
        target = CIStateAverage(((0, 1, 0, 1), (0, 1, 1, 3)))
        def nuclear(radius):
            return -2/radius
        f = CSFOrbitalFunctional(csfs, data, 2, nuclear, state_average=target)
        inputs = np.load(output/'geometry_inputs.npz', allow_pickle=False)
        keys = tuple(map(tuple, inputs['orbital_keys'].tolist()))
        arrays = {}
        for index in (0, 1, 2, 7, 8):
            waves = {key: tuple(part.copy() for part in inputs[f'waves_{index}'][i])
                     for i, key in enumerate(keys)}
            arrays[f'H_{index}'] = matrix.build_ci_matrix(csfs, waves, keys, data, 2, ns,
                                                        nuclear_potential=nuclear)
        baseline = np.load(output/'CI_00.npz', allow_pickle=False)
        H, C, energies = arrays['H_0'], baseline['mixing'], baseline['energies']
        residual = float(np.max(np.linalg.norm(H@C-C*energies, axis=0)))
        gram = float(np.max(np.abs(C.T@C-np.eye(len(csfs)))))
        if not np.isfinite((residual, gram)).all() or residual > 1e-8 or gram > 1e-10:
            raise ValueError('saved complete CI does not validate against frozen ordinary H')
        ci = CIResult(energies.copy(), C.copy())
        waves = {key: tuple(part.copy() for part in inputs['waves_0'][i]) for i, key in enumerate(keys)}
        f.level_tracking = CIBasisTracking(target, csfs, ci, waves, f.metric)
        _, gradient, _ = f.evaluate(waves, ci_result=ci)
        arrays['baseline_gradient'] = np.array([gradient[key] for key in keys])
        np.savez_compressed(output/'frozen_comparators.npz', **arrays)
        (output/'frozen_comparator_validation.json').write_text(json.dumps(dict(
            source_variational_sha256=hashlib.sha256(Path(variational.__file__).read_bytes()).hexdigest(),
            source_matrix_sha256=hashlib.sha256(Path(matrix.__file__).read_bytes()).hexdigest(),
            complete_saved_CI_column_residual=residual, complete_saved_CI_Gram=gram,
            comparator_indices=[0, 1, 2, 7, 8]), indent=2)+'\n')
    finally:
        flush()


if __name__ == '__main__':
    main()
