"""Replay saved physical DC/DCB and CCSD matrices using the final native source.

This verifies consumers and retained states, not a second radial assembly.
Use --output for an extracted-bundle replay without changing capture files.
"""
from dataclasses import asdict
import argparse
import hashlib
import json
from pathlib import Path
import resource
import sys

import numpy as np

from atomx.angular.csf_hamiltonian import csf_signature
from atomx.config.csf import get_shared_orbital_basis
from atomx.methods.ci_effective import FiniteCIModel, static_breit_second_order_control
from atomx.methods.coupled_cluster import compute
from atomx.physics.csf.effective_one_body import CIReferenceBinding

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE.parent / 'yb_diagnosis_repair_2026_10_08'))
from common import load_reference
from probe_yb import plans


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=HERE / 'final_source_replay.json')
    args = parser.parse_args()
    resource.setrlimit(resource.RLIMIT_AS, (2 * 1024**3, 2 * 1024**3))
    import atomx
    assert Path(atomx.__file__).resolve().is_relative_to((ROOT / 'src').resolve())
    sources = {str(f.relative_to(ROOT)): hashlib.sha256(f.read_bytes()).hexdigest()
               for f in sorted((ROOT / 'src/atomx').rglob('*.py'))}
    core, reference, _, identity = load_reference('v68_default')
    rows = {}
    for label, plan in plans(2).items():
        stem = label.replace('/', '_')
        record = json.loads((HERE / (stem + '_result.json')).read_text())
        assert record['reference']['orbitals_workspace_sha256'] == identity['orbitals_workspace_sha256']
        assert record['reference']['core_sha256'] == identity['core_sha256']
        csfs, p = plan.materialize()
        ids = tuple(csf_signature(c).key for c in csfs)
        # Exact complete-generation replay, including all coupling trees.
        captured = json.loads((HERE / 'preflight.json').read_text())['2:' + label]
        assert json.loads(json.dumps(ids)) == captured['CSF_identities']
        keys = get_shared_orbital_basis(csfs)
        binding = CIReferenceBinding.capture(
            [(*reference.orbitals_dict[key], key[1]) for key in keys], {},
            core._bsp_data, 70, core._bsp_data['ns'], core._nuclear_potential)
        models = {}
        row = {}
        with np.load(HERE / (stem + '_sd_breit.npz')) as saved:
            assert tuple(map(int, saved['p_indices'])) == p
            for terms, name in [(('dirac-coulomb',), 'DC'),
                                (('dirac-coulomb', 'static-breit'), 'DCB')]:
                model = FiniteCIModel(saved[name], p, reference_identity='Yb:final-source:' + label,
                                      csf_identities=ids, radial_binding=binding,
                                      hamiltonian_terms=terms)
                root = model.solve(record[name]['bracket_Ha'])
                en, vec = np.linalg.eigh(saved[name])
                nearest = int(np.argmin(abs(en - root.energy_Ha)))
                energy_error = abs(root.energy_Ha - record[name]['total_Ha'])
                anchor_error = abs(root.energy_Ha - en[nearest])
                projector = np.max(abs(np.outer(root.coefficients, root.coefficients)
                                       - np.outer(saved[name + '_vector'], saved[name + '_vector'])))
                assert energy_error < 1e-10 and anchor_error < 1e-10 and projector < 1e-8
                assert root.p_weight > .5 and root.residual_Ha < 1e-10
                models[name] = model
                row[name] = dict(energy_error_Ha=float(energy_error),
                                 dense_anchor_error_Ha=float(anchor_error),
                                 saved_projector_error=float(projector),
                                 residual_Ha=root.residual_Ha)
            control = static_breit_second_order_control(models['DC'], models['DCB'], saved['H0'])
            errors = {k: abs(v - record['restricted_H0_control'][k])
                      for k, v in asdict(control).items() if k != 'h0_diagonal_Ha'}
            assert max(errors.values()) < 1e-12
            row['restricted_control_errors_Ha'] = errors
        rows[label] = row
    cc_record = json.loads((HERE / 'native_ccsd_anchor.json').read_text())
    with np.load(HERE / 'native_ccsd_anchor.npz') as saved:
        cc = compute(saved['hamiltonian'], tuple(map(int, saved['determinant_masks'])),
                     int(saved['reference_mask']), spinor_count=len(saved['spinors']),
                     reference_identity=cc_record['provenance']['reference_identity'])
        cc_error = abs(cc.total_energy_Ha - cc_record['CC_energy_Ha'])
        cc_projector = np.max(abs(np.outer(cc.coefficients, cc.coefficients)
                                 - np.outer(saved['CC_vector'], saved['CC_vector'])))
        assert cc_error < 1e-10 and cc_projector < 1e-8
        assert cc.full_eigenstate_residual_Ha < 1e-10
        assert cc_record['sources'] == sources
    result = dict(passed=True, source_sha256=sources, imported_atomx=str(Path(atomx.__file__).resolve()),
                  five_sector_replay=rows, CCSD_energy_error_Ha=float(cc_error),
                  CCSD_projector_error=float(cc_projector),
                  CCSD_full_residual_Ha=cc.full_eigenstate_residual_Ha,
                  scope='Saved physical matrices and complete CSF generator replay; no new radial assembly')
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'source_sha256'}, indent=2))


if __name__ == '__main__':
    main()
