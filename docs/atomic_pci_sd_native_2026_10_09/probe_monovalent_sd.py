"""Reproducible finite native standalone SD feedback and physical-state panel."""
import hashlib
import itertools
import json
import os
import runpy
import shutil
from pathlib import Path

import numpy as np
from scipy.optimize import root

from atomx.angular.recoupling import twoj_of_kappa
from atomx.angular.wigner import wigner_3j
from atomx.methods.dirac_fock import DFSolver, Subshell
from atomx.methods.mbpt import build_virtual_basis
from atomx.methods.monovalent_sd import one_body_matrix_element
from atomx.numerics.fermion import apply_terms
from atomx.physics.csf.spinor_integrals import build_spinor_integrals
from atomx.properties.transitions.angular import angular_eL, selection_rules
from atomx.properties.transitions.multipole import radial_eL, radial_eL_velocity
from atomx.properties.transitions.rates import einstein_A_eL, oscillator_strength

OWNER = Path(__file__).resolve().parent
ROOT = OWNER.parents[1]
OUT = OWNER/'monovalent_sd006'


def main():
    OUT.mkdir(exist_ok=False)
    contract = json.loads((OWNER/'monovalent_sd004/contract.json').read_text())
    contract.update(panel='all M of 2s,3s,2p1/2,2p3/2 in the complete common twelve-spinor pool',
                    angular_gate=1e-9, physical_line_gate=1e-11,
                    lines='2p1/2,2p3/2 -> 2s; actual SD energy difference; length/velocity x finite/long wavelength',
                    property='normalized Eq6 SD-state bilinear; explicit lowest-M Wigner-Eckart components and shared native photon kernels; no gauge-agreement assertion',
                    load=os.getloadavg())
    (OUT/'contract.json').write_text(json.dumps(contract, indent=2)+'\n')
    hashes = {}
    for path in list((ROOT/'src/atomx').rglob('*.py'))+[ROOT/'tests/test_monovalent_sd.py',
                 ROOT/'tools/build_pci_sd_backend.py', Path(__file__).resolve()]:
        rel = path.relative_to(ROOT)
        target = OUT/'source'/rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        hashes[str(rel)] = hashlib.sha256(path.read_bytes()).hexdigest()
    (OUT/'source_sha256.json').write_text(json.dumps(hashes, indent=2)+'\n')
    helpers = runpy.run_path(str(ROOT/'tests/test_monovalent_sd.py'))

    class Factory:
        def mktemp(self, name):
            target = OUT/name
            target.mkdir()
            return target

    core, core_result = helpers['live_core'].__wrapped__(Factory())
    labels = core.integrals.labels
    n, o = len(labels), core.integrals.noccupied
    arrays = dict(core_amplitudes=core_result.amplitudes, core_source=core.source,
                  core_denominators=core.denominators, labels=np.array(labels),
                  one_body_Ha=core.integrals.one_body_Ha, fock_Ha=core.integrals.fock_Ha)
    (OUT/'core_identity.json').write_text(json.dumps(dict(core_result.provenance), indent=2)+'\n')
    g = np.empty((n,)*4)
    for indices in itertools.product(range(n), repeat=4):
        g[indices] = .5*core.integrals.eri(*indices).real
    arrays['direct_half_antisym_eri_Ha'] = g
    records, states = [], {}
    raising = []
    for p, (nr, k, m2) in enumerate(labels):
        destination = (nr, k, m2+2)
        if destination in labels:
            j, m = twoj_of_kappa(k)/2, m2/2
            raising.append((np.sqrt(j*(j+1)-m*(m+1)), ((labels.index(destination), True), (p, False))))
    lowering = tuple((value, ((operations[1][0], True), (operations[0][0], False)))
                     for value, operations in raising)
    for key in ((2, -1), (3, -1), (2, 1), (2, -2)):
        for m in range(-twoj_of_kappa(key[1]), twoj_of_kappa(key[1])+1, 2):
            problem = helpers['problem_for']((core, core_result), key, m)
            result = problem.solve()
            size = problem.plan['amplitudes']
            random = np.random.default_rng(41319).normal(size=size)*.013
            cs, cd, vs, vd = helpers['amplitude_tensors'](problem, random)
            rs, rd = helpers['paper_rhs'](g, cs, cd, vs, vd, problem.valence_index)
            diagram = [rs[problem.valence_index]+problem.orbital_energy_Ha]
            for particles, holes, _ in problem.excitations:
                diagram.append(rs[particles[0]] if len(particles) == 1 else rd[particles[0], particles[1], holes[1]])
            diagram = np.array(diagram)
            diagram[1:] += problem.diagonal_gaps_Ha*random
            f = core.integrals.fock_Ha.real
            off = f-np.diag(np.diag(f))
            fterms = tuple((off[p, q], ((p, True), (q, False)))
                           for p, q in itertools.product(range(n), repeat=2) if off[p, q] != 0)
            correction, _ = helpers['dense_linked'](problem, random, fterms)
            action = problem.linked_action(random)
            dense, _ = helpers['dense_linked'](problem, random)
            zero, _ = helpers['dense_linked'](problem, np.zeros(size))
            columns = []
            for i in range(size):
                unit = np.zeros(size)
                unit[i] = 1
                value, _ = helpers['dense_linked'](problem, unit)
                columns.append(value-zero)
            linear = np.column_stack(columns)
            fixed = np.linalg.solve(linear[1:], -zero[1:])

            def independent_residual(x):
                value = zero+linear@x[:-1]
                value[0] -= problem.orbital_energy_Ha+x[-1]
                value[1:] -= x[-1]*x[:-1]
                return value

            independent = root(independent_residual, np.r_[fixed, 0.])
            independent_raw = float(np.max(abs(independent_residual(independent.x))))
            _, (masks, h, tc, sv) = helpers['dense_linked'](problem, result.amplitudes)
            psi = (np.eye(len(masks))+tc+sv)[:, masks.index(problem.reference_mask)]
            psi /= np.linalg.norm(psi)
            state = dict(zip(result.determinants, result.coefficients, strict=True))
            reconstructed = np.array([state.get(mask, 0) for mask in masks])
            j, projection = twoj_of_kappa(key[1])/2, m/2
            ladder = apply_terms(lowering, apply_terms(tuple(raising), state))
            jdefect = max(abs(ladder.get(mask, 0)+(projection*(projection+1)-j*(j+1))*state.get(mask, 0))
                          for mask in ladder.keys() | state.keys())
            errors = dict(primary_equation_Ha=float(np.max(abs(action-diagram-correction))),
                          independent_action_Ha=float(np.max(abs(action-dense))),
                          independent_root_Ha=float(abs(independent.x[-1]-result.correlation_energy_Ha)),
                          independent_amplitudes=float(np.max(abs(independent.x[:-1]-result.amplitudes))),
                          independent_raw_Ha=independent_raw,
                          signed_state=float(np.max(abs(psi-reconstructed))),
                          projected_Ha=result.projected_residual_Ha, energy_Ha=result.energy_residual_Ha,
                          J2_state=jdefect)
            passed = (independent.success and errors['primary_equation_Ha'] < 1e-11
                      and errors['independent_action_Ha'] < 1e-11 and errors['independent_root_Ha'] < 1e-11
                      and errors['independent_amplitudes'] < 1e-10 and independent_raw < 1e-10
                      and errors['signed_state'] < 1e-11 and max(errors['projected_Ha'], errors['energy_Ha']) < 1e-10
                      and jdefect < contract['angular_gate'])
            record = dict(key=key, m2=m, plan=dict(problem.plan), errors=errors, passed=bool(passed),
                          total_energy_Ha=result.total_energy_Ha, attachment_energy_Ha=result.attachment_energy_Ha,
                          deltaE_Ha=result.correlation_energy_Ha, wavefunction_norm=result.wavefunction_norm,
                          full_eigenstate_residual_Ha=result.full_eigenstate_residual_Ha,
                          energy_expectation_Ha=result.expectation_energy_Ha,
                          feedback_amplitude_change=float(np.max(abs(result.amplitudes-fixed))),
                          expanded_core_residual_Ha=problem.expanded_core_residual_Ha,
                          continuation=result.continuation_path)
            records.append(record)
            tag = f'{key[0]}_{key[1]}_{m}'
            for name, value in dict(source=problem.source_Ha, linear=problem.linear_map_Ha,
                                   gaps=problem.diagonal_gaps_Ha, amplitudes=result.amplitudes,
                                   masks=result.determinants, coefficients=result.coefficients,
                                   random=random, primary_diagram=diagram+correction, action=action,
                                   dense_action=dense, independent_solution=independent.x,
                                   fixed_amplitudes=fixed, full_masks=masks, full_H=h,
                                   core_T=tc, valence_S=sv, full_state=psi).items():
                arrays[tag+'_'+name] = np.asarray(value)
            states[key, m] = result
            np.savez_compressed(OUT/'arrays.npz', **arrays)
            (OUT/'states.json').write_text(json.dumps(records, indent=2)+'\n')
            if not passed:
                raise AssertionError('complete monovalent signed equation/root/state/J gate failed')
    multiplet_errors = {str(key): max(row['attachment_energy_Ha'] for row in records if tuple(row['key']) == key)
                        - min(row['attachment_energy_Ha'] for row in records if tuple(row['key']) == key)
                        for key in ((2, -1), (3, -1), (2, 1), (2, -2))}
    if max(multiplet_errors.values()) > 1e-11:
        raise AssertionError('full magnetic multiplet energies disagree')
    # Reconstruct the explicit fixture radial input independently; certify
    # its native binding before any photon operator is consumed.
    df = DFSolver(Z=4, subshells=[Subshell(1, -1, 2)], ns=30, ks=7).solve()
    orbitals = {(1, -1): df.orbitals_by_subshell()[1, -1][:2]}
    for k in (-1, 1, -2):
        _, band = build_virtual_basis(df, k, e_max=50, potential_mode='VN', orthogonal_to_occupied=True)
        for i, item in enumerate(band[:2 if k == -1 else 1]):
            orbitals[i+2, k] = item[:2]
    rebound = build_spinor_integrals(orbitals, df._bsp_data, 4, 30, occupied_keys=((1, -1),),
                                    virtual_keys=core.plan['virtual_keys'], nuclear_potential=df._nuclear_potential,
                                    representation='on_demand')
    if rebound.identity['binding'] != core.integrals.identity['binding']:
        raise AssertionError('photon operator radial input binding differs')
    for key, (p, q) in orbitals.items():
        arrays[f'radial_{key[0]}_{key[1]}_P'] = p
        arrays[f'radial_{key[0]}_{key[1]}_Q'] = q
    lines = []
    lower = states[(2, -1), -1]
    for key in ((2, 1), (2, -2)):
        ju, jl = twoj_of_kappa(key[1]), 1
        upper = states[key, -ju]
        omega = upper.total_energy_Ha-lower.total_energy_Ha
        q2 = -ju+1
        factor = (-1)**ju*wigner_3j(ju, 2, jl, ju, q2, -1)
        for gauge, approximation in itertools.product(('length', 'velocity'), ('finite_wavelength', 'long_wavelength')):
            operator = np.zeros((n, n))
            radial = {}
            for p, (na, ka, ma) in enumerate(labels):
                for q, (nb, kb, mb) in enumerate(labels):
                    if ma != mb+q2 or not selection_rules(kb, ka, 1, 'E'):
                        continue
                    pair = ((na, ka), (nb, kb))
                    if pair not in radial:
                        pi, qi = orbitals[nb, kb]
                        pf, qf = orbitals[na, ka]
                        if gauge == 'velocity':
                            value = radial_eL_velocity(pi, qi, pf, qf, df._bsp_data, 1,
                                kappa_i=kb, kappa_f=ka, omega=omega, long_wavelength=approximation == 'long_wavelength')
                        else:
                            value = radial_eL(pi, qi, pf, qf, df._bsp_data, 1, kappa_i=kb, kappa_f=ka,
                                              omega=omega if approximation == 'finite_wavelength' else None)
                        radial[pair] = angular_eL(kb, ka, 1)*value
                    operator[p, q] = (-1)**((twoj_of_kappa(ka)-ma)//2)*wigner_3j(
                        twoj_of_kappa(ka), 2, twoj_of_kappa(kb), -ma, q2, mb)*radial[pair]
            value = one_body_matrix_element(upper, lower, operator)/factor
            signed = 0.
            bu = dict(zip(upper.determinants, upper.coefficients, strict=True))
            for mask, coeff in zip(lower.determinants, lower.coefficients, strict=True):
                for p, q in itertools.product(range(n), repeat=2):
                    target, sign = helpers['elementary'](mask, ((p, True), (q, False)))
                    if sign:
                        signed += bu.get(target, 0)*coeff*sign*operator[p, q]
            error = abs(value-signed/factor)
            if error > 1e-11:
                raise AssertionError('normalized SD photon operator consumption failed')
            strength = value**2
            f, gf = oscillator_strength(omega, strength, 1, jl, ju)
            lines.append(dict(upper_key=key, lower_key=(2, -1), omega_Ha=omega, upper_J2=ju, lower_J2=jl,
                              gauge=gauge, photon_approximation=approximation, signed_reduced_au=float(value),
                              line_strength=float(strength), A_per_s=float(einstein_A_eL(omega, strength, 1, ju)),
                              f=float(f), gf=float(gf), independent_error_au=float(error),
                              scope='finite SD state bilinear, not experimental/space or length-velocity convergence'))
            arrays[f'E1_{key[1]}_{gauge}_{approximation}'] = operator
    np.savez_compressed(OUT/'arrays.npz', **arrays)
    (OUT/'lines.json').write_text(json.dumps(lines, indent=2)+'\n')
    if any(hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != sha for name, sha in hashes.items()):
        raise AssertionError('actual dirty source changed during probe')
    summary = dict(status='pass-finite', states=len(records), lines=len(lines), arrays=len(arrays),
                   maximum_errors={key: max(row['errors'][key] for row in records) for key in records[0]['errors']},
                   magnetic_multiplet_energy_defects_Ha=multiplet_errors,
                   max_full_eigenstate_residual_Ha=max(row['full_eigenstate_residual_Ha'] for row in records),
                   max_line_consumption_error_au=max(row['independent_error_au'] for row in lines),
                   source_after_matches=True, production_hpc_achieved=False, experimental_accuracy_certified=False)
    (OUT/'comparison.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
