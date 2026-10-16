"""Final screened002 source-only stdlib audit; no consumer execution/import."""
import ast
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import time

HERE = Path(__file__).resolve().parent
TOP = HERE.parents[1]
REPO = TOP.parents[1]
BASE = TOP/'screened_core_virtual_discriminator002'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    start = time.monotonic()
    admission = json.loads((HERE/'contract.json').read_text())
    os.sched_setaffinity(0,admission['resources']['affinity'])
    cap = admission['resources']['AS_bytes']
    resource.setrlimit(resource.RLIMIT_AS,(cap,cap))
    signal.alarm(admission['resources']['wall_seconds'])
    checks = []

    def gate(name,value):
        checks.append(dict(name=name,passed=bool(value)))
        if not value:
            raise ValueError(name)

    for name,digest in admission['inputs_sha256'].items():
        gate('input:'+name,sha(REPO/name) == digest)
    c = json.loads((BASE/'contract.json').read_text())
    prep = json.loads((BASE/'preparation_verification.json').read_text())
    gate('preparation_metadata_passed',prep['all_preparation_checks_passed'] is True)
    fixed,caps,gates = c['fixed_conditions'],c['resources'],c['acceptance']
    source_root = REPO/fixed['frozen_source_root']
    source = json.loads((TOP/'public_mixed_resume_run002/source_identity.json').read_text())
    gate('original_source_epoch',source['source_map_sha256'] == fixed['source_map_sha256'] == admission['source_map_sha256'])
    gate('complete353source',len(source['all_atomx_python_sha256']) == fixed['full_atomx_source_count'] == 353)
    actual_names = {'src/'+str(p.relative_to(source_root)) for p in (source_root/'atomx').rglob('*.py')}
    gate('complete_archive_names',actual_names == set(source['all_atomx_python_sha256']))
    for name,digest in source['all_atomx_python_sha256'].items():
        gate('frozen353:'+name,sha(source_root/name.removeprefix('src/')) == digest)
    for name,digest in c['audited_critical_source_sha256'].items():
        gate('critical_source:'+name,source['all_atomx_python_sha256'][name] == digest)
    for role,row in c['saved_operator_roles'].items():
        gate('saved_operator_role:'+role,sha(REPO/row['path']) == row['sha256'])
    first = json.loads((REPO/c['saved_operator_roles']['qualified_first_saved_audit']['path']).read_text())
    gate('qualified_first116',first['all_independent_saved_array_checks_passed'] is True and first['first_kappa_complete_roots'] == 116)
    field = json.loads((REPO/c['saved_operator_roles']['screened_saved_audit']['path']).read_text())
    gate('prior_saved_field_scalar_anchor',field['all_independent_saved_evidence_checks_passed'] is True)
    gate('sameaccepted160',fixed['checkpoint']['manifest_sha256'] == '1ce49cd2bcc49bb22ed7bf21ec8f1e21fe3f37f945a2ef5ca3a3669e1a15c93c')
    gate('original_metric',fixed['original_metric_sha256'] == 'aae3a596009a671fd905f3c97f935cf17408a344a3d08cced5fbfb3ab2dcb6fe')
    gate('original_nonstationarity_preserved',fixed['checkpoint']['stationarity_gates_passed'] is False and
         gates['original_stationarity_residual'] == 1e-7 and gates['original_energy_indicator_Ha'] == 1e-8)
    gate('complete_remaining_sizes',c['numerical_method']['remaining_spectrum_sizes'] == [116,114,113,114])
    sizes = c['numerical_method']['remaining_spectrum_sizes']
    gate('complete_pair_budget',caps['maximum_attempted_pair_rotations'] == sum(n*(n-1)//2 for n in sizes)*12 == 310560)
    gate('complete_residual_budget',caps['maximum_saved_residual_evaluations'] == 4*13 == 52)
    for key in ('screened_complement_SVD_calls_budget','screened_original_P_full_gvd_spectral_controls','full_spectrum_refinement_attempts'):
        gate('four_remaining:'+key,caps[key] == 4)
    for key in ('screened_JK_builder_calls_budget','initial_one_body_matrix_kappa_count',
                'external_geometry_hS_build_calls','additional_virtual_control_hS_builds',
                'closed_core_one_body_calls_budget','scalar_radial_callback_calls_budget',
                'scalar_Slater_calls_budget','nuclear_Dirac_spectral_calls','orbital_gradients_budget','optimizer_calls_budget'):
        gate('zero_external_work:'+key,caps[key] == 0)
    gate('two_completeCI',caps['complete_CI_calls_budget'] == 2)
    gate('four_completeH',caps['total_many_electron_H_builds_budget'] == 4 and
         caps['implicit_public_CI_H_builds_budget'] == caps['explicit_archive_H_builds_budget'] == 2)
    gate('internal_hS_budget',caps['internal_CI_onebody_builds_budget'] == 28)
    gate('original_spectral_gates',gates['CI_residual_Ha'] == 1e-8 and gates['CI_orthogonality'] == 1e-10 and
         gates['screened_full_spectrum_constrained_residual_relative'] == 1e-8 and
         gates['screened_full_spectrum_SGram_error'] == 1e-10 and
         gates['screened_minimum_cluster_gap_Ha'] == 1e-10)
    run,helper = (BASE/'run.py').read_text(),(BASE/'refine_spectrum.py').read_text()
    runtime,method = ast.parse(run),ast.parse(helper)
    functions = {n.name:n for n in runtime.body if isinstance(n,ast.FunctionDef)}
    execute = ast.get_source_segment(run,functions['execute'])
    gate('authorization_before_physical_import',execute.index('authorize(binding_path)') < execute.index('import numpy as np'))
    for name in ('metadata','authorize'):
        segment = ast.get_source_segment(run,functions[name])
        gate('metadata_only:'+name,'import numpy' not in segment and 'import atomx' not in segment)
    for node in runtime.body:
        if isinstance(node,ast.Import):
            gate('top_level_stdlib:'+str(node.lineno),not any(a.name.startswith(('numpy','scipy','atomx')) for a in node.names))
        elif isinstance(node,ast.ImportFrom):
            gate('top_level_stdlib:'+str(node.lineno),not node.module.startswith(('numpy','scipy','atomx')))
    for node in method.body:
        if isinstance(node,(ast.Import,ast.ImportFrom)):
            gate('helper_stdlib_only:'+str(node.lineno),isinstance(node,ast.ImportFrom) or all(a.name == 'time' for a in node.names))
    for token in ('first["normalized_physical_vectors"]','first["full_coefficients"]','qualified_first_original_control',
                  'norms = np.sum(mapped * (metric.astype(np.longdouble) @ mapped), axis=0)',
                  'complement = null_space(frozen.T @ metric @ projector)',
                  'eigh(projected_h, projected_s, eigvals_only=False, driver="gvd")',
                  'refined = refine(np, h=h, s=metric, b=basis, initial=coefficients_initial',
                  'np.array_equal(first["original_P"], projector)',
                  'np.array_equal(first["frozen"], frozen)',
                  '"build_dirac_matrices", "internal_CI_onebody_attempts"',
                  'ci_solve(space, current, keys, data, 70, 123, solver="dense", n_states=None',
                  'include_breit=False, max_determinants=2000',
                  'h = build_ci_matrix(space, current, keys, data, 70, 123',
                  'tracker.resolve(space, ci, current, geometry.metric)',
                  'np.repeat(.2, 5)', 'original_stationarity_passed=False',
                  'expanded_stationarity_evaluated=False','full_spectrum_or_globalground_certified=False',
                  'original_nuclear_projector_bytes_preserved=True','output' if False else 'result.json'):
        gate('execution_invariant:'+token,token in run)
    forbidden_calls = []
    for node in ast.walk(runtime):
        if isinstance(node,ast.Call):
            name = node.func.id if isinstance(node.func,ast.Name) else (node.func.attr if isinstance(node.func,ast.Attribute) else '')
            if name in ('build_dirac_matrices','solve_dirac','DFSolver','_build_hf_matrix_dbsr','optimize_orbitals','closed_core_one_body','compute_slater_rk_cell'):
                forbidden_calls.append(name)
    gate('no_external_physical_builders',not forbidden_calls)
    for token in ('n = b.shape[1]','calls[\'complete_spectrum_refinement_attempts\'] += 1',
                  'defect / 2 + 3 * d2 / 8 - 5 * d3 / 16 + 35 * d4 / 128',
                  'for i in range(n - 1):','for j in range(i + 1, n):','if cross == 0:',
                  'np.copysign(np.longdouble(1), cross) if difference == 0',
                  'np.hypot(difference, cross)','vectors = np.asarray(b @ coeff, dtype=float)',
                  'vectors /= np.asarray(np.sqrt(norms), dtype=float)',
                  'constrained = b.T @ (hv - sv * rayleigh)',
                  'row["original_full_physical_residual_relative_max"] <= gates["screened_full_spectrum_constrained_residual_relative"]',
                  'row["full_Gram_error"] <= gates["screened_full_spectrum_SGram_error"]',
                  'np.argsort(final["actual_Rayleigh_Ha"], kind="stable")',
                  'save("failed_partial_iteration"'):
        gate('complete_method_invariant:'+token,token in helper)
    gate('attempt_recorded_before_numeric_work',helper.index("calls['complete_spectrum_refinement_attempts'] += 1") < helper.index('projected_h, projected_s ='))
    gate('no_subset_or_spectral_fallback','subset_by_index' not in run+helper and 'subset_by_value' not in run+helper)
    for name,digest in admission['inputs_sha256'].items():
        gate('unchanged_after:'+name,sha(REPO/name) == digest)
    report=dict(all_passed=all(x['passed'] for x in checks),passed_count=len(checks),checks=checks,
         source_map_sha256=fixed['source_map_sha256'],contract_sha256=sha(HERE/'contract.json'),
         inputs_sha256=admission['inputs_sha256'],verdict='No unresolved material source finding; bounded root-only execution may proceed.',
         corrected_draft_findings=['InheritedbuildJK/norefinement/all116prose corrected to savedfieldandremainingfullsizes.','Refinementattempt now recorded atentry.','Firstκ archive normalizationfactors now follow originalLDmap-to-float normalization without changing reusedphysicalvectors.'],
         reviewer_calls=dict(Atomx_imports=0,NumPy_imports=0,consumer_imports=0,H=0,CI=0,gradient=0,Dirac=0,optimizer=0),
         actual_resources=dict(affinity=sorted(os.sched_getaffinity(0)),AS_bytes=cap,elapsed_seconds=time.monotonic()-start,max_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
         live_consumer_run_by_reviewer=False,numerical_qualification_passed=None,original_stationarity_passed=False,
         scope='Source/AST/hash readiness only; all numericalspectral,CI,physicaltracking and budget outcomes require the authorized actual execution and separate saved-evidence audit.')
    (HERE/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(all_source_controls_passed=report['all_passed'],controls=len(checks),newscience=0,elapsed_seconds=report['actual_resources']['elapsed_seconds'])))


if __name__ == '__main__':
    main()
