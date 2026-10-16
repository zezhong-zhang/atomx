"""Stdlib-only inspection of existing source and stored JSON metadata.

This does not import Atomx/NumPy, load physical arrays, run a physical
builder, apply the proposed patch, or make a performance measurement.
"""

import ast
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    contract = json.loads((HERE / 'contract.json').read_text())
    checks = []

    def check(name, passed):
        checks.append({'name': name, 'passed': bool(passed)})
        if not passed:
            raise ValueError('Source/metadata check failed: ' + name)

    for name, expected in contract['inputs_sha256'].items():
        check('input_' + name, sha(REPO / name) == expected)
    identity = json.loads((REPO / contract['source_identity']).read_text())
    check('frozen_356_map', identity['source_map_sha256'] == contract['source_map_sha256'])
    archive = REPO / identity['source_archive_directory']
    for name, expected in identity['all_atomx_python_sha256'].items():
        check('current_' + name, sha(REPO / name) == expected)
        check('archive_' + name, sha(archive / name.removeprefix('src/')) == expected)
    check('356_files', len(identity['all_atomx_python_sha256']) == 356)

    matrix = (REPO / 'src/atomx/physics/csf/matrix.py').read_text()
    functional = (REPO / 'src/atomx/methods/csf/variational.py').read_text()
    convol = (REPO / 'src/atomx/numerics/bspline/convol.py').read_text()
    nodes = {node.name: node for node in ast.parse(matrix).body
             if isinstance(node, ast.FunctionDef)}
    producer = ast.get_source_segment(matrix, nodes['_bspline_pair_potential'])
    evaluator = ast.get_source_segment(matrix, nodes['_radial_evaluator'])
    check('CI_still_dense_bank', '_convol_case1_t(' in producer and
          '_convol_case1_semiseparable_t' not in producer)
    check('existing_bounded_density_cache', '@lru_cache(maxsize=512)' in evaluator)
    check('existing_bounded_pair_cache', '@lru_cache(maxsize=pair_capacity)' in evaluator)
    check('existing_bounded_radial_cache', '@lru_cache(maxsize=8192)' in evaluator)
    check('existing_pair_role_order', 'source = tuple(sorted((key_b, key_d)))' in evaluator)
    check('gradient_existing_factored_branch', 'ns >= SEMISEPARABLE_DIRECT_NS' in functional)
    check('gradient_existing_action_reuse', 'actions[right] = pp @ pr, qq @ qr' in functional)
    check('fixed_threshold_96', 'SEMISEPARABLE_DIRECT_NS = 96' in convol)
    check('exact_overlap_plus_prefix_suffix', all(text in convol for text in
          ('def _convol_case1_semiseparable_t', 'prefix[jp, before]',
           'suffix[jp, after]', 'near[ip, jp, i, j - first_near]')))

    proposal = (HERE / 'proposed_matrix.patch').read_text()
    check('proposal_bound', sha(HERE / 'proposed_matrix.patch') == contract['proposal_sha256'])
    check('proposal_single_shared_file', proposal.count('--- a/src/') == 1 and
          '--- a/src/atomx/physics/csf/matrix.py' in proposal)
    check('no_proposal_screening', all(text not in proposal.lower() for text in
          ('coefficient_cutoff', 'fastmath', 'gradient_keys', 'n_states')))

    progress = json.loads((REPO / contract['original_progress']).read_text())
    rotation = json.loads((REPO / contract['rotation_progress']).read_text())
    density = json.loads((REPO / contract['signed_density']).read_text())
    statistics = {}
    for label in ('unfiltered', 'actual_gradient_retained'):
        rows = density[label]
        sources = defaultdict(set)
        terms = Counter()
        for term, _coefficient in rows:
            terms[term[0]] += 1
            if term[0] != 'h':
                _, rank, a, b, c, d = term
                a, b, c, d = map(tuple, (a, b, c, d))
                for left, right, source_a, source_b in ((a, b, c, d), (c, d, a, b)):
                    source = (rank, *sorted((source_a, source_b)))
                    sources[source].update((left, right))
        statistics[label] = dict(
            saved_term_count=len(rows), negative_terms=sum(v < 0 for _, v in rows),
            one_body_terms=terms['h'], two_body_terms=terms['r'],
            unique_gradient_source_pairs=len(sources),
            implied_gradient_component_contractions=4 * len(sources),
            implied_reused_two_component_actions=2 * sum(map(len, sources.values())),
            provenance='Stored signed JSON keys only; no new RDM or radial evaluation. '
                       'Not a measurement of CI source cache misses/evictions.')
    check('saved_unfiltered_12403', statistics['unfiltered']['saved_term_count'] == 12403)
    check('saved_actual_retained_10738', statistics['actual_gradient_retained']['saved_term_count'] == 10738)
    check('original_4H4CI1g', {key: progress['completed'][key] for key in ('H', 'CI', 'gradient')}
          == {'H': 4, 'CI': 4, 'gradient': 1})
    check('rotation_3H3CI0g', {key: rotation['completed'][key] for key in ('H', 'CI', 'gradient')}
          == {'H': 3, 'CI': 3, 'gradient': 0})
    check('original_failure_preserved', progress['phase'] == 'direction0_reported_derivative')
    check('rotation_failure_preserved', rotation['phase'] == 'direction1_NEW_Rayleigh_derivative')
    outputs = dict(
        status='source-only proposal; no physical or timing qualification',
        all_passed=all(c['passed'] for c in checks), checks_count=len(checks), checks=checks,
        source_map_sha256=contract['source_map_sha256'],
        inputs_sha256=contract['inputs_sha256'], contract_sha256=sha(HERE / 'contract.json'),
        verifier_sha256=sha(Path(__file__)), proposal_sha256=sha(HERE / 'proposed_matrix.patch'),
        original_actual_completed=progress['completed'], rotation_actual_completed=rotation['completed'],
        saved_density_key_statistics=statistics,
        physical_and_performance_calls=dict(H=0, CI=0, gradient=0, radial=0,
                                            eigensolve=0, optimizer=0, timing_measurement=0),
        actual29_dominant_stage=None, speedup_claim=None,
        physical_gradient_qualification=False, optimizer_execution_authorized=False)
    (HERE / 'review.json').write_text(json.dumps(outputs, indent=2) + '\n')
    print(json.dumps({'all_passed': outputs['all_passed'], 'checks_count': len(checks),
                      'saved_density_key_statistics': statistics}))


if __name__ == '__main__':
    main()
