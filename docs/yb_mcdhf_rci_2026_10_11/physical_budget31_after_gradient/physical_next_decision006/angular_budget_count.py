"""Integer-only sparse-work census; no Atomx/NumPy/CSF or determinant arrays.

Count joint magnetic occupations with a four-letter local generating function:
empty, bra only, ket only, common. The number of unequal modes is 0, 2 or 4
for a Slater-Condon connection. Distinct occupation patterns have disjoint
determinant sets, so their joint counts can be added without duplicate pairs.
"""

import hashlib
import itertools
import json
import resource
import signal
import sys
import time
from collections import Counter
from functools import lru_cache
from math import comb
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
OLD = HERE.parent / "physical_next_decision005/missing_4f12_budget.json"
MAP = REPO / "docs/yb_mcdhf_rci_2026_10_11/eu_yb_comparison001/core_core_discriminator_preparation001/execution_preparation002/actual/complete_basis_and_configuration_maps.json"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@lru_cache(None)
def magnetic(tj, q):
    work = {(0, 0): 1}
    for m in range(-tj, tj + 1, 2):
        nxt = dict(work)
        for (n, total), count in work.items():
            if n < q:
                key = n + 1, total + m
                nxt[key] = nxt.get(key, 0) + count
        work = nxt
    return {m: n for (q0, m), n in work.items() if q0 == q}


@lru_cache(None)
def pattern_m(pattern):
    out = {0: 1}
    for _, k, q in pattern:
        nxt = Counter()
        for m, number in out.items():
            for local, count in magnetic(2 * abs(k) - 1, q).items():
                nxt[m + local] += number * count
        out = nxt
    return dict(out)


def missing_patterns(rows):
    core = [(1, -1, 2), (2, -1, 2), (2, 1, 2), (2, -2, 4),
            (3, -1, 2), (3, 1, 2), (3, -2, 4), (3, 2, 4), (3, -3, 6),
            (4, -1, 2), (4, 1, 2), (4, -2, 4), (4, 2, 4), (4, -3, 6),
            (5, -1, 2), (5, 1, 2), (5, -2, 4)]
    out = set()
    for row in rows:
        shells = [(4, "f", 12)] + row["valence_occupation"]
        choices = []
        for n, letter, q in shells:
            ell = "spdfghik".index(letter)
            if ell == 0:
                choices.append((((n, -1, q),),))
            else:
                choices.append(tuple(tuple(x for x in
                    ((n, ell, lo), (n, -(ell + 1), q - lo)) if x[2])
                    for lo in range(max(0, q - 2 * ell - 2), min(q, 2 * ell) + 1)))
        for split in itertools.product(*choices):
            out.add(tuple(sorted(core + [item for group in split for item in group])))
    return out


@lru_cache(None)
def joint_shell(tj, qa, qb):
    """Joint two-subset M counts with at most four unequal magnetic modes."""
    width = tj + 1
    if qa + qb > width:
        # Complement both sets: same unequal modes, both M signs reverse.
        return tuple((e, -ma, -mb, count)
                     for e, ma, mb, count in joint_shell(tj, width - qa, width - qb))
    states = {(0, 0, 0, 0, 0): 1}
    for index, m in enumerate(range(-tj, tj + 1, 2)):
        remaining = width - index - 1
        nxt = Counter()
        for (a, b, e, ma, mb), count in states.items():
            for da, db in ((0, 0), (1, 0), (0, 1), (1, 1)):
                aa, bb, ee = a + da, b + db, e + (da != db)
                if aa > qa or bb > qb or aa + remaining < qa or bb + remaining < qb or ee > 4:
                    continue
                nxt[aa, bb, ee, ma + da * m, mb + db * m] += count
        states = nxt
    return tuple((e, ma, mb, count) for (a, b, e, ma, mb), count in states.items()
                 if a == qa and b == qb)


@lru_cache(None)
def joint_pattern(descriptor):
    counts = {(0, 0, 0): 1}
    for tj, qa, qb in descriptor:
        nxt = Counter()
        for (e, ma, mb), number in counts.items():
            for de, a, b, factor in joint_shell(tj, qa, qb):
                if e + de <= 4:
                    nxt[e + de, ma + a, mb + b] += number * factor
        counts = nxt
    return {(e, j2, j2): counts.get((e, j2, j2), 0)
            for e in (0, 2, 4) for j2 in (1, 3, 5)}


def descriptor(pa, pb):
    a = {(n, k): q for n, k, q in pa}
    b = {(n, k): q for n, k, q in pb}
    keys = a.keys() | b.keys()
    if sum(abs(a.get(key, 0) - b.get(key, 0)) for key in keys) > 4:
        return None
    local = []
    for n, k in keys:
        qa, qb = a.get((n, k), 0), b.get((n, k), 0)
        width = 2 * abs(k)
        if qa == qb and qa in (0, width):
            continue
        local.append((width - 1, qa, qb))
    return tuple(sorted(local))


def local_validation():
    """Independent small local-subset checks, never a full atomic determinant."""
    checks = 0
    for tj in (1, 3, 5, 7):
        modes = tuple(range(-tj, tj + 1, 2))
        for qa in range(tj + 2):
            for qb in range(tj + 2):
                expected = Counter()
                for aa in itertools.combinations(range(tj + 1), qa):
                    for bb in itertools.combinations(range(tj + 1), qb):
                        unequal = len(set(aa) ^ set(bb))
                        if unequal <= 4:
                            expected[unequal, sum(modes[i] for i in aa), sum(modes[i] for i in bb)] += 1
                actual = {(e, ma, mb): count for e, ma, mb, count in joint_shell(tj, qa, qb)}
                assert actual == dict(expected), (tj, qa, qb)
                checks += 1
    return checks


def block_budget(patterns, j2):
    patterns = tuple(p for p in patterns if pattern_m(p).get(j2, 0) - pattern_m(p).get(j2 + 2, 0) > 0)
    dims = [pattern_m(p).get(j2, 0) for p in patterns]
    csfs = [pattern_m(p).get(j2, 0) - pattern_m(p).get(j2 + 2, 0) for p in patterns]
    projection_bound = sum(d * c for d, c in zip(dims, csfs, strict=True))
    by_kind = Counter()
    descriptions = Counter()
    one_body_singles = Counter()
    frozen = set((n, k) for n, k, q in patterns[0] if q == 2 * abs(k))
    for p in patterns:
        frozen &= {(n, k) for n, k, q in p if q == 2 * abs(k)}
    frozen_n = sum(2 * abs(k) for n, k in frozen)
    active_n = 69 - frozen_n
    for ia, pa in enumerate(patterns):
        for ib in range(ia, len(patterns)):
            desc = descriptor(pa, patterns[ib])
            if desc is not None:
                multiplicity = 1 if ia == ib else 2
                descriptions[desc] += multiplicity
                da = {(n, k): q for n, k, q in pa}
                db = {(n, k): q for n, k, q in patterns[ib]}
                differences = [(key, da.get(key, 0) - db.get(key, 0)) for key in da.keys() | db.keys()
                               if da.get(key, 0) != db.get(key, 0)]
                if (len(differences) == 2 and sorted(q for key, q in differences) == [-1, 1]
                        and differences[0][0][1] == differences[1][0][1]):
                    one_body_singles[desc] += multiplicity
    print("block", j2, len(patterns), sum(dims), sum(csfs), "joint descriptors", len(descriptions), flush=True)
    for desc, multiplicity in descriptions.items():
        joint = joint_pattern(desc)
        for e in (0, 2, 4):
            by_kind[e] += multiplicity * joint.get((e, j2, j2), 0)
    assert by_kind[0] == sum(dims)
    ordered = sum(by_kind.values())
    diagonal = by_kind[0]
    triangular = (ordered + diagonal) // 2
    diagonal_h_terms = sum(d * len(p) for p, d in zip(patterns, dims, strict=True))
    onebody_pair_entries = sum(multiplicity * joint_pattern(desc).get((2, j2, j2), 0)
                               for desc, multiplicity in one_body_singles.items())
    # At least these one-body entries survive: diagonal occupations are positive,
    # and same-kappa magnetic-preserving radial singles carry a single +/-1 h key.
    h_nnz_lower_bound = diagonal_h_terms + onebody_pair_entries
    max_rank_terms = 8  # largest retained 2j is 7; kmax<=7 in two_body_angular.
    diag_terms_bound = 69 + 2 * comb(69, 2) * max_rank_terms
    single_terms_bound = 1 + 2 * 68 * max_rank_terms
    double_terms_bound = 2 * max_rank_terms
    table_nnz_bound = (diagonal * diag_terms_bound + by_kind[2] * single_terms_bound
                       + by_kind[4] * double_terms_bound)
    triangular_table_nnz_bound = (diagonal * diag_terms_bound + by_kind[2] // 2 * single_terms_bound
                                  + by_kind[4] // 2 * double_terms_bound)
    # Lower bound for distinct objects simultaneously alive in current
    # _connected_pairs return list-comprehension. Container headers/capacity,
    # spectator dicts/lists, and allocator overhead are deliberately excluded.
    object_bytes_per_pair = sys.getsizeof((0, 0)) + 2 * (sys.getsizeof([None]) - sys.getsizeof([])) + sys.getsizeof(1)
    # Every active determinant contributes choose(active N,2) spectator memberships.
    memberships = sum(dims) * comb(active_n, min(2, active_n))
    # These are structural candidates, before angular-key cancellation.
    return {
        "jj_patterns": len(patterns), "raw_M_determinants": sum(dims), "CSFs": sum(csfs),
        "frozen_spinors": frozen_n, "active_electrons_for_connected_pairs": active_n,
        "spectator_bucket_memberships_exact": memberships,
        "joint_pattern_descriptors": len(descriptions),
        "connected_ordered_pairs_by_spinor_replacements_exact": {str(e // 2): by_kind[e] for e in (0, 2, 4)},
        "connected_ordered_pairs_exact": ordered,
        "connected_triangular_pairs_exact": triangular,
        "spectator_bucket_ordered_pair_visits_before_dedup_exact": sum(by_kind[2 * replacements] * comb(active_n - replacements, active_n - 2) for replacements in (0, 1, 2)),
        "spectator_bucket_triangular_pair_visits_before_dedup_exact": (diagonal * comb(active_n, 2) + by_kind[2] // 2 * (active_n - 1) + by_kind[4] // 2),
        "one_body_table_nnz_ordered_exact_lower_bound": h_nnz_lower_bound,
        "coefficient_table_ordered_nnz_conservative_upper_bound": table_nnz_bound,
        "coefficient_table_triangular_nnz_conservative_upper_bound": triangular_table_nnz_bound,
        "coefficient_table_upper_bound_reason": "Before collection, each diagonal has at most69+16*C(69,2) terms, each single at most1+16*68, and each double at most16; canonical-key collection only reduces these. Bounds are loose, not measured nnz.",
        "connected_pairs_return_live_objects_lower_bound_bytes": ordered * object_bytes_per_pair - min(ordered, 257) * sys.getsizeof(1),
        "connected_pairs_return_live_objects_lower_bound_scope": "One min-size encoded Python int in the pair set, decoded2-tuple, one slot in sorted(pairs), and one output-list slot per structural connection. Excludes set table, all container headers/over-allocation, integer-size growth, spectator buckets and allocator overhead. Current source lifetime, CPython object sizes only; not overall peak prediction.",
        "spectator_membership_pointer_payload_lower_bound_bytes": memberships * (sys.getsizeof([None]) - sys.getsizeof([])),
        "projection_nnz_upper_bound_sum_pattern_D_times_C": projection_bound,
        "projection_CSR_FP64_int32_upper_bound_bytes": 12 * projection_bound + 4 * (sum(dims) + 1),
        "largest_dense_per_pattern_projection_FP64_bytes": max(8 * d * c for d, c in zip(dims, csfs, strict=True)),
        "largest_single_pattern_raw_M": max(dims),
        "Hdet_CSR_FP64_int32_structural_upper_bound_bytes": 12 * ordered + 4 * (sum(dims) + 1),
        "pair_coordinates_int64_ordered_payload_upper_bound_bytes": 16 * ordered,
        "pair_coordinates_int64_triangular_payload_upper_bound_bytes": 16 * triangular,
        "density_five_root_products_FP64_ordered_upper_bound_bytes": 8 * 5 * ordered,
    }


def main():
    resource.setrlimit(resource.RLIMIT_AS, (2 * 1024**3, 2 * 1024**3))
    signal.alarm(180)
    start = time.monotonic()
    validation_cases = local_validation()
    old = json.loads(OLD.read_text())
    saved = json.loads(MAP.read_text())
    missing = missing_patterns(old["complete_missing_family"]["complete_NR_occupancy_rows"])
    models = {}
    for arm in ("baseline", "expanded"):
        old_patterns = {tuple(map(tuple, p)) for p in saved[arm]["selected_occupations"]}
        patterns = old_patterns | missing
        block = {}
        for j2, parity in ((1, 1), (3, 1), (5, 1), (1, -1), (3, -1)):
            # Dirac l(k)=k for k>0; -k-1 otherwise.
            same = tuple(p for p in patterns if (-1) ** sum((k if k > 0 else -k - 1) * q for n, k, q in p) == parity)
            item = block_budget(same, j2)
            expected = old["existing_and_proposed_models"][arm + "_plus_complete_4f12"]["by_block"][f"2J={j2} parity={parity}"]
            assert item["CSFs"] == expected["CSFs"]
            assert item["raw_M_determinants"] == expected["raw_M_determinants_native_selected_patterns"]
            block[f"2J={j2} parity={parity}"] = item
        models[arm + "_plus_complete_4f12"] = block
    result = {
        "status": "INTEGER_SOURCE_BUDGET_NO_H_CI_NPZ_OR_FULL_DETERMINANT_ENUMERATION",
        "source": {str(OLD.relative_to(REPO)): sha(OLD), str(MAP.relative_to(REPO)): sha(MAP),
                   "src/atomx/angular/csf_hamiltonian.py": sha(REPO / "src/atomx/angular/csf_hamiltonian.py"),
                   str(Path(__file__).resolve().relative_to(REPO)): sha(Path(__file__).resolve())},
        "method": __doc__,
        "validation": {"all_local_tj_1_3_5_7_qa_qb_combinations_vs_independent_subsets": validation_cases,
                       "all_10_raw_and_CSF_dimensions_match_prior_integer_census": True},
        "models": models,
        "python_construction_payload": {
            "interpreter": sys.version,
            "measured_object_bytes": {"positive_small_int": sys.getsizeof(1), "pair_tuple": sys.getsizeof((0, 0)),
                                      "list_reference_slot": sys.getsizeof([None]) - sys.getsizeof([]),
                                      "float": sys.getsizeof(1.0)},
            "coefficient_triplet_parameterization": "Let T be actual signed nonzero angular-table entries and P be actual retained determinant pairs after zero-term rejection. The three Python row/column/data lists alone hold at least24*T bytes of reference slots. With FP64/32-bit CSR, the final table holds12*T+4*(P+1) bytes while those lists are still alive. Int64 pair coordinates add16*P bytes while pairs Python tuples/list hold at least64*P bytes. Thus a conservative simultaneous payload floor is36*T+84*P+4, before data-float objects, row-index integers, all container capacities/headers, key dictionaries, CSR/COO conversion temporaries, radial caches or previously cached blocks. T and P have not been computed here, so this is a parameterization, not an actual peak estimate.",
            "float_inclusive_estimate": "If each collected table entry has its own Python float, add24*T bytes; this gives60*T+84*P+4 before the other omissions. This common implementation-object estimate is conditional, not a rigorous lower bound or admission.",
            "connection_and_triplet_lifetimes": "During angular evaluation the decoded full candidate connection list remains alive as the for-loop iterator while output Python triplets and retained-pair list grow. At CSR conversion the connection iterator has ended, but Python triplets, output pair tuples, CSR conversion arrays and int64 coordinates overlap. A final CSR-only footprint omits both stages.",
        },
        "source_audit": {
            "angular_source_matches_frozen_f043": sha(REPO / "src/atomx/angular/csf_hamiltonian.py") == "287d1cc46d4735a6f112754a945eadad7a06252f28d66dbab516eef046f85167",
            "connected_pairs_lines_1080_1115": "Common N-2 spectator buckets avoid D^2 scans, but current implementation retains all buckets, global encoded pair set, sorted encoded list and decoded pair list simultaneously at return.",
            "coefficient_table_lines_1118_1160": "Consumes connected list and retains Python rows/cols/data/pairs lists, keys/key_index, then constructs CSR and int64 coordinates. Final sparse payload is not construction peak.",
            "projection_lines_1353_1374": "Constructs one small dense pattern projection at a time, converts to CSR and assembles block_diag. sum_pattern(D_pattern*C_pattern) bounds nonzero coefficients; global D*C zero rectangle is absent on sparse path.",
            "matrix_frozen_f043_lines_646_649": "build_ci_matrix calls exact_block_matrix without radial_symmetry=True. Existing H operator path therefore uses ordered connections, not the optional static triangular representation.",
            "density_lines_26_31": "CIRadialDensity uses ordered block_terms and dense projection by default. Sparse projection reuse removes global inter-pattern zeros but not angular-table construction.",
        },
        "streaming_equivalence_next_change_not_implemented_here": {
            "definition": "Keep the same spectator buckets. For rows i in ascending determinant order, union all indices from buckets obtained by removing the same min(2,active_N) occupied non-frozen modes of determinant i. Sort that row-local set; emit(i,j), restricting j>=i only for triangular=True.",
            "proof": "The old global set contains(i,j) iff i and j occur in a common N-2 spectator bucket. The proposed row union has exactly that membership predicate. Sorting each row produces the same order as sorting i*dimension+j globally. This preserves every connected pair, duplicates, triangular policy and deterministic first-encounter radial-key order. For active_N0/1 retain the source min(2,N) rule.",
            "memory_scope": "Removes global encoded pair set, its sorted list, and decoded full connection list. Retains spectator buckets and one row-neighbour set; maximum row set<=D. It does not remove coefficient-table Python triplets, final table CSR, projection, H or Davidson workspaces.",
            "follow_on": "Chunk exact table rows into CSR instead of full Python triplet lists, preserving first-encounter keys, signed collection and zero rules. Quantify actual table nnz/final sparse payload before admitting full4f12. No amplitude threshold, configuration reduction, external backend or change of physics.",
            "validation_needed": "Compare all streamed ordered/triangular pairs against old source on small signed cases and full old A/B; require exact pairs/key order and matched table coefficients/operator residuals. This note is a source-equivalence proposal, not an implemented or qualified interface.",
        },
        "admission": {
            "full_4f12_H_CI_admitted": False,
            "reason": "Exact structural work and sparse projection bound are established. Actual coefficient nnz, cache residency and Python triplet construction peak are still unknown; no8GiB fit assertion follows from a final CSR bound. No assertion that a full exact sparse run is impossible.",
            "next_bottleneck": "35-44million ordered Slater-Condon candidates per block and simultaneous Python connection/triplet materialization, rather than the absent global dense determinant H.",
        },
        "limits": ["Connected pairs are exact structural Slater-Condon candidates, not nonzero Hamiltonian entries.",
                   "Projection nnz is an upper bound; no coupling coefficients or floating angular values were formed.",
                   "Coefficient-table nnz and Python construction peak remain unmeasured; no full CI admission or timing claim."],
        "execution": {"elapsed_s_uncontrolled": time.monotonic() - start,
                      "peakRSS_MiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                      "imports": "stdlib only", "H_CI_NPZ_calls": 0,
                      "resources": {"CPU_affinity": [28], "RLIMIT_AS_bytes": 2 * 1024**3, "alarm_seconds": 180},
                      "joint_pattern_cache_entries": joint_pattern.cache_info().currsize},
    }
    (HERE / "angular_budget.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["execution"]), flush=True)


if __name__ == "__main__":
    main()
