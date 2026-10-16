"""Integer/metadata-only complete 4f12 numerical execution preparation.

This does not import Atomx/NumPy, open NPZ, enumerate atomic determinants or
CSFs, construct a coefficient table/Hamiltonian, or launch an eigensolver.
"""

from __future__ import annotations

import collections
import functools
import hashlib
import itertools
import json
import re
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "pyproject.toml").is_file())
BASE = HERE.parent.parent
INPUTS = {
    "missing": BASE / "physical_next_decision005/missing_4f12_budget.json",
    "angular_budget": BASE / "physical_next_decision006/angular_budget.json",
    "graph": BASE / "physical_next_decision008/full_f12_connections001/actual/result.json",
    "basis_maps": ROOT / "docs/yb_mcdhf_rci_2026_10_11/eu_yb_comparison001/"
    "core_core_discriminator_preparation001/execution_preparation002/actual/"
    "complete_basis_and_configuration_maps.json",
    "layout": ROOT / "docs/yb_mcdhf_rci_2026_10_11/screened_f_next_step208/"
    "gradient_proposal001/execution_preparation001/actual/new_layout.json",
    "fixed_contract": BASE / "physical_next_decision006/execution001/contract.json",
}
SECTORS = ((1, 1), (3, 1), (5, 1), (1, -1), (3, -1))


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, separators=(",", ":")).encode()).hexdigest()


def nr_patterns(configuration):
    """Independent NR-shell splitting, including all allowed jj occupations."""
    choices = []
    for token in configuration.split():
        match = re.fullmatch(r"(\d+)([spdfghik])(\d+)", token)
        if match is None:
            raise ValueError("unrecognized NR shell: " + token)
        n, letter, q = match.groups()
        n, q, ell = int(n), int(q), "spdfghik".index(letter)
        if ell == 0:
            choices.append((((n, -1, q),),))
        else:
            choices.append(tuple(tuple(x for x in ((n, ell, low),
                                                   (n, -ell-1, q-low)) if x[2])
                                 for low in range(max(0, q-2*ell-2), min(q, 2*ell)+1)))
    return {tuple(sorted(x for group in split for x in group))
            for split in itertools.product(*choices)}


@functools.lru_cache(None)
def magnetic_shell(twoj, q):
    # Fixed-q coefficients of product_m (1 + z*x^m); integer arithmetic only.
    work = {(0, 0): 1}
    for m in range(-twoj, twoj+1, 2):
        nxt = collections.Counter(work)
        for (count, total), value in work.items():
            if count < q:
                nxt[count+1, total+m] += value
        work = nxt
    return {m: value for (count, m), value in work.items() if count == q}


@functools.lru_cache(None)
def magnetic_pattern(pattern):
    work = {0: 1}
    for _n, k, q in pattern:
        nxt = collections.Counter()
        for total, number in work.items():
            for m, value in magnetic_shell(2*abs(k)-1, q).items():
                nxt[total+m] += number*value
        work = nxt
    return work


def parity(pattern):
    return (-1)**sum((k if k > 0 else -k-1)*q for _n, k, q in pattern)


def block(patterns, j2, p):
    selected = sorted(x for x in patterns if parity(x) == p
                      and magnetic_pattern(x).get(j2, 0)
                      - magnetic_pattern(x).get(j2+2, 0) > 0)
    return dict(patterns=selected, patterns_sha256=digest(selected),
                jj_patterns=len(selected),
                raw_M_determinants=sum(magnetic_pattern(x).get(j2, 0) for x in selected),
                CSFs=sum(magnetic_pattern(x).get(j2, 0)
                         - magnetic_pattern(x).get(j2+2, 0) for x in selected))


def prepare():
    started = time.perf_counter()
    old, saved, budget = (read(INPUTS[k]) for k in ("missing", "basis_maps", "angular_budget"))
    graph, fixed = read(INPUTS["graph"]), read(INPUTS["fixed_contract"])
    rows = old["complete_missing_family"]["complete_NR_occupancy_rows"]
    assert len(rows) == 82
    core = "1s2 2s2 2p6 3s2 3p6 3d10 4s2 4p6 4d10 5s2 5p6 4f12"
    extra = [dict(NR_index=i, family="complete_4f12_plus_3v",
                  configuration=core+" "+" ".join(f"{n}{letter}{q}"
                                                   for n, letter, q in row["valence_occupation"]))
             for i, row in enumerate(rows)]
    missing_patterns = set().union(*(nr_patterns(row["configuration"]) for row in extra))
    assert len(missing_patterns) == 1080
    models = {}
    for arm, key in (("A", "baseline"), ("B", "expanded")):
        nr = saved[key]["configurations"]
        complete_nr = [dict(NR_index=i, family=row["family"], configuration=row["configuration"])
                       for i, row in enumerate(nr+extra)]
        assert len({row["configuration"] for row in complete_nr}) == len(complete_nr)
        patterns = set().union(*(nr_patterns(row["configuration"]) for row in complete_nr))
        by_sector = {}
        for j2, p in SECTORS:
            label = f"2J={j2} parity={p}"
            result = block(patterns, j2, p)
            expected = budget["models"][key+"_plus_complete_4f12"][label]
            for field in ("jj_patterns", "raw_M_determinants", "CSFs"):
                assert result[field] == expected[field], (arm, label, field)
            result["NR_inclusion_complete"] = True
            by_sector[label] = result
        models[arm] = dict(NR_count=len(complete_nr), NR_configurations=complete_nr,
                           selected_CSFs=sum(v["CSFs"] for v in by_sector.values()),
                           sectors=by_sector)
    assert models["A"]["selected_CSFs"] == 44707
    assert models["B"]["selected_CSFs"] == 45686
    assert all(set(tuple(map(tuple, x)) for x in models["A"]["sectors"][label]["patterns"])
               <= set(tuple(map(tuple, x)) for x in row["patterns"])
               for label, row in models["B"]["sectors"].items())
    assert models["B"]["sectors"]["2J=1 parity=1"]["patterns_sha256"] == graph["patterns_sha256"]
    write(HERE / "complete_models.json", dict(status="COMPLETE_INTEGER_MODELS_NO_SCIENCE",
                                             missing_NR=extra, models=models))

    all_patterns = set().union(*(set(tuple(map(tuple, x)) for x in v["patterns"])
                                for v in models["B"]["sectors"].values()))
    shell_cases = sorted({(2*abs(k)-1, q) for pattern in all_patterns for _n, k, q in pattern})
    census = []
    for j, q in shell_cases:
        counts = magnetic_shell(j, q)
        calls = [dict(J2=m, shape=[counts.get(m+2, 0), number],
                      nullity=number-counts.get(m+2, 0))
                 for m, number in sorted(counts.items()) if m >= 0]
        census.append(dict(twoj=j, q=q, nullspace_calls=len(calls), calls=calls))
    keys = [tuple(z) for row in read(INPUTS["layout"]) for z in row["keys"]]
    assert len(keys) == len(set(keys)) == 31
    oriented = {}
    for rank in range(8):
        oriented[rank] = sum(abs((2*abs(a[1])-1)-(2*abs(c[1])-1)) <= 2*rank
                             <= 2*abs(a[1])+2*abs(c[1])-2
                             and ((a[1] if a[1] > 0 else -a[1]-1)
                                  +(c[1] if c[1] > 0 else -c[1]-1)+rank) % 2 == 0
                             for a in keys for c in keys)
    one_body = sum(n*n for n in collections.Counter(k for _n, k in keys).values())
    radial_keys = one_body + sum(n*n for n in oriented.values())
    source_pairs = sum(sum(a <= c
                           and abs(abs(a[1])-abs(c[1])) <= rank <= abs(a[1])+abs(c[1])-1
                           and ((a[1] if a[1] > 0 else -a[1]-1)
                                +(c[1] if c[1] > 0 else -c[1]-1)+rank) % 2 == 0
                           for a in keys for c in keys) for rank in range(8))
    assert radial_keys == 384706
    assert source_pairs == 706
    budgets = {}
    for label, row in budget["models"]["expanded_plus_complete_4f12"].items():
        p, d = row["connected_ordered_pairs_exact"], row["raw_M_determinants"]
        csr = 12*p+4*(d+1)
        budgets[label] = dict(
            ordered_structural_pairs=p, triangular_structural_pairs=(p+d)//2,
            Hdet_CSR_max_bytes=csr,
            typed_float64_int64_CSR_build_buffers_max_bytes=16*p+8*(d+1),
            Hdet_symmetry_three_CSR_plus_int32_transpose_payload_bound=4*csr,
            projection_CSR_bound_bytes=row["projection_CSR_FP64_int32_upper_bound_bytes"],
            local_pattern_diagonal_largest_dense_projection_bytes=row["largest_dense_per_pattern_projection_FP64_bytes"],
            local_pattern_diagonal_submatrix_dense_upper_bound_bytes=8*row["largest_single_pattern_raw_M"]**2,
            native_pair_terms_calls_cap=p,
            local_numeric_radial_cache_misses_cap=radial_keys,
            graph_evidence_scope="Only1+ graph measured; other graph dimensions from exact integer census.")
    contract = dict(
        status="SOURCE_ONLY_FULL_PHYSICAL_RUN_PLAN_NOT_ADMITTED",
        driver_sha256=sha(Path(__file__)),
        inputs={name: dict(path=str(path.relative_to(ROOT)), sha256=sha(path))
                for name, path in INPUTS.items()},
        complete_models_sha256=sha(HERE/"complete_models.json"),
        fixed=fixed["fixed"], UIDs=fixed["UIDs"],
        physical_delta="Add complete82NR4f12+3v in existing31radials toA91/B100; fixedoriginalaccepted13.",
        operator="Native unshifted Dirac-Coulomb, original69e/Z70/A175Fermi/P/S; noBreit/QED/empiricalshift.",
        angular_census=census,
        angular_nullspace_total_calls_cold=sum(x["nullspace_calls"] for x in census),
        radial_key_integer_bound=dict(orbital_keys=keys, oriented_pair_counts_by_rank=oriented,
                                     h_key_upper_bound=one_body, h_plus_r_key_upper_bound=radial_keys,
                                     unordered_rank_source_pair_bound=source_pairs,
                                     unordered_density_pair_bound=31*32//2,
                                     native_pair_potential_LRU_capacity=1024,
                                     native_density_LRU_capacity=512,
                                     reason="Allowed reducedC rank/triangle/parity ordered orbitalpairs; sum_k Pk^2 plus equal-kappa h. Cancellations may reduce, never increase."),
        memory_per_B_sector=budgets,
        resources_proposed=dict(threads=1, AS_bytes=8*1024**3, inner_seconds=1200,
                                outer_seconds=1215, host_load1_max=16,
                                host_MemAvailable_GiB_min=32,
                                sector_CPUs=[28,29,30], max_simultaneous_workers=3),
        resource_status="8GiB/1200s is bounded trial cap, not measured fit/runtime prediction. Rootmust first qualify current source on complete smallA/B.",
        source_api_requirement=dict(build_ci_matrix="angular_storage='streamed', as_operator=True, max_determinants=92715",
                                    coefficient_table_calls_cap=0,
                                    dense_CSF_or_raw_H_calls_cap=0,
                                    exact_signed_core_contraction=True,
                                    diagonal="For eachpattern p, diag(Cp.T Hdet[Sp,Sp] Cp); separate signed projectionperm. No globalHdet@projection temporary.",
                                    expected_full_spectrum_tracker_calls_cap=0),
        solver=dict(native="atomx.methods.ci.davidson_eigh", per_arm_sector_n_states=2,
                    tol_L2_Ha=1e-9, final_direct_L2_residual_Ha_max=1e-8,
                    max_iter=200, max_subspace=22, selected_Gram_maxabs=1e-10,
                    operator_matvec_columns_cap_per_arm=1000),
        nested_A_B_identity=dict(
            compare="OneBfullH/operator, exactsignature/occupationrestriction givesAfullH. No4f12/NR/CSFamplitudedropwithin either declaredarm.",
            requirements=["EveryA CSFsignature occurs exactlyonce inB withsame signed CFPphase/couplingtree.",
                          "Every originalA/B selectedsignature occurs inmatching enlargedarm; keepnative fullpatternorder mapping.",
                          "Reducedmatrixaction=Aembed.T H_B Aembed; diagA=Bdiag[Aindices].",
                          "Current dirtysource sameH qualification againstowned originalA/B E/C/H is prerequisite."]),
        state_identity=dict(
            original_reference="Ownedaccepted13 A/B signedtarget C, already physicallytracked tooriginal531in004; embedsameorbitalsbyfullCSFsignature. Directoriginal531overlap isnotclaimed: earlierorbitals needmixed-orbital overlap.",
            rank=1, signed_overlap_squared_strictly_greater_than=0.5,
            proof="For normalizedreference r and completeorthonormal eigenbasis{u_j}, sum_j|r.Tu_j|^2=1. A selectedisolatedroot withweight>.5 outranks every omittedorthogonalroot. Finiteresidual/Gram acceptance remainsrequired; finite residual alone doesnot certify omittedspectralgap.",
            stop_if=["No selectedroot crossesstrict.5", "targetsubspace degenerate/ambiguous", "signature/phase embedding nonunique", "rootenergy or residualnonfinite"],
            not_a_public_full_tracker=True, not_a_retained_snapshot=True),
        results_required=["All5signedtarget energies forAfull/Bfull andsameboundoriginalA/B references", "deltaE state anddeltaTransition relativeonefresh6s ground perarm", "D/Pcentroids andfine-structure shifts", "completepaircounts/raw/pattern/projectionhashes", "directfulloperator residuals andstateoverlaps", "caps/caches/time/RSS/source/file bindings"],
        stop="First source/input/census/resource/solver/identity failure cancels remainingworkers; preservepartial. Noauto retry, changedcaps/gates, modelcut, alternatebackend or appendedcorrections.",
        forbidden_claims=["MCDHFstationarity", "orbitalresponse", "fullspectrum orNISTaccuracy", "SCIremainderbound", "parallel speedup fromsingle trials", "Breit/QED corrections", "fullpublicstateowner qualification"],
        scientific_execution_performed=False,
        elapsed_integer_preparation_seconds=time.perf_counter()-started,
    )
    write(HERE / "plan.json", contract)
    print(json.dumps(dict(status=contract["status"], NR_A=models["A"]["NR_count"],
                          NR_B=models["B"]["NR_count"], angular_cases=len(census),
                          cold_nullspaces=contract["angular_nullspace_total_calls_cold"],
                          plan_sha256=sha(HERE/"plan.json"),
                          models_sha256=sha(HERE/"complete_models.json"))))


if __name__ == "__main__":
    prepare()
