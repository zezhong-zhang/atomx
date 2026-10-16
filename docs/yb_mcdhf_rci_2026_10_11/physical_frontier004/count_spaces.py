"""Independent kinematic counts only: no Atomx import or physical evaluation.

Closed cores have M=0.  A determinant with holes h and particles p has
2M=sum(p.2m)-sum(h.2m).  Within each complete jj occupation pattern,
N(J)=D(M=J)-D(M=J+1).  Atomx's raw angular construction retains patterns
with N(J)>0, not all magnetic determinants with M=J.
"""

from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
import json


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def subshell(name, twoj, orbital_l):
    return [(name, twoj, orbital_l, m) for m in range(-twoj, twoj + 1, 2)]


CORE = {
    "4f": subshell("4f5/2", 5, 3) + subshell("4f7/2", 7, 3),
    "5p": subshell("5p1/2", 1, 1) + subshell("5p3/2", 3, 1),
    "5s": subshell("5s1/2", 1, 0),
}
BASE_PARTICLES = (subshell("6s1/2", 1, 0) + subshell("5d3/2", 3, 2)
                  + subshell("5d5/2", 5, 2) + subshell("6p1/2", 1, 1)
                  + subshell("6p3/2", 3, 1))
LAYER = (subshell("7s1/2", 1, 0) + subshell("6d3/2", 3, 2)
         + subshell("6d5/2", 5, 2) + subshell("7p1/2", 1, 1)
         + subshell("7p3/2", 3, 1))


def magnetic_patterns(holes, particles, hole_number, particle_number):
    patterns = defaultdict(Counter)
    magnetic_dimension = 0
    for h in combinations(holes, hole_number):
        h_occ = tuple(sorted(Counter(s[0] for s in h).items()))
        h_m = sum(s[3] for s in h)
        h_l = sum(s[2] for s in h)
        for p in combinations(particles, particle_number):
            p_occ = tuple(sorted(Counter(s[0] for s in p).items()))
            parity = -1 if (h_l + sum(s[2] for s in p)) % 2 else 1
            pattern = (h_occ, p_occ, parity)
            patterns[pattern][sum(s[3] for s in p) - h_m] += 1
            magnetic_dimension += 1
    return patterns, magnetic_dimension


def add_patterns(target, source):
    for pattern, histogram in source.items():
        target[pattern].update(histogram)


def summarize(patterns, magnetic_dimension, nr_config_count, radial_count):
    max_twoj = max(max(histogram) for histogram in patterns.values())
    blocks = []
    for twoj in range(1, max_twoj + 1, 2):
        for parity in (-1, 1):
            count_csf = count_raw = count_patterns = count_all_m = 0
            for pattern, histogram in patterns.items():
                if pattern[2] != parity:
                    continue
                at_j = histogram[twoj]
                multiplicity = at_j - histogram[twoj + 2]
                assert multiplicity >= 0, (pattern, twoj, multiplicity)
                count_all_m += at_j
                count_csf += multiplicity
                if multiplicity:
                    count_raw += at_j
                    count_patterns += 1
            if count_csf:
                blocks.append({"twoJ": twoj, "parity": parity,
                               "CSFs": count_csf,
                               "unique_patterns": count_patterns,
                               "raw_M_equals_J_determinants": count_raw,
                               "all_patterns_M_equals_J": count_all_m})
    return {
        "NR_configuration_count": nr_config_count,
        "radial_spinor_count": radial_count,
        "full_magnetic_dimension": magnetic_dimension,
        "CSFs": sum(b["CSFs"] for b in blocks),
        "symmetry_blocks": len(blocks),
        "max_block_CSFs": max(b["CSFs"] for b in blocks),
        "max_block_raw_M_equals_J": max(b["raw_M_equals_J_determinants"] for b in blocks),
        "max_all_patterns_M_equals_J": max(b["all_patterns_M_equals_J"] for b in blocks),
        "raw_pair_work_upper": sum(b["raw_M_equals_J_determinants"] ** 2 for b in blocks),
        "dense_CSF_matrix_bytes_all_blocks": 8 * sum(b["CSFs"] ** 2 for b in blocks),
        "dense_raw_matrix_bytes_all_blocks": 8 * sum(b["raw_M_equals_J_determinants"] ** 2 for b in blocks),
        "target_symmetry_blocks": [b for b in blocks if (b["twoJ"], b["parity"]) in {(1, 1), (3, 1), (5, 1), (1, -1), (3, -1)}],
        "blocks": blocks,
    }


def model(double_core=None, extended_layer=False):
    all_holes = [s for family in CORE.values() for s in family]
    particles = BASE_PARTICLES + LAYER if extended_layer else BASE_PARTICLES
    patterns, count0 = magnetic_patterns([], particles, 0, 1)
    one_hole, count1 = magnetic_patterns(all_holes, particles, 1, 2)
    add_patterns(patterns, one_hole)
    count2 = 0
    if double_core:
        holes = all_holes if double_core == "all" else CORE[double_core]
        two_holes, count2 = magnetic_patterns(holes, particles, 2, 3)
        add_patterns(patterns, two_holes)
    nr_count = 69 if extended_layer else 21
    if double_core:
        nr_count += 54 if double_core == "all" else 9
    count = count0 + count1 + count2
    assert count <= 250000
    return summarize(patterns, count, nr_count, 29 if extended_layer else 24)


def main():
    archive = json.loads((ROOT / "balanced_cv_frontier/result.json").read_text())["spaces"]["complete_union"]
    baseline = model()
    keys = ("CSFs", "symmetry_blocks", "max_block_CSFs", "max_block_raw_M_equals_J",
            "raw_pair_work_upper", "dense_CSF_matrix_bytes_all_blocks",
            "dense_raw_matrix_bytes_all_blocks", "full_magnetic_dimension")
    controls = [{"name": k, "actual": baseline[k], "archive": archive[k], "pass": baseline[k] == archive[k]} for k in keys]
    for actual, old in zip(baseline["blocks"], archive["blocks"], strict=True):
        for k in ("twoJ", "parity", "CSFs", "unique_patterns", "raw_M_equals_J_determinants"):
            controls.append({"name": f"J{actual['twoJ']}/parity{actual['parity']}/{k}",
                             "actual": actual[k], "archive": old[k], "pass": actual[k] == old[k]})
    assert all(c["pass"] for c in controls), [c for c in controls if not c["pass"]]
    models = {"baseline531": baseline,
              "baseline_plus_complete_5p_double_holes": model("5p"),
              "baseline_plus_complete_4f_double_holes": model("4f"),
              "baseline_plus_all_complete_core_double_holes": model("all"),
              "baseline_plus_one_balanced_sdp_virtual_layer": model(extended_layer=True)}
    base_work = baseline["raw_pair_work_upper"]
    base_targets = sum(b["raw_M_equals_J_determinants"] ** 2 for b in baseline["target_symmetry_blocks"])
    for value in models.values():
        targets = value["target_symmetry_blocks"]
        value["five_complete_target_sectors"] = {
            "CSFs": sum(b["CSFs"] for b in targets),
            "sector_count": len(targets),
            "raw_M_equals_J_determinants_sum_across_sectors": sum(b["raw_M_equals_J_determinants"] for b in targets),
            "raw_pair_work_upper": sum(b["raw_M_equals_J_determinants"] ** 2 for b in targets),
            "dense_CSF_matrix_bytes": 8 * sum(b["CSFs"] ** 2 for b in targets),
            "dense_raw_matrix_bytes": 8 * sum(b["raw_M_equals_J_determinants"] ** 2 for b in targets),
            "exact_symmetry_bound": "For scalar DC/static Breit and five specified J/parity target states, omitted sectors have exactly zero state amplitude and Hamiltonian coupling by symmetry. Each selected sector is complete: no occupation, configuration or amplitude pruning. Original embedded target norm is one; expanded target overlap is unmeasured and must satisfy unchanged signed-overlap matching rules.",
            "claim_limit": "This subset supports five-state EOL/target RCI only. It does not certify all-spectrum roots, global finite-H ground identity, or omitted-sector experiments. Baseline full-versus-sector CI energies, signed vectors and complete EOL gradients must match before this route is used.",
        }
        value["all_block_raw_pair_work_ratio_to531"] = value["raw_pair_work_upper"] / base_work
        value["five_target_raw_pair_work_ratio_to531"] = sum(b["raw_M_equals_J_determinants"] ** 2 for b in value["target_symmetry_blocks"]) / base_targets
        value["within_original_raw_per_block_cap_2000"] = value["max_block_raw_M_equals_J"] <= 2000
    output = {
        "status": "pure_kinematic_preflight_no_physics_evaluation",
        "method": "Pauli-complete magnetic determinants grouped by jj occupation pattern; CSF multiplicity D(J)-D(J+1); raw angular count includes only patterns with nonzero CSF multiplicity at that J.",
        "failed_first_control_preserved": {"mistake": "Using aggregate D(M=J) as implementation raw angular count retains zero-J-multiplicity patterns.", "baseline_aggregate_max": baseline["max_all_patterns_M_equals_J"], "archived_and_pattern_resolved_max": baseline["max_block_raw_M_equals_J"], "consequence": "First max-raw equality assertion failed (275 vs232); no physics computation occurred; corrected counting matches all88 archived scalar/block controls."},
        "controls": controls,
        "all_controls_pass": all(c["pass"] for c in controls),
        "models": models,
        "resource_limits": "Dense matrix bytes and raw pair ratios are algebraic counts, not peak RSS, operation FLOPs, runtime or parallel-efficiency bounds; radial/angular caches and coefficient tables also consume memory. All prospective models require actual public CSF/raw-support preflight before any Hamiltonian allocation. No source, CSF solver or gradient imported.",
    }
    (HERE / "kinematic_preflight.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({"controls": len(controls), "all_pass": output["all_controls_pass"], "models": {k: {q: v[q] for q in ("NR_configuration_count", "CSFs", "full_magnetic_dimension", "max_block_CSFs", "max_block_raw_M_equals_J", "five_target_raw_pair_work_ratio_to531", "within_original_raw_per_block_cap_2000")} for k, v in models.items()}}, indent=2))


if __name__ == "__main__":
    main()
