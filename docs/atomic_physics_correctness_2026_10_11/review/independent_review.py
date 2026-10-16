"""Replay saved finite evidence independently; never launch a science solver."""

import ast
import hashlib
import itertools
import json
import sys
import tarfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.linalg import expm

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def arrays(path):
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def maximum(value):
    return float(np.max(np.abs(value), initial=0))


def function_subset(path, names, namespace):
    tree = ast.parse(path.read_text())
    chosen = ast.Module(
        body=[node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names],
        type_ignores=[],
    )
    exec(compile(chosen, str(path), "exec"), namespace)


def semantic_ast(path):
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if (
            isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
            and node.body
            and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant)
            and isinstance(node.body[0].value.value, str)
        ):
            node.body.pop(0)
    return ast.dump(tree, include_attributes=False)


def main():
    contract = json.loads((HERE / "contract.json").read_bytes())
    bindings = contract["inputs"]
    paths = {key: REPO / value["path"] for key, value in bindings.items()}
    assert all(sha(paths[key]) == value["sha256"] for key, value in bindings.items())
    gate = contract["gate_Ha"]
    snapshot = REPO / "docs/atomic_physics_correctness_2026_10_11/sd/source"
    sys.path.insert(0, str(snapshot / "src"))
    from atomx.angular.recoupling import l_of_kappa
    from atomx.angular.sd_rho import kappa_from_partial, rho_coefficient
    from atomx.methods.linearized_sd import plan_scalar_sd

    namespace = dict(itertools=itertools, np=np, l_of_kappa=l_of_kappa,
                     rho_coefficient=rho_coefficient, kappa_from_partial=kappa_from_partial)
    function_subset(paths["frozen_core"], {"_channels"}, namespace)
    function_subset(paths["frozen_lift"], {"_core_terms", "_symmetry"}, namespace)
    q = arrays(paths["quotient_arrays"])
    loc = arrays(paths["localization_arrays"])
    metadata = json.loads(paths["quotient_metadata"].read_bytes())
    occupied = ((1, -1), (2, -1))
    virtual = ((3, -1), (4, -1), (2, -2))
    keys = occupied + virtual
    labels = tuple((nr, k, m) for nr, k in keys for m in range(-2 * abs(k) + 1, 2 * abs(k), 2))
    assert labels == tuple(map(tuple, metadata["labels"]))
    groups = tuple(tuple(key for key in keys if key[1] == kappa_from_partial(i)) for i in range(1, 4))
    plan = dict(occupied_keys=occupied, virtual_keys=virtual, groups=groups,
                channels=namespace["_channels"](occupied, 1, "rho"), dimensions={"KK": 3})
    core = SimpleNamespace(plan=plan, integrals=SimpleNamespace(labels=labels, noccupied=4))
    masks = tuple(map(int, q["masks"]))
    positions = {mask: i for i, mask in enumerate(masks)}
    assert masks == tuple(sum(1 << bit for bit in row) for row in itertools.combinations(range(12), 4))
    reference = positions[15]
    ranks = np.array([(15 & ~mask).bit_count() for mask in masks])
    qsd = (ranks > 0) & (ranks <= 2)
    assert np.array_equal(qsd, q["qsd"])

    def elementary(mask, operations):
        phase = 1
        for bit, create in reversed(operations):
            if bool(mask & (1 << bit)) == create:
                return None, 0
            phase *= (-1) ** ((mask & ((1 << bit) - 1)).bit_count())
            mask ^= 1 << bit
        return mask, phase

    def apply(terms, vector):
        result = np.zeros(len(masks))
        for j in np.flatnonzero(vector):
            for value, operations in terms:
                target, phase = elementary(masks[j], operations)
                if phase:
                    result[positions[target]] += phase * value * vector[j]
        return result

    count = len(q["source_Ha"])
    unit = np.eye(count)
    reference_vector = np.eye(len(masks))[reference]
    terms = [namespace["_core_terms"](core, column) for column in unit]
    B = np.column_stack([apply(t, reference_vector) for t in terms])
    H = q["H_Ha"]
    L = np.column_stack([H @ B[:, i] - apply(t, H[:, reference]) for i, t in enumerate(terms)])
    P = np.eye(count)
    coordinates = metadata["coordinates"]
    for row in coordinates:
        if "holes" not in row or row["holes"][0] != row["holes"][1] or row["groups"][0] != row["groups"][1]:
            continue
        paired = [r for r in coordinates if r.get("holes") == row["holes"]
                  and r.get("groups") == row["groups"] and r.get("rank") == row["rank"]
                  and r.get("particles") == row["particles"][::-1]]
        assert len(paired) == 1
        i, j = row["index"], paired[0]["index"]
        P[i] = 0
        P[i, i] += 0.5
        P[i, j] += 0.5
    eig, vectors = np.linalg.eigh(P)
    domain = vectors[:, eig > 0.5]
    _, singular, vh = np.linalg.svd(B @ domain, full_matrices=True)
    physical_rank = int(np.sum(singular > 1e-12))
    null = domain @ vh[physical_rank:].T
    K = q["K_Ha"]
    mismatch = B @ K + L
    raw = q["raw_random_t"]
    solved = q["solved_t"]
    sd = dict(
        raw_coordinates=count, pair_domain_dimension=domain.shape[1],
        physical_dimension=physical_rank,
        independently_counted_scalar_sd_dimension=plan_scalar_sd(occupied, virtual).amplitudes,
        constrained_null_dimension=null.shape[1],
        lift_reconstruction_error=maximum(B - q["B"]),
        commutator_reconstruction_error_Ha=maximum(L - q["L_Ha"]),
        projector_reconstruction_error=maximum(P - q["P"]),
        projector_idempotence_error=maximum(P @ P - P),
        excluded_directions_are_lift_null=maximum(B @ (unit - P)),
        raw_random_failure_Ha=maximum((mismatch @ raw)[qsd]),
        symmetrized_random_error_Ha=maximum((mismatch @ P @ raw)[qsd]),
        complete_domain_column_error_Ha=maximum((mismatch @ P)[qsd]),
        constrained_null_action_error_Ha=maximum((B @ K @ null)[qsd]),
        domain_invariance_error_Ha=maximum((unit - P) @ K @ P),
        source_domain_error_Ha=maximum(q["source_Ha"] - P @ q["source_Ha"]),
        solved_domain_error=maximum(solved - P @ solved),
        independently_solved_QSD_residual_Ha=maximum((H[:, reference] + L @ solved)[qsd]),
        native_solved_raw_residual_Ha=maximum(K @ solved - q["source_Ha"]),
        localization_same_H_error_Ha=maximum(H - loc["H_Ha"]),
        declared_equations_pass_on_complete_finite_domain=True,
        missing_diagram_inference_from_failed_random=False,
        Yb_or_omitted_physics_certificate=False,
        production_core_nondoc_AST_unchanged=(semantic_ast(paths["frozen_core"]) == semantic_ast(REPO / "src/atomx/methods/core_sd.py")),
    )
    assert physical_rank == sd["independently_counted_scalar_sd_dimension"] == 17
    assert domain.shape[1] == 30 and null.shape[1] == 13
    assert sd["excluded_directions_are_lift_null"] < gate
    assert sd["raw_random_failure_Ha"] > 1e-3
    for key, value in sd.items():
        if key.endswith(("_error_Ha", "_error")) and isinstance(value, float):
            assert value < gate, (key, value)
    assert sd["independently_solved_QSD_residual_Ha"] < gate
    assert sd["native_solved_raw_residual_Ha"] < 1e-12
    assert sd["production_core_nondoc_AST_unchanged"]

    g = arrays(paths["guard_arrays"])
    result = json.loads(paths["guard_result"].read_bytes())
    request = json.loads(paths["guard_request"].read_bytes())
    hb = expm(-g["backend_tcore3"]) @ g["full_h3_Ha"] @ expm(g["backend_tcore3"])
    omega, target, E = g["backend_omega"], g["backend_heff_Ha"], result["core_energy_Ha"]
    residual = hb @ omega - omega @ (target + E * np.eye(8))
    root_result = json.loads(paths["guard_root_result"].read_bytes())
    guard = dict(
        hbar_reconstruction_error_Ha=maximum(hb - g["backend_hbar3_Ha"]),
        complete_120_by_8_Bloch_residual_Ha=maximum(residual),
        retained_independent_heff_error_Ha=maximum(target - g["exact_heff_Ha"]),
        all_8_roots_vs_FCI_Ha=maximum(np.sort(root_result["sector_energies_Ha"]) - np.linalg.eigvalsh(g["full_h3_Ha"])[:8]),
        intermediate_normalization_error=maximum(omega[g["model_indices"]] - np.eye(8)),
        source_and_archive_bindings=len(result["source_and_inputs"]),
        residual_scopes_explicit="omitted-rank" in request["residual_scope"],
        guard_scope="Nonzero actual occupied-virtual Fock in sector0h1p only; before export/launch; no coupling deletion or tolerance change.",
        trusted_provenance_limit="Manifest checks bind trusted builder provenance; they do not authenticate an arbitrary malicious manifest copying the accepted source hash onto another binary.",
        highest_level_reached=False,
    )
    assert guard["complete_120_by_8_Bloch_residual_Ha"] < gate
    assert guard["retained_independent_heff_error_Ha"] < gate
    assert guard["all_8_roots_vs_FCI_Ha"] < 1e-10
    with tarfile.open(paths["guard_archive"]) as archive:
        entries = {m.name: archive.extractfile(m).read() for m in archive.getmembers() if m.isfile()}
    for source, digest in result["source_and_inputs"].items():
        path = Path(source).resolve()
        assert sha(path) == digest
        assert hashlib.sha256(entries[str(path.relative_to(REPO))]).hexdigest() == digest
    coverage = json.loads(paths["coverage"].read_bytes())
    for bound in coverage["inputs"].values():
        epoch = paths["coverage"].parent / "source_epoch" / bound["path"]
        assert sha(epoch if epoch.exists() else REPO / bound["path"]) == bound["sha256"]
    four_core = next(r for r in coverage["finite_fscc_rows"] if r["case"] == "four_core" and r["backend"] == "corrected")
    assert four_core["attachment_retained_residual_Ha"] < gate
    assert four_core["attachment_full_residual_Ha"] == four_core["unretained_attachment_rank34_residual_Ha"] > 1e-3
    assert all(sha(paths[key]) == value["sha256"] for key, value in bindings.items())
    evidence = dict(status="pass_complete_finite_domain_and_guard_review", contract_sha256=sha(HERE / "contract.json"),
                    reviewer_script_sha256=sha(Path(__file__)), inputs=bindings, sd=sd, fscc_guard=guard,
                    retained_vs_omitted_rank_distinction_preserved=True,
                    new_orbital_or_amplitude_solves=0, source_mutations=0, highest_level_reached=False)
    output = HERE / "evidence.json"
    if output.exists():
        assert json.loads(output.read_bytes()) == evidence, "saved review changed; retain original evidence"
    else:
        output.write_text(json.dumps(evidence, indent=2) + "\n")
    print(json.dumps({"status": evidence["status"], "sd": sd, "fscc_guard": guard}, indent=2))


if __name__ == "__main__":
    main()
