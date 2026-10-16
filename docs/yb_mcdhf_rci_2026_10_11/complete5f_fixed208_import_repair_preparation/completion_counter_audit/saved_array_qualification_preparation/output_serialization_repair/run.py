"""Separately admitted saved-array qualification; no physical producers.

The original5f expected14 h/S execution gate remains FAIL. Default performs
only byte/source/JSON authorization. NumPy is imported only after exact root
binding, and is used for algebra over eighteen immutable archives once each.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import resource
import signal
import sys
import time
import traceback
from collections import Counter
from collections.abc import Mapping
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "pyproject.toml").exists())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for piece in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(piece)
    return digest.hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    temporary = path.with_name(path.name + ".part")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def metadata():
    contract, inputs = load(HERE / "contract.json"), load(HERE / "inputs.json")
    if sha(HERE / "inputs.json") != contract["inputs_sha256"]:
        raise ValueError("Frozen saved input map changed")
    for role, row in inputs.items():
        if sha(ROOT / row["path"]) != row["sha256"]:
            raise ValueError("Frozen saved input changed: " + role)
    if sum(row["kind"] == "NPZ" for row in inputs.values()) != 18:
        raise ValueError("Exact eighteen unique NPZ inputs required")
    if len({row["path"] for row in inputs.values()}) != len(inputs):
        raise ValueError("No duplicate input role/archive loads")
    failure = load(ROOT / inputs["original_failure"]["path"])
    receipt = load(ROOT / inputs["original_receipt"]["path"])
    if (receipt["exit_code"] != 1 or failure["scientific_success_claim"]
            or not failure["message"].endswith("expected_completed_CI_internal_hS")
            or failure["completed"]["CI_internal_hS"] != 7):
        raise ValueError("Original failed execution boundary changed")
    return contract, inputs


def authorize(path):
    contract, inputs = metadata()
    binding = load(path)
    expected = dict(execution_authorized_by_root=True,
                    preparation_contract_sha256=sha(HERE / "contract.json"),
                    driver_sha256=sha(__file__),
                    preparation_sha256=sha(HERE / "preparation.json"),
                    inputs_sha256=sha(HERE / "inputs.json"),
                    input_roles=inputs,
                    original_failed_receipt_sha256=inputs["original_receipt"]["sha256"],
                    source_map_sha256=contract["source_map_sha256"],
                    new_physical_producer_calls=0)
    if any(binding.get(key) != value for key, value in expected.items()):
        raise ValueError("Missing/mismatched exact root saved-array authorization")
    return contract, inputs, binding


def execute(binding_path):
    contract, inputs, binding = authorize(binding_path)
    output = HERE / "actual"
    output.mkdir(exist_ok=False)
    started = time.monotonic()
    checks, loaded, models, spectra = [], [], {}, []

    def check(name, error=0.0, limit=0.0):
        if len(checks) >= contract["resources"]["maximum_numerical_check_rows"]:
            raise RuntimeError("Prospective numerical-check cap")
        value = float(error)
        row = dict(name=name, error=value, limit=float(limit),
                   passed=value >= 0 and value <= limit)
        checks.append(row)
        if not row["passed"]:
            raise ValueError("Saved-array qualification gate failed: " + name)

    def deadline(_signum, _frame):
        raise TimeoutError("Original120s saved-algebra cap")

    try:
        cpu = contract["resources"]["CPU"]
        if cpu not in os.sched_getaffinity(0):
            raise RuntimeError("Exact CPU31 unavailable; no substitute")
        os.sched_setaffinity(0, {cpu})
        for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                     "BLIS_NUM_THREADS", "NUMBA_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
            os.environ[name] = "1"
        memory = int(contract["resources"]["address_space_GiB"] * 1024**3)
        resource.setrlimit(resource.RLIMIT_AS, (memory, memory))
        signal.signal(signal.SIGALRM, deadline)
        signal.alarm(contract["resources"]["wall_seconds"])
        import numpy as np

        def forbidden(*_args, **_kwargs):
            raise RuntimeError("No eigen/SVD/factorization/solve or physical producer")

        for name in ("eig", "eigh", "eigvals", "eigvalsh", "svd", "qr", "solve",
                     "inv", "pinv", "cholesky", "lstsq"):
            setattr(np.linalg, name, forbidden)
        if any(name == "atomx" or name.startswith("atomx.") for name in sys.modules):
            raise RuntimeError("Atomx must never be imported by saved algebra")

        def archive(role):
            if role in loaded or len(loaded) >= 18 or inputs[role]["kind"] != "NPZ":
                raise RuntimeError("Only eighteen declared unique one-time loads")
            loaded.append(role)
            with np.load(ROOT / inputs[role]["path"], allow_pickle=False) as values:
                return {key: values[key].copy() for key in values.files}

        def contents(value):
            if isinstance(value, np.ndarray):
                return (value.shape, value.dtype.str, value.tobytes())
            if isinstance(value, Mapping):
                return tuple((key, contents(item)) for key, item in sorted(value.items()))
            if isinstance(value, (tuple, list)):
                return tuple(contents(item) for item in value)
            if isinstance(value, (str, int, float, bool, np.number, type(None))):
                return value
            raise TypeError("Unsupported saved binding value")

        def digest(value):
            return hashlib.sha256(repr(contents(value)).encode()).hexdigest()

        def same_bytes(a, b):
            return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()

        gates = contract["acceptance"]
        literal, geometry, hs = archive("literal208"), archive("geometry"), archive("hS_geometry")
        manifest = load(ROOT / inputs["literal208_manifest"]["path"])
        layout = load(ROOT / inputs["layout"]["path"])
        basis = load(ROOT / inputs["basis"]["path"])
        declared_models = load(ROOT / inputs["models"]["path"])["models"]
        nominal = {row["label"]: row["NIST_nominal_cm1"] for row in
                   load(ROOT / inputs["nominal_NIST"]["path"])["DC_rows"]}
        old_keys = tuple(map(tuple, literal["keys"].tolist()))
        data = {key[6:]: value.item() if value.ndim == 0 else value
                for key, value in geometry.items() if key.startswith("basis_")}
        check("literal29keys", old_keys != tuple(map(tuple, contract["original29keys"])))
        check("original3451coordinates", literal["x"].shape != (3451,))
        check("wide_precision_exceeds_float64", np.finfo(np.longdouble).eps >= np.finfo(float).eps)
        projectors, fp = {}, hashlib.sha256()
        for i, row in enumerate(layout):
            keys, shape = tuple(map(tuple, row["keys"])), tuple(row["shape"])
            p = literal[f"projector_{i}"]
            projectors[keys[0][1]] = p
            fp.update(repr((list(keys), shape)).encode())
            fp.update(np.ascontiguousarray(p).tobytes())
            check("P119shape/" + str(i), p.shape != (246, 119))
            metric = hs[f"s_{keys[0][1]}"]
            check("PactualSGram/" + str(i), np.max(abs(p.T @ metric @ p - np.eye(119))),
                  gates["saved_P_metric_Gram_error"])
            mapped = p @ literal["x"][slice(*row["slice"])].reshape(shape)
            for j, key in enumerate(keys):
                index = old_keys.index(key)
                check("literalP_x_samewave/" + str(key), not same_bytes(
                    mapped[:, j], np.concatenate((literal["P"][index], literal["Q"][index]))))
        fp.update(np.ascontiguousarray(literal["inverse_diagonal"]).tobytes())
        for i in range(len(layout)):
            for block in literal[f"inverse_blocks_{i}"]:
                fp.update(np.ascontiguousarray(block).tobytes())
        check("originalmetricfingerprint", fp.hexdigest() != contract["metric_sha256"])

        constants = ast.parse((ROOT / inputs["constants_source"]["path"]).read_text())
        literals = {node.targets[0].id: ast.literal_eval(node.value) for node in constants.body
                    if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
                    and node.targets[0].id in ("CODATA", "DEFAULT_CODATA_VERSION")}
        codata = literals["CODATA"][literals["DEFAULT_CODATA_VERSION"]]
        hartree_cm1 = codata["_Ha_J"] / (100.0 * codata["_hplanck"] * codata["_c"])
        target_uids = {tuple(uid): label for label, uid in contract["target_UIDs"]}
        expected_shape = {"baseline": 1046, "complete5f": 1595}
        for name in ("baseline", "complete5f"):
            a = archive(name + "_CI")
            owner = load(ROOT / inputs[name + "_owner"]["path"])
            tracked = load(ROOT / inputs[name + "_tracking"]["path"])
            summary = load(ROOT / inputs[name + "_evidence"]["path"])
            n = expected_shape[name]
            h, e, c = a["H"], a["energies"], a["mixing"]
            keys = tuple(map(tuple, a["keys"].tolist()))
            check(name + "/completeHCE_shapes", h.shape != (n, n) or c.shape != (n, n) or e.shape != (n,))
            check(name + "/real_finite64", any(x.dtype != np.dtype(float) or not np.isfinite(x).all()
                                               for x in (h, e, c, a["P"], a["Q"])))
            check(name + "/wholeHsymmetry", np.max(abs(h - h.T)), gates["matrix_hermiticity_Ha"])
            residuals = np.sqrt(np.sum((h @ c - c * e)**2, axis=0))
            check(name + "/whole_eigenresidual", np.max(residuals), gates["CI_residual_Ha"])
            check(name + "/whole_signedGram", np.max(abs(c.T @ c - np.eye(n))), gates["CI_orthogonality"])
            check(name + "/ownerDC", owner["hamiltonian_terms"] != ["dirac-coulomb"])
            check(name + "/ownerkeys", owner["orbital_keys"] != a["keys"].tolist()
                  or owner["orbital_basis_keys"] != a["keys"].tolist())
            binding = owner["radial_binding"]
            check(name + "/ownerworkspace_digest", digest(data) != binding["geometry"])
            check(name + "/ownerZns", binding["Z"] != 70 or binding["ns"] != 123 or binding["core"] != [])
            check(name + "/signedspinor_binding", binding["external"] != [digest((pp, qq, int(key[1])))
                  for key, pp, qq in zip(keys, a["P"], a["Q"], strict=True)])
            for key in old_keys:
                j, k = keys.index(key), old_keys.index(key)
                check(name + "/old29bytewave/" + str(key), not (same_bytes(a["P"][j], literal["P"][k])
                      and same_bytes(a["Q"][j], literal["Q"][k])))
            check(name + "/only_declared_radialkeys", set(keys) != set(old_keys) | ({(5, 3), (5, -4)} if name == "complete5f" else set()))
            for kappa in projectors:
                metric = hs[f"s_{kappa}"].astype(np.longdouble)
                vectors = np.column_stack([np.concatenate((a["P"][i], a["Q"][i]))
                                          for i, key in enumerate(keys) if key[1] == kappa]).astype(np.longdouble)
                p = projectors[kappa].astype(np.longdouble)
                check(name + "/actualradialGram/" + str(kappa), np.max(abs(vectors.T @ metric @ vectors
                      - np.eye(vectors.shape[1]))), gates["actual_radial_Gram_error"])
                absent = vectors - p @ (p.T @ metric @ vectors)
                check(name + "/originalP_leakage/" + str(kappa), max(0.0, float(np.max(
                      np.sum(absent * (metric @ absent), axis=0)))), gates["nuclear_projector_leakage_S_norm_squared"])
            signatures = basis[name]["retained_signatures"]
            decoded = [json.loads(sig) for sig in signatures]
            check(name + "/complete_signedCSFcount", len(signatures) != n or len(set(signatures)) != n)
            expected_owner = [[69, sig[3], [sig[0], sig[2], sig[1]]] for sig in decoded]
            check(name + "/whole_signedowner_order", owner["CSFs"] != expected_owner)
            sectors = [(int(sig[2]), int(sig[3])) for sig in decoded]
            target_sectors = {uid[:2] for uid in target_uids}
            retained_from_full = [sig for sig in basis[name]["full_signatures"]
                                  if tuple(json.loads(sig)[-2:]) in target_sectors]
            check(name + "/complete_parent_and_target_sector_order", len(basis[name]["full_signatures"])
                  != declared_models[name]["full_CSFs"] or retained_from_full != signatures)
            expected_blocks = {(row["twoJ"], row["parity"]): row["CSFs_integer_multiplicity"]
                               for row in declared_models[name]["complete_target_blocks"]}
            check(name + "/all_declared_sector_counts", Counter(sectors) != expected_blocks)
            blocks = {}
            for sector in sorted(set(sectors)):
                rows = [i for i, label in enumerate(sectors) if label == sector]
                other = [i for i, label in enumerate(sectors) if label != sector]
                check(name + "/crossH/" + str(sector), np.max(abs(h[np.ix_(rows, other)])),
                      gates["cross_symmetry_zero_Ha"])
                population = np.sum(c[rows]**2, axis=0)
                check(name + "/symmetry_purity/" + str(sector), np.any((population > 1e-9) & (population < 1-1e-9)))
                columns = np.flatnonzero(population > 1-1e-9)
                check(name + "/complete_block/" + str(sector), len(columns) != len(rows))
                blocks[sector] = columns[np.argsort(e[columns], kind="stable")]
            check(name + "/same_five_complete_blocks", set(blocks) != set(uid[:2] for uid in target_uids))
            weights = np.asarray(tracked["weights"], dtype=float)
            check(name + "/five_equal_weights", weights.shape != (n,) or np.count_nonzero(weights) != 5
                  or not np.array_equal(weights[weights > 0], np.repeat(.2, 5)))
            selected, rows_by_label = {}, {row["label"]: row for row in summary["rows"]}
            for group in tracked["tracked"]:
                check(name + "/complete_nondegenerate_span/" + str(group["reference_levels"]),
                      len(group["reference_levels"]) != 1 or len(group["selected_levels"]) != 1)
                uid, selector = tuple(group["reference_levels"][0]), tuple(group["selected_levels"][0])
                check(name + "/declaredUID/" + str(uid), uid not in target_uids or selector[:2] != uid[:2])
                column = int(blocks[selector[:2]][selector[2]])
                selected[uid] = selector
                check(name + "/weight_selector/" + str(uid), weights[column] != .2)
                probability = group["principal_overlap_squared"]
                check(name + "/principal_span_gate/" + str(uid), len(probability) != 1
                      or not (.5 < probability[0] <= 1 + 1e-8))
                same = blocks[selector[:2]][abs(e[blocks[selector[:2]]] - e[column]) < 1e-10]
                check(name + "/degenerate_equalweights/" + str(uid), np.max(abs(weights[same] - weights[column])), 1e-14)
                row = rows_by_label[target_uids[uid]]
                check(name + "/savedUIDcolumn_energy/" + str(uid), row["reference_UID"] != list(uid)
                      or row["selected_level"] != list(selector) or row["column"] != column or row["energy_Ha"] != float(e[column]))
                check(name + "/sigma_notinvented/" + str(uid), row["NIST_uncertainty_cm1"] is not None)
                check(name + "/sameNISTnominal/" + str(uid), row["NIST_nominal_cm1"] != nominal[row["label"]])
            check(name + "/fiveuniqueUIDs", set(selected) != set(target_uids))
            check(name + "/EOL_arithmetic", abs(float(weights @ e) - summary["EOL_reported_Ha"]), 0)
            lower = rows_by_label["6s_1/2"]["energy_Ha"]
            for label, row in rows_by_label.items():
                excitation = (row["energy_Ha"] - lower) * hartree_cm1
                check(name + "/excitation_arithmetic/" + label, abs(excitation - row["excitation_cm1"]), 0)
                check(name + "/gap_arithmetic/" + label, abs(excitation - row["NIST_nominal_cm1"] - row["nominal_gap_cm1"]), 0)
                relative = ((excitation-row["NIST_nominal_cm1"])/row["NIST_nominal_cm1"]
                            if row["NIST_nominal_cm1"] else None)
                check(name + "/relativegap_arithmetic/" + label, relative != row["nominal_relative_gap"])
                if name == "complete5f":
                    old = models["baseline"]["rows"][label]
                    du = row["energy_Ha"] - old["energy_Ha"]
                    dl = lower - models["baseline"]["rows"]["6s_1/2"]["energy_Ha"]
                    check(name + "/upperlower_arithmetic/" + label, row["absolute_delta_upper_Ha"] != du
                          or row["absolute_delta_lower_Ha"] != dl or row["transition_delta_upper_minus_lower_cm1"] != (du-dl)*hartree_cm1)
            for multiplet, akey, bkey, degeneracy in (("5d", "5d_3/2", "5d_5/2", (4,6)),
                                                     ("6p", "6p_1/2", "6p_3/2", (2,4))):
                centroid = sum(g*rows_by_label[k]["excitation_cm1"] for g,k in zip(degeneracy,(akey,bkey),strict=True))/sum(degeneracy)
                split = rows_by_label[bkey]["excitation_cm1"]-rows_by_label[akey]["excitation_cm1"]
                row = next(x for x in summary["multiplets"] if x["label"] == multiplet)
                check(name + "/multiplet/" + multiplet, row["centroid_cm1"] != centroid or row["splitting_cm1"] != split
                      or row["degeneracy_weights"] != list(degeneracy))
                if name == "complete5f":
                    old = next(x for x in models["baseline"]["summary"]["multiplets"] if x["label"] == multiplet)
                    check(name + "/multiplet_delta/" + multiplet, row["delta_centroid_cm1"] != centroid-old["centroid_cm1"]
                          or row["delta_splitting_cm1"] != split-old["splitting_cm1"])
            rms = float(np.sqrt(np.mean([row["nominal_gap_cm1"]**2 for row in summary["rows"]
                                       if row["nominal_relative_gap"] is not None])))
            check(name + "/RMS_arithmetic", rms != summary["nominal_four_gap_RMS_cm1"])
            check(name + "/preserved_failed_scopes", summary["original_stationarity_passed"]
                  or summary["expanded_stationarity_evaluated"] or summary["full_spectrum_certified"]
                  or summary["nominal_four_1percent_goal_passed"])
            if name == "baseline":
                check("literal208_H_E_C_byteidentity", any(not same_bytes(a[k],literal[j])
                      for k,j in (("H","H"),("energies","energies"),("mixing","mixing"))))
                check("literal208_owner_exact", owner != manifest["owner"])
                check("literal208_UIDselectors", selected != {tuple(x["UID"]):tuple(x["selected_level"]) for x in manifest["targets"]})
            models[name] = dict(archive=a, owner=owner, summary=summary, rows=rows_by_label,
                                signatures=signatures, blocks=blocks,
                                all_root_residuals_Ha=residuals.tolist())

        old, new = models["baseline"], models["complete5f"]
        for key in ("geometry", "nuclear", "Z", "ns", "core"):
            check("owner_same_cachedgeometry_nucleus/"+key,
                  old["owner"]["radial_binding"][key] != new["owner"]["radial_binding"][key])
        positions = {sig:i for i,sig in enumerate(new["signatures"])}
        mapping = [positions[sig] for sig in old["signatures"]]
        check("signed_nested1046_mapping", mapping != load(ROOT/inputs["mapping"]["path"])["mapping"])
        check("whole_nested1046_H", np.max(abs(new["archive"]["H"][np.ix_(mapping,mapping)]-old["archive"]["H"])),
              gates["nested_signed_old1046_H_error_Ha"])
        for sector in sorted(old["blocks"]):
            old_e = old["archive"]["energies"][old["blocks"][sector]]
            new_e = new["archive"]["energies"][new["blocks"][sector]][:len(old_e)]
            check("complete_sector_Ritz/"+str(sector), max(0.,float(np.max(new_e-old_e))),
                  gates["nested_ordered_Ritz_eigenvalue_increase_Ha"])
        raw_kappas = []
        for i in range(1,8):
            raw = archive("raw_hS_"+str(i))
            kappa = int(raw["kappa"])
            raw_kappas.append(kappa)
            logical = np.tril(raw["h"])+np.tril(raw["h"],-1).T
            check("cached_raw_hS_exact/"+str(i), not same_bytes(logical,hs[f"h_{kappa}"])
                  or not same_bytes(raw["s"],hs[f"s_{kappa}"]))
        check("all_seven_kappas_once", Counter(raw_kappas) != Counter(projectors.keys()))

        virtual_rows = load(ROOT/inputs["virtual_controls"]["path"])
        for i,key in enumerate(((5,3),(5,-4))):
            op = archive("virtual_operator_"+str(i))
            q = archive("qualified_virtual_"+str(i))
            initial = archive("initial_virtual_"+str(i))
            h,s,b,p = [op[k].astype(np.longdouble) for k in ("h","s","basis","originalP")]
            vectors = q["physical"].astype(np.longdouble)
            check("virtual118_complete/"+str(i), vectors.shape != (246,118))
            check("virtual_original_hSP/"+str(i), not same_bytes(op["h"],hs[f"h_{key[1]}"])
                  or not same_bytes(op["s"],hs[f"s_{key[1]}"])
                  or not same_bytes(op["originalP"],projectors[key[1]]))
            frozen_index = old_keys.index((4,key[1]))
            check("virtual_original4f_byteidentity/"+str(i), not same_bytes(op["frozen"][:,0],
                  np.concatenate((literal["P"][frozen_index],literal["Q"][frozen_index]))))
            check("virtual_basis_exact_originalP_complement/"+str(i), not same_bytes(op["basis"],
                  op["originalP"]@op["complement"]))
            check("whole_complement_SGram/"+str(i), np.max(abs(b.T@s@b-np.eye(118))),gates["screened_full_spectrum_SGram_error"])
            check("whole_complement_frozen4f/"+str(i),np.max(abs(op["frozen"].T.astype(np.longdouble)@s@b)),gates["actual_radial_Gram_error"])
            hv,sv = h@vectors,s@vectors
            rayleigh = np.sum(vectors*hv,axis=0)/np.sum(vectors*sv,axis=0)
            constrained = b.T@(hv-sv*rayleigh)
            absres = np.sqrt(np.sum(constrained**2,axis=0))
            relres = absres/np.maximum(1.,np.maximum(abs(rayleigh),np.sqrt(np.sum((b.T@hv)**2,axis=0))))
            check("virtual_all118_SGram/"+str(i),np.max(abs(vectors.T@sv-np.eye(118))),gates["screened_full_spectrum_SGram_error"])
            check("virtual_all118_constrainedres/"+str(i),np.max(relres),gates["screened_full_spectrum_constrained_residual_relative"])
            check("virtual_all118_actualRayleigh/"+str(i),np.max(abs(rayleigh-q["actual_Rayleigh_Ha"])),1e-12)
            coeff = q["coefficients"].astype(np.longdouble)
            reconstructed = np.asarray(b@coeff,dtype=float)
            reconstructed /= np.asarray(q["normalization_factors"],dtype=float)
            check("virtual_fullcoeff_normalizedphysical_identity/"+str(i),np.max(abs(reconstructed-q["physical"])),1e-12)
            raw = initial["raw_normalized"].astype(np.longdouble)
            raw_rayleigh = np.sum(raw*(h@raw),axis=0)/np.sum(raw*(s@raw),axis=0)
            check("raw_all118_actualRayleigh_record/"+str(i),np.max(abs(raw_rayleigh-initial["raw_Rayleigh_Ha"])),1e-12)
            raw_index = int(np.argmin(raw_rayleigh))
            check("virtual_same_minimum_Rayleigh/"+str(i),abs(rayleigh[0]-np.min(raw_rayleigh)),gates["virtual_original_P_minimum_Ha"])
            gap = rayleigh[1]-rayleigh[0]
            check("virtual_unique_minimum/"+str(i),gap <= gates["virtual_minimum_separation_Ha"])
            overlap = float(vectors[:,0]@s@raw[:,raw_index])**2
            check("virtual_minimum_span/"+str(i),overlap <= gates["virtual_original_P_minimum_span_overlap_squared"])
            coords = p.T@s@vectors[:,0]
            pivot = int(np.argmax(abs(coords)))
            phase = 1. if coords[pivot]>=0 else -1.
            seed = q["phased_seed"]
            check("virtual_only_fixed_phase/"+str(i),float(q["phase"]) != phase or not same_bytes(seed,phase*q["unsigned_seed"]))
            check("virtual_unsignedlowest_fullvector_identity/"+str(i),not same_bytes(q["unsigned_seed"],q["physical"][:,0]))
            j = tuple(map(tuple,new["archive"]["keys"].tolist())).index(key)
            check("virtual_CI_seed_bytes/"+str(i),not same_bytes(seed,np.concatenate((new["archive"]["P"][j],new["archive"]["Q"][j]))))
            row=virtual_rows[i]
            check("virtual_scalar_record/"+str(i),row["key"]!=list(key) or row["phase"]!=phase
                  or row["sign_coordinate_index"]!=pivot or row["whole_spectrum_rows"]!=118)
            check("virtual_record_energy_gap/"+str(i),max(abs(float(rayleigh[0])-row["actual_nuclear_minimum_Ha"]),
                  abs(float(gap)-row["actual_lowest_gap_Ha"])),1e-12)
            spectra.append(dict(key=key,lowest_actual_Rayleigh_Ha=float(rayleigh[0]),gap_Ha=float(gap),
                                all118_absres_Ha=[float(v) for v in absres],all118_relative_residual=[float(v) for v in relres]))

        check("exact_18_once_NPZloads",len(loaded)!=18)
        check("original14_expectedfailure_unchanged",load(ROOT/inputs["original_failure"]["path"])["completed"]["CI_internal_hS"]!=7
              or sha(ROOT/inputs["original_failure"]["path"])!=inputs["original_failure"]["sha256"])
        for role,row in inputs.items():
            if sha(ROOT/row["path"]) != row["sha256"]:
                raise ValueError("Frozen input changed during independent audit: "+role)
        write(output/"verification.json",dict(saved_array_qualification_passed=True,checks_count=len(checks),checks=checks,
              load_roles_once=loaded,NPZ_loads=len(loaded),eigensolves_SVD_Atomx_H_CI_g_Fock_Dirac_optimizer_calls=0,
              original_failed_execution_preserved=True,original_expected14gate_passed=False,
              physics_complete=False,expanded_MCDHF_stationarity_evaluated=False,
              same_point_responses={name:model["summary"] for name,model in models.items()},
              all_root_residuals_Ha={name:model["all_root_residuals_Ha"] for name,model in models.items()},
              complete_virtual_spectra=spectra,contract_sha256=sha(HERE/"contract.json"),driver_sha256=sha(__file__),
              inputs_sha256=sha(HERE/"inputs.json"),root_binding_sha256=sha(binding_path),
              elapsed_uncontrolled_s=time.monotonic()-started,peak_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
              limitations=contract["limitations"]))
    except Exception as error:
        write(output/"failure.json",dict(status="independent_saved_array_qualification_failed_no_retry",
              message=str(error),traceback=traceback.format_exc(),checks=checks,load_roles_once=loaded,NPZ_loads=len(loaded),
              new_scientific_producer_calls=0,original_expected14gate_passed=False,original_failed_execution_preserved=True,
              contract_sha256=sha(HERE/"contract.json"),driver_sha256=sha(__file__),
              elapsed_uncontrolled_s=time.monotonic()-started))
        raise
    finally:
        signal.alarm(0)


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--authorize",action="store_true")
    parser.add_argument("--execute",action="store_true")
    parser.add_argument("--binding",type=Path)
    args=parser.parse_args()
    if args.execute:
        if not args.binding or args.authorize:
            parser.error("Only exact root --execute --binding can read arrays")
        execute(args.binding)
    elif args.authorize:
        if not args.binding:
            parser.error("--authorize requires binding")
        authorize(args.binding)
        print("Exact saved-array authorization PASS;0 arrays/importscience/producers")
    else:
        metadata()
        print("Frozen metadata PASS;0 arrays/importscience/producers")
