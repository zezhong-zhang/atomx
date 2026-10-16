"""Explicit private consumer refactor of the frozen qualification003 chain.

No numerical imports occur at module load. Numerical/operator implementations
remain public Atomx; this module owns only the inherited identity and validation
consumer. The preceding qualification and its driver are never modified.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]


@dataclass(frozen=True)
class PreparedPoint:
    csfs: object
    waves: object
    data: object
    nuclear: object
    functional: object
    saved: object
    gradient: object
    geometry: object
    layout: object
    inverse: object
    originals: object
    raw_hs: object
    targets: object
    reference_targets: object


@dataclass(frozen=True)
class ValidatedTrial:
    point: PreparedPoint
    ci: object
    h: object
    energy: float
    public_origins: object


def prepare_point(contract, raw_hs, gate, close, plain):
    import numpy as np

    import atomx.methods.csf.variational as variational
    import atomx.physics.csf.tracking as tracking
    from atomx.angular.csf_hamiltonian import csf_signature
    from atomx.config.csf import enumerate_jj_csfs
    from atomx.methods.ci import CIResult
    from atomx.numerics.bspline.orbital_optimization import _residual_metric_sha256
    from atomx.physics.csf.state_average import CIStateAverage
    from atomx.potentials.nuclear import make_nuclear_potential

    def load(role):
        path = REPO / contract["inputs"][role]["path"]
        if path.suffix == ".npz":
            with np.load(path, allow_pickle=False) as saved:
                return {k: saved[k].copy() for k in saved.files}
        return json.loads(path.read_text())

    saved, gradient, geometry, radial = (load(k) for k in ("saved_point", "saved_gradient", "screened_geometry", "geometry_workspace"))
    for public, stored in (("radial_gradient", "production_radial"),
                           ("raw_coordinate_gradient", "production_raw_coordinate"),
                           ("tangent_gradient", "production_tangent"),
                           ("preconditioned_gradient", "production_preconditioned")):
        gradient[public] = gradient[stored]
    waves = {tuple(map(int, key)): (p.copy(), q.copy()) for key, p, q in
             zip(saved["keys"], saved["P"], saved["Q"], strict=True)}
    data = {k[6:]: v.item() if v.ndim == 0 else v.copy() for k, v in radial.items() if k.startswith("basis_")}
    models = load("screened_models")["models"]
    old = [c for row in models["baseline"]["configs"] for c in enumerate_jj_csfs(row["config"])]
    full = [c for row in models["balanced_virtual"]["configs"] for c in enumerate_jj_csfs(row["config"])]
    sectors = {(j, p) for j, p, _ in contract["fixed"]["target_blocks"]}
    csfs = [c for c in full if (c.total_J2, c.parity) in sectors]
    gate("complete_dimensions_531_2163_1046_29", (len(old), len(full), len(csfs), len(waves)) == (531, 2163, 1046, 29))
    gate("complete69electrons", {c.n_electrons for c in csfs} == {69})
    for j, p, count in contract["fixed"]["target_blocks"]:
        gate(f"complete_sector_{j}_{p}", sum((c.total_J2, c.parity) == (j, p) for c in csfs) == count)
    signatures = [json.dumps(plain((s.subshells, s.coupling, c.total_J2, c.parity)), separators=(",", ":"))
                  for c in csfs for s in [csf_signature(c)]]
    gate("complete_signed_order", signatures == load("screened_basis")["balanced_virtual"]["retained_signatures"])
    targets = load("tracking_binding")["targets"]
    average = CIStateAverage(tuple((t["twoJ"], t["parity"], t["models"]["DC"]["local_root"], 1.) for t in targets))
    metric = {k: geometry[f"s_{k}"] for k in {key[1] for key in waves}}
    reference = load("tracking_ci")
    old_waves = {tuple(map(int, k)): (p.copy(), q.copy()) for k, p, q in
                 zip(radial["orbital_keys"], radial["P"], radial["Q"], strict=True)}
    tracker = tracking.CIBasisTracking(average, old, CIResult(reference["energies_Ha"], reference["mixing"]),
                                      old_waves, metric, max_determinant_pairs=1_000_000)
    nuclear = make_nuclear_potential(70, "fermi", A=175)
    functional = variational.CSFOrbitalFunctional(csfs, data, 70, nuclear, state_average=average,
        level_tracking=tracker, retain_hamiltonian=True, max_determinants=2000)
    gate("exact29radialkey_order", functional.keys == tuple(map(tuple, saved["keys"])))
    originals = {}
    for k, h, s in raw_hs:
        close(f"original_Lh_{k}", functional.one_body[k], np.tril(h) + np.tril(h, -1).T)
        close(f"saved_Lh_{k}", functional.one_body[k], geometry[f"h_{k}"])
        gate(f"saved_Lh_bytes_{k}", functional.one_body[k].tobytes() == geometry[f"h_{k}"].tobytes())
        close(f"saved_metric_{k}", s, metric[k])
        gate(f"saved_metric_bytes_{k}", s.dtype == metric[k].dtype and s.tobytes() == metric[k].tobytes())
        originals[k] = (h, s)
    rows = load("saved_layout")
    layout = [(tuple(map(tuple, row["keys"])), gradient[f"projector_{i}"], tuple(row["shape"]),
               slice(*row["slice"])) for i, row in enumerate(rows)]
    inverse = [gradient[f"inverse_blocks_{i}"] for i in range(7)]
    gate("original3451_metric_bytes", _residual_metric_sha256(layout, gradient["inverse_diagonal"], inverse)
         == contract["fixed"]["residual_metric_sha256"])
    close("exact3451coordinate_shape", gradient["x"].size, 3451)
    for i, (keys, projector, shape, where) in enumerate(layout):
        y = gradient["x"][where].reshape(shape)
        coefficients = projector @ y
        for col, key in enumerate(keys):
            close(f"original_coordinate_wave_{key}", np.concatenate(waves[key]), coefficients[:, col])
        close(f"coordinate_Gram_{i}", y.T @ y, np.eye(shape[1]), 1e-10)
        c = np.column_stack([np.concatenate(waves[key]) for key in keys])
        s = metric[keys[0][1]]
        close(f"radial_Gram_{i}", c.T @ s @ c, np.eye(shape[1]), 1e-10)
        outside = c - projector @ (projector.T @ s @ c)
        close(f"no_pair_leakage_{i}", np.sum(outside * (s @ outside), axis=0), 0., 1e-12)
    reference_targets = load("corrected_baseline_result")["targets"]
    return PreparedPoint(csfs, waves, data, nuclear, functional, saved, gradient,
                         geometry, layout, inverse, originals, raw_hs, targets, reference_targets)


def validate_trial(point, energy, ci, centered_calls, gate, close):
    import numpy as np

    import atomx.methods.ci as ci_module

    csfs, waves, functional = point.csfs, point.waves, point.functional
    data, nuclear = point.data, point.nuclear
    h = ci.hamiltonian_for(csfs, waves, functional.keys, data, 70, 123,
                          nuclear_potential=nuclear, max_determinants=2000)
    groups = ci_module._group_csfs_by_J_parity(csfs)
    gate("complete_five_centered_public_blocks", len(groups) == len(centered_calls) == 5)
    public_origins = [{"sector": sector, **call} for (sector, _), call in zip(groups, centered_calls, strict=True)]
    close("actual_provenance_centered_origins", [call["origin_Ha"] for call in centered_calls],
          ci.provenance["dense_centering_shifts_Ha"])
    close("whole_H_exact_symmetric", h, h.T)
    close("full_CI_column_norm_residual", np.max(np.linalg.norm(h @ ci.mixing_coefficients
          - ci.mixing_coefficients * ci.state_energies, axis=0)), 0., 1e-8)
    close("full_CI_Gram", ci.mixing_coefficients.T @ ci.mixing_coefficients, np.eye(1046), 1e-10)
    gate("read_only_owned_H", not h.flags.writeable)
    mutated = {k: (p.copy(), q.copy()) for k, (p, q) in waves.items()}
    mutated[functional.keys[0]][0][0] = np.nextafter(mutated[functional.keys[0]][0][0], np.inf)
    try:
        ci.hamiltonian_for(csfs, mutated, functional.keys, data, 70, 123,
                           nuclear_potential=nuclear, max_determinants=2000)
    except ValueError:
        pass
    else:
        raise ValueError("Stale physical point accepted")
    gate("stale_owner_rejected", True)
    return ValidatedTrial(point, ci, h, energy, public_origins)


def finish_checked_chain(point, trial, gradient_result, phase, contract,
                         output, write, gate, close):
    import numpy as np

    from atomx.physics.csf.levels import ci_level_blocks

    functional, csfs, saved, gradient = point.functional, point.csfs, point.saved, point.gradient
    layout, inverse = point.layout, point.inverse
    raw_hs, originals = point.raw_hs, point.originals
    targets, reference_targets = point.targets, point.reference_targets
    ci, h, energy = trial.ci, trial.h, trial.energy
    energy2, radial_g, returned = gradient_result
    validation_start = time.monotonic()
    gate("same_trial_CI_object", returned is ci)
    close("same_reported_objective", energy2, energy)
    raw = np.zeros(3451)
    for keys, projector, _shape, where in layout:
        raw[where] = (projector.T @ np.column_stack([radial_g[k] for k in keys])).ravel()

    def tangent(vector):
        answer = vector.copy()
        for _, _, shape, where in layout:
            y, g = gradient["x"][where].reshape(shape), vector[where].reshape(shape)
            a = y.T @ g
            answer[where] = (g - .5 * y @ (a + a.T)).ravel()
        return answer

    g = tangent(raw)
    inverse_g = g.copy()
    for (_, _, shape, where), blocks in zip(layout, inverse, strict=True):
        a = g[where].reshape(shape)
        inverse_g[where] = np.column_stack([block @ a[:, j] for j, block in enumerate(blocks)]).ravel()
    pg = tangent(inverse_g)
    close("saved_complete_radial_gradient", np.array([radial_g[k] for k in functional.keys]), gradient["radial_gradient"], 1e-9)
    close("saved_original_raw_g", raw, gradient["raw_coordinate_gradient"], 1e-9)
    close("saved_original_tangent_g", g, gradient["tangent_gradient"], 1e-9)
    close("saved_original_metric_pg", pg, gradient["preconditioned_gradient"], 1e-10)
    weights = functional.last_state_weights
    gate("original_equal5weights", np.count_nonzero(weights) == 5 and np.array_equal(weights[weights > 0], np.repeat(.2, 5)))
    _, blocks = ci_level_blocks(csfs, ci, require_full=True)
    write(output / "sector_columns.json", [{"sector": sector, "columns": columns,
          "CSF_rows": [i for i, c in enumerate(csfs) if (c.total_J2, c.parity) == sector]}
          for sector, columns in blocks.items()])
    write(output / "signed_density.json", [[key, float(value)] for key, value in functional.last_weights.items()])
    bindings = {}
    for row in functional.last_tracking_diagnostics:
        gate("rank_one_binding", len(row["reference_levels"]) == len(row["selected_levels"]) == 1)
        uid = tuple(row["reference_levels"][0])
        gate("unique_UID_" + str(uid), uid not in bindings and tuple(row["selected_levels"][0][:2]) == uid[:2])
        bindings[uid] = row
    target_rows = []
    for target in targets:
        uid = target["twoJ"], target["parity"], target["models"]["DC"]["local_root"]
        selected = bindings[uid]["selected_levels"][0]
        column = int(blocks[tuple(selected[:2])][selected[2]])
        old_target = next(row for row in reference_targets if row.get("label") == target["label"])
        gate("same_original_target_" + target["label"], column == old_target["column"])
        target_rows.append({"label": target["label"], "UID": uid, "column": column,
                            "energy_Ha": float(ci.state_energies[column]), "tracking": bindings[uid]})
    gate("exact_five_UID_support", len(bindings) == len(target_rows) == 5)
    for k, raw_h, s in raw_hs[7:]:
        close(f"same_raw_h_{k}", raw_h, originals[k][0])
        close(f"same_S_{k}", s, originals[k][1])
    h_limit = 0. if phase == "ordinary" else contract["gates"]["candidate_H_entry_Ha"]
    close("saved_full_H", h, saved["H"], h_limit)
    close("historical_unshifted_full_roots_source_evolution", ci.state_energies, saved["energies"],
          contract["gates"]["candidate_all_roots_Ha"])
    if phase == "ordinary":
        gate("saved_H_bytes", h.dtype == saved["H"].dtype and h.tobytes() == saved["H"].tobytes())
    write(output / "targets_metric.json", {"targets": target_rows,
          "residual_metric_sha256": contract["fixed"]["residual_metric_sha256"],
          "residual": float(np.linalg.norm(pg)), "indicator_Ha": float(.5 * abs(g @ pg)),
          "stationarity_passed": bool(np.linalg.norm(pg) <= 1e-7),
          "indicator_passed": bool(.5 * abs(g @ pg) <= 1e-8),
          "original_physical_failures_retained": True, "reported_EO_Ha": energy})
    write(output / "centered_origins.json", trial.public_origins)
    owner = output / "owner.json"
    write(owner, ci.hamiltonian_snapshot.identity_payload())
    validation_seconds = time.monotonic() - validation_start
    start = time.monotonic()
    arrays = output / "arrays.npz"
    values = {"H": h, "energies": ci.state_energies, "mixing": ci.mixing_coefficients,
              "radial_gradient": np.array([radial_g[k] for k in functional.keys]),
              "raw_g": raw, "g": g, "pg": pg, "weights": weights, "x": gradient["x"],
              "keys": np.array(functional.keys), "P": saved["P"], "Q": saved["Q"]}
    with arrays.open("wb") as handle:
        np.savez_compressed(handle, **values)
        handle.flush()
        os.fsync(handle.fileno())
    with np.load(arrays, allow_pickle=False) as stored:
        for name, value in values.items():
            close("archive_exact_" + name, stored[name], value)
    return {"directory": str(output), "arrays": str(arrays), "owner": str(owner),
            "validation_metadata_seconds": validation_seconds,
            "checked_compressed_archive_seconds": time.monotonic() - start}


def compare_saved_chains(contract, ordinary_paths, candidate_paths, check, *, same_worker=False):
    import numpy as np

    def load(directory, name):
        return json.loads((directory / name).read_text())

    path0, path1 = Path(ordinary_paths), Path(candidate_paths)
    with np.load(path0 / "arrays.npz", allow_pickle=False) as ar:
        ordinary = {k: ar[k].copy() for k in ar.files}
    with np.load(path1 / "arrays.npz", allow_pickle=False) as ar:
        candidate = {k: ar[k].copy() for k in ar.files}
    for name, limit in (("H", 1e-9), ("energies", 1e-8), ("radial_gradient", 1e-9),
                        ("raw_g", 1e-9), ("g", 1e-9), ("pg", 1e-10), ("weights", 0.),
                        ("x", 0.), ("keys", 0.), ("P", 0.), ("Q", 0.)):
        check("whole_" + name, np.max(abs(candidate[name] - ordinary[name])), limit)
    if same_worker:
        for name in ("H", "energies"):
            check("owncold_bytes_" + name,
                  ordinary[name].dtype != candidate[name].dtype
                  or ordinary[name].shape != candidate[name].shape
                  or ordinary[name].tobytes() != candidate[name].tobytes(), 0.)
    energies0, energies1 = ordinary["energies"], candidate["energies"]
    blocks = []
    targets = load(path0, "targets_metric.json")["targets"]
    cols = load(path0, "sector_columns.json")
    for row in cols:
        columns = row["columns"]
        csf_rows = row["CSF_rows"]
        outside = sorted(set(range(1046)) - set(csf_rows))
        for label, values in (("ordinary", ordinary), ("candidate", candidate)):
            check(label + "_complete_sector_exact_zeros_" + str(row["sector"]),
                  np.max(abs(values["mixing"][np.ix_(outside, columns)]), initial=0.), 0.)
        start = 0
        for stop in range(1, len(columns) + 1):
            if (stop == len(columns) or
                    min(abs(energies0[columns[stop]] - energies0[columns[stop-1]]),
                        abs(energies1[columns[stop]] - energies1[columns[stop-1]])) > contract["gates"]["cluster_gap_Ha"]):
                cluster = columns[start:stop]
                c0, c1 = ordinary["mixing"][np.ix_(csf_rows, cluster)], candidate["mixing"][np.ix_(csf_rows, cluster)]
                error = np.max(abs(c0 @ c0.T - c1 @ c1.T))
                check("signed_cluster_" + str(cluster), error, contract["gates"]["signed_cluster_span"])
                blocks.append(cluster)
                start = stop
    reference_partition = json.loads((REPO / contract["inputs"]["common_shift_result"]["path"]).read_text())
    check("complete_unchanged_original_cluster_partition",
          blocks != [row["columns"] for row in reference_partition["shifted_complete_cluster_spans"]], 0.)
    density0 = load(path0, "signed_density.json")
    density1 = load(path1, "signed_density.json")
    terms0 = {json.dumps(row[0]): row[1] for row in density0}
    terms1 = {json.dumps(row[0]): row[1] for row in density1}
    check("complete_same_signed_RDM_support", set(terms0) != set(terms1), 0.)
    check("complete_signed_RDM_coefficients", max(abs(terms0[k] - terms1[k]) for k in terms0), 1e-9)
    check("same_complete_sector_columns", load(path0, "sector_columns.json") != load(path1, "sector_columns.json"), 0.)
    check("complete_same_physical_owner", load(path0, "owner.json") != load(path1, "owner.json"), 0.)
    current = load(path1, "targets_metric.json")["targets"]
    check("same5targetUIDbindings",
          [(r["label"], r["UID"], r["column"]) for r in targets] !=
          [(r["label"], r["UID"], r["column"]) for r in current], 0.)
    origins0 = load(path0, "centered_origins.json")
    origins1 = load(path1, "centered_origins.json")
    check("same_per_sector_nearest_integer_centering", [(r["sector"], r["origin_Ha"]) for r in origins0] !=
          [(r["sector"], r["origin_Ha"]) for r in origins1], 0.)
    return blocks
