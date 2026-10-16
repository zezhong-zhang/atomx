"""Fixed occupied-p/mixed-j core first-Born and projected-action oracle."""

import hashlib
import importlib.util
import itertools
import json
import os
import resource
import sys
import time
import traceback
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent
SOURCE = OUT / "source"
sys.path[:0] = [str(SOURCE / name) for name in ("src", "tests", "tools")]

from test_monovalent_sd import full_operator  # noqa: E402

from atomx.methods.core_sd import CoreSD, plan_core_sd  # noqa: E402
from atomx.methods.dirac_fock import DFSolver, Subshell  # noqa: E402
from atomx.methods.mbpt import build_virtual_basis  # noqa: E402
from atomx.methods.monovalent_sd import _core_terms  # noqa: E402
from atomx.numerics.bspline.coulomb_features import (  # noqa: E402
    PolynomialCellCoulombFeatures,
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(name, values):
    (OUT / name).write_text(json.dumps(values, indent=2) + "\n")


def independent_H_terms(integrals):
    """All actual one/two-body terms; no native SD source/RHS."""
    n = len(integrals.labels)
    terms = [(float(integrals.frozen_energy_Ha), ())]
    for p, q in itertools.product(range(n), repeat=2):
        value = float(integrals.one_body_Ha[p, q].real)
        if value:
            terms.append((value, ((p, True), (q, False))))
    for p, q in itertools.combinations(range(n), 2):
        for r, s in itertools.combinations(range(n), 2):
            value = float(integrals.eri(p, q, r, s).real)
            if value:
                terms.append((value, ((p, True), (q, True), (s, False), (r, False))))
    return tuple(terms)


def main():
    start = time.perf_counter()
    contract = json.loads((OUT / "contract.json").read_text())
    assert digest(__file__) == contract["probe_sha256"]
    caps = contract["resource_caps"]
    resource.setrlimit(resource.RLIMIT_AS, (caps["address_space_bytes"],) * 2)
    assert all(digest(SOURCE / p) == h for p, h in contract["source_sha256"].items())
    for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "NUMBA_NUM_THREADS", "MKL_NUM_THREADS"):
        assert os.environ[key] == "1", key
    save("probe_identity.json", dict(sha256=digest(__file__), load=os.getloadavg()))
    fixed = contract["fixed"]
    occupied = tuple(map(tuple, fixed["occupied_keys"]))
    virtual = tuple(map(tuple, fixed["virtual_keys"]))
    bounds = contract["preallocation_bound"]
    assert bounds["H_elementary_action_bound"] + bounds["T_elementary_action_bound"] < caps["elementary_action_bound"]
    assert bounds["H_and_T_dense_bytes"] * 2 < caps["independent_materialized_payload_bytes"]
    save("initial_admission.json", dict(bounds, pass_=True, no_orbital_solve_yet=True))
    radial = DFSolver(
        Z=fixed["Z"], ns=fixed["ns"], ks=fixed["ks"],
        subshells=[Subshell(n, k, 2 * abs(k)) for n, k in occupied],
    ).solve()
    save("radial_status.json", dict(converged=bool(radial.converged), iterations=radial.iterations, residual=radial.residual, energy_Ha=radial.total_energy))
    assert radial.converged, "fixed radial solve did not converge"
    orbitals = {key: radial.orbitals_by_subshell()[key][:2] for key in occupied}
    virtual_energies = {}
    for key in virtual:
        _, states = build_virtual_basis(
            radial, key[1], e_max=50, potential_mode="VN", orthogonal_to_occupied=True,
        )
        assert states, (key, "no declared virtual orbital")
        orbitals[key] = states[0][:2]
        virtual_energies[str(key)] = states[0][2]
    features = PolynomialCellCoulombFeatures(radial._bsp_data)
    plan = plan_core_sd(occupied, virtual, features.identity)
    save("native_admission.json", dict(plan=plan, feature_identity=dict(features.identity), virtual_energies_Ha=virtual_energies, native_payload_cap_bytes=caps["native_payload_bytes"]))
    assert plan["complete_payload_bound_bytes"] < caps["native_payload_bytes"]
    spec = importlib.util.spec_from_file_location("pcore_builder", SOURCE / "tools/build_pci_sd_backend.py")
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    manifest = builder.build(
        os.environ["ATOMX_PCI_SOURCE"], OUT / "backend", plan["dimensions"],
        os.environ["ATOMX_PCI_COMPILER"], runtime_directory=os.environ["ATOMX_PCI_RUNTIME"],
    )
    core = CoreSD(
        orbitals, radial._bsp_data, fixed["Z"], fixed["ns"],
        occupied_keys=occupied, virtual_keys=virtual, backend_directory=OUT / "backend",
        nuclear_potential=radial._nuclear_potential, radial_representation="polynomial",
        memory_bytes=caps["native_payload_bytes"],
    )
    print("native source prepared; independent complete fixed-N H next", flush=True)
    integrals = core.integrals
    n, electrons = len(integrals.labels), integrals.noccupied
    assert n == fixed["spinors"] and electrons == fixed["core_electrons"]
    masks, H = full_operator(n, electrons, independent_H_terms(integrals))
    reference = (1 << electrons) - 1
    ref_index = masks.index(reference)
    qsd = np.array([0 < (reference & ~mask).bit_count() <= 2 for mask in masks])
    eps = integrals.fock_Ha.diagonal().real
    energies = np.array([sum(eps[i] for i in range(n) if mask & (1 << i)) for mask in masks])
    gaps = energies[ref_index] - energies
    source = H[:, ref_index].copy()
    source[ref_index] = 0
    degenerate = (abs(gaps) < contract["gates"]["zero_source_diagnostic_division_threshold_Ha"]) & (np.arange(len(masks)) != ref_index)
    degenerate_source = float(np.max(abs(source[degenerate]), initial=0))
    expected_wave = np.divide(source, gaps, out=np.zeros(len(masks)), where=abs(gaps) >= 1e-8)
    born = core.source / core.denominators
    _, T = full_operator(n, electrons, _core_terms(core, born))
    _, native_zero = full_operator(n, electrons, _core_terms(core, -core.source))
    native_action_rho = core.action(born) - core.source
    _, native_action = full_operator(n, electrons, _core_terms(core, native_action_rho))
    expected_action = H[:, ref_index] + H @ T[:, ref_index] - T @ H[:, ref_index]
    expected_energy = float(H[ref_index] @ expected_wave)
    native_energy = core.energy(born)
    errors = dict(
        first_order_signed_max_error=float(np.max(abs(T[:, ref_index] - expected_wave))),
        core_MBPT2_energy_error_Ha=abs(native_energy - expected_energy),
        source_signed_max_error_Ha=float(np.max(abs(source[qsd] + native_zero[qsd, ref_index]))),
        first_Born_projected_commutator_action_error_Ha=float(np.max(abs(expected_action[qsd] + native_action[qsd, ref_index]))),
        native_core_energy_consumer_error_Ha=abs(native_energy - float(H[ref_index] @ T[:, ref_index])),
        Hamiltonian_Hermiticity_max_error_Ha=float(np.max(abs(H - H.T))),
        coupled_degenerate_source_Ha=degenerate_source,
    )
    topology = {}
    labels = integrals.labels
    for i, mask in enumerate(masks):
        holes = [bit for bit in range(electrons) if not mask & (1 << bit)]
        if not 0 < len(holes) <= 2:
            continue
        particles = [bit for bit in range(electrons, n) if mask & (1 << bit)]
        name = str((tuple(sorted(labels[h][1] for h in holes)), tuple(sorted(labels[p][1] for p in particles))))
        row = topology.setdefault(name, dict(complete_component_count=0, nonzero_source_components=0, source_norm_squared_Ha2=0.0, MBPT2_Ha=0.0, max_first_order_signed_error=0.0, max_projected_action_error_Ha=0.0))
        row["complete_component_count"] += 1
        row["nonzero_source_components"] += int(source[i] != 0)
        row["source_norm_squared_Ha2"] += float(source[i] ** 2)
        row["MBPT2_Ha"] += float(H[ref_index, i] * expected_wave[i])
        row["max_first_order_signed_error"] = max(row["max_first_order_signed_error"], abs(float(T[i, ref_index] - expected_wave[i])))
        row["max_projected_action_error_Ha"] = max(row["max_projected_action_error_Ha"], abs(float(expected_action[i] + native_action[i, ref_index])))
    np.savez_compressed(
        OUT / "arrays.npz", H_Ha=H, T_first_Born=T, masks=np.array(masks, dtype=np.uint64), qsd=qsd,
        source_fixedN_Ha=source, source_native_rho_Ha=core.source, denominators_native_Ha=core.denominators,
        native_first_Born_rho=born, expected_first_order_wave=expected_wave,
        expected_commutator_action_Ha=expected_action, native_encoded_action_Ha=native_action[:, ref_index],
        native_encoded_source_Ha=native_zero[:, ref_index], one_body_Ha=integrals.one_body_Ha,
        fock_Ha=integrals.fock_Ha, gaps_Ha=gaps,
        **{f"orbital_{nr}_{k}_{part}": orbitals[nr, k][part] for nr, k in occupied + virtual for part in (0, 1)},
    )
    save("coordinates.json", dict(labels=labels, occupied_keys=occupied, virtual_keys=virtual, raw_channels=plan["channels"], reference_index=ref_index, reference_mask=reference, topology=topology))
    source_unchanged = all(digest(SOURCE / p) == h for p, h in contract["source_sha256"].items())
    passed = all(errors[key] < contract["gates"][key] for key in errors) and source_unchanged
    result = dict(
        pass_=passed, errors=errors, expected_core_MBPT2_Ha=expected_energy, native_core_MBPT2_Ha=native_energy,
        spinors=n, core_electrons=electrons, core_determinants=len(masks), raw_rho_amplitudes=len(born),
        scalar_physical_plan_amplitudes=plan["independent_scalar_amplitudes"], topology=topology,
        source_unchanged=source_unchanged, source_contract_sha256=digest(OUT / "contract.json"),
        probe_sha256=digest(__file__), arrays_sha256=digest(OUT / "arrays.npz"), backend_manifest=manifest,
        scope=contract["scope"], full_arbitrary_rho_domain_derived=False, production_certificate=False,
        source_or_consumer_mutation=False, skip_count=0, wall_seconds=time.perf_counter() - start,
        peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    )
    save("result.json", result)
    print(json.dumps({k: v for k, v in result.items() if k in ("pass_", "errors", "wall_seconds", "peak_RSS_KiB")}), flush=True)
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        save("failure.json", dict(type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc(), scientific_gate_pass=False, no_skip=True))
        raise
