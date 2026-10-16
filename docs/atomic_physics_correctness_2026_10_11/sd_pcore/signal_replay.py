"""Separate saved full affine action from interaction-only Born action."""

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

OUT = Path(__file__).resolve().parent
sys.path[:0] = [str(OUT / "source" / name) for name in ("src", "tests")]

from test_monovalent_sd import elementary  # noqa: E402

from atomx.methods.monovalent_sd import _core_terms  # noqa: E402


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    result = json.loads((OUT / "result.json").read_text())
    admission = json.loads((OUT / "native_admission.json").read_text())
    coordinates = json.loads((OUT / "coordinates.json").read_text())
    assert digest(OUT / "arrays.npz") == result["arrays_sha256"]
    a = np.load(OUT / "arrays.npz")
    H, T, qsd, gaps = a["H_Ha"], a["T_first_Born"], a["qsd"], a["gaps_Ha"]
    ref, mask = coordinates["reference_index"], coordinates["reference_mask"]
    source, native = a["source_fixedN_Ha"], a["native_encoded_action_Ha"]
    commutator = H @ T[:, ref] - T @ H[:, ref]
    diagonal_Fock_commutator = -gaps * T[:, ref]
    residual_interaction_commutator = commutator - diagonal_Fock_commutator
    # Numerical D*tborn need not equal source bit-for-bit. Apply the exact
    # represented correction to distinguish K-D from K*tborn-source.
    plan = admission["plan"]
    for key in ("occupied_keys", "virtual_keys", "groups"):
        plan[key] = tuple(tuple(tuple(x) if isinstance(x, list) else x for x in group) for group in plan[key])
    labels = tuple(map(tuple, coordinates["labels"]))
    representation = SimpleNamespace(plan=plan, integrals=SimpleNamespace(labels=labels, noccupied=10))
    delta_rho = a["source_native_rho_Ha"] - a["denominators_native_Ha"] * a["native_first_Born_rho"]
    delta = np.zeros(len(source))
    positions = {int(m): i for i, m in enumerate(a["masks"])}
    for coefficient, operations in _core_terms(representation, delta_rho):
        target, sign = elementary(mask, operations)
        if sign:
            delta[positions[target]] += coefficient * sign
    native_interaction_only = native + delta
    native_full_homogeneous = native - a["native_encoded_source_Ha"]
    full_affine = source + commutator
    magnitudes = dict(
        source_max_Ha=float(np.max(abs(source[qsd]))),
        first_Born_wave_max=float(np.max(abs(T[qsd, ref]))),
        full_homogeneous_commutator_max_Ha=float(np.max(abs(commutator[qsd]))),
        diagonal_Fock_commutator_max_Ha=float(np.max(abs(diagonal_Fock_commutator[qsd]))),
        interaction_only_commutator_max_Ha=float(np.max(abs(residual_interaction_commutator[qsd]))),
        full_affine_action_max_Ha=float(np.max(abs(full_affine[qsd]))),
        native_source_D_Born_roundoff_max_Ha=float(np.max(abs(delta_rho))),
        lifted_D_Born_roundoff_max_Ha=float(np.max(abs(delta))),
    )
    errors = dict(
        full_affine_action_error_Ha=float(np.max(abs((full_affine + native)[qsd]))),
        full_homogeneous_action_error_Ha=float(np.max(abs((commutator + native_full_homogeneous)[qsd]))),
        interaction_only_action_error_Ha=float(np.max(abs((residual_interaction_commutator + native_interaction_only)[qsd]))),
    )
    ratios = dict(
        full_affine_relative_error=errors["full_affine_action_error_Ha"] / magnitudes["full_affine_action_max_Ha"],
        interaction_only_relative_error=errors["interaction_only_action_error_Ha"] / magnitudes["interaction_only_commutator_max_Ha"],
        source_relative_error=result["errors"]["source_signed_max_error_Ha"] / magnitudes["source_max_Ha"],
        first_Born_relative_error=result["errors"]["first_order_signed_max_error"] / magnitudes["first_Born_wave_max"],
        MBPT2_relative_error=result["errors"]["core_MBPT2_energy_error_Ha"] / abs(result["expected_core_MBPT2_Ha"]),
    )
    roundoff = float(np.finfo(float).eps * np.linalg.norm(H, ord=np.inf))
    branch_signals = {}
    for i, m in enumerate(a["masks"]):
        holes = [bit for bit in range(10) if not int(m) & (1 << bit)]
        if not 0 < len(holes) <= 2:
            continue
        particles = [bit for bit in range(10, 14) if int(m) & (1 << bit)]
        name = str((tuple(sorted(labels[h][1] for h in holes)), tuple(sorted(labels[p][1] for p in particles))))
        row = branch_signals.setdefault(name, dict(source_max_Ha=0.0, interaction_action_max_Ha=0.0, occupied_p=any(labels[h][1] in (1, -2) for h in holes), mixed_j_holes=len({abs(labels[h][1]) for h in holes}) > 1, complete_components=0))
        row["source_max_Ha"] = max(row["source_max_Ha"], abs(float(source[i])))
        row["interaction_action_max_Ha"] = max(row["interaction_action_max_Ha"], abs(float(residual_interaction_commutator[i])))
        row["complete_components"] += 1
    for name, row in branch_signals.items():
        row["source_above_H_roundoff_scale"] = row["source_max_Ha"] > roundoff
        row["MBPT2_Ha"] = result["topology"][name]["MBPT2_Ha"]
    summary = dict(
        pass_=max(errors.values()) < 1e-11, saved_array_replay_only=True,
        new_orbital_or_amplitude_solves=0, native_API_calls=0, representation_only_view_not_a_CoreSD_result=True,
        magnitudes=magnitudes, errors=errors, relative_errors=ratios,
        H_operator_roundoff_scale_Ha=roundoff, branch_signals=branch_signals,
        interpretation="Full affine error includes source/diagonal cancellation; interaction-only K-D removes diagonal Fock using exact saved Born direction plus lifted floating division correction. No vector components discarded.",
        full_arbitrary_rho_domain_derived=False, scope="Saved zero/Born coefficient topology only",
        inputs_sha256={name: digest(OUT / name) for name in ("arrays.npz", "result.json", "coordinates.json", "native_admission.json", "source/src/atomx/methods/monovalent_sd.py", "source/tests/test_monovalent_sd.py")},
        signal_replay_sha256=digest(Path(__file__)),
    )
    np.savez_compressed(OUT / "signal_arrays.npz", full_homogeneous_commutator_Ha=commutator, diagonal_Fock_commutator_Ha=diagonal_Fock_commutator, residual_interaction_commutator_Ha=residual_interaction_commutator, native_interaction_only_Ha=native_interaction_only, lifted_roundoff_correction_Ha=delta)
    summary["signal_arrays_sha256"] = digest(OUT / "signal_arrays.npz")
    (OUT / "signal.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(dict(magnitudes=magnitudes, errors=errors, relative_errors=ratios), indent=2))


if __name__ == "__main__":
    main()
