"""Read-only first-Born evidence replay and source/array/link bindings."""

import hashlib
import itertools
import json
import re
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent
REPO = OUT.parents[2]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    contract = json.loads((OUT / "contract.json").read_text())
    result = json.loads((OUT / "result.json").read_text())
    assert digest(OUT / "contract.json") == result["source_contract_sha256"]
    assert digest(OUT / "probe.py") == contract["probe_sha256"] == result["probe_sha256"]
    assert digest(OUT / "arrays.npz") == result["arrays_sha256"]
    current_drift = []
    for path, sha in contract["source_sha256"].items():
        assert digest(OUT / "source" / path) == sha, path
        if digest(REPO / path) != sha:
            current_drift.append(path)
    coordinates = json.loads((OUT / "coordinates.json").read_text())
    labels = coordinates["labels"]
    for n, k in contract["fixed"]["occupied_keys"] + contract["fixed"]["virtual_keys"]:
        assert [m for nr, ka, m in labels if (nr, ka) == (n, k)] == list(range(-2 * abs(k) + 1, 2 * abs(k), 2))
    a = np.load(OUT / "arrays.npz")
    masks = np.array([sum(1 << i for i in row) for row in itertools.combinations(range(14), 10)], dtype=np.uint64)
    assert np.array_equal(a["masks"], masks)
    H, T, qsd = a["H_Ha"], a["T_first_Born"], a["qsd"]
    ref = coordinates["reference_index"]
    eps = a["fock_Ha"].diagonal().real
    e0 = sum(eps[:10])
    gaps = np.array([e0 - sum(eps[i] for i in range(14) if int(mask) & (1 << i)) for mask in masks])
    assert np.array_equal(gaps, a["gaps_Ha"])
    source = H[:, ref].copy()
    source[ref] = 0
    wave = np.divide(source, gaps, out=np.zeros(len(masks)), where=abs(gaps) >= 1e-8)
    action = H[:, ref] + H @ T[:, ref] - T @ H[:, ref]
    metrics = dict(
        first_order_signed_max_error=float(np.max(abs(T[:, ref] - wave))),
        source_signed_max_error_Ha=float(np.max(abs(source[qsd] + a["native_encoded_source_Ha"][qsd]))),
        first_Born_projected_commutator_action_error_Ha=float(np.max(abs(action[qsd] + a["native_encoded_action_Ha"][qsd]))),
        independent_MBPT2_Ha=float(H[ref] @ wave),
    )
    for key in metrics:
        expected = result["expected_core_MBPT2_Ha"] if key == "independent_MBPT2_Ha" else result["errors"][key]
        assert metrics[key] == expected, (key, metrics[key], expected)
    assert np.array_equal(wave, a["expected_first_order_wave"])
    assert np.array_equal(action, a["expected_commutator_action_Ha"])
    assert all(result["errors"][key] < contract["gates"][key] for key in result["errors"])
    links = re.findall(r"\[[^\]]+\]\(([^)\n]+)\)", (OUT / "README.md").read_text())
    assert all((OUT / link).exists() for link in links)
    verification = dict(
        pass_=True, replay_only=True, new_orbital_or_amplitude_solves=0,
        frozen_source_hash_count=len(contract["source_sha256"]), current_source_drift=current_drift,
        full_magnetic_multiplets_verified=True, complete_fixed_N_basis_verified=True,
        QSD_signed_components=int(qsd.sum()), reconstructed_metrics=metrics,
        link_count=len(links), missing_links=[], production_code_mutations=0,
        evidence_sha256={name: digest(OUT / name) for name in ("contract.json", "probe.py", "result.json", "arrays.npz", "coordinates.json", "README.md", "verify.py", "native_admission.json", "radial_status.json", "probe.log")},
        scope="Integrity and saved first-Born equation replay; no new complete-domain certificate",
    )
    (OUT / "verification.json").write_text(json.dumps(verification, indent=2) + "\n")
    print(json.dumps(verification, indent=2))


if __name__ == "__main__":
    main()
