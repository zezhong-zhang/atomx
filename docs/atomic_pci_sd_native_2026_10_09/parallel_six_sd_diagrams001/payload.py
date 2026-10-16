"""Fixed complete signed Hamiltonian and physical density payload creation."""

import json
from pathlib import Path

import numpy as np
from test_sd_pair_diagrams import _phase_change, fixed_integrals, fixed_native_model

from atomx.numerics.bspline.coulomb_features import CellCoulombFeatures
from atomx.physics.csf.spinor_integrals import SpinorIntegrals


def create(path):
    path = Path(path)
    if path.exists():
        raise FileExistsError("fixed scientific input is never overwritten")
    model = fixed_native_model()
    dc = fixed_integrals(model)
    energies = {x[:2]: dc.fock_Ha[i, i].real for i, x in enumerate(dc.labels)}
    arrays = dict(
        amplitudes_basis=np.eye(10),
        amplitudes_signed=np.arange(1, 11) * (0.013 + 0.021j),
    )
    identities = {}
    for mode in ("dc", "static", "dynamic", "complex_phase"):
        a = (
            _phase_change(dc)
            if mode == "complex_phase"
            else fixed_integrals(model, mode, energies)
        )
        for name in ("one_body_Ha", "fock_Ha", "packed_eri_Ha"):
            arrays[f"{mode}_{name}"] = getattr(a, name)
        arrays[f"{mode}_labels"] = np.asarray(a.labels, dtype=np.int64)
        arrays[f"{mode}_constants"] = np.asarray(
            [a.noccupied, a.frozen_energy_Ha, a.reference_energy_Ha]
        )
        identities[mode] = dict(a.identity)
    feature = CellCoulombFeatures(model[0]._bsp_data, memory_bytes=16 * 1024**2)
    keys = tuple(dict.fromkeys(label[:2] for label in dc.labels))
    samples = np.asarray([feature.sample(*model[1][key]) for key in keys]).T
    arrays["radial_keys"] = np.asarray(keys, dtype=np.int64)
    arrays["radial_samples"] = samples
    # Every ordered density between all five retained radial orbitals; no
    # orbital pair/radial component/coordinate omitted and no sign fitting.
    arrays["density_full"] = np.ascontiguousarray(
        (samples[:, :, None] * samples[:, None, :]).reshape(len(feature.radii), 25)
    )
    for name in ("radii", "outer_indices", "edge_left", "edge_right"):
        arrays[f"density_{name}"] = getattr(feature, name)
    for rank in range(4):  # complete possible electric ranks through p3/2
        for name, value in zip(
            ("rise", "fall", "weights"), feature.rank_arrays(rank), strict=True
        ):
            arrays[f"density_rank{rank}_{name}"] = value
        arrays[f"density_rank{rank}_public"] = np.asarray(
            [
                feature.apply_density(arrays["density_full"][:, i], rank)
                for i in range(25)
            ]
        ).T
    arrays["identities_json"] = np.asarray(
        json.dumps(identities, default=repr, sort_keys=True)
    )
    arrays["density_identity_json"] = np.asarray(
        json.dumps(dict(feature.identity), default=repr, sort_keys=True)
    )
    arrays["dynamic_frequencies_Ha"] = np.asarray([energies[key] for key in keys])
    np.savez_compressed(path, **arrays)


def load_hamiltonian(data, mode):
    noccupied, frozen, reference = data[f"{mode}_constants"]
    return SpinorIntegrals(
        tuple(tuple(map(int, row)) for row in data[f"{mode}_labels"]),
        int(noccupied),
        data[f"{mode}_one_body_Ha"],
        data[f"{mode}_fock_Ha"],
        data[f"{mode}_packed_eri_Ha"],
        float(frozen),
        float(reference),
        json.loads(str(data["identities_json"]))[mode],
    )


class FrozenDensityFeature:
    """Replay exactly the archived public feature geometry and weights."""

    def __init__(self, data):
        self.data = data
        for name in ("radii", "outer_indices", "edge_left", "edge_right"):
            setattr(self, name, data[f"density_{name}"])

    def rank_arrays(self, rank):
        if isinstance(rank, bool) or not isinstance(rank, int) or not 0 <= rank <= 3:
            raise ValueError("fixed complete rank 0..3 required")
        return tuple(
            self.data[f"density_rank{rank}_{name}"]
            for name in ("rise", "fall", "weights")
        )
