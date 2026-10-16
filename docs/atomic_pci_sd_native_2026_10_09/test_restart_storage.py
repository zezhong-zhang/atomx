"""Private real-operator interruption/replay and committed-payload rejection."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tests"))
from restart_storage import RestartStorage  # noqa: E402
from test_valence_sd import live_valence  # noqa: E402, F401


def test_real_restart_interruption_and_fresh_certification(live_valence, tmp_path):  # noqa: F811
    core, _, problem = live_valence
    key = problem.plan["valence_keys"][0]
    index = next(i for i, label in enumerate(core.integrals.labels) if label[:2] == key)
    energy = float(core.integrals.fock_Ha[index, index].real + 0.075)
    reference = problem.solve(key, energy, residual_tolerance_Ha=1e-11)
    store = RestartStorage(
        tmp_path / "saved",
        dict(test="same actual core and complete coordinate binding"),
    )
    identities = {}

    def save_and_interrupt(snapshot):
        store(snapshot)
        identities.update(
            {
                k: v
                for k, v in snapshot.items()
                if k not in ("numerators", "stage", "refinement", "ordinal")
            }
        )
        raise RuntimeError("requested interruption after committed restart")

    with pytest.raises(RuntimeError, match="requested interruption"):
        problem.solve(key, energy, restart=2, iteration_callback=save_and_interrupt)
    assert list(store.directory.glob("root_*.npz"))
    assert store.load("response", identities, reference.numerators.size) is None
    initial = store.load("root", identities, reference.numerators.size)
    restored = problem.solve(
        key, energy, initial_numerators=initial, residual_tolerance_Ha=1e-11
    )
    assert restored.residual_max_Ha < 1e-11
    assert restored.derivative_residual_max_Ha < 1e-11
    for name in (
        "numerators",
        "numerator_derivative",
        "sigma1_Ha",
        "sigma1_derivative",
    ):
        assert np.max(abs(getattr(restored, name) - getattr(reference, name))) < 1e-11
    for field, changed in (
        ("key", (99, -1)),
        ("energy_Ha", energy + 1),
        ("core_amplitudes_sha256", "changed"),
        ("packed_coordinates_sha256", ("changed",)),
        ("denominators_sha256", "changed"),
        ("valence_source_sha256", (("changed", "source"),)),
    ):
        with pytest.raises(ValueError, match="identity differs"):
            store.load("root", dict(identities, **{field: changed}), initial.size)
    with pytest.raises(ValueError, match="identity differs"):
        RestartStorage(
            store.directory, dict(test="different source/input manifest")
        ).load("root", identities, initial.size)
    with pytest.raises(ValueError, match="complete finite real"):
        store.load("root", identities, initial.size + 1)
    head = json.loads((store.directory / "root.json").read_text())
    with (store.directory / head["file"]).open("ab") as stream:
        stream.write(b"corruption")
    with pytest.raises(ValueError, match="hash differs"):
        store.load("root", identities, initial.size)


def test_uncommitted_generation_and_resource_guard(tmp_path):
    store = RestartStorage(tmp_path, {}, cap_bytes=1)
    snapshot = dict(stage="root", ordinal=1, certified=False, numerators=np.zeros(3))
    with pytest.raises(MemoryError, match="cap/free space"):
        store(snapshot)
    assert not list(tmp_path.glob("*.npz"))
    (tmp_path / "root_000001.npz.partial").write_bytes(b"incomplete")
    assert store.load("root", {}, 3) is None
