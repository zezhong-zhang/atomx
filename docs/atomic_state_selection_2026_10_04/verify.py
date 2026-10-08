"""Replay the existing physical trajectory and protect prior scientific files."""

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np

from atomx.scattering.models import ContinuumModel
from atomx.xfel import CoherentGaussianPulse, TwoElectronCoherentModel
from atomx.xfel.atomic_rates import _source_identity

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PREVIOUS = ROOT / "docs/xfel_state_average_2026_10_03"
TRAJECTORY = ROOT / "docs/xfel_ci_reuse_2026_10_03"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    previous = json.loads((PREVIOUS / "identity.json").read_text())
    protected = dict(previous["protected_previous_artifacts_sha256"])
    protected.update(
        {
            name: digest
            for name, digest in previous["files_sha256"].items()
            if name.startswith("docs/xfel_state_average_2026_10_03/")
        }
    )
    for name, digest in protected.items():
        assert sha(ROOT / name) == digest, name
    for name, digest in previous["archived_SCF_h5_sha256"].items():
        assert sha(ROOT / name) == digest, name
    for name, digest in previous["external_reference_files_sha256"].items():
        assert sha(Path(name)) == digest, name

    spec = importlib.util.spec_from_file_location("previous_ci_bench", TRAJECTORY / "bench.py")
    benchmark = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(benchmark)
    value = benchmark.atom()
    recorded = json.loads((TRAJECTORY / "bench.json").read_text())
    pulse = CoherentGaussianPulse(recorded["panels"][0]["carrier_eV"], 1e17, 10)
    model = TwoElectronCoherentModel(
        value,
        pulse,
        photon_numbers={i: int(i != 0) for i in range(5)},
        ionic_photon_numbers={i: 0 for i in range(3)},
        continuum_model=ContinuumModel.shooting(n_grid=16000, r_max=80, orbital_removal="none"),
    )
    dynamics = model.solve(pulse, after_pulse_fs=100)
    with np.load(TRAJECTORY / "density.npz") as archived:
        np.testing.assert_array_equal(archived["state_ids"], dynamics.state_ids)
        np.testing.assert_array_equal(archived["density_matrices"], dynamics.density_matrices)
        difference = float(np.max(abs(archived["density_matrices"] - dynamics.density_matrices)))
    sources = (
        "AGENTS.md", "docs/ATOMIC_STATE_DESIGN.md", "docs/XFEL_GAP_ANALYSIS.md",
        "src/atomx/__init__.py", "src/atomx/physics/csf/levels.py",
        "src/atomx/physics/csf/state_average.py", "src/atomx/xfel/mcdhf_coherent.py",
        "src/atomx/xfel/two_electron_coherent.py", "tests/test_ci_level_selection.py",
        "tests/test_xfel_mcdhf_coherent.py", "tests/test_xfel_ci_continuum.py",
    )
    files = [ROOT / name for name in sources] + list(HERE.glob("*"))
    identity = {
        "atomx_source_tree_sha256": _source_identity(),
        "source_state": "working tree; no commit or push",
        "protected_previous_artifacts_sha256": protected,
        "archived_SCF_h5_sha256": previous["archived_SCF_h5_sha256"],
        "external_reference_files_sha256": previous["external_reference_files_sha256"],
        "default_density_replay_max_error": difference,
        "files_sha256": {
            str(p.relative_to(ROOT)): sha(p)
            for p in sorted(files)
            if p.is_file() and p.name not in ("identity.json", "verify.log")
        },
    }
    (HERE / "identity.json").write_text(json.dumps(identity, indent=2) + "\n")
    print("Default ground-only complete density replay: max difference", difference)
    print(
        "Protected", len(protected), "previous artifacts and",
        len(previous["archived_SCF_h5_sha256"]), "SCF snapshots; reference hashes unchanged.",
    )


if __name__ == "__main__":
    main()
