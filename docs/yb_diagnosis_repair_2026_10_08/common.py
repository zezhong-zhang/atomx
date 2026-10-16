"""Identity-preserving loader for the retained Yb V68 reference."""
from pathlib import Path
import hashlib
import json
import pickle
import numpy as np

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
OLD = ROOT.parent / "rare_earth_yb_spectrum_2026_10_08"
CORE = "1s2 2s2 2p6 3s2 3p6 3d10 4s2 4p6 4d10 5s2 5p6 4f14"
KEYS = {"6s_1/2": (6, -1), "5d_3/2": (5, 2), "5d_5/2": (5, -3),
        "6p_1/2": (6, 1), "6p_3/2": (6, -2)}
EXPERIMENT = {"6s_1/2": 0., "5d_3/2": 22960.80, "5d_5/2": 24332.69,
              "6p_1/2": 27061.82, "6p_3/2": 30392.23}


def save(path, value):
    def convert(x):
        if isinstance(x, np.ndarray): return x.tolist()
        if isinstance(x, np.generic): return x.item()
        raise TypeError(type(x).__name__)
    Path(path).write_text(json.dumps(value, indent=2, default=convert)+"\n")


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_identity():
    return {str(p.relative_to(REPO)): sha(p) for p in sorted((REPO/"src/atomx").rglob("*.py"))}


def load_reference(name="v68_default"):
    from atomx.config.csf import enumerate_jj_csfs
    from atomx.methods.dirac_fock import SCFResult
    from atomx.potentials.nuclear import make_nuclear_potential
    base = OLD/"results"/name
    if not json.loads((base/"status.json").read_text())["passed"]:
        raise ValueError("baseline numerical gates failed")
    with (base/"result.pkl").open("rb") as f: result = pickle.load(f)
    info = json.loads((base/"core.json").read_text())
    shells = enumerate_jj_csfs(CORE)[0].subshells
    raw = {}
    for sh in shells:
        P, Q = result.orbitals_dict[sh.n, sh.kappa]
        raw.setdefault(sh.kappa, []).append((P.copy(), Q.copy(), sh.occupation))
    core = SCFResult(converged=info["converged"], iterations=info["iterations"],
        residual=info["residual"], total_energy=info["total_energy_hartree"],
        orbital_energies=result.orbital_energies, subshells=shells, orbitals_raw=raw,
        Z=70, _bsp_data=result.bsp_data, solver_diagnostics=info["solver_diagnostics"])
    core._nuclear_potential = result._nuclear_potential = make_nuclear_potential(70, "fermi")
    levels = json.loads((base/"structure.json").read_text())["levels"]
    identity = {"base": name, "orbitals_workspace_sha256": sha(base/"orbitals_workspace.npz"),
                "core_sha256": sha(base/"core.json"), "source_sha256": source_identity()}
    assert sum(s.occupation for s in core.subshells) == 68
    return core, result, levels, identity
