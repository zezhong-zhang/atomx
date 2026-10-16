"""Complete matched fixed-energy Sigma workflow; no new physical formulas."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import resource
import sys
import time
from unittest.mock import patch

import numpy as np

from atomx.methods.dirac_fock import SCFResult, Subshell
from atomx.methods.mbpt_matrix import build_mbpt2_matrix_self_energy
from atomx.potentials.nuclear import make_nuclear_potential

KAPPAS = (-1, 1, -2, 2, -3)


def canonical(k, a, b, c, d):
    x, y = sorted((tuple(sorted((a, c))), tuple(sorted((b, d)))))
    return (k, *x, *y)


def load_context(directory):
    arrays = np.load(directory / "radial_inputs.npz")
    rows = arrays["orbital_rows"]
    states = [
        (arrays[f"P_{i}"], arrays[f"Q_{i}"], int(row[2]), float(row[3]))
        for i, row in enumerate(rows)
    ]
    workspace = json.loads((directory / "workspace_scalars.json").read_text())
    for name in arrays.files:
        if name.startswith("workspace_"):
            workspace[name.removeprefix("workspace_")] = arrays[name]
    meta = json.loads((directory / "core_metadata.json").read_text())
    subshells = [Subshell(**row) for row in meta["subshells"]]
    raw = {}
    for i, row in enumerate(rows):
        if row[4]:
            raw.setdefault(int(row[2]), []).append(
                (states[i][0], states[i][1], 2 * abs(int(row[2])))
            )
    core = SCFResult(
        converged=meta["converged"], iterations=meta["iterations"],
        residual=meta["residual"], total_energy=meta["total_energy"],
        orbital_energies=meta["orbital_energies"], subshells=subshells,
        orbitals_raw=raw, Z=10, _bsp_data=workspace,
    )
    core._nuclear_potential = make_nuclear_potential(10, "point")
    blocks = {k: [state for i, state in enumerate(states) if not rows[i, 4] and state[2] == k]
              for k in KAPPAS}
    by_array = {(id(state[0]), id(state[1])): i for i, state in enumerate(states)}
    table = {tuple(int(value) for value in row[:5]): float(row[5])
             for row in np.loadtxt(directory / "integrals.txt")}
    if len(states) != 19 or len(table) != 37350:
        raise ValueError("complete matched input identity changed")

    def factory(targets, partner, kets, data, cache):
        b = by_array[id(partner[0]), id(partner[1])]

        def evaluator(target):
            a = by_array[id(target[0]), id(target[1])]

            def radial(k, pc, qc, pd, qd):
                c = by_array[id(pc), id(qc)]
                d = by_array[id(pd), id(qd)]
                return table[canonical(k, a, b, c, d)]

            return radial

        return [evaluator(target) for target in targets]

    def basis(_, k, **kwargs):
        if kwargs["potential_mode"] != "VN" or not kwargs["orthogonal_to_occupied"]:
            raise ValueError("changed virtual-field contract")
        return [], blocks[k]

    def action():
        output = []
        for k in (-1, -2, -3):
            ids = [i for i, state in enumerate(states) if not rows[i, 4] and state[2] == k]
            targets = [states[i] for i in ids]
            energy = targets[0][3]
            terms = []
            with patch("atomx.methods.mbpt.build_virtual_basis", side_effect=basis), \
                 patch("atomx.methods.mbpt_matrix._radial_table_evaluators", side_effect=factory):
                sigma = build_mbpt2_matrix_self_energy(
                    core, targets, reference_identity=f"Ne:matched:AMBiT:{k}",
                    virtual_kappas=KAPPAS, energy_window_au=50., n_virt_max=3,
                    term_recorder=lambda p, r, d: terms.append((p, r, d)),
                )
            matrices = {diagram: sum((r / (energy - p) for p, r, d in terms if d == diagram),
                                     np.zeros((3, 3))) for diagram in ("2p1h", "2h1p")}
            total = sigma.evaluate(energy)
            for ii, a in enumerate(ids):
                for jj, b in enumerate(ids):
                    output.append([a, b, energy, matrices["2p1h"][ii, jj],
                                   matrices["2h1p"][ii, jj], total[ii, jj]])
        return np.asarray(output)

    return action


def main():
    resource.setrlimit(resource.RLIMIT_AS, (8 * 1024**3, 8 * 1024**3))
    directory, output = map(Path, sys.argv[1:])
    started = time.perf_counter()
    action = load_context(directory)
    setup_seconds = time.perf_counter() - started
    action()  # All 27 entries and three components, angular-cache warmup only.
    started = time.perf_counter()
    actual = action()
    action_seconds = time.perf_counter() - started
    np.savez_compressed(output.with_suffix(".npz"), output=actual)
    sources = {str(Path(module.__file__).resolve()):
               hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
               for module in tuple(sys.modules.values())
               if getattr(module, "__file__", None) and Path(module.__file__).is_file()}
    output.write_text(json.dumps(dict(setup_seconds=setup_seconds, action_seconds=action_seconds,
                                     max_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                                     actual_imported_sources_sha256=sources), indent=2) + "\n")
    output.with_suffix(".maps").write_text(Path("/proc/self/maps").read_text())


if __name__ == "__main__":
    main()
