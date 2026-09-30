"""Live FAC WaveFuncTable comparison for exact-energy Dirac continuum waves.

Run with FAC_SFAC_PATH pointing at sfac and its runtime libraries available.
The FAC timing includes radial optimization and is not a like-for-like
solver-speed comparison. Atomx timings exclude the structure solve and JIT
compilation; first-wave timing includes grid/potential preparation.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import tempfile
from pathlib import Path
from time import perf_counter

import numpy as np

from atomx.potentials.static_field import static_hartree_potential

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fac_runner import run_sfac  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from atomx import Atom  # noqa: E402
from atomx.constants import HARTREE_EV  # noqa: E402
from atomx.numerics.quadrature import p_rel  # noqa: E402
from atomx.numerics.shooting.dirac_continuum import (  # noqa: E402
    solve_dirac_continuum_shooting,
)
from atomx.scattering.distorted_wave.solver import (  # noqa: E402
    _shooting_points,
)

CASES = {"Fe": (26, [(-1, 1000.), (-1, 40000.), (-5, 10000.)]),
         "Xe": (54, [(-1, 200000.)])}
FIELDS = ("element", "Z", "kappa", "energy_eV", "n_radial", "fac_ilast",
          "atomx_over_fac_P", "ratio_cv", "amplitude_error_vs_sqrt_2_over_pi",
          "atomx_first_ms", "atomx_warm_ms", "fac_case_s")


def _fac_wave(path: Path):
    ilast = None
    rows = []
    for line in path.read_text().splitlines():
        if line.startswith("#"):
            if line.strip().startswith("#  ilast"):
                ilast = int(line.split("=", 1)[1])
            continue
        parts = line.split()
        if len(parts) >= 6 and ilast is not None and len(rows) <= ilast:
            rows.append((float(parts[1]), float(parts[4])))
    if ilast is None or len(rows) != ilast + 1:
        raise ValueError(f"incomplete FAC wave dump: {path}")
    array = np.asarray(rows)
    return ilast, array[:, 0], array[:, 1]


def run_panel(workdir: Path):
    # Compile once before timing. This does not prepare any panel potential.
    solve_dirac_continuum_shooting(
        1., -1, lambda r: -1./r, r_max=20., n_grid=500,
        Z_residual=1.)
    rows = []
    for element, (Z, cases) in CASES.items():
        case_dir = workdir / element
        case_dir.mkdir(parents=True, exist_ok=True)
        lines = [f"SetAtom('{element}')", "Config('ground', '1s2')",
                 "OptimizeRadial(['ground'])"]
        for kappa, energy_eV in cases:
            name = f"{element}_k{kappa}_e{int(energy_eV)}.wav"
            lines.append(f"WaveFuncTable('{name}', 0, {kappa}, {energy_eV})")
        deck = case_dir / "continuum.sf"
        deck.write_text("\n".join(lines) + "\n")
        _, fac_case_s, _ = run_sfac(deck, workdir=case_dir)

        atom = Atom(Z=Z, config="1s2", charge=Z-2,
                    engine="bspline", corrections="none")
        atom.solve()
        if not atom.converged:
            raise RuntimeError(f"atomx {element} structure did not converge")
        potential = static_hartree_potential(atom._result, screening="mean")
        radius = float(atom._bsp_data["t"][-1])
        for kappa, energy_eV in cases:
            name = f"{element}_k{kappa}_e{int(energy_eV)}.wav"
            ilast, fac_r, fac_P = _fac_wave(case_dir / name)
            energy = energy_eV / HARTREE_EV
            n_grid = _shooting_points(radius, p_rel(energy))
            args = (energy, kappa, potential)
            kwargs = dict(r_max=radius, n_grid=n_grid, Z_residual=Z-1.)
            start = perf_counter()
            r, P, _Q, _phase = solve_dirac_continuum_shooting(*args, **kwargs)
            first_ms = 1000 * (perf_counter() - start)
            warm = []
            for _ in range(5):
                start = perf_counter()
                solve_dirac_continuum_shooting(*args, **kwargs)
                warm.append(1000 * (perf_counter() - start))

            sampled = np.interp(fac_r, r, P)
            mask = np.abs(fac_P) > 0.05 * np.max(np.abs(fac_P))
            ratio = sampled[mask] / fac_P[mask]
            median = float(np.median(ratio))
            cv = float(np.std(ratio) / abs(median))
            error = median / math.sqrt(2. / math.pi) - 1.
            if cv > .015 or abs(error) > .015:
                raise AssertionError(
                    f"{element} kappa={kappa} E={energy_eV}: "
                    f"FAC wave shape/normalization disagree: cv={cv}, error={error}")
            rows.append(dict(element=element, Z=Z, kappa=kappa,
                             energy_eV=energy_eV, n_radial=n_grid,
                             fac_ilast=ilast, atomx_over_fac_P=median,
                             ratio_cv=cv,
                             amplitude_error_vs_sqrt_2_over_pi=error,
                             atomx_first_ms=first_ms,
                             atomx_warm_ms=float(np.median(warm)),
                             fac_case_s=fac_case_s))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path,
                        default=Path(__file__).with_name("results_fac_continuum.csv"))
    parser.add_argument("--workdir", type=Path)
    args = parser.parse_args()
    if args.workdir is None:
        with tempfile.TemporaryDirectory(prefix="atomx-fac-continuum-") as directory:
            rows = run_panel(Path(directory))
    else:
        rows = run_panel(args.workdir)
    with args.out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(f"{row['element']:2} κ={row['kappa']:2} E={row['energy_eV']:7.0f} eV "
              f"P/P_FAC={row['atomx_over_fac_P']:.7f} "
              f"shape_cv={row['ratio_cv']:.4%} "
              f"first={row['atomx_first_ms']:.2f} ms "
              f"warm={row['atomx_warm_ms']:.2f} ms")


if __name__ == "__main__":
    main()
