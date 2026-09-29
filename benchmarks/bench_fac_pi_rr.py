"""Benchmark AtomX photoionisation (PI) and radiative recombination (RR)
against FAC (sfac).

Standard reference set:
1. H-like Fe^{25+} 1s ionisation: ground-state σ_PI(ω) and σ_RR(E_e) for
   capture into 1s.  Pure single-channel hydrogenic test — clean
   reference because a closed-form Stobbe-Dirac value exists.
2. He-like Fe^{24+} 1s² ionisation: σ_PI(ω) for the 1s² ground state,
   σ_RR(E_e) for capture into He-like ground from H-like 1s.
3. Li-like Fe^{23+}: σ_RR for capture into 1s²2s, 1s²2p_{1/2}, 1s²2p_{3/2}.

For each case we compare the σ at the FAC-defined NUSR electron-energy
grid.  Output:
  - benchmarks/results_fac_pi_rr.csv  with rows
    case, transition, E_eV, sigma_atomx_1e20cm2, sigma_fac_1e20cm2, rel_diff
  - benchmarks/results_fac_pi_rr.log  with full sfac stdout for triage.
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fac_runner import parse_levels, parse_rr, run_sfac  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from atomx import Atom, ContinuumModel  # noqa: E402
from atomx.constants import BOHR_RADIUS_M, HARTREE_EV  # noqa: E402
from atomx.scattering.photoionization import photoionization_cross_section  # noqa: E402
from atomx.scattering.radiative_recombination import rr_cross_section  # noqa: E402

# NOTE (S5 finding 3): ``ContinuumModel.fac_residual_core`` is a *shooting*
# continuum model and ``rr_cross_section`` supports only the E1 multipole on
# that branch, so the panel is evaluated E1-only.  The SCF solve must be the
# B-spline engine: the shooting engine leaves ``SCFResult._bsp_data = None``
# and the RR kernel raises "SCFResult is missing _bsp_data".

A0_CM2_X1E20 = (BOHR_RADIUS_M * 100.0) ** 2 * 1e20  # = 0.28003 (a₀² in 10⁻²⁰ cm²)
FAC_CONTINUUM = ContinuumModel.fac_residual_core(n_grid=8000)


def atomx_li_capture_results():
    """Optimize each Li-like final level with its actual jj occupation.

    A ground-state CI solve of ``1s2 2p1`` leaves the 2p3/2 occupation
    zero. Its spectator orbital is not the relaxed 2P3/2 capture state.
    """
    from atomx.methods.dirac_fock import DiracFock, Subshell
    results = {}
    for kappa in (-1, 1, -2):
        results[kappa] = DiracFock().solve(
            Z=26, subshells=[Subshell(1, -1, 2), Subshell(2, kappa, 1)])
        if not results[kappa].converged:
            raise RuntimeError(f"Li-like final kappa={kappa} did not converge")
    return results


# ---------------------------------------------------------------------------
# FAC-side script template
# ---------------------------------------------------------------------------

FAC_PIRR_SF = """\
SetAtom('{element}')
{closed_lines}
{config_lines}
ConfigEnergy(0)
OptimizeRadial({optimise!r})
ConfigEnergy(1)
{structure_lines}
MemENTable('{prefix}.lev.b')
PrintTable('{prefix}.lev.b', '{prefix}.lev', 1)

SetSE(-1)
SetVP(0)
SetMS(0,0)
SetBreit(0)

# electron-energy grid
SetUsrPEGrid({n_e}, {e_min}, {e_max}, {z_eff})
SetPEGrid({n_e}, {e_min}, {e_max}, {z_eff})

RRTable('{prefix}.rr.b', {recombined!r}, {recombining!r})
PrintTable('{prefix}.rr.b', '{prefix}.rr', 1)
"""


def write_fac_script(workdir: Path, element: str, prefix: str,
                     closed: List[str],
                     configs: Dict[str, str],
                     optimise: List[str],
                     structure_groups: List[str],
                     recombined: List[str],
                     recombining: List[str],
                     n_e: int = 6, e_min: float = 50.0, e_max: float = 1.6e4,
                     z_eff: float = 25.0) -> Path:
    """Render the sfac script for a PI/RR run."""
    closed_lines = "\n".join(f"Closed('{s}')" for s in closed)
    config_lines = "\n".join(f"Config('{c}', group='{g}')" for g, c in configs.items())
    structure_lines = "\n".join(
        f"Structure('{prefix}.lev.b', ['{g}'])" for g in structure_groups
    )
    src = FAC_PIRR_SF.format(
        element=element, prefix=prefix,
        closed_lines=closed_lines, config_lines=config_lines,
        optimise=optimise, structure_lines=structure_lines,
        recombined=recombined, recombining=recombining,
        n_e=n_e, e_min=e_min, e_max=e_max, z_eff=z_eff,
    )
    path = workdir / f"{prefix}.sf"
    path.write_text(src)
    return path


# ---------------------------------------------------------------------------
# AtomX evaluator
# ---------------------------------------------------------------------------


def atomx_pi_for_orbital(atom: "Atom", orbital_label: str,
                          photon_energies_au: np.ndarray) -> np.ndarray:
    """Compute σ_PI(ω) [a₀²] for the named bound orbital using AtomX's
    B-spline continuum solver.

    Returns the partial cross section for ionising that orbital (no
    statistical occupation factor — the caller multiplies by q_a if
    they want the configuration-summed σ_PI).
    """
    res = atom._result
    bsp_data = res._bsp_data
    Z = res.Z
    sh = next(s for s in res.subshells if s.label == orbital_label)
    eps = res.orbital_energies[orbital_label]
    orb_list = res.orbitals_raw[sh.kappa]
    kappa_subshells = [s for s in res.subshells if s.kappa == sh.kappa]
    idx = kappa_subshells.index(sh)
    P_b, Q_b, _ = orb_list[idx]
    return photoionization_cross_section(P_b, Q_b, sh.kappa, eps,
                                          photon_energies_au, Z, bsp_data)


# ---------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------


def run_he_like_fe(workdir: Path) -> List[Dict]:
    """He-like Fe^{24+} 1s²: PI of 1s² + RR (Fe^{25+} → He-like)."""
    element = "Fe"
    Z = 26
    sfac_path = write_fac_script(
        workdir, element, "fe24",
        closed=[],
        configs={"g1": "1s1", "g2": "1s2"},
        optimise=["g2"],
        structure_groups=["g1", "g2"],
        recombined=["g2"], recombining=["g1"],
        n_e=6, e_min=1.0, e_max=2.0e2, z_eff=25.0,
    )
    print(f"  [FAC] running {sfac_path.name} ...")
    stdout, t_fac, _ = run_sfac(sfac_path, workdir=workdir)
    Path(workdir / "fe24.sfac.log").write_text(stdout)
    print(f"  [FAC] done in {t_fac:.2f}s")

    rr_rows = parse_rr(workdir / "fe24.rr")
    print(f"  [FAC] parsed {len(rr_rows)} RR transitions")

    print("  [AtomX] He-like Fe^{24+} solve ...")
    t0 = time.perf_counter()
    atom = Atom(Z=Z, config="1s2", corrections="none", engine="bspline")
    atom.solve()
    res = atom._result
    eps_1s = res.orbital_energies["1s1/2"]
    I_1s_eV = -eps_1s * HARTREE_EV
    print(f"  [AtomX] I_1s(He-like) = {I_1s_eV:.3f} eV")

    # Use AtomX's RR helper directly
    out = []
    for rr in rr_rows:
        E_eV = rr["E_eV"]
        sigma_RR_fac = rr["sigma_RR_1e20cm2"]
        # FAC's printed RR cross section is divided by the recombining
        # ion's level degeneracy (see faclib/dbase.c:PrintRRTable), so
        # pass the FAC free-level 2J+1 as AtomX's g_initial.
        rrres = rr_cross_section(
            res, E_eV, g_initial=rr["twoJ_f"] + 1,
            continuum_model=FAC_CONTINUUM,
            multipoles=("E1",),
        )
        t_atomx = time.perf_counter() - t0
        sigma_atomx_1e20 = rrres.sigma_total_a0_sq * A0_CM2_X1E20
        rel = (sigma_atomx_1e20 - sigma_RR_fac) / sigma_RR_fac
        for k in range(len(E_eV)):
            out.append({
                "case": "Fe25+ -> Fe24+(1s²) RR",
                "transition": f"FAC ILEV {rr['free']}->{rr['bound']}",
                "E_eV": float(E_eV[k]),
                "sigma_atomx_1e20cm2": float(sigma_atomx_1e20[k]),
                "sigma_fac_1e20cm2": float(sigma_RR_fac[k]),
                "rel_diff": float(rel[k]),
                "t_fac_s": float(t_fac),
                "t_atomx_s": float(t_atomx),
            })
    return out


def run_li_like_fe(workdir: Path) -> List[Dict]:
    """Li-like Fe^{23+} (1s² 2s) RR — port of FAC fe24_pirr demo."""
    element = "Fe"
    sfac_path = write_fac_script(
        workdir, element, "fe23",
        closed=[],
        configs={"n1": "1s2", "n2": "1s1 2*1", "rn2": "1s2 2*1"},
        optimise=["rn2"],
        structure_groups=["n1", "n2", "rn2"],
        recombined=["rn2"], recombining=["n1"],
        n_e=6, e_min=50.0, e_max=1.6e4, z_eff=25.0,
    )
    print(f"  [FAC] running {sfac_path.name} ...")
    stdout, t_fac, _ = run_sfac(sfac_path, workdir=workdir)
    Path(workdir / "fe23.sfac.log").write_text(stdout)
    print(f"  [FAC] done in {t_fac:.2f}s")

    rr_rows = parse_rr(workdir / "fe23.rr")
    print(f"  [FAC] parsed {len(rr_rows)} RR transitions")

    # AtomX side — solve the final-state configurations that appear in
    # FAC's rn2 group: 1s²2s, 1s²2p_{1/2}, 1s²2p_{3/2}.  FAC's printed
    # RR rows are level-resolved, so comparing every bound row to 2s
    # capture artificially inflated the old gap.
    print("  [AtomX] Li-like Fe^{23+} final-state solves ...")
    t0 = time.perf_counter()
    capture_results = atomx_li_capture_results()

    level_by_ilev = {int(lv["ILEV"]): lv for lv in parse_levels(workdir / "fe23.lev")}

    def _capture_target(bound_ilev: int):
        lv = level_by_ilev.get(bound_ilev, {})
        cfg = str(lv.get("config", ""))
        twoJ = int(lv.get("TWOJ", -1))
        if "2s1" in cfg:
            return capture_results[-1], ["2s1/2"]
        if "2p1" in cfg and twoJ == 1:
            return capture_results[1], ["2p1/2"]
        if "2p1" in cfg and twoJ == 3:
            return capture_results[-2], ["2p3/2"]
        return None, []

    out = []
    for rr in rr_rows:
        E_eV = rr["E_eV"]
        sigma_RR_fac = rr["sigma_RR_1e20cm2"]
        res, capture_orbitals = _capture_target(int(rr["bound"]))
        if res is None:
            continue
        rrres = rr_cross_section(
            res, E_eV, g_initial=rr["twoJ_f"] + 1,
            capture_orbitals=capture_orbitals,
            continuum_model=FAC_CONTINUUM,
            multipoles=("E1",),
        )
        t_atomx = time.perf_counter() - t0
        sigma_atomx_1e20 = rrres.sigma_total_a0_sq * A0_CM2_X1E20
        rel = (sigma_atomx_1e20 - sigma_RR_fac) / sigma_RR_fac
        for k in range(len(E_eV)):
            out.append({
                "case": "Fe24+ -> Fe23+(1s²2s) RR",
                "transition": f"FAC ILEV {rr['free']}->{rr['bound']}",
                "E_eV": float(E_eV[k]),
                "sigma_atomx_1e20cm2": float(sigma_atomx_1e20[k]),
                "sigma_fac_1e20cm2": float(sigma_RR_fac[k]),
                "rel_diff": float(rel[k]),
                "t_fac_s": float(t_fac),
                "t_atomx_s": float(t_atomx),
            })
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    out_dir = Path(__file__).resolve().parent
    workroot = out_dir / "_pi_rr_work"
    workroot.mkdir(exist_ok=True)

    rows: List[Dict] = []
    cases = [
        ("He-like Fe24+ (1s2)", run_he_like_fe),
        ("Li-like Fe23+ (1s2 2s)", run_li_like_fe),
    ]
    for name, fn in cases:
        print(f"\n=== {name} ===")
        wd = workroot / name.replace(" ", "_").replace("(", "").replace(")", "").replace("²", "2").replace("⁺", "+").replace("{","").replace("}","")
        wd.mkdir(exist_ok=True)
        try:
            rows.extend(fn(wd))
        except Exception as e:  # pragma: no cover  -- want a clear error
            print(f"  !! FAILED: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()

    csv_path = out_dir / "results_fac_pi_rr.csv"
    if not rows:
        # Never truncate the committed panel: a case that raises (see the
        # module docstring) used to overwrite the good CSV with a bare header.
        print("No rows produced — leaving the committed CSV untouched")
        return 1
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["case", "transition", "E_eV",
                                            "sigma_atomx_1e20cm2",
                                            "sigma_fac_1e20cm2",
                                            "rel_diff", "t_fac_s",
                                            "t_atomx_s"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nWrote {len(rows)} rows to {csv_path}")

    # Quick summary by case
    from collections import defaultdict
    by_case: Dict[str, List[float]] = defaultdict(list)
    for r in rows:
        by_case[r["case"]].append(abs(r["rel_diff"]))
    print("\nMax |rel diff| per case:")
    for case, vals in by_case.items():
        if vals:
            print(f"  {case:50s} max={max(vals):8.2e}  median={sorted(vals)[len(vals)//2]:8.2e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
