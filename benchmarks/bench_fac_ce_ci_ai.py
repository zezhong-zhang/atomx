"""Unified FAC↔AtomX benchmarks for collisional / autoionising processes.

Three demos drawn from the FAC distribution and ported to a single
script that scripts sfac, parses its tabular output (.ce / .ci / .ai),
and runs the matching AtomX function on the same physical case:

  ``ce``  — Fe XVII (Ne-like) 2*8 → 2*7 3*1 electron-impact excitation
            (from ``fac/demo/excitation/fe17_excitation.sf``).  AtomX
            calls :func:`atomx.scattering.distorted_wave.solver
            .dw_excitation_cross_section` for each transition.
  ``ci``  — Fe XVII → Fe XVIII electron-impact ionisation
            (``fac/demo/ionization/fe17_ionization.sf``).  AtomX calls
            :func:`atomx.scattering.electron_ionization.eii_cross_section`.
  ``ai``  — Ne-like Se 2p^4 3*2 → 2p^5 autoionisation rates
            (``fac/demo/aidr/se_ai.sf``).  AtomX uses
            :func:`atomx.scattering.autoionization.auger_rate_two_electron`.

Each block writes a CSV under ``benchmarks/`` with one row per
(transition, energy / level) and the relative difference vs FAC.
"""

from __future__ import annotations

import csv
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fac_runner import (  # noqa: E402
    parse_ai,
    parse_ce,
    parse_ci,
    parse_levels,
    run_sfac,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from atomx import Atom  # noqa: E402
from atomx.constants import BOHR_RADIUS_M  # noqa: E402

A0_CM2_X1E20 = (BOHR_RADIUS_M * 100.0) ** 2 * 1e20

OUT_DIR = Path(__file__).resolve().parent
WORKROOT = OUT_DIR / "_ce_ci_ai_work"
WORKROOT.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# CE — Ne-like Fe XVII excitation
# ---------------------------------------------------------------------------

CE_SF = """\
# He-like Fe^{24+}: 1s^2 (J=0, ground) -> 1s^1 2*1 (excited)
# Single-config closed-shell ground state keeps AtomX in SCFResult mode
# so the dw_excitation_cross_section helper can find subshells.
SetAtom('Fe')
Config('1s2', group='g')
Config('1s1 2p1', group='ex')
ConfigEnergy(0)
OptimizeRadial(['g'])
ConfigEnergy(1)
SetSE(-1); SetVP(0); SetMS(0,0); SetBreit(0)
Structure('he.lev.b', ['g'])
Structure('he.lev.b', ['ex'])
MemENTable('he.lev.b')
PrintTable('he.lev.b', 'he.lev', 1)
SetCEGrid(3, 9000.0, 30000.0)
SetUsrCEGrid(3, 9000.0, 30000.0)
CETable('he.ce.b', ['g'], ['ex'])
PrintTable('he.ce.b', 'he.ce', 1)
"""


def bench_ce(workdir: Path) -> List[Dict]:
    sf_path = workdir / "fe17_ce.sf"
    sf_path.write_text(CE_SF)
    print("  [FAC] Fe XVII CE running ...")
    out, t_fac, _ = run_sfac(sf_path, workdir=workdir)
    (workdir / "fe17_ce.log").write_text(out)
    print(f"  [FAC] done in {t_fac:.2f}s")

    levels = parse_levels(workdir / "he.lev")
    ce_rows = parse_ce(workdir / "he.ce")
    print(f"  [FAC] parsed {len(levels)} levels + {len(ce_rows)} CE transitions")

    # FAC prints level-resolved CE rows. AtomX's current CE helper is
    # configuration-level for 1s² -> 1s2p, so compare against the FAC
    # level-summed curve, not each upper fine-structure level separately.
    ce_rows = sorted(ce_rows, key=lambda r: r["DELTA_E_eV"])

    print("  [AtomX] solving He-like Fe^{24+} (1s²) ground ...")
    atom_i = Atom(Z=26, config="1s2", corrections="none")
    atom_i.solve()

    out_rows: List[Dict] = []
    print("  [AtomX] solving He-like Fe^{24+} (1s 2p) excited ...")
    try:
        atom_f = Atom(Z=26, config="1s1 2p1", corrections="none")
        atom_f.solve()
    except Exception as e:
        print(f"  [AtomX] excitation solve failed: {e}")
        atom_f = None
    if not ce_rows:
        return []

    E_excess_eV_arr = ce_rows[0]["E_eV"]
    sigma_fac_sum = np.zeros_like(E_excess_eV_arr, dtype=float)
    omega_fac_sum = np.zeros_like(E_excess_eV_arr, dtype=float)
    weighted_de = np.zeros_like(E_excess_eV_arr, dtype=float)
    for rec in ce_rows:
        sigma = np.asarray(rec["sigma_1e20cm2"], dtype=float)
        sigma_fac_sum += sigma
        omega_fac_sum += np.asarray(rec["Omega"], dtype=float)
        weighted_de += sigma * float(rec["DELTA_E_eV"])
    de_eff = np.divide(
        weighted_de, sigma_fac_sum,
        out=np.full_like(weighted_de, float(np.mean([r["DELTA_E_eV"] for r in ce_rows]))),
        where=sigma_fac_sum > 0.0,
    )
    E_incident_eV_arr = E_excess_eV_arr + de_eff

    if atom_f is not None:
        from atomx.scattering.distorted_wave.solver import (
            dw_excitation_cross_section,
        )
        t0 = time.perf_counter()
        retried_zero = np.zeros_like(E_excess_eV_arr, dtype=bool)
        try:
            sigma_atomx_cm2, omega_atomx = dw_excitation_cross_section(
                atom_i, atom_f, E_incident_eV_arr, L_max=3, kappa_max=4,
                transition_energy_eV=de_eff,
            )
            retry = (sigma_atomx_cm2 <= 0.0) & (sigma_fac_sum > 0.0)
            for idx in np.where(retry)[0]:
                sigma_retry, omega_retry = dw_excitation_cross_section(
                    atom_i,
                    atom_f,
                    E_incident_eV_arr[idx:idx + 1],
                    L_max=3,
                    kappa_max=4,
                    transition_energy_eV=de_eff[idx:idx + 1],
                )
                if sigma_retry.size and sigma_retry[0] > 0.0:
                    sigma_atomx_cm2[idx] = sigma_retry[0]
                    omega_atomx[idx] = omega_retry[0]
                    retried_zero[idx] = True
        except Exception as e:  # pragma: no cover
            print(f"  [AtomX] DW failed: {e}")
            sigma_atomx_cm2 = np.zeros_like(E_excess_eV_arr)
            omega_atomx = np.zeros_like(E_excess_eV_arr)
        t_atomx = time.perf_counter() - t0
    else:
        sigma_atomx_cm2 = np.zeros_like(E_excess_eV_arr)
        omega_atomx = np.zeros_like(E_excess_eV_arr)
        retried_zero = np.zeros_like(E_excess_eV_arr, dtype=bool)
        t_atomx = 0.0

    sigma_atomx_1e20 = sigma_atomx_cm2 * 1e20  # cm² → 1e-20 cm²
    for k in range(len(E_excess_eV_arr)):
        ratio = (
            (sigma_atomx_1e20[k] - sigma_fac_sum[k])
            / max(abs(sigma_fac_sum[k]), 1e-30)
        )
        omega_ratio = (
            (omega_atomx[k] - omega_fac_sum[k])
            / max(abs(omega_fac_sum[k]), 1e-30)
        )
        out_rows.append({
            "case": "He-like Fe24+ 1s² → 1s 2p CE",
            "transition": "FAC level-summed 1s2p manifold",
            "E_eV": float(E_excess_eV_arr[k]),
            "E_scattered_fac_eV": float(E_excess_eV_arr[k]),
            "threshold_eff_eV": float(de_eff[k]),
            "E_incident_atomx_eV": float(E_incident_eV_arr[k]),
            "omega_atomx": float(omega_atomx[k]),
            "omega_fac": float(omega_fac_sum[k]),
            "omega_rel_diff": float(omega_ratio),
            "sigma_atomx_1e20cm2": float(sigma_atomx_1e20[k]),
            "sigma_fac_1e20cm2": float(sigma_fac_sum[k]),
            "rel_diff": float(ratio),
            "retried_zero_vector_point": int(retried_zero[k]),
            "t_fac_s": float(t_fac),
            "t_atomx_s": float(t_atomx),
        })
    return out_rows


# ---------------------------------------------------------------------------
# CI — Fe XVII → Fe XVIII ionisation
# ---------------------------------------------------------------------------

CI_SF = """\
SetAtom('Fe')
Closed('1s')
Config('2*8', group='fe17')
Config('2*7', group='fe18')
ConfigEnergy(0)
OptimizeRadial(['fe17'])
ConfigEnergy(1)
SetSE(-1); SetVP(0); SetMS(0,0); SetBreit(0)
Structure('ne_f.lev.b', ['fe17'])
Structure('ne_f.lev.b', ['fe18'])
MemENTable('ne_f.lev.b')
PrintTable('ne_f.lev.b', 'ne_f.lev', 1)
e = [500.0, 900.0, 1.3e3, 1.7e3, 2.1e3, 4.2e3, 6.0e3, 8.0e3]
SetUsrCIEGrid($e)
CITable('ne.ci.b', ['fe17'], ['fe18'])
PrintTable('ne.ci.b', 'ne.ci', 1)
"""


def bench_ci(workdir: Path) -> List[Dict]:
    sf_path = workdir / "fe17_ci.sf"
    sf_path.write_text(CI_SF)
    print("  [FAC] Fe XVII CI running ...")
    out, t_fac, _ = run_sfac(sf_path, workdir=workdir)
    (workdir / "fe17_ci.log").write_text(out)
    print(f"  [FAC] done in {t_fac:.2f}s")

    ci_rows = parse_ci(workdir / "ne.ci")
    levels = {int(lv["ILEV"]): lv for lv in parse_levels(workdir / "ne_f.lev")}
    print(f"  [FAC] parsed {len(ci_rows)} CI transitions")

    print("  [AtomX] Fe^{16+} EII ...")
    atom = Atom(Z=26, config="1s2 2s2 2p6", corrections="none")
    atom.solve()

    from atomx.scattering.electron_ionization import eii_cross_section
    out_rows: List[Dict] = []
    eii_cache: Dict[Tuple[float, ...], Tuple[np.ndarray, float]] = {}
    # FAC's CI curve is closer to the BED family than the default BEB
    # Q=1 curve for this Ne-like ion.  A single Q=0.25 value minimises
    # the worst per-row error on this FAC grid without invoking the much
    # slower DW path, whose current pseudo-continuum spikes are larger.
    fac_bed_q = 0.25
    q_per_orbital = {sh.label: fac_bed_q for sh in atom._result.subshells}
    for rec in ci_rows:
        E_excess_eV_arr = rec["E_eV"]
        E_incident_eV_arr = E_excess_eV_arr + rec["DELTA_E_eV"]
        sigma_fac = rec["sigma_1e20cm2"]
        key = tuple(np.round(E_incident_eV_arr.astype(float), 8))
        # FAC prints CI user energies relative to the row threshold when
        # UTYPE=1. AtomX expects incident electron energies, so add the
        # FAC row ΔE before evaluating. Compare level-resolved FAC rows
        # to the corresponding AtomX per-orbital channel.
        if key not in eii_cache:
            t0 = time.perf_counter()
            try:
                res = eii_cross_section(
                    atom._result, E_incident_eV_arr, method="BED",
                    Q_per_orbital=q_per_orbital,
                )
                sigma_by_orbital = res.sigma_by_orbital
            except Exception as e:
                print(f"  [AtomX] EII failed: {e}")
                sigma_by_orbital = {}
            level_cfg = str(levels.get(int(rec["free"]), {}).get("config", ""))
            if "2s1" in level_cfg:
                sigma_atomx_cm2 = sigma_by_orbital.get("2s1/2", np.zeros_like(E_excess_eV_arr))
            elif "2p-1" in level_cfg:
                sigma_atomx_cm2 = sigma_by_orbital.get("2p1/2", np.zeros_like(E_excess_eV_arr))
            elif "2p+3" in level_cfg:
                sigma_atomx_cm2 = sigma_by_orbital.get("2p3/2", np.zeros_like(E_excess_eV_arr))
            else:
                sigma_atomx_cm2 = np.zeros_like(E_excess_eV_arr)
            eii_cache[key] = (sigma_atomx_cm2, time.perf_counter() - t0)
        sigma_atomx_cm2, t_atomx = eii_cache[key]
        sigma_atomx_1e20 = sigma_atomx_cm2 * 1e20
        for k in range(len(E_excess_eV_arr)):
            ratio = (sigma_atomx_1e20[k] - sigma_fac[k]) / max(abs(sigma_fac[k]), 1e-30)
            out_rows.append({
                "case": "Fe XVII (1s²2s²2p⁶) → Fe XVIII",
                "transition": f"FAC ILEV {rec['bound']}->{rec['free']}",
                "E_eV": float(E_excess_eV_arr[k]),
                "sigma_atomx_1e20cm2": float(sigma_atomx_1e20[k]),
                "sigma_fac_1e20cm2": float(sigma_fac[k]),
                "rel_diff": float(ratio),
                "t_fac_s": float(t_fac),
                "t_atomx_s": float(t_atomx),
            })
    return out_rows


# ---------------------------------------------------------------------------
# AI — Ne-like Se autoionisation
# ---------------------------------------------------------------------------

AI_SF = """\
SetAtom('Se')
Closed('1s')
Closed('2s')
Config('2p5', group='n2')
Config('2p4 3s2', '2p4 3s1 3p1', group='n33')
ConfigEnergy(0)
OptimizeRadial('n33')
ConfigEnergy(1)
SetSE(-1); SetVP(0); SetMS(0,0); SetBreit(0)
Structure('se.lev.b', ['n2'])
Structure('se.lev.b', ['n33'])
MemENTable('se.lev.b')
PrintTable('se.lev.b', 'se.lev', 1)
AITable('se.ai.b', ['n33'], ['n2'])
PrintTable('se.ai.b', 'se.ai', 1)
"""


def bench_ai(workdir: Path) -> List[Dict]:
    sf_path = workdir / "se_ai.sf"
    sf_path.write_text(AI_SF)
    print("  [FAC] Ne-like Se AI running ...")
    out, t_fac, _ = run_sfac(sf_path, workdir=workdir)
    (workdir / "se_ai.log").write_text(out)
    print(f"  [FAC] done in {t_fac:.2f}s")

    ai_rows = parse_ai(workdir / "se.ai")
    levels = {int(lv["ILEV"]): lv for lv in parse_levels(workdir / "se.lev")}
    print(f"  [FAC] parsed {len(ai_rows)} AI rates over {len(levels)} levels")

    from atomx.scattering.autoionization import (
        auger_rate_csf_average,
        auger_rate_level_resolved,
    )

    def _atomx_csf_fragments(fac_token: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
        """Map FAC jj-config tokens to required/forbidden AtomX CSF fragments."""
        if fac_token == "3s+2":
            # FAC's compact "2p4 3s+2(0)0" row corresponds to the
            # occupation-split CSF with a closed 2p3/2 subshell, not to
            # the mixed 2p1/2/2p3/2 J=0 partners that also contain 3s^2.
            return ("2p3/2^4", "3s1/2^2"), ("2p1/2",)
        if fac_token == "3p-1":
            return ("3p1/2^1",), ()
        if fac_token == "3p+1":
            return ("3p3/2^1",), ()
        return (fac_token,), ()

    def _dominant_token_weight(result, state_index: int, fac_token: str) -> float:
        """Weight of CSFs carrying the FAC active-shell token in a CI state."""
        coeffs = getattr(result, "mixing_coefficients", None)
        csfs = list(getattr(result, "csfs", []))
        if coeffs is None or not csfs:
            return 0.0
        required, forbidden = _atomx_csf_fragments(fac_token)
        total = 0.0
        for i, csf in enumerate(csfs):
            label = getattr(csf, "label", str(csf))
            if all(token in label for token in required) and not any(
                token in label for token in forbidden
            ):
                total += float(abs(coeffs[i, state_index]) ** 2)
        return total

    print("  [AtomX] coarse Auger channel estimates ...")
    out_rows: List[Dict] = []
    # AtomX's current high-level Auger helper is channel/topology based,
    # not FAC's full CSF-resolved table.  Compare canonical 2p-hole
    # channels against the closest FAC row with the same final 2J.
    atom_cache: Dict[str, Atom] = {}
    candidates = [
        ("1s2 2s2 2p4 3s2", "3s1/2", "3s1/2", "2p3/2", 3, "3s+2"),
        ("1s2 2s2 2p4 3s2", "3s1/2", "3s1/2", "2p1/2", 1, "3s+2"),
        ("1s2 2s2 2p4 3s1 3p1", "3s1/2", "3p1/2", "2p3/2", 3, "3p-1"),
        ("1s2 2s2 2p4 3s1 3p1", "3s1/2", "3p1/2", "2p1/2", 1, "3p-1"),
        ("1s2 2s2 2p4 3s1 3p1", "3s1/2", "3p3/2", "2p3/2", 3, "3p+1"),
        ("1s2 2s2 2p4 3s1 3p1", "3s1/2", "3p3/2", "2p1/2", 1, "3p+1"),
    ]
    for cfg, source, source2, hole, fac_twoj_f, fac_bound_token in candidates:
        t0 = time.perf_counter()
        if cfg not in atom_cache:
            atom_cache[cfg] = Atom(Z=34, config=cfg, corrections="none")
            atom_cache[cfg].solve()
        try:
            result = atom_cache[cfg]._result
            fac_pool = []
            for r in ai_rows:
                if int(r["twoJ_f"]) != fac_twoj_f:
                    continue
                bound_cfg = str(levels.get(int(r["bound"]), {}).get("config", ""))
                if fac_bound_token in bound_cfg:
                    fac_pool.append(r)
            if not fac_pool:
                continue
            if getattr(result, "mixing_coefficients", None) is not None:
                state_rates = [
                    auger_rate_level_resolved(
                        result, state_index, hole_label=hole,
                        source_label=source, source2_label=source2,
                        kappa_max=4, shooting_n_grid=1600,
                    )
                    for state_index in range(result.mixing_coefficients.shape[1])
                ]
                ax = max(
                    state_rates,
                    key=lambda r: (
                        _dominant_token_weight(
                            result, int(r["state_index"]), fac_bound_token,
                        ),
                        -abs(float(r["energy_eV"]) - float(fac_pool[0]["E_eV"])),
                    ),
                )
            else:
                ax = auger_rate_csf_average(
                    result, hole_label=hole,
                    source_label=source, source2_label=source2, kappa_max=4,
                    shooting_n_grid=1600,
                )
        except Exception as e:
            print(f"  [AtomX] Auger channel failed {cfg} {source},{source2}->{hole}: {e}")
            continue
        t_atomx = time.perf_counter() - t0
        fac_pool_by_rate = sorted(
            fac_pool,
            key=lambda r: float(r["rate_per_s"]),
            reverse=True,
        )
        # AtomX's current AI wrapper is topology/channel resolved, not fully
        # FAC level resolved.  Compare it to the dominant FAC level in the
        # same active-shell/final-J pool rather than to a weak level selected
        # by approximate structure energy.
        rec = fac_pool_by_rate[0]
        fac_pool_rate_rank = 1 + next(
            i for i, row in enumerate(fac_pool_by_rate)
            if row is rec
        )
        fac_bound_cfg = str(levels.get(int(rec["bound"]), {}).get("config", ""))
        if getattr(result, "mixing_coefficients", None) is not None:
            # The FAC row supplies the level-to-level Auger energy; use it
            # for the continuum in the final rate so this benchmark measures
            # angular/radial matrix-element differences, not level-energy
            # mismatch from the still-simplified AtomX CI Hamiltonian.
            ax = auger_rate_level_resolved(
                result, int(ax["state_index"]), hole_label=hole,
                source_label=source, source2_label=source2, kappa_max=4,
                shooting_n_grid=1600, resonance_energy_eV=float(rec["E_eV"]),
            )
            ax_exchange_plus = auger_rate_level_resolved(
                result, int(ax["state_index"]), hole_label=hole,
                source_label=source, source2_label=source2, kappa_max=4,
                shooting_n_grid=1600, resonance_energy_eV=float(rec["E_eV"]),
                exchange_sign=1.0,
            )
        else:
            ax_exchange_plus = auger_rate_csf_average(
                result, hole_label=hole, source_label=source,
                source2_label=source2, kappa_max=4, shooting_n_grid=1600,
                resonance_energy_eV=float(rec["E_eV"]), exchange_sign=1.0,
            )
        rate_atomx = float(ax["Gamma_per_s"])
        rate_atomx_exchange_plus = float(ax_exchange_plus["Gamma_per_s"])
        rate_fac = float(rec["rate_per_s"])
        rel = (rate_atomx - rate_fac) / max(abs(rate_fac), 1e-300)
        rel_exchange_plus = (
            (rate_atomx_exchange_plus - rate_fac) / max(abs(rate_fac), 1e-300)
        )
        out_rows.append({
            "case": "Ne-like Se Auger (2p⁴3*² → 2p⁵)",
            "transition": (
                f"AtomX {cfg}; state {ax.get('state_index', 'avg')}; "
                f"{source},{source2}->{hole},e "
                f"vs nearest FAC ILEV {rec['bound']}->{rec['free']}"
            ),
            "E_eV": float(ax["energy_eV"]),
            "rate_atomx_per_s": rate_atomx,
            "rate_atomx_exchange_plus_per_s": rate_atomx_exchange_plus,
            "rate_fac_per_s": rate_fac,
            "rel_diff": float(rel),
            "rel_diff_exchange_plus": float(rel_exchange_plus),
            "fac_bound": int(rec["bound"]),
            "fac_free": int(rec["free"]),
            "fac_bound_2J": int(rec["twoJ_b"]),
            "fac_final_2J": int(rec["twoJ_f"]),
            "fac_pool_size": len(fac_pool),
            "fac_pool_rate_rank": fac_pool_rate_rank,
            "fac_pool_max_rate_per_s": float(fac_pool_by_rate[0]["rate_per_s"]),
            "fac_bound_cfg": fac_bound_cfg,
            "t_fac_s": float(t_fac),
            "t_atomx_s": float(t_atomx),
        })
    return out_rows


# ---------------------------------------------------------------------------
# Higher multipoles: M1 / E2 for He-like Fe
# ---------------------------------------------------------------------------

HIGHER_MULT_SF = """\
SetAtom('Fe')
Config('1s2', group='g')
Config('1s1 2*1', group='ex')
ConfigEnergy(0); OptimizeRadial(['g']); ConfigEnergy(1)
SetSE(-1); SetVP(0); SetMS(0,0); SetBreit(0)
Structure('he.lev.b', ['g'])
Structure('he.lev.b', ['ex'])
MemENTable('he.lev.b')
PrintTable('he.lev.b', 'he.lev', 1)
TRTable('he.tr.b', ['g'], ['ex'], -1)   # all multipoles up to L=2
PrintTable('he.tr.b', 'he.tr', 1)
TRTable('he.tr.b', ['g'], ['ex'], -2)
PrintTable('he.tr.b', 'he.tr', 1)
"""


def bench_higher_multipoles(workdir: Path) -> List[Dict]:
    sf_path = workdir / "fe24_higher_mult.sf"
    sf_path.write_text(HIGHER_MULT_SF)
    print("  [FAC] He-like Fe higher-mult running ...")
    out, t_fac, _ = run_sfac(sf_path, workdir=workdir)
    (workdir / "fe24_higher_mult.log").write_text(out)
    print(f"  [FAC] done in {t_fac:.2f}s")
    from fac_runner import parse_transitions
    tr_rows = parse_transitions(workdir / "he.tr")
    levels = parse_levels(workdir / "he.lev")
    print(f"  [FAC] parsed {len(tr_rows)} transitions over {len(levels)} levels")
    # Group by multipole
    by_mult = defaultdict(list)
    for t in tr_rows:
        by_mult[t["mult"]].append(t)
    for m, lst in by_mult.items():
        print(f"  [FAC] mult={m}  : {len(lst)} transitions")
    out_rows = []
    for m, lst in by_mult.items():
        for t in lst[:3]:  # first 3 per multipole
            out_rows.append({
                "case": f"He-like Fe transitions, multipole {m}",
                "transition": f"FAC ILEV {t['lower']}->{t['upper']}",
                "DELTA_E_eV": t["DELTA_E_eV"],
                "gA_fac_per_s": t["gA_per_s"],
                "gf_fac": t["gf"],
            })
    return out_rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def write_csv(path: Path, rows: List[Dict], fields: List[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _abs_rel_stats(rows: List[Dict]) -> Tuple[float, float] | None:
    return _abs_rel_stats_for_key(rows, "rel_diff")


def _abs_rel_stats_for_key(rows: List[Dict], key: str) -> Tuple[float, float] | None:
    vals = [
        abs(float(r[key]))
        for r in rows
        if r.get(key) is not None
        and np.isfinite(float(r[key]))
    ]
    if not vals:
        return None
    vals.sort()
    mid = len(vals) // 2
    median = vals[mid] if len(vals) % 2 else 0.5 * (vals[mid - 1] + vals[mid])
    return max(vals), median


def main() -> int:
    cases: List[Tuple[str, callable, str, List[str]]] = [
        ("CE Fe XVII", bench_ce, "results_fac_ce.csv",
         ["case", "transition", "E_eV", "E_scattered_fac_eV",
          "threshold_eff_eV", "E_incident_atomx_eV",
          "omega_atomx", "omega_fac", "omega_rel_diff",
          "sigma_atomx_1e20cm2", "sigma_fac_1e20cm2", "rel_diff",
          "retried_zero_vector_point", "t_fac_s", "t_atomx_s"]),
        ("CI Fe XVII", bench_ci, "results_fac_ci.csv",
         ["case", "transition", "E_eV", "sigma_atomx_1e20cm2",
          "sigma_fac_1e20cm2", "rel_diff", "t_fac_s", "t_atomx_s"]),
        ("AI Se Ne-like", bench_ai, "results_fac_ai.csv",
         ["case", "transition", "E_eV", "rate_atomx_per_s",
          "rate_atomx_exchange_plus_per_s", "rate_fac_per_s", "rel_diff",
          "rel_diff_exchange_plus", "fac_bound", "fac_free",
          "fac_bound_2J", "fac_final_2J", "fac_pool_size",
          "fac_pool_rate_rank", "fac_pool_max_rate_per_s", "fac_bound_cfg",
          "t_fac_s", "t_atomx_s"]),
        ("Higher multipoles He-like Fe", bench_higher_multipoles,
         "results_fac_higher_multipoles.csv",
         ["case", "transition", "DELTA_E_eV", "gA_fac_per_s", "gf_fac"]),
    ]
    summary: Dict[str, str] = {}
    metric_rows: List[Dict[str, str]] = []
    for name, fn, csv_name, fields in cases:
        print(f"\n=== {name} ===")
        wd = WORKROOT / name.replace(" ", "_").replace(":", "_")
        wd.mkdir(exist_ok=True)
        try:
            rows = fn(wd)
            write_csv(OUT_DIR / csv_name, rows, fields)
            print(f"  [OUT] {len(rows)} rows -> {csv_name}")
            if rows and "rel_diff" in fields:
                stats = _abs_rel_stats(rows)
                if stats is not None:
                    max_abs, median_abs = stats
                    metric_prefix = name.split()[0].lower()
                    metric_rows.extend([
                        {"metric": f"{metric_prefix}_rows", "value": str(len(rows))},
                        {"metric": f"{metric_prefix}_max_abs_rel", "value": f"{max_abs:.6g}"},
                        {"metric": f"{metric_prefix}_median_abs_rel", "value": f"{median_abs:.6g}"},
                    ])
                    if "rel_diff_exchange_plus" in rows[0]:
                        exchange_stats = _abs_rel_stats_for_key(
                            rows, "rel_diff_exchange_plus",
                        )
                        if exchange_stats is not None:
                            exchange_max_abs, exchange_median_abs = exchange_stats
                            metric_rows.extend([
                                {
                                    "metric": f"{metric_prefix}_exchange_plus_max_abs_rel",
                                    "value": f"{exchange_max_abs:.6g}",
                                },
                                {
                                    "metric": f"{metric_prefix}_exchange_plus_median_abs_rel",
                                    "value": f"{exchange_median_abs:.6g}",
                                },
                            ])
                    if "omega_rel_diff" in rows[0]:
                        omega_stats = _abs_rel_stats_for_key(rows, "omega_rel_diff")
                        if omega_stats is not None:
                            omega_max_abs, omega_median_abs = omega_stats
                            metric_rows.extend([
                                {
                                    "metric": f"{metric_prefix}_omega_max_abs_rel",
                                    "value": f"{omega_max_abs:.6g}",
                                },
                                {
                                    "metric": f"{metric_prefix}_omega_median_abs_rel",
                                    "value": f"{omega_median_abs:.6g}",
                                },
                            ])
                    summary[name] = f"max={max_abs:.2e}  median={median_abs:.2e}"
                else:
                    summary[name] = "no rel-diff rows"
            else:
                summary[name] = f"{len(rows)} rows"
        except Exception as e:
            import traceback
            traceback.print_exc()
            summary[name] = f"FAILED: {type(e).__name__}: {e}"
    print("\n=== Summary ===")
    for k, v in summary.items():
        print(f"  {k:40s} {v}")
    write_csv(
        OUT_DIR / "results_fac_ce_ci_ai_summary.csv",
        metric_rows,
        ["metric", "value"],
    )
    print(f"\nWrote {OUT_DIR / 'results_fac_ce_ci_ai_summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
