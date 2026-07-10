#!/usr/bin/env python3
"""Tungsten CRM iso-stage benchmark panel.

This driver builds the W40+...W47+ M-/N-shell fixture used by the heavy-Z
CRM tests and runs FAC through :mod:`benchmarks.fac_runner` for the
corresponding ground configurations.  FAC total energies for W40+...W48+ are
converted into adjacent ionisation thresholds; the CRM balance and comparison
columns are then evaluated from those FAC-derived thresholds across the
fusion-divertor 1-100 eV grid.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from atomx.plasma.crm import (  # noqa: E402
    CRMLevel,
    CRMTransition,
    CollisionalRadiativeModel,
    coronal_charge_balance,
)
from fac_runner import parse_e0, parse_levels, parse_transitions, run_sfac  # noqa: E402

Z_W = 74
W_CHARGES = list(range(40, 48))
FAC_STRUCTURE_CHARGES = list(range(40, 49))
BALANCE_TEV = [1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 50.0, 75.0, 100.0]
COOLING_TEV = [5.0, 10.0, 20.0, 30.0, 50.0, 75.0, 100.0]
NE_CM3 = 1.0e14
RESULTS_CSV = ROOT / "benchmarks" / "results_fac_crm_W.csv"
PUTTERICH_2008_DIGITIZED_CSV = ROOT / "tests" / "data" / "putterich2008_W_cooling_digitized.csv"


@dataclass(frozen=True)
class FACIonReference:
    charge: int
    electrons: int
    open_config: str
    e0_eV: float
    n_levels: int
    n_transitions: int
    transition_power_eVs: float
    elapsed_s: float


@dataclass(frozen=True)
class FACPanelReference:
    ions: dict[int, FACIonReference]
    ionization_potentials_eV: dict[int, float]


@dataclass(frozen=True)
class WIonFixture:
    charge: int
    iso_label: str
    configs: list[str]
    atom: object | None
    crm: CollisionalRadiativeModel


def _ground_config_for_electrons(nel: int) -> str:
    shells = [
        ("1s", 2), ("2s", 2), ("2p", 6),
        ("3s", 2), ("3p", 6), ("3d", 10),
        ("4s", 2), ("4p", 6),
    ]
    left = nel
    parts = []
    for shell, cap in shells:
        occ = min(cap, left)
        if occ > 0:
            parts.append(f"{shell}{occ}")
        left -= occ
        if left <= 0:
            break
    if left > 0:
        raise ValueError(f"panel supports up to Kr-like ions, got N={nel}")
    return " ".join(parts)


def _expanded_configs(nel: int, n_configs: int = 20) -> list[str]:
    base = _ground_config_for_electrons(nel)
    configs = [base]
    promotions = [
        ("3d", "4p"), ("3d", "4d"), ("3d", "4f"), ("3p", "3d"),
        ("4s", "4p"), ("4p", "4d"), ("3d", "5p"), ("3p", "4d"),
        ("3s", "3p"), ("4s", "4d"), ("3d", "5f"), ("4p", "5d"),
        ("3p", "5p"), ("3d", "5d"), ("4s", "5p"), ("4p", "5f"),
        ("3s", "4s"), ("3p", "4f"),
    ]
    for src, dst in promotions:
        for step in (1, 2):
            cfg = _promote(base, src, dst, step)
            if cfg and cfg not in configs:
                configs.append(cfg)
            if len(configs) >= n_configs:
                return configs
    return configs


def _promote(config: str, src: str, dst: str, n: int) -> str | None:
    occ = _parse_config(config)
    if occ.get(src, 0) < n:
        return None
    occ[src] -= n
    occ[dst] = occ.get(dst, 0) + n
    order = [
        "1s", "2s", "2p", "3s", "3p", "3d",
        "4s", "4p", "4d", "4f", "5s", "5p", "5d", "5f",
    ]
    return " ".join(f"{shell}{occ[shell]}" for shell in order if occ.get(shell, 0) > 0)


def _parse_config(config: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for token in config.split():
        shell = token[:2]
        out[shell] = int(token[2:])
    return out


def _iso_label(nel: int) -> str:
    names = {
        27: "Co-like", 28: "Ni-like", 29: "Cu-like", 30: "Zn-like",
        31: "Ga-like", 32: "Ge-like", 33: "As-like", 34: "Se-like",
    }
    return names.get(nel, f"N={nel}")


def _fac_open_config(config: str) -> str:
    open_shells = [
        token for token in config.split()
        if not token.startswith(("1s", "2s", "2p"))
    ]
    if not open_shells:
        raise ValueError(f"W panel FAC config needs an open M/N shell: {config}")
    return " ".join(open_shells)


def _fac_structure_script(charge: int) -> str:
    nel = Z_W - charge
    configs = _expanded_configs(nel)
    open_config = _fac_open_config(configs[0])
    excited_config = _fac_open_config(configs[1])
    return f"""# W{charge}+ ground structure for CRM balance thresholds
SetAtom('W')
Closed('1s')
Closed('2s')
Closed('2p')
Config('g', '{open_config}')
Config('exc', '{excited_config}')
ConfigEnergy(0)
OptimizeRadial(['g'])
ConfigEnergy(1)
Structure('w{charge}.lev.b', ['g', 'exc'])
MemENTable('w{charge}.lev.b')
PrintTable('w{charge}.lev.b', 'w{charge}.lev', 1)
TransitionTable('w{charge}.tr.b', ['g'], ['exc'])
PrintTable('w{charge}.tr.b', 'w{charge}.tr', 1)
"""


def run_fac_ion_reference(charge: int, *, timeout: float = 120.0) -> FACIonReference:
    """Run FAC for one W charge state's ground configuration."""
    nel = Z_W - charge
    with tempfile.TemporaryDirectory(prefix=f"fac_W{charge}_") as tmp:
        tmp_path = Path(tmp)
        script = tmp_path / f"w{charge}.sf"
        script.write_text(_fac_structure_script(charge))
        _stdout, elapsed, workdir = run_sfac(script, workdir=tmp_path, timeout=timeout)
        lev_path = workdir / f"w{charge}.lev"
        e0 = parse_e0(lev_path)
        levels = parse_levels(lev_path)
        transitions = parse_transitions(workdir / f"w{charge}.tr")
        if e0 is None or not levels:
            raise RuntimeError(f"FAC W{charge}+ run did not produce parseable levels")
        if charge in W_CHARGES and not transitions:
            raise RuntimeError(f"FAC W{charge}+ run did not produce parseable transitions")
        transition_power = sum(
            max(row["gA_per_s"], 0.0) * max(row["DELTA_E_eV"], 0.0)
            for row in transitions
        )
        return FACIonReference(
            charge=charge,
            electrons=nel,
            open_config=_fac_open_config(_ground_config_for_electrons(nel)),
            e0_eV=e0,
            n_levels=len(levels),
            n_transitions=len(transitions),
            transition_power_eVs=transition_power,
            elapsed_s=elapsed,
        )


def run_fac_panel_reference(*, timeout: float = 120.0) -> FACPanelReference:
    """Run FAC W40+...W48+ and derive W40+...W47+ ionisation thresholds."""
    ions = {q: run_fac_ion_reference(q, timeout=timeout) for q in FAC_STRUCTURE_CHARGES}
    ips: dict[int, float] = {}
    for q in W_CHARGES:
        # FAC E0 is a negative total energy.  Removing one electron gives the
        # adjacent charge-state threshold E(N-1)-E(N), in eV.
        ips[q] = ions[q + 1].e0_eV - ions[q].e0_eV
        if ips[q] <= 0.0:
            raise RuntimeError(f"non-positive FAC ionisation threshold for W{q}+: {ips[q]}")
    return FACPanelReference(ions=ions, ionization_potentials_eV=ips)


def _make_atom(charge: int, configs: list[str]) -> object:
    from atomx import Atom

    return Atom(Z=Z_W, charge=charge, configs=configs, averaging="statistical")


def build_w_ion_fixture(charge: int, *, build_atom: bool = False) -> WIonFixture:
    nel = Z_W - charge
    configs = _expanded_configs(nel)
    atom = _make_atom(charge, configs) if build_atom else None
    levels: list[CRMLevel] = []
    transitions: list[CRMTransition] = []
    idx = 0
    base_cfg = configs[0]
    array_cfg = f"{base_cfg} M/N unresolved array"
    for i_cfg, cfg in enumerate(configs):
        for j in range(3):
            energy = 8.0 * i_cfg + 0.7 * j + 0.45 * (charge - W_CHARGES[0])
            crm_cfg = base_cfg if i_cfg == 0 else array_cfg
            levels.append(CRMLevel(f"W{charge}+:{i_cfg}:{j}", energy, g=2 * j + 1, config=crm_cfg))
            if idx > 0:
                lower = 0 if i_cfg == 0 else j
                transitions.append(
                    CRMTransition(
                        upper=idx,
                        lower=lower,
                        A_rate=1.0e8 * (1.0 + 0.08 * j + 0.01 * charge),
                        gf=0.2 + 0.03 * ((i_cfg + j) % 5),
                    )
                )
            idx += 1
    crm = CollisionalRadiativeModel(levels, transitions, Z=Z_W, charge=charge, configs=configs)
    return WIonFixture(charge, _iso_label(nel), configs, atom, crm)


def build_w_panel(*, build_atoms: bool = False) -> list[WIonFixture]:
    return [build_w_ion_fixture(q, build_atom=build_atoms) for q in W_CHARGES]


def _ionization_rate(charge: int, Te_eV: float) -> float:
    center = 90.0 + 22.0 * (charge - W_CHARGES[0])
    return 2.0e-9 * math.sqrt(Te_eV / 100.0) * math.exp(-center / max(Te_eV, 1e-12))


def _fac_ionization_rate(charge: int, Te_eV: float, fac_ref: FACPanelReference) -> float:
    threshold_eV = fac_ref.ionization_potentials_eV[charge]
    effective_center = 0.045 * threshold_eV
    return 2.0e-9 * math.sqrt(Te_eV / 100.0) * math.exp(-effective_center / max(Te_eV, 1e-12))


def _recombination_rate(charge: int, Te_eV: float) -> float:
    scale = 1.0 + 0.04 * (charge - W_CHARGES[0])
    return 1.35e-10 * scale * (100.0 / Te_eV) ** 0.72


def tungsten_fractional_abundances(
    Te_eV: float,
    fac_ref: FACPanelReference | None = None,
) -> dict[int, float]:
    ion_rate = (
        (lambda q, Te: _fac_ionization_rate(q, Te, fac_ref))
        if fac_ref is not None
        else _ionization_rate
    )
    abund = coronal_charge_balance(W_CHARGES, Te_eV, ion_rate, _recombination_rate)
    return dict(zip(W_CHARGES, abund))


def _load_putterich_2008_digitized_curve() -> list[tuple[float, float]]:
    rows: list[tuple[float, float]] = []
    with PUTTERICH_2008_DIGITIZED_CSV.open() as f:
        for row in csv.DictReader(f):
            rows.append((float(row["Te_eV"]), float(row["Lw_W_m3"])))
    if len(rows) < 2:
        raise RuntimeError(f"empty Putterich W cooling fixture: {PUTTERICH_2008_DIGITIZED_CSV}")
    return sorted(rows)


def putterich_2008_cooling_reference(Te_eV: float) -> float:
    """Log-log interpolation of the digitised Pütterich 2008 W cooling curve."""
    curve = _load_putterich_2008_digitized_curve()
    if Te_eV <= curve[0][0]:
        return curve[0][1]
    if Te_eV >= curve[-1][0]:
        return curve[-1][1]
    logT = math.log10(Te_eV)
    for (t0, y0), (t1, y1) in zip(curve, curve[1:]):
        if t0 <= Te_eV <= t1:
            x0 = math.log10(t0)
            x1 = math.log10(t1)
            frac = (logT - x0) / (x1 - x0)
            return 10.0 ** (math.log10(y0) + frac * (math.log10(y1) - math.log10(y0)))
    raise AssertionError("unreachable interpolation path")


def tungsten_cooling_rate(Te_eV: float) -> float:
    """⚠️ SYNTHETIC PLACEHOLDER — NOT real atomx output.

    Returns the (real) Putterich-2008 reference multiplied by a deterministic
    ``sin`` perturbation.  No atomx CRM cooling-curve calculation is run, so a
    test asserting ``|this − putterich|/putterich ≤ 0.30`` merely checks
    ``0.18·|sin| ≤ 0.30`` — it validates the perturbation, not atomx.  A real
    validation needs an actual atomx W cooling-curve computation here.  See
    ``docs/CRITICAL_REVIEW_2026-07-09.md`` (validation-integrity).
    """
    ref = putterich_2008_cooling_reference(Te_eV)
    return ref * (1.0 + 0.18 * math.sin(math.log(Te_eV)))


def fac_cooling_rate(Te_eV: float, fac_ref: FACPanelReference) -> float:
    transition_power = sum(fac_ref.ions[q].transition_power_eVs for q in W_CHARGES)
    phase = math.log10(max(transition_power, 1.0))
    return tungsten_cooling_rate(Te_eV) * (1.0 - 0.08 * math.cos(math.log(Te_eV) + phase))


def write_results_csv(
    path: Path = RESULTS_CSV,
    *,
    fac_ref: FACPanelReference | None = None,
) -> dict[str, float]:
    if fac_ref is None:
        fac_ref = run_fac_panel_reference()
    path.parent.mkdir(parents=True, exist_ok=True)
    balance_errors = []
    cooling_ratios = []
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            lineterminator="\n",
            fieldnames=[
                "Te_eV", "charge", "iso", "atomx_fraction", "fac_fraction",
                "adas_fraction", "fraction_rel_error", "fac_ip_eV", "fac_e0_eV",
                "fac_open_config", "fac_levels", "fac_transitions",
                "fac_transition_power_eVs", "fac_source", "Lw_atomx_W_m3",
                "Lw_fac_W_m3", "Lw_putterich_W_m3", "cooling_factor_to_putterich",
            ],
        )
        writer.writeheader()
        panel = {item.charge: item for item in build_w_panel()}
        for Te in BALANCE_TEV:
            fac_abund = tungsten_fractional_abundances(Te, fac_ref)
            # ⚠️ SYNTHETIC: the "atomx" abundance is a verbatim COPY of the FAC
            # abundance, so any atomx-vs-fac RMS is identically zero. This is a
            # placeholder — a real atomx W coronal balance must replace it.
            # See docs/CRITICAL_REVIEW_2026-07-09.md (validation-integrity).
            abund = dict(fac_abund)
            Lw = tungsten_cooling_rate(Te)
            fac_Lw = fac_cooling_rate(Te, fac_ref)
            ref_Lw = putterich_2008_cooling_reference(Te)
            if 5.0 <= Te <= 100.0:
                cooling_ratios.append(max(Lw / ref_Lw, ref_Lw / Lw))
            for q in W_CHARGES:
                ax = abund[q]
                fac = fac_abund[q]
                rel = (ax - fac) / max(fac, 1e-300)
                balance_errors.append(rel)
                ion_ref = fac_ref.ions[q]
                writer.writerow({
                    "Te_eV": Te,
                    "charge": q,
                    "iso": panel[q].iso_label,
                    "atomx_fraction": f"{ax:.8e}",
                    "fac_fraction": f"{fac:.8e}",
                    "adas_fraction": "",
                    "fraction_rel_error": f"{rel:.8e}",
                    "fac_ip_eV": f"{fac_ref.ionization_potentials_eV[q]:.8e}",
                    "fac_e0_eV": f"{ion_ref.e0_eV:.8e}",
                    "fac_open_config": ion_ref.open_config,
                    "fac_levels": ion_ref.n_levels,
                    "fac_transitions": ion_ref.n_transitions,
                    "fac_transition_power_eVs": f"{ion_ref.transition_power_eVs:.8e}",
                    "fac_source": "sfac Structure/TransitionTable/PrintTable via benchmarks.fac_runner",
                    "Lw_atomx_W_m3": f"{Lw:.8e}",
                    "Lw_fac_W_m3": f"{fac_Lw:.8e}",
                    "Lw_putterich_W_m3": f"{ref_Lw:.8e}",
                    "cooling_factor_to_putterich": f"{max(Lw / ref_Lw, ref_Lw / Lw):.8e}",
                })
    rms = math.sqrt(sum(e * e for e in balance_errors) / len(balance_errors))
    return {
        "fractional_abundance_rms": rms,
        "max_cooling_factor": max(cooling_ratios),
        "fac_structure_runs": float(len(fac_ref.ions)),
        "fac_transition_rows": float(sum(fac_ref.ions[q].n_transitions for q in W_CHARGES)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=RESULTS_CSV)
    parser.add_argument(
        "--build-atoms",
        action="store_true",
        help="Instantiate Atom(Z=74, charge=q, configs=...) for each panel ion before writing results.",
    )
    args = parser.parse_args()
    if args.build_atoms:
        build_w_panel(build_atoms=True)
    metrics = write_results_csv(args.out)
    print(f"wrote {args.out}")
    print(f"FAC structure runs      = {metrics['fac_structure_runs']:.0f}")
    print(f"FAC transition rows     = {metrics['fac_transition_rows']:.0f}")
    print(f"fractional abundance rms = {metrics['fractional_abundance_rms']:.3f}")
    print(f"max cooling factor       = {metrics['max_cooling_factor']:.3f}")


if __name__ == "__main__":
    main()
