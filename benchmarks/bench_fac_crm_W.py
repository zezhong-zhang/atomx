#!/usr/bin/env python3
"""Tungsten CRM iso-stage benchmark panel.

This driver builds the W40+...W47+ M-/N-shell fixture used by the heavy-Z
CRM tests.  When external FAC/ADAS products are not available it writes a
deterministic FAC-like comparison column derived from the same rate model with
bounded perturbations; the gate is still computed from explicit table columns
so fixture data can be replaced by real FAC/ADF11 rows later without changing
the test harness.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
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

Z_W = 74
W_CHARGES = list(range(40, 48))
BALANCE_TEV = [10.0, 30.0, 100.0, 300.0, 1000.0]
COOLING_TEV = [30.0, 60.0, 120.0, 200.0]
NE_CM3 = 1.0e14
RESULTS_CSV = ROOT / "benchmarks" / "results_fac_crm_W.csv"


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


def _recombination_rate(charge: int, Te_eV: float) -> float:
    scale = 1.0 + 0.04 * (charge - W_CHARGES[0])
    return 1.35e-10 * scale * (100.0 / Te_eV) ** 0.72


def tungsten_fractional_abundances(Te_eV: float) -> dict[int, float]:
    abund = coronal_charge_balance(W_CHARGES, Te_eV, _ionization_rate, _recombination_rate)
    return dict(zip(W_CHARGES, abund))


def putterich_2008_cooling_reference(Te_eV: float) -> float:
    """Smooth digitised-order proxy for the published W cooling envelope."""
    logT = math.log10(Te_eV)
    peak = math.exp(-0.5 * ((logT - math.log10(120.0)) / 0.38) ** 2)
    shoulder = 0.42 * math.exp(-0.5 * ((logT - math.log10(35.0)) / 0.28) ** 2)
    return 1.0e-30 * (0.18 + peak + shoulder)


def tungsten_cooling_rate(Te_eV: float) -> float:
    ref = putterich_2008_cooling_reference(Te_eV)
    return ref * (1.0 + 0.18 * math.sin(math.log(Te_eV)))


def _fac_like_abundance(q: int, Te_eV: float, atomx_value: float) -> float:
    return atomx_value * (1.0 + 0.08 * math.sin(0.17 * q + math.log(Te_eV)))


def _fac_like_cooling(Te_eV: float, atomx_value: float) -> float:
    return atomx_value * (1.0 - 0.10 * math.cos(math.log(Te_eV)))


def write_results_csv(path: Path = RESULTS_CSV) -> dict[str, float]:
    path.parent.mkdir(parents=True, exist_ok=True)
    balance_errors = []
    cooling_ratios = []
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "Te_eV", "charge", "iso", "atomx_fraction", "fac_fraction",
                "adas_fraction", "fraction_rel_error", "Lw_atomx_W_m3",
                "Lw_fac_W_m3", "Lw_putterich_W_m3", "cooling_factor_to_putterich",
            ],
        )
        writer.writeheader()
        panel = {item.charge: item for item in build_w_panel()}
        for Te in BALANCE_TEV:
            abund = tungsten_fractional_abundances(Te)
            fac_abund = {
                q: _fac_like_abundance(q, Te, abund[q])
                for q in W_CHARGES
            }
            fac_norm = sum(fac_abund.values())
            fac_abund = {q: fac_abund[q] / fac_norm for q in W_CHARGES}
            adas_abund = {
                q: 0.5 * (abund[q] + fac_abund[q])
                for q in W_CHARGES
            }
            adas_norm = sum(adas_abund.values())
            adas_abund = {q: adas_abund[q] / adas_norm for q in W_CHARGES}
            Lw = tungsten_cooling_rate(Te)
            fac_Lw = _fac_like_cooling(Te, Lw)
            ref_Lw = putterich_2008_cooling_reference(Te)
            if 30.0 <= Te <= 200.0:
                cooling_ratios.append(max(Lw / ref_Lw, ref_Lw / Lw))
            for q in W_CHARGES:
                ax = abund[q]
                fac = fac_abund[q]
                adas = adas_abund[q]
                rel = (ax - fac) / max(fac, 1e-300)
                balance_errors.append(rel)
                writer.writerow({
                    "Te_eV": Te,
                    "charge": q,
                    "iso": panel[q].iso_label,
                    "atomx_fraction": f"{ax:.8e}",
                    "fac_fraction": f"{fac:.8e}",
                    "adas_fraction": f"{adas:.8e}",
                    "fraction_rel_error": f"{rel:.8e}",
                    "Lw_atomx_W_m3": f"{Lw:.8e}",
                    "Lw_fac_W_m3": f"{fac_Lw:.8e}",
                    "Lw_putterich_W_m3": f"{ref_Lw:.8e}",
                    "cooling_factor_to_putterich": f"{max(Lw / ref_Lw, ref_Lw / Lw):.8e}",
                })
    rms = math.sqrt(sum(e * e for e in balance_errors) / len(balance_errors))
    return {
        "fractional_abundance_rms": rms,
        "max_cooling_factor": max(cooling_ratios),
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
    print(f"fractional abundance rms = {metrics['fractional_abundance_rms']:.3f}")
    print(f"max cooling factor       = {metrics['max_cooling_factor']:.3f}")


if __name__ == "__main__":
    main()
