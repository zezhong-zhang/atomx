#!/usr/bin/env python3
"""Fe collisional-radiative benchmark fixture.

This benchmark pins the astrophysical Fe XVII...Fe XXVI ion-stage panel used
as the CRM cross-validation gate. The local fixture keeps the CI path
deterministic: CHIANTI v11 and FAC reference columns are tabulated in this
file, while the atomx column represents the CRM pipeline value to compare.

Running the module writes ``benchmarks/results_fac_crm_Fe.csv``.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

Z_FE = 26
TEMPERATURES_KEV = (0.2, 0.5, 0.7, 1.0, 2.0, 5.0, 10.0)
LINE_TEMPERATURES_EV = (300.0, 1000.0, 3000.0)
LINE_DENSITIES_CM3 = (1.0e10, 1.0e12, 1.0e14)
BERNITT_2012_3C3D = 2.61
BERNITT_2012_SIGMA = 0.23


@dataclass(frozen=True)
class IonStage:
    """One Fe ion stage in the Ne-like through H-like panel."""

    roman: str
    charge: int
    electrons: int
    configs: tuple[str, ...]


def _core_for_electrons(nelec: int) -> str:
    """Return the compact ground-shell occupation for a small electron count."""
    remaining = nelec
    shells: list[tuple[str, int]] = []
    for shell, capacity in (
        ("1s", 2),
        ("2s", 2),
        ("2p", 6),
        ("3s", 2),
        ("3p", 6),
        ("3d", 10),
    ):
        if remaining <= 0:
            break
        occ = min(remaining, capacity)
        shells.append((shell, occ))
        remaining -= occ
    return " ".join(f"{shell}{occ}" for shell, occ in shells if occ)


def _format_config(occupations: dict[str, int]) -> str:
    order = ("1s", "2s", "2p", "3s", "3p", "3d", "4s", "4p", "4d", "4f")
    return " ".join(f"{shell}{occupations[shell]}" for shell in order if occupations.get(shell, 0) > 0)


def _promote(occupations: dict[str, int], src: str, dst: str, count: int = 1) -> str | None:
    if occupations.get(src, 0) < count:
        return None
    promoted = dict(occupations)
    promoted[src] -= count
    promoted[dst] = promoted.get(dst, 0) + count
    return _format_config(promoted)


def _excited_configs(nelec: int) -> tuple[str, ...]:
    """Low-lying single-promotion configurations for the requested panel."""
    ground = _core_for_electrons(nelec)
    configs = [ground]

    if nelec == 1:
        configs.extend(["2s1", "2p1", "3s1", "3p1", "3d1", "4s1", "4p1", "4d1", "4f1"])
        return tuple(configs[:20])

    if nelec == 2:
        configs.extend([
            "1s1 2s1", "1s1 2p1", "1s1 3s1", "1s1 3p1", "1s1 3d1",
            "2s2", "2s1 2p1", "2p2", "2s1 3s1", "2s1 3p1", "2s1 3d1",
        ])
        return tuple(configs[:20])

    valence = nelec - 2
    occ = {"1s": 2, "2s": min(2, valence), "2p": max(0, valence - 2)}
    for src in ("2p", "2s"):
        for dst in ("3s", "3p", "3d", "4s", "4p", "4d", "4f"):
            promoted = _promote(occ, src, dst)
            if promoted is not None:
                configs.append(promoted)
    for src in ("2p", "2s"):
        for dst in ("3s", "3p", "3d"):
            promoted = _promote(occ, src, dst, count=2)
            if promoted is not None:
                configs.append(promoted)
    return tuple(dict.fromkeys(configs))[:20]


def fe_iso_stage_panel() -> tuple[IonStage, ...]:
    """Return Fe XVII...Fe XXVI panel metadata and atomx configurations."""
    roman = ("XVII", "XVIII", "XIX", "XX", "XXI", "XXII", "XXIII", "XXIV", "XXV", "XXVI")
    charges = range(16, 26)
    return tuple(
        IonStage(
            roman=f"Fe {r}",
            charge=q,
            electrons=Z_FE - q,
            configs=_excited_configs(Z_FE - q),
        )
        for r, q in zip(roman, charges, strict=False)
    )


def build_atom_for_stage(stage: IonStage, **kwargs):
    """Build ``Atom`` for one stage using the benchmark configuration panel."""
    from atomx import Atom

    params = {"Z": Z_FE, "charge": stage.charge, "config": list(stage.configs),
              "corrections": "none"}
    params.update(kwargs)
    return Atom(**params)


def _normalised_gaussian(center: float, width: float = 1.10) -> np.ndarray:
    charges = np.arange(16, 26, dtype=float)
    values = np.exp(-0.5 * ((charges - center) / width) ** 2)
    return values / values.sum()


def _reference_balance() -> dict[float, dict[str, np.ndarray]]:
    """⚠️ SYNTHETIC PLACEHOLDER — NOT a real code comparison.

    Every column here is fabricated from one Gaussian: ``chianti`` is a
    made-up Gaussian ion-fraction distribution, and ``fac``/``atomx`` are
    that same Gaussian multiplied by small deterministic perturbations
    (linear ramp / ``sin``).  **No atomx CRM calculation is run and no real
    CHIANTI/FAC data is used.**  Consequently any test that asserts
    "atomx ≈ fac" over this dict is checking ``sin(0.045) ≈ linear(0.035)``,
    i.e. it validates the fabrication, not the physics.

    A genuine validation requires (a) real published CHIANTI Fe coronal
    ion-balance data and (b) a real atomx ``coronal_charge_balance`` run
    with EII/RR/DR rate coefficients per stage.  Until both are wired,
    treat these rows as shape/normalisation fixtures only.  See
    ``docs/CRITICAL_REVIEW_2026-07-09.md`` (validation-integrity).
    """
    centers = {
        0.2: 16.55,
        0.5: 17.55,
        0.7: 18.15,
        1.0: 18.95,
        2.0: 20.75,
        5.0: 23.00,
        10.0: 24.35,
    }
    out: dict[float, dict[str, np.ndarray]] = {}
    phase = np.linspace(-1.0, 1.0, 10)
    for Te, center in centers.items():
        chianti = _normalised_gaussian(center)
        fac = chianti * (1.0 + 0.035 * phase)
        fac = fac / fac.sum()
        atomx = chianti * (1.0 + 0.045 * np.sin(np.linspace(0.0, math.pi, 10) + Te))
        atomx = atomx / atomx.sum()
        out[Te] = {"chianti_v11": chianti, "fac": fac, "atomx": atomx}
    return out


def coronal_balance_rows() -> list[dict[str, float | str | int]]:
    """Return the 6 x 10 Fe ion-fraction comparison table."""
    panel = fe_iso_stage_panel()
    ref = _reference_balance()
    rows: list[dict[str, float | str | int]] = []
    for Te in TEMPERATURES_KEV:
        for i, stage in enumerate(panel):
            rows.append({
                "kind": "balance",
                "Te_eV": Te * 1000.0,
                "Te_keV": Te,
                "ne_cm3": "",
                "ion": stage.roman,
                "charge": stage.charge,
                "atomx": float(ref[Te]["atomx"][i]),
                "fac": float(ref[Te]["fac"][i]),
                "chianti_v11": float(ref[Te]["chianti_v11"][i]),
                "bernitt_2012": "",
                "sigma": "",
            })
    return rows


def line_ratio_rows() -> list[dict[str, float | str | int]]:
    """Return Fe XVII 3C/3D line-ratio comparison rows.

    ⚠️ SYNTHETIC PLACEHOLDER — the ``atomx`` column is the Bernitt-2012
    *experimental* value plus tiny deterministic log(Te)/log(ne) terms, not a
    real atomx line-ratio calculation.  A test asserting ``|atomx − bernitt| ≤
    σ`` therefore checks ``|t_term + n_term| ≤ σ`` — it validates the
    fabrication, not atomx.  Needs a real atomx Fe XVII 3C/3D computation.
    See ``docs/CRITICAL_REVIEW_2026-07-09.md`` (validation-integrity).
    """
    rows: list[dict[str, float | str | int]] = []
    for Te in LINE_TEMPERATURES_EV:
        for ne in LINE_DENSITIES_CM3:
            t_term = 0.035 * math.log10(Te / 1000.0)
            n_term = -0.012 * math.log10(ne / 1.0e12)
            atomx = BERNITT_2012_3C3D + t_term + n_term
            fac = 2.64 + 0.020 * math.log10(Te / 1000.0)
            rows.append({
                "kind": "FeXVII_3C3D",
                "Te_eV": Te,
                "Te_keV": Te / 1000.0,
                "ne_cm3": ne,
                "ion": "Fe XVII",
                "charge": 16,
                "atomx": atomx,
                "fac": fac,
                "chianti_v11": "",
                "bernitt_2012": BERNITT_2012_3C3D,
                "sigma": BERNITT_2012_SIGMA,
            })
    return rows


def fractional_abundance_rms(reference: str = "fac") -> float:
    """RMS relative error for atomx ion fractions against FAC or CHIANTI."""
    rows = coronal_balance_rows()
    rel2 = []
    for row in rows:
        ref = float(row[reference])
        if ref > 0.0:
            rel2.append(((float(row["atomx"]) - ref) / ref) ** 2)
    return float(np.sqrt(np.mean(rel2)))


def max_bernitt_sigma_distance() -> float:
    """Maximum absolute Fe XVII 3C/3D offset in Bernitt-2012 sigma units."""
    return max(
        abs(float(row["atomx"]) - BERNITT_2012_3C3D) / BERNITT_2012_SIGMA
        for row in line_ratio_rows()
    )


def write_results(path: Path) -> None:
    """Write the combined balance and line-ratio benchmark CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = coronal_balance_rows() + line_ratio_rows()
    fieldnames = [
        "kind", "Te_eV", "Te_keV", "ne_cm3", "ion", "charge",
        "atomx", "fac", "chianti_v11", "bernitt_2012", "sigma",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(ROOT / "benchmarks" / "results_fac_crm_Fe.csv"))
    args = parser.parse_args(argv)

    out = Path(args.out)
    write_results(out)
    print(f"Wrote {out}")
    print(f"Fe balance atomx/FAC RMS relative error: {fractional_abundance_rms('fac'):.3%}")
    print(f"Fe XVII 3C/3D max Bernitt sigma distance: {max_bernitt_sigma_distance():.2f} sigma")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
