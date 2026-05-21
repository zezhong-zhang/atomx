"""Shared FAC-parity matrix helpers.

The matrix benchmark intentionally keeps the frozen CI gate independent of
``sfac``: benchmark scripts can regenerate FAC columns, while tests reuse only
the atomx-side row calculation and compare it with the committed golden CSV.
"""

from __future__ import annotations

import csv
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from atomx import Atom
from atomx.constants import HARTREE_EV
from atomx.physics.dirac import hydrogenic_dirac_energy
from atomx.prop.transitions.rates import einstein_A_eL
from atomx.sampling import Sampling
from fac_runner import parse_e0, parse_levels, parse_transitions, run_sfac

MATRIX_Z = [10, 18, 26, 36, 54, 74, 82]
MATRIX_ISO = ["He-like", "Li-like", "Be-like", "Ne-like", "Ar-like"]
QUICK_SKIP_Z = {74, 82}

CSV_COLUMNS = [
    "Z",
    "iso",
    "atomx_E_eV",
    "fac_E_eV",
    "atomx_first_E1_eV",
    "fac_first_E1_eV",
    "atomx_first_A_s",
    "fac_first_A_s",
    # Worst |A_v/A_l - 1| across all E1 channels in the ground-config
    # DHF solve (Gu 2008 §3.2 gauge-invariance diagnostic).  Closed-shell
    # ground configs with no s↔p E1 channel record 0.0 by construction;
    # other rows pin the structure-code numerics tight enough that a 10 %
    # perturbation anywhere in the radial / angular pipeline shifts the
    # value past the frozen tolerance.
    "atomx_gauge_ratio_max",
    "t_atomx",
    "t_sfac",
]


@dataclass(frozen=True)
class IsoSpec:
    nel: int
    ground: str
    excited: str
    fac_closed: tuple[str, ...]
    fac_ground: str
    fac_excited: str


ISO_SPECS = {
    "He-like": IsoSpec(2, "1s2", "1s1 2p1", (), "1s2", "1s1 2p1"),
    "Li-like": IsoSpec(3, "1s2 2s1", "1s2 2p1", ("1s",), "2s1", "2p1"),
    "Be-like": IsoSpec(4, "1s2 2s2", "1s2 2s1 2p1", ("1s",), "2s2", "2s1 2p1"),
    "Ne-like": IsoSpec(10, "1s2 2s2 2p6", "1s2 2s2 2p5 3s1", ("1s",), "2*8", "2*7 3s1"),
    "Ar-like": IsoSpec(
        18,
        "1s2 2s2 2p6 3s2 3p6",
        "1s2 2s2 2p6 3s2 3p5 4s1",
        ("1s", "2s", "2p", "3s"),
        "3p6",
        "3p5 4s1",
    ),
}

ELEMENT_SYMBOLS = {
    1: "H", 2: "He", 3: "Li", 4: "Be", 5: "B", 6: "C", 7: "N", 8: "O", 9: "F",
    10: "Ne", 11: "Na", 12: "Mg", 13: "Al", 14: "Si", 15: "P", 16: "S", 17: "Cl",
    18: "Ar", 19: "K", 20: "Ca", 21: "Sc", 22: "Ti", 23: "V", 24: "Cr", 25: "Mn",
    26: "Fe", 27: "Co", 28: "Ni", 29: "Cu", 30: "Zn", 31: "Ga", 32: "Ge", 33: "As",
    34: "Se", 35: "Br", 36: "Kr", 37: "Rb", 38: "Sr", 39: "Y", 40: "Zr", 41: "Nb",
    42: "Mo", 43: "Tc", 44: "Ru", 45: "Rh", 46: "Pd", 47: "Ag", 48: "Cd", 49: "In",
    50: "Sn", 51: "Sb", 52: "Te", 53: "I", 54: "Xe", 55: "Cs", 56: "Ba", 57: "La",
    58: "Ce", 59: "Pr", 60: "Nd", 61: "Pm", 62: "Sm", 63: "Eu", 64: "Gd", 65: "Tb",
    66: "Dy", 67: "Ho", 68: "Er", 69: "Tm", 70: "Yb", 71: "Lu", 72: "Hf", 73: "Ta",
    74: "W", 75: "Re", 76: "Os", 77: "Ir", 78: "Pt", 79: "Au", 80: "Hg", 81: "Tl",
    82: "Pb", 83: "Bi", 84: "Po", 85: "At", 86: "Rn",
}

FAC_TEMPLATE = """SetAtom('{symbol}', {Z})
{closed_cmds}
Config('gnd', '{ground}')
Config('exc', '{excited}')
ConfigEnergy(0)
OptimizeRadial(['gnd'])
ConfigEnergy(1)
Structure('matrix.lev.b', ['gnd', 'exc'])
MemENTable('matrix.lev.b')
PrintTable('matrix.lev.b', 'matrix.lev', 1)
TransitionTable('matrix.tr.b', ['gnd'], ['exc'])
PrintTable('matrix.tr.b', 'matrix.tr', 1)
"""

E1_PROXY_ORBITALS = {
    "He-like": ((1, -1), (2, 1)),
    "Li-like": ((2, -1), (2, -2)),
    "Be-like": ((2, -1), (2, -2)),
    "Ne-like": ((2, 1), (3, -1)),
    "Ar-like": ((3, 1), (4, -1)),
}


def iter_matrix(quick: bool = False):
    """Yield physical ``(Z, iso)`` matrix cells."""
    for Z in MATRIX_Z:
        if quick and Z in QUICK_SKIP_Z:
            continue
        for iso in MATRIX_ISO:
            spec = ISO_SPECS[iso]
            if Z >= spec.nel:
                yield Z, iso


def atomx_e1_proxy(Z: int, iso: str) -> tuple[float, float]:
    """Cheap atomx-derived proxy for the first E1 line in the CI gate.

    Full MCDHF transition solves are useful benchmark outputs but too slow for
    the frozen CI tripwire.  This proxy uses atomx's hydrogenic Dirac energies
    plus the shared Einstein-A implementation, so physics-changing edits to
    core constants/formulas still trip the golden gate.
    """
    spec = ISO_SPECS[iso]
    (n_lo, k_lo), (n_hi, k_hi) = E1_PROXY_ORBITALS[iso]
    z_eff = max(1, Z - spec.nel + 1)
    e_lo = hydrogenic_dirac_energy(n_lo, k_lo, z_eff)
    e_hi = hydrogenic_dirac_energy(n_hi, k_hi, z_eff)
    omega = abs(e_hi - e_lo)
    return omega * HARTREE_EV, einstein_A_eL(omega, line_strength=1.0, L=1, tj_i=1)


def _gauge_ratio_max(atom) -> float:
    """Worst |A_v/A_l - 1| across the atom's ground-config E1 channels.

    Closed-shell ground configs (1s², 1s²2s, 1s²2s² …) have no E1-allowed
    orbital pair and return ``0.0``.  Open-shell or non-degenerate-l
    ground configs (Ne-like, Ar-like …) drive a real ratio that pins the
    structure-code numerics; a 10 % perturbation in the radial /
    angular pipeline shifts this value past the frozen tolerance.
    """
    transitions = atom.transitions(multipoles=["E1"], gauge="both")
    devs = [
        abs(t.A_velocity / t.A_length - 1.0)
        for t in transitions
        if t.A_length is not None and t.A_velocity is not None
        and t.A_length > 0.0
    ]
    return max(devs) if devs else 0.0


def run_atomx_matrix_cell(Z: int, iso: str) -> dict[str, float | int | str]:
    spec = ISO_SPECS[iso]
    t0 = time.perf_counter()
    atom = Atom(
        Z=Z,
        charge=Z - spec.nel,
        config=spec.ground,
        breit=False,
        qed=False,
        sampling=Sampling.quick,
    )
    atom.solve()
    first_e1_eV, first_A_s = atomx_e1_proxy(Z, iso)
    gauge_ratio_max = _gauge_ratio_max(atom)
    elapsed = time.perf_counter() - t0
    return {
        "Z": Z,
        "iso": iso,
        "atomx_E_eV": atom.total_energy * HARTREE_EV,
        "atomx_first_E1_eV": first_e1_eV,
        "atomx_first_A_s": first_A_s,
        "atomx_gauge_ratio_max": gauge_ratio_max,
        "t_atomx": elapsed,
    }


def _build_fac_script(Z: int, iso: str) -> str:
    spec = ISO_SPECS[iso]
    closed_cmds = "\n".join(f"Closed('{shell}')" for shell in spec.fac_closed)
    if not closed_cmds:
        closed_cmds = "# no closed shells"
    return FAC_TEMPLATE.format(
        symbol=ELEMENT_SYMBOLS[Z],
        Z=Z,
        closed_cmds=closed_cmds,
        ground=spec.fac_ground,
        excited=spec.fac_excited,
    )


def _first_fac_e1(levels: list[dict], transitions: list[dict]) -> tuple[float, float]:
    e1 = [tr for tr in transitions if tr["mult"] in (0, None)]
    if not e1:
        return 0.0, 0.0
    first = min(e1, key=lambda tr: tr["DELTA_E_eV"])
    up_level = next((lv for lv in levels if lv["ILEV"] == first["upper"]), None)
    g_up = (up_level["TWOJ"] + 1) if up_level is not None else 1
    return first["DELTA_E_eV"], first["gA_per_s"] / g_up


def run_fac_matrix_cell(Z: int, iso: str, timeout: float = 300.0) -> dict[str, float]:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        sf = tmp_path / "matrix.sf"
        sf.write_text(_build_fac_script(Z, iso))
        _stdout, elapsed, wd = run_sfac(sf, workdir=tmp_path, timeout=timeout)
        lev = wd / "matrix.lev"
        tr = wd / "matrix.tr"
        levels = parse_levels(lev)
        transitions = parse_transitions(tr)
        first_e1_eV, first_A_s = _first_fac_e1(levels, transitions)
        return {
            "fac_E_eV": parse_e0(lev) or 0.0,
            "fac_first_E1_eV": first_e1_eV,
            "fac_first_A_s": first_A_s,
            "t_sfac": elapsed,
        }


def run_matrix(quick: bool = False, include_fac: bool = True, timeout: float = 300.0) -> list[dict]:
    rows = []
    for Z, iso in iter_matrix(quick=quick):
        row = run_atomx_matrix_cell(Z, iso)
        if include_fac:
            try:
                row.update(run_fac_matrix_cell(Z, iso, timeout=timeout))
            except Exception:
                row.update({
                    "fac_E_eV": 0.0,
                    "fac_first_E1_eV": 0.0,
                    "fac_first_A_s": 0.0,
                    "t_sfac": 0.0,
                })
        else:
            row.update({
                "fac_E_eV": 0.0,
                "fac_first_E1_eV": 0.0,
                "fac_first_A_s": 0.0,
                "t_sfac": 0.0,
            })
        rows.append(row)
    return rows


def write_matrix_csv(rows: list[dict], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "Z": int(row["Z"]),
                "iso": row["iso"],
                "atomx_E_eV": f"{float(row['atomx_E_eV']):.8e}",
                "fac_E_eV": f"{float(row['fac_E_eV']):.8e}",
                "atomx_first_E1_eV": f"{float(row['atomx_first_E1_eV']):.8e}",
                "fac_first_E1_eV": f"{float(row['fac_first_E1_eV']):.8e}",
                "atomx_first_A_s": f"{float(row['atomx_first_A_s']):.8e}",
                "fac_first_A_s": f"{float(row['fac_first_A_s']):.8e}",
                "atomx_gauge_ratio_max": f"{float(row['atomx_gauge_ratio_max']):.8e}",
                "t_atomx": f"{float(row['t_atomx']):.6f}",
                "t_sfac": f"{float(row['t_sfac']):.6f}",
            })


def print_matrix_summary(rows: list[dict]) -> None:
    hdr = (
        f"{'Z':>3}  {'iso':>8}  {'E_ax(eV)':>14}  {'E_fac(eV)':>14}  "
        f"{'E1_ax':>10}  {'A_ax':>11}  {'t_ax':>7}  {'t_sfac':>7}"
    )
    print(hdr)
    print("-" * len(hdr))
    for row in rows:
        print(
            f"{int(row['Z']):>3}  {row['iso']:>8}  "
            f"{float(row['atomx_E_eV']):14.3f}  {float(row['fac_E_eV']):14.3f}  "
            f"{float(row['atomx_first_E1_eV']):10.3f}  "
            f"{float(row['atomx_first_A_s']):11.3e}  "
            f"{float(row['t_atomx']):7.3f}  {float(row['t_sfac']):7.3f}"
        )
