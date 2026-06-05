#!/usr/bin/env python3
"""Generate benchmarks/PERFORMANCE.md from all benchmark CSVs.

Run from the repo root:
    python scripts/perf_dashboard.py
"""

from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BENCH = ROOT / "benchmarks"
OUT = BENCH / "PERFORMANCE.md"


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _element(z: int) -> str:
    sym = {
        1: "H", 2: "He", 3: "Li", 4: "Be", 5: "B", 6: "C", 7: "N", 8: "O",
        10: "Ne", 18: "Ar", 26: "Fe", 36: "Kr", 54: "Xe", 74: "W", 79: "Au",
        82: "Pb", 92: "U",
    }
    return sym.get(z, f"Z{z}")


def _matrix_table() -> str:
    rows = _read_csv(BENCH / "results_fac_matrix.csv")
    if not rows:
        return "*No matrix results yet.*\n"
    lines = [
        "### FAC Parity Matrix",
        "",
        "| Z | iso | E atomx (eV) | E FAC (eV) | rel Δ | t_atomx (s) | t_sfac (s) | ratio |",
        "|--:|:----|-------------:|-----------:|------:|------------:|-----------:|------:|",
    ]
    for r in rows:
        z = int(r["Z"])
        e_ax = float(r["atomx_E_eV"])
        e_fac = float(r["fac_E_eV"])
        rel = abs(e_ax - e_fac) / max(abs(e_fac), 1e-30)
        t_ax = float(r["t_atomx"])
        t_fac = float(r["t_sfac"])
        ratio = t_ax / t_fac if t_fac > 0 else float("inf")
        lines.append(
            f"| {z} {_element(z)} | {r['iso']} | {e_ax:,.1f} | {e_fac:,.1f} | {rel:.2e} | {t_ax:.2f} | {t_fac:.2f} | {ratio:.1f}× |"
        )
    return "\n".join(lines) + "\n"


def _structure_table() -> str:
    rows = _read_csv(BENCH / "results_fac_structure.csv")
    if not rows:
        return ""
    lines = [
        "",
        "### DHF+QED Structure",
        "",
        "| Z | iso | E atomx (eV) | E FAC (eV) | rel Δ | t_atomx (s) | t_fac (s) |",
        "|--:|:----|-------------:|-----------:|------:|------------:|----------:|",
    ]
    for r in rows:
        z = int(r["Z"])
        e_ax = float(r["E_atomx_eV"])
        e_fac = float(r["E_fac_eV"])
        rel = abs(e_ax - e_fac) / max(abs(e_fac), 1e-30)
        t_ax = float(r["t_atomx_s"])
        t_fac = float(r["t_fac_s"])
        lines.append(
            f"| {z} {_element(z)} | {r['iso']} | {e_ax:,.1f} | {e_fac:,.1f} | {rel:.2e} | {t_ax:.2f} | {t_fac:.2f} |"
        )
    return "\n".join(lines) + "\n"


def _shooting_table() -> str:
    rows = _read_csv(BENCH / "results_shooting.csv")
    if not rows:
        return ""
    lines = [
        "",
        "### Shooting vs radial (DHFS)",
        "",
        "| Z | nuclear | E atomx (Ha) | E radial (Ha) | rel Δ | t_warm (s) | t_radial (s) | speedup |",
        "|--:|:--------|-------------:|--------------:|------:|-----------:|-------------:|--------:|",
    ]
    for r in rows:
        z = int(r["Z"])
        e_ax = float(r["E_atomx_Eh"])
        e_rad = float(r["E_radial_Eh"])
        rel = abs(e_ax - e_rad) / max(abs(e_rad), 1e-30)
        t_w = float(r.get("t_atomx_warm_s", r.get("t_atomx_cold_s", 0)))
        t_r = float(r["t_radial_s"])
        sp = t_r / t_w if t_w > 0 else float("inf")
        lines.append(
            f"| {z} {_element(z)} | {r['nuclear']} | {e_ax:.6f} | {e_rad:.6f} | {rel:.2e} | {t_w:.3f} | {t_r:.3f} | {sp:.1f}× |"
        )
    return "\n".join(lines) + "\n"


def _local_exchange_table() -> str:
    rows = _read_csv(BENCH / "results_local_exchange.csv")
    if not rows:
        return ""
    lines = [
        "",
        "### Local-exchange DFS vs sfac",
        "",
        "| Z | E shooting (eV) | E sfac (eV) | rel Δ | t_shooting (s) | t_sfac (s) |",
        "|--:|----------------:|------------:|------:|---------------:|-----------:|",
    ]
    for r in rows:
        z = int(r["Z"])
        e_sh = float(r["E_shooting_eV"])
        e_fac = float(r["E_sfac_eV"])
        rel = abs(e_sh - e_fac) / max(abs(e_fac), 1e-30)
        t_sh = float(r["t_shooting_s"])
        t_fac = float(r["t_sfac_s"])
        lines.append(
            f"| {z} {_element(z)} | {e_sh:,.3f} | {e_fac:,.3f} | {rel:.2e} | {t_sh:.3f} | {t_fac:.3f} |"
        )
    return "\n".join(lines) + "\n"


def _crm_tables() -> str:
    parts = []
    for name, label in [("Fe", "Fe"), ("W", "W"), ("Au", "Au")]:
        rows = _read_csv(BENCH / f"results_fac_crm_{name}.csv")
        if not rows:
            continue
        balance = [r for r in rows if r.get("kind") == "balance"]
        if not balance:
            continue
        parts.append(f"\n### CRM {label}\n")
        parts.append(f"{len(balance)} balance rows in `results_fac_crm_{name}.csv`.\n")
    return "".join(parts)


def _dr_table() -> str:
    rows = _read_csv(BENCH / "results_fac_dr.csv")
    if not rows:
        return ""
    return "\n### Dielectronic Recombination (Fe XVII → Fe XVI)\n\nSee `results_fac_dr.csv`.\n"


def main() -> None:
    sections = [
        "# atomx Performance Dashboard",
        "",
        "Auto-generated by `scripts/perf_dashboard.py`.",
        "",
        "---",
        "",
        _matrix_table(),
        _structure_table(),
        _shooting_table(),
        _local_exchange_table(),
        _crm_tables(),
        _dr_table(),
    ]
    OUT.write_text("\n".join(s for s in sections if s) + "\n")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
