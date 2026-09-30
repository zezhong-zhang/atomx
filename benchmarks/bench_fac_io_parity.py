#!/usr/bin/env python3
"""FAC I/O parity bench — HYP-187 acceptance gate.

For a handful of small ions (H, He+, Ne-like Mg, Fe XVII) this script:

1. Runs ``sfac`` and dumps the LEV / TR / CE tables it would write.
2. Runs the equivalent ``atomx`` calculation and writes the same three
   tables via :mod:`atomx.io.fac_writer`.
3. Reads both sides back with :func:`atomx.io.read_lev` /
   :func:`atomx.io.read_tra` / :func:`atomx.io.read_ce` and prints a
   side-by-side table.

Exits 0 when every cell completes (passing parity is *not* the goal
here — we just want the formats to round-trip cleanly and the numbers
to be comparable).  CE rows are skipped when atomx can't yet produce
the cross-section (the CE writer is data-driven; see HYP-141 Sp2).

Usage
-----
    python benchmarks/bench_fac_io_parity.py [--no-fac]

``--no-fac`` skips the sfac side, which is useful when sfac isn't
installed on the host.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "benchmarks"))

from atomx import Atom, Correlation  # noqa: E402
from atomx.io import (  # noqa: E402
    read_lev,
    read_tra,
    write_lev,
    write_tra,
)

try:
    from fac_runner import run_sfac  # noqa: E402
    HAS_SFAC_RUNNER = True
except Exception:
    HAS_SFAC_RUNNER = False


# ---------------------------------------------------------------------------
# Per-ion specs
# ---------------------------------------------------------------------------

class IonSpec:
    """Minimal description of a small ion to bench."""

    def __init__(self, label: str, Z: int, charge: int, configs: list[str],
                 fac_script: str):
        self.label = label
        self.Z = Z
        self.charge = charge
        self.configs = configs
        self.fac_script = fac_script

    @property
    def nele(self) -> int:
        return self.Z - self.charge


IONS: list[IonSpec] = [
    IonSpec(
        label="H (1s,2p)",
        Z=1, charge=0,
        configs=["1s1", "2p1"],
        fac_script="""SetAtom('H', 1)
Config('gnd', '1s1')
Config('exc', '2p1')
ConfigEnergy(0)
OptimizeRadial(['gnd'])
ConfigEnergy(1)
Structure('out.lev.b', ['gnd', 'exc'])
MemENTable('out.lev.b')
PrintTable('out.lev.b', 'out.lev', 1)
TransitionTable('out.tr.b', ['gnd'], ['exc'], -1)
PrintTable('out.tr.b', 'out.tr', 1)
""",
    ),
    IonSpec(
        label="He+ (1s,2p)",
        Z=2, charge=1,
        configs=["1s1", "2p1"],
        fac_script="""SetAtom('He', 2)
Config('gnd', '1s1')
Config('exc', '2p1')
ConfigEnergy(0)
OptimizeRadial(['gnd'])
ConfigEnergy(1)
Structure('out.lev.b', ['gnd', 'exc'])
MemENTable('out.lev.b')
PrintTable('out.lev.b', 'out.lev', 1)
TransitionTable('out.tr.b', ['gnd'], ['exc'], -1)
PrintTable('out.tr.b', 'out.tr', 1)
""",
    ),
    IonSpec(
        label="Ne-like Mg (3s↔3p)",
        Z=12, charge=2,
        configs=["1s2 2s2 2p6", "1s2 2s2 2p5 3s1", "1s2 2s2 2p5 3p1"],
        fac_script="""SetAtom('Mg', 12)
Closed('1s', '2s')
Config('gnd', '2p6')
Config('exc1', '2p5 3s1')
Config('exc2', '2p5 3p1')
ConfigEnergy(0)
OptimizeRadial(['gnd'])
ConfigEnergy(1)
Structure('out.lev.b', ['gnd', 'exc1', 'exc2'])
MemENTable('out.lev.b')
PrintTable('out.lev.b', 'out.lev', 1)
TransitionTable('out.tr.b', ['gnd', 'exc1'], ['exc1', 'exc2'], -1)
PrintTable('out.tr.b', 'out.tr', 1)
""",
    ),
    IonSpec(
        label="Fe XVII (2p,3s/3p/3d)",
        Z=26, charge=16,
        configs=[
            "1s2 2s2 2p6",
            "1s2 2s1 2p6 3s1",
            "1s2 2s1 2p6 3p1",
            "1s2 2s1 2p6 3d1",
        ],
        fac_script="""SetAtom('Fe', 26)
Closed('1s')
Config('gnd', '2s2 2p6')
Config('e3s', '2s1 2p6 3s1')
Config('e3p', '2s1 2p6 3p1')
Config('e3d', '2s1 2p6 3d1')
ConfigEnergy(0)
OptimizeRadial(['gnd'])
ConfigEnergy(1)
Structure('out.lev.b', ['gnd', 'e3s', 'e3p', 'e3d'])
MemENTable('out.lev.b')
PrintTable('out.lev.b', 'out.lev', 1)
TransitionTable('out.tr.b', ['gnd'], ['e3s', 'e3p', 'e3d'], -1)
PrintTable('out.tr.b', 'out.tr', 1)
""",
    ),
]


# ---------------------------------------------------------------------------
# Per-ion drivers
# ---------------------------------------------------------------------------

def _run_atomx(spec: IonSpec, outdir: Path) -> dict:
    """Solve in atomx and write LEV + TR via the new IO layer.

    Returns ``{lev: rows, tr: rows, t: seconds}``.
    """
    t0 = time.perf_counter()
    atom = Atom(
                Z=spec.Z,
                charge=spec.charge,
                config=spec.configs,
                corrections="none",
                correlation=Correlation(j_resolved_csfs=True),
           ).solve()
    write_lev(atom, outdir / "atomx.lev")
    write_tra(atom, outdir / "atomx.tr", multipole="E1")
    elapsed = time.perf_counter() - t0
    lev_rows = read_lev(outdir / "atomx.lev")
    tr_rows = read_tra(outdir / "atomx.tr")
    return {"lev": lev_rows, "tr": tr_rows, "t": elapsed, "atom": atom}


def _run_sfac(spec: IonSpec, outdir: Path) -> dict | None:
    """Run sfac, dump LEV + TR, return parsed rows.  ``None`` if sfac fails."""
    if not HAS_SFAC_RUNNER:
        return None
    sf = outdir / "sfac.sf"
    sf.write_text(spec.fac_script)
    t0 = time.perf_counter()
    try:
        _stdout, _t, wd = run_sfac(sf, workdir=outdir)
    except Exception as e:
        return {"error": str(e), "t": time.perf_counter() - t0}
    elapsed = time.perf_counter() - t0
    lev_rows = read_lev(wd / "out.lev")
    tr_rows = read_tra(wd / "out.tr")
    return {"lev": lev_rows, "tr": tr_rows, "t": elapsed}


def _diff_levels(ax_rows: list[dict], fac_rows: list[dict]) -> list[tuple]:
    """Pair levels by ascending energy and report (idx, eV_atomx, eV_fac, Δ)."""
    out = []
    nshared = min(len(ax_rows), len(fac_rows))
    ax_E = sorted(r["ENERGY_eV"] for r in ax_rows)
    fa_E = sorted(r["ENERGY_eV"] for r in fac_rows)
    for i in range(nshared):
        a, f = ax_E[i], fa_E[i]
        out.append((i, a, f, a - f))
    return out


def _diff_transitions(ax_rows: list[dict], fac_rows: list[dict]) -> list[tuple]:
    """Pair transitions by closest ΔE and report (ΔE_ax, ΔE_fac, A_ax, A_fac)."""
    if not ax_rows or not fac_rows:
        return []
    fac_by_dE = sorted(fac_rows, key=lambda r: r["DELTA_E_eV"])
    out = []
    for ax in sorted(ax_rows, key=lambda r: r["DELTA_E_eV"]):
        f = min(fac_by_dE, key=lambda r: abs(r["DELTA_E_eV"] - ax["DELTA_E_eV"]))
        out.append((ax["DELTA_E_eV"], f["DELTA_E_eV"],
                    ax["A_per_s"], f["A_per_s"]))
    return out


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _print_report(spec: IonSpec, ax: dict, fac: dict | None) -> None:
    print(f"\n=== {spec.label}  Z={spec.Z}  NELE={spec.nele} ===")
    print(f"  atomx t = {ax['t']:.2f}s; {len(ax['lev'])} levels, {len(ax['tr'])} transitions")
    if fac is None:
        print("  (sfac runner not available — skipping FAC side)")
        return
    if "error" in fac:
        print(f"  sfac failed: {fac['error']}")
        return
    print(f"  sfac  t = {fac['t']:.2f}s; {len(fac['lev'])} levels, {len(fac['tr'])} transitions")

    print("\n  Levels (eV above ground):")
    print(f"  {'idx':>3}  {'atomx':>12}  {'fac':>12}  {'Δ':>10}")
    for idx, a, f, d in _diff_levels(ax["lev"], fac["lev"]):
        print(f"  {idx:>3}  {a:12.4f}  {f:12.4f}  {d:10.4f}")

    print("\n  E1 transitions (paired by closest ΔE):")
    print(f"  {'ΔE_ax':>10}  {'ΔE_fac':>10}  {'A_ax':>11}  {'A_fac':>11}  {'A_ratio':>8}")
    for de_ax, de_fa, ga_ax, ga_fa in _diff_transitions(ax["tr"], fac["tr"]):
        ratio = ga_ax / ga_fa if ga_fa else float("nan")
        print(f"  {de_ax:10.3f}  {de_fa:10.3f}  {ga_ax:11.3e}  {ga_fa:11.3e}  {ratio:8.3f}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--no-fac", action="store_true",
                   help="skip the sfac side of the comparison")
    p.add_argument("--out", default=str(ROOT / "benchmarks" / "results_fac_io_parity.log"))
    args = p.parse_args()

    log_path = Path(args.out)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_lines: list[str] = []

    class _Tee:
        def __init__(self, sink): self.sink = sink
        def write(self, s):
            sys.__stdout__.write(s)
            self.sink.append(s)
        def flush(self):
            sys.__stdout__.flush()
    sys.stdout = _Tee(log_lines)  # type: ignore[assignment]

    print("FAC I/O parity bench — HYP-187")
    print(f"Tested ions: {len(IONS)}")

    rc = 0
    for spec in IONS:
        with tempfile.TemporaryDirectory(prefix="fac_io_") as td:
            outdir = Path(td)
            try:
                ax = _run_atomx(spec, outdir)
            except Exception as e:
                print(f"\n=== {spec.label} ===\n  atomx failed: {e}")
                rc = 1
                continue
            fac = None if args.no_fac else _run_sfac(spec, outdir)
            try:
                _print_report(spec, ax, fac)
            except Exception as e:
                print(f"  report failed: {e}")
                rc = 1

    sys.stdout = sys.__stdout__  # type: ignore[assignment]
    log_path.write_text("".join(log_lines))
    print(f"\nWrote {log_path}")
    sys.exit(rc)


if __name__ == "__main__":
    main()
