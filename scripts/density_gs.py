#!/usr/bin/env python3
"""Ground-state density: generate per-element HDF5 and compare with WF_TXT.

For each requested Z, runs B-spline Dirac-Hartree-Fock (jj-averaged
DBSR_HF-style, ``term=AV``) with Breit + QED and writes an HDF5 file
with one group per occupied relativistic subshell (datasets ``r``,
``density``, ``density_large``, ``density_small``, ``wave_large``,
``wave_small``; attrs ``n``, ``l``, ``spin``, ``orbital``, ``nq``,
``small_component_ratio``, ``speed``). The radial density convention
is ``int_0^inf rho(r) dr = nq`` per subshell, so the total atomic
density ``sum_i rho_i(r)`` integrates to Z.

Then, if ``/Users/zhangzz/code/ASAP/WF_TXT/<Z>.txt`` exists, compares
the summed atomx density against the WF_TXT reference (which has a
final ``R,Density`` section, same convention) on the intersection of
the two supports and writes a PNG + appends a row to REPORT.md.

Examples
--------
    python scripts/density_gs.py Ag Cu 26 -v
    python scripts/density_gs.py 1-20 -v
    python scripts/density_gs.py H-Ne Cu Ag --force -v
    python scripts/density_gs.py --all --out /tmp/density
    python scripts/density_gs.py 79 92 103 108 --no-compare   # generate only
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import h5py
import numpy as np

from atomx import Atom
from atomx.config import ground_state_config
from atomx.config.jj_coupling import generate_jj_average
from atomx.constants import atomic_numbers, element_symbol

LOG = logging.getLogger("density_gs")

# c in atomic units (=1/alpha); used only for the ``speed`` orbital attr.
C_SPEED = 137.03599908421
# Trim each subshell grid where density_large drops below this.
TAIL_THRESHOLD = 1e-17

DEFAULT_OUT = "output"
DEFAULT_WF_TXT_DIR = Path("/Users/zhangzz/code/ASAP/WF_TXT")


# ---------------------------------------------------------------------------
# CLI: element-token parsing
# ---------------------------------------------------------------------------

def _resolve_single(tok: str) -> int:
    try:
        Z = int(tok)
    except ValueError:
        if tok not in atomic_numbers:
            raise SystemExit(f"Unknown element token: {tok!r}")
        Z = atomic_numbers[tok]
    if not 1 <= Z <= 118:
        raise SystemExit(f"Z={Z} out of range 1..118")
    return Z


def parse_elements(tokens: list[str]) -> list[int]:
    """Accepts symbols ('Ag'), Z ints ('47'), and ranges ('1-10', '1..20',
    '1:86', 'H-Ne'). Multiple tokens may be mixed."""
    out: list[int] = []
    for raw in tokens:
        tok = raw.strip()
        if not tok:
            continue
        sep = next((c for c in ("..", ":", "-") if c in tok and
                    len(tok.split(c)) == 2 and all(tok.split(c))), None)
        if sep:
            lo, hi = (_resolve_single(s.strip()) for s in tok.split(sep))
            step = 1 if lo <= hi else -1
            out.extend(range(lo, hi + step, step))
        else:
            out.append(_resolve_single(tok))
    return out


# ---------------------------------------------------------------------------
# DHF solve + per-orbital payload
# ---------------------------------------------------------------------------

def reference_subshells(Z: int):
    """jj-averaged neutral ground-state subshells, sorted by (n, l, j)."""
    subshells, _ = generate_jj_average(
        ground_state_config(Z), averaging="statistical")
    keep = [(int(s.n), int(s.kappa), float(s.occupation), s.label)
            for s in subshells if s.occupation > 1e-10]

    def key(item):
        n, kappa, _, _ = item
        l = kappa if kappa > 0 else -kappa - 1
        j = abs(kappa) - 0.5
        return (n, l, j)
    keep.sort(key=key)
    return keep


def build_orbital_payload(orb, nq: float, grid: np.ndarray) -> dict:
    """Per-orbital attrs + datasets on atomx's native Gauss-Legendre grid."""
    grid_g = np.asarray(grid, dtype=np.float64)
    P_g = np.asarray(orb.P, dtype=np.float64)
    Q_g = (np.asarray(orb.Q, dtype=np.float64)
           if orb.Q is not None else np.zeros_like(P_g))

    # Pin global phase so P(r->0+) > 0 (atomx's SCF picks an arbitrary sign).
    probe = int(np.argmax(np.abs(P_g) > 1e-6 * np.abs(P_g).max()))
    if P_g[probe] < 0:
        P_g, Q_g = -P_g, -Q_g

    density_large_full = nq * P_g ** 2
    density_small_full = nq * Q_g ** 2
    density_full = density_large_full + density_small_full
    small_component_ratio = float(np.trapezoid(density_small_full, grid_g))

    # speed = sqrt(2 <T_NR>) / c on the large component.
    l_orb = int(orb.l)
    dP = np.gradient(P_g, grid_g)
    d2P = np.gradient(dP, grid_g)
    centrifugal = np.zeros_like(grid_g)
    mask = grid_g > 0
    centrifugal[mask] = P_g[mask] ** 2 / grid_g[mask] ** 2
    T_kin = (
        -0.5 * float(np.trapezoid(P_g * d2P, grid_g))
        + 0.5 * l_orb * (l_orb + 1) * float(np.trapezoid(centrifugal, grid_g))
    )
    speed = float(np.sqrt(2.0 * max(T_kin, 0.0)) / C_SPEED)

    keep = np.flatnonzero(density_large_full >= TAIL_THRESHOLD)
    last = int(keep[-1]) + 1 if keep.size else len(P_g)
    sl = slice(0, last)
    spin = float((abs(int(orb.kappa)) - 0.5) - l_orb)

    return {
        "attrs": {
            "n": int(orb.n),
            "l": l_orb,
            "spin": spin,
            "orbital": str(orb.label),
            "nq": float(nq),
            "small_component_ratio": small_component_ratio,
            "speed": speed,
        },
        "datasets": {
            "r": grid_g[sl],
            "density": density_full[sl],
            "density_large": density_large_full[sl],
            "density_small": density_small_full[sl],
            "wave_large": P_g[sl],
            "wave_small": Q_g[sl],
        },
    }


def write_h5(path: Path, Z: int, symbol: str, config_str: str, payloads: list) -> None:
    """Atomic write via tmp file + os.replace."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        with h5py.File(tmp, "w") as f:
            meta = f.create_group("metadata")
            meta.attrs["Z"] = np.int64(Z)
            meta.attrs["element"] = symbol
            meta.attrs["method"] = "gs"
            meta.attrs["core_hole"] = np.bool_(False)
            meta.attrs["oxidation"] = np.int64(0)
            meta.attrs["electron_configuration"] = config_str

            for i, payload in enumerate(payloads):
                g = f.create_group(str(i))
                for k, v in payload["attrs"].items():
                    if isinstance(v, bool):
                        g.attrs[k] = np.bool_(v)
                    elif isinstance(v, (int, np.integer)):
                        g.attrs[k] = np.int64(int(v))
                    elif isinstance(v, (float, np.floating)):
                        g.attrs[k] = np.float64(float(v))
                    else:
                        g.attrs[k] = v
                for name, arr in payload["datasets"].items():
                    g.create_dataset(name, data=np.asarray(arr, dtype=np.float64))
        os.replace(tmp, path)
    except Exception:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        raise


def generate(Z: int, out_path: Path, args) -> bool:
    """Run DHF for one element and write the h5 file. Returns success flag."""
    symbol = element_symbol(Z)
    t0 = time.time()
    try:
        config_str = ground_state_config(Z)
        if args.correlation:
            atom = Atom(Z=Z, breit=args.breit, qed=args.qed,
                        correlation=args.correlation)
        else:
            atom = Atom(Z=Z, breit=args.breit, qed=args.qed)
        atom.solve(tol=args.tol, max_iter=args.max_iter)
    except Exception as e:
        LOG.error("[FAIL] %s Z=%d solve error: %s", symbol, Z, e)
        return False

    if not atom.converged:
        LOG.error("[FAIL] %s Z=%d did not converge after %d iterations",
                  symbol, Z, atom.iterations)
        return False

    grid = np.asarray(atom.result.waves.grid, dtype=np.float64)
    orb_by_nk = {(int(o.n), int(o.kappa)): o
                 for o in atom.result.waves.orbitals.values()}

    payloads = []
    for (n, kappa, nq, label) in reference_subshells(Z):
        orb = orb_by_nk.get((n, kappa))
        if orb is None:
            LOG.error("[FAIL] %s: no orbital for (n=%d, kappa=%d, %s)",
                      symbol, n, kappa, label)
            return False
        payloads.append(build_orbital_payload(orb, nq, grid))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_h5(out_path, Z, symbol, config_str, payloads)

    LOG.info("[OK] %s Z=%d | %d orbitals | iters=%d | %.1fs",
             symbol, Z, len(payloads), atom.iterations, time.time() - t0)
    return True


# ---------------------------------------------------------------------------
# WF_TXT comparison
# ---------------------------------------------------------------------------

def load_wf_txt_density(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Extract (r, rho) from the trailing 'R,Density' section of a WF_TXT file."""
    lines = path.read_text().splitlines()
    start = next((i + 1 for i, ln in enumerate(lines)
                  if ln.strip().startswith("R,Density")), None)
    if start is None:
        raise ValueError(f"No 'R,Density' section in {path}")
    rs, ds = [], []
    for line in lines[start:]:
        parts = line.strip().rstrip(",").split(",")
        if len(parts) < 2:
            continue
        try:
            rs.append(float(parts[0]))
            ds.append(float(parts[1]))
        except ValueError:
            continue
    return np.asarray(rs), np.asarray(ds)


def load_atomx_total_density(
    h5_path: Path,
) -> tuple[np.ndarray, np.ndarray, list[tuple[str, float]]]:
    """Sum per-subshell densities onto a common log grid (each subshell is
    tail-trimmed to its own r-range)."""
    with h5py.File(h5_path, "r") as f:
        keys = sorted([k for k in f.keys() if k != "metadata"], key=int)
        subshells = [(str(f[k].attrs["orbital"]), float(f[k].attrs["nq"]))
                     for k in keys]
        r_min = min(float(f[k]["r"][0]) for k in keys)
        r_max = max(float(f[k]["r"][-1]) for k in keys)
        r_common = np.geomspace(r_min, r_max, 4000)
        rho_total = np.zeros_like(r_common)
        for k in keys:
            rho_total += np.interp(
                r_common, f[k]["r"][:], f[k]["density"][:],
                left=0.0, right=0.0,
            )
    return r_common, rho_total, subshells


def compare(Z: int, h5_path: Path, wf_path: Path, plots_dir: Path) -> dict:
    """Compute L1/L2 errors and emit a side-by-side PNG."""
    import matplotlib.pyplot as plt  # local import: plotting is optional

    symbol = element_symbol(Z)
    r_wf, rho_wf = load_wf_txt_density(wf_path)
    r_at, rho_at, subs = load_atomx_total_density(h5_path)

    N_wf = float(np.trapezoid(rho_wf, r_wf))
    N_at = float(np.trapezoid(rho_at, r_at))

    r_lo = max(r_wf[0], r_at[0])
    r_hi = min(r_wf[-1], r_at[-1])
    r_cmp = np.geomspace(r_lo, r_hi, 2000)
    rho_wf_i = np.interp(r_cmp, r_wf, rho_wf)
    rho_at_i = np.interp(r_cmp, r_at, rho_at)
    diff = rho_at_i - rho_wf_i

    l1 = float(np.trapezoid(np.abs(diff), r_cmp))
    denom = float(np.trapezoid(rho_wf_i ** 2, r_cmp)) ** 0.5
    l2_rel = float(np.trapezoid(diff ** 2, r_cmp)) ** 0.5 / denom if denom else float("nan")

    fig, (ax, ax2) = plt.subplots(
        2, 1, figsize=(7, 6), sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )
    ax.plot(r_wf, rho_wf, label="WF_TXT", lw=1.8, color="C0")
    ax.plot(r_at, rho_at, label="atomx (DHF)", lw=1.2, color="C3", ls="--")
    ax.set_xscale("log")
    ax.set_ylabel(r"$\rho(r)$  [electrons / $a_0$]")
    ax.set_title(f"{symbol}  (Z={Z})   "
                 f"N$_{{WF\\_TXT}}$={N_wf:.4f}   N$_{{atomx}}$={N_at:.4f}")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    ax2.semilogx(r_cmp, diff, color="k", lw=1.0)
    ax2.axhline(0, color="grey", lw=0.5)
    ax2.set_xlabel(r"$r$  [$a_0$]")
    ax2.set_ylabel(r"$\Delta\rho$")
    ax2.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    plots_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(plots_dir / f"{symbol}_Z{Z}_density.png", dpi=130)
    plt.close(fig)

    subshells_str = ", ".join(
        f"{o}^{int(nq) if float(nq).is_integer() else nq}" for o, nq in subs
    )
    return {
        "Z": Z, "element": symbol,
        "N_wf_txt": N_wf, "N_atomx": N_at,
        "L1": l1, "L2_rel": l2_rel,
        "subshells": subshells_str,
    }


def write_report(rows: list[dict], plots_dir: Path) -> Path:
    md = [
        "# atomx vs WF_TXT  —  ground-state radial charge density\n",
        "Per-element comparison. `N = ∫ρ(r)dr` should equal Z. "
        "`L1 = ∫|Δρ|dr` is the integrated electron mismatch. "
        "`L2_rel = ‖Δρ‖₂ / ‖ρ_wf‖₂` is the relative shape error on "
        "the intersection of the two r-supports.\n",
        "| Z | El | atomx subshells | N (WF_TXT) | N (atomx) | L1 [e⁻] | L2_rel |",
        "|---:|:---|:---|---:|---:|---:|---:|",
    ]
    for r in rows:
        md.append(
            f"| {r['Z']} | {r['element']} | `{r['subshells']}` "
            f"| {r['N_wf_txt']:.4f} | {r['N_atomx']:.4f} "
            f"| {r['L1']:.3e} | {r['L2_rel']:.3e} |"
        )
    path = plots_dir / "REPORT.md"
    path.write_text("\n".join(md) + "\n")
    return path


# ---------------------------------------------------------------------------
# Top-level
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("elements", nargs="*",
                    help="Symbols, Z ints, or ranges (e.g. 'Ag 47 1-10 H-Ne').")
    ap.add_argument("--all", action="store_true",
                    help="Iterate Z=1..118 (overrides positional elements).")
    ap.add_argument("--out", default=DEFAULT_OUT, type=Path,
                    help=f"Output directory for h5 files (default: {DEFAULT_OUT}/).")
    ap.add_argument("--wf-txt-dir", default=DEFAULT_WF_TXT_DIR, type=Path,
                    help=f"WF_TXT reference directory (default: {DEFAULT_WF_TXT_DIR}).")
    ap.add_argument("--plots-dir", default=None, type=Path,
                    help="Where to write PNGs + REPORT.md. "
                         "Default: <out>/wf_txt_compare/.")
    ap.add_argument("--correlation", default=None, choices=[None, "s", "sd"],
                    help="None (default, jj-averaged DHF) | 's' | 'sd' (slow).")
    ap.add_argument("--breit", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--qed", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--max-iter", type=int, default=300)
    ap.add_argument("--tol", type=float, default=1e-6)
    ap.add_argument("--force", action="store_true",
                    help="Re-run SCF even if the h5 file already exists.")
    ap.add_argument("--no-compare", action="store_true",
                    help="Generate h5 only; skip WF_TXT comparison.")
    ap.add_argument("-v", "--verbose", action="count", default=0,
                    help="-v for INFO, -vv for DEBUG.")
    args = ap.parse_args(argv)

    level = (logging.DEBUG if args.verbose >= 2
             else logging.INFO if args.verbose == 1
             else logging.WARNING)
    logging.basicConfig(level=level,
                        format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")

    if args.all:
        Zs = list(range(1, 119))
    elif args.elements:
        Zs = parse_elements(args.elements)
    else:
        ap.error("Provide element tokens or --all")

    plots_dir = args.plots_dir or (args.out / "wf_txt_compare")

    fails: list[int] = []
    rows: list[dict] = []
    for Z in Zs:
        symbol = element_symbol(Z)
        h5_path = args.out / f"{symbol}.h5"
        if h5_path.exists() and not args.force:
            LOG.info("[SKIP] %s Z=%d (h5 exists at %s)", symbol, Z, h5_path)
        elif not generate(Z, h5_path, args):
            fails.append(Z)
            continue

        if args.no_compare:
            continue
        wf_path = args.wf_txt_dir / f"{Z}.txt"
        if not wf_path.exists():
            LOG.info("[--] %s Z=%d: no WF_TXT reference at %s", symbol, Z, wf_path)
            continue
        try:
            row = compare(Z, h5_path, wf_path, plots_dir)
        except Exception as e:
            LOG.error("[FAIL] %s Z=%d compare error: %s", symbol, Z, e)
            fails.append(Z)
            continue
        rows.append(row)
        LOG.info("[CMP] %s Z=%d  L1=%.3e  L2_rel=%.3e",
                 symbol, Z, row["L1"], row["L2_rel"])

    if rows:
        report = write_report(rows, plots_dir)
        LOG.info("Wrote %s and %d plots in %s", report, len(rows), plots_dir)
        print(f"\nReport: {report}")
        print(f"Plots:  {plots_dir}")

    if fails:
        LOG.warning("Summary: %d/%d elements failed (Z=%s)",
                    len(fails), len(Zs), fails)
        return 1
    LOG.info("Summary: all %d elements succeeded", len(Zs))
    return 0


if __name__ == "__main__":
    sys.exit(main())
