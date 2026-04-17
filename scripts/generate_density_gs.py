#!/usr/bin/env python3
"""MCDHF + QED + Breit ground-state density generator.

Runs B-spline Dirac-Hartree-Fock (the DBSR_HF-style jj-averaged AVerage
of configurations) with Breit and QED corrections on neutral ground-state
atoms via the ``atomx`` library, and writes per-element HDF5 files matching
the schema of ``/Users/zhangzz/code/coredensity/database/density_gs/Ag.h5``.

By default uses single-reference DHF with jj-averaging — fractional
occupations distributed statistically over j- and j+ subshells (matches
DBSR_HF's ``term=AV`` mode). Pass ``--correlation sd`` for true
multi-reference MCDHF with singles+doubles CSF excitations; note that
this is orders of magnitude slower and not needed for ground-state
densities.

Schema (verified against reference Ag.h5):

    root/              (no attrs)
        metadata/      attrs: Z, element, method="gs", core_hole=False,
                              oxidation=0, electron_configuration
        0/, 1/, ...    one numbered group per reference-occupied j-subshell,
                       ordered by (n, l, j) ascending. Each group:
            attrs    : n, l, spin (j-l), orbital ("1s1/2"), nq,
                       small_component_ratio, speed
            datasets : r, density, density_large, density_small,
                       wave_large, wave_small   (all 1D, tail-trimmed)

Conventions (verified against the reference Ag.h5):
    r-grid is atomx's native Gauss-Legendre quadrature grid
        (bsp_data["gr"].ravel()) — the natural, per-interval-dense sampling
        of the B-spline solution. Dense in the tail, which matters for
        outer-shell densities.
    wave_large = P(r), wave_small = Q(r), normalized so int(P^2+Q^2)dr=1.
        Global phase pinned so P(r->0+) > 0 (SCF returns ±P arbitrarily).
    density_large = nq*P^2, density_small = nq*Q^2, density = sum.
    small_component_ratio = int(density_small) dr = nq * int Q^2 dr
        (integrated small-component electron count for the subshell).
    speed = sqrt(2*<T_NR>)/c, where <T_NR> is the non-relativistic kinetic
        expectation -1/2 <P|d^2/dr^2|P> + l(l+1)/2 <P|r^-2|P>. Matches the
        reference Ag.h5 convention.

Examples
--------
    python scripts/generate_density_gs.py Ag Cu 26 -v
    python scripts/generate_density_gs.py 1-20 -v              # Z=1..20
    python scripts/generate_density_gs.py 1..86 -v             # Z=1..86
    python scripts/generate_density_gs.py H-Ne Cu Ag -v        # mixed
    python scripts/generate_density_gs.py Ag --verify-ag -v
    python scripts/generate_density_gs.py --all --out /tmp/density
    python scripts/generate_density_gs.py 1-118 -v             # full PT → ./output/
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import List, Tuple

import h5py
import numpy as np

from atomx import Atom
from atomx.config import ground_state_config
from atomx.config.jj_coupling import generate_jj_average
from atomx.constants import atomic_numbers, element_symbol

LOG = logging.getLogger("generate_density_gs")

# Speed of light in atomic units (c = 1/alpha).
C_SPEED = 137.03599908421

DEFAULT_OUT = "output"  # relative to cwd; use --out to override
REFERENCE_AG = "/Users/zhangzz/code/coredensity/database/density_gs/Ag.h5"

# Tail truncation threshold: trim each subshell grid where density_large
# drops below this floor. Matches varying lengths in the reference Ag.h5.
TAIL_THRESHOLD = 1e-17


# ----------------------------------------------------------------------
# CLI plumbing
# ----------------------------------------------------------------------

def _resolve_single(tok: str) -> int:
    """Resolve one token (symbol or Z int) to a Z."""
    try:
        Z = int(tok)
    except ValueError:
        if tok not in atomic_numbers:
            raise SystemExit(f"Unknown element token: {tok!r}")
        Z = atomic_numbers[tok]
    if not 1 <= Z <= 118:
        raise SystemExit(f"Z={Z} out of range 1..118")
    return Z


def parse_elements(tokens: List[str]) -> List[int]:
    """Resolve a list of symbol/Z/range tokens into a list of Z ints.

    Accepted token forms:
        "Ag"      — symbol
        "47"      — Z integer
        "1-10"    — inclusive Z range
        "1..10"   — inclusive Z range
        "1:10"    — inclusive Z range
        "H-Ne"    — symbol range (H through Ne)

    Ranges may be ascending or descending. Duplicates are preserved in
    the order they appear; de-duplication happens implicitly via the
    skip-existing check.
    """
    out: List[int] = []
    for raw in tokens:
        tok = raw.strip()
        if not tok:
            continue
        # Try range syntax first. Check in order: "..", ":", "-" (last
        # because a symbol can legitimately start with a letter; "-"
        # between two numeric/symbol endpoints is a range).
        sep = None
        for candidate in ("..", ":", "-"):
            if candidate in tok:
                parts = tok.split(candidate)
                if len(parts) == 2 and parts[0] and parts[1]:
                    sep = candidate
                    break
        if sep is not None:
            lo_tok, hi_tok = tok.split(sep)
            lo = _resolve_single(lo_tok.strip())
            hi = _resolve_single(hi_tok.strip())
            step = 1 if lo <= hi else -1
            out.extend(range(lo, hi + step, step))
        else:
            out.append(_resolve_single(tok))
    return out


# ----------------------------------------------------------------------
# Solver wrapper
# ----------------------------------------------------------------------

def solve_element(
    Z: int, *, correlation: str | None, breit: bool, qed: bool,
    max_iter: int, tol: float,
) -> Tuple[Atom, str]:
    """Solve for ground-state orbitals. Returns (atom, config_str).

    ``correlation=None`` runs single-reference B-spline DHF with jj-average
    fractional occupations (DBSR_HF ``term=AV`` equivalent). This is fast
    (seconds to minutes) and sufficient for ground-state densities.

    ``correlation="s"`` or ``"sd"`` uses atomx's MCDHFSolver which adds
    CSF excitations + CI mixing on top of the shared-orbital DHF. Much
    slower and generally unnecessary for the density task.
    """
    config_str = ground_state_config(Z)
    kwargs = dict(Z=Z, breit=breit, qed=qed)
    if correlation:
        kwargs["correlation"] = correlation
    atom = Atom(**kwargs)
    atom.solve(tol=tol, max_iter=max_iter)
    return atom, config_str


def reference_subshells(Z: int):
    """Enumerate reference-occupied j-subshells for the neutral ground state.

    Uses the same ``generate_jj_average`` helper that atomx itself uses to
    distribute electrons across j- and j+ (kappa=+l and kappa=-(l+1))
    subshells with statistical (2j+1) weights. Filters subshells whose
    average occupation is zero, then sorts by (n, l, j) ascending.
    """
    config_str = ground_state_config(Z)
    subshells, _ = generate_jj_average(config_str, averaging="statistical")
    out = []
    for sh in subshells:
        if sh.occupation > 1e-10:
            out.append((int(sh.n), int(sh.kappa), float(sh.occupation), sh.label))

    def sort_key(item):
        n, kappa, _nq, _lbl = item
        l = kappa if kappa > 0 else -kappa - 1
        j = abs(kappa) - 0.5
        return (n, l, j)

    out.sort(key=sort_key)
    return out


# ----------------------------------------------------------------------
# Per-orbital payload construction
# ----------------------------------------------------------------------

def build_orbital_payload(orb, nq: float, grid: np.ndarray) -> dict:
    """Compute attrs + datasets for one orbital group.

    Arrays are written on atomx's native Gauss-Legendre quadrature grid
    (the raveled ``bsp_data["gr"]``). This is the natural grid for the
    B-spline solution: it samples each spline interval with Gauss points,
    so the tail receives the same per-interval density as the bulk. A
    geometric/log resample would concentrate points near the origin at
    the cost of the tail — avoided here on purpose.
    """
    grid_g = np.asarray(grid, dtype=np.float64)
    P_g = np.asarray(orb.P, dtype=np.float64)
    Q_g = (np.asarray(orb.Q, dtype=np.float64)
           if orb.Q is not None else np.zeros_like(P_g))

    # Sign convention: P(r) > 0 near the origin. Atomx's SCF picks an
    # arbitrary overall phase per orbital; pin it to match DBSR_HF / the
    # reference Ag.h5 so (wave_large, wave_small) are reproducible.
    # Probe at the first sample where |P| is clearly above numerical floor
    # (inside the first lobe, before any physical node).
    probe = int(np.argmax(np.abs(P_g) > 1e-6 * np.abs(P_g).max()))
    if P_g[probe] < 0:
        P_g = -P_g
        Q_g = -Q_g

    # Densities on the full Gauss grid.
    p2 = P_g ** 2
    q2 = Q_g ** 2
    density_large_full = nq * p2
    density_small_full = nq * q2
    density_full = density_large_full + density_small_full

    # Integrated small-component density (matches Ag.h5 convention).
    small_component_ratio = float(np.trapezoid(density_small_full, grid_g))

    # Non-rel kinetic expectation on the large component:
    # <T_NR> = -1/2 <P|d^2/dr^2|P> + l(l+1)/2 <P|r^-2|P>; speed = sqrt(2T)/c.
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

    # Tail truncation: drop trailing samples where density_large has
    # fallen below the numerical floor (matches the varying per-subshell
    # grid lengths in the reference Ag.h5).
    keep = np.flatnonzero(density_large_full >= TAIL_THRESHOLD)
    last = int(keep[-1]) + 1 if keep.size else len(P_g)
    sl = slice(0, last)

    j = abs(int(orb.kappa)) - 0.5
    spin = float(j - l_orb)  # +0.5 for kappa<0 (j=l+1/2), -0.5 for kappa>0

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


# ----------------------------------------------------------------------
# HDF5 writer (atomic via tmp file)
# ----------------------------------------------------------------------

def write_h5(path: Path, Z: int, symbol: str, config_str: str, payloads: list) -> None:
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
            try:
                tmp.unlink()
            except OSError:
                pass
        raise


# ----------------------------------------------------------------------
# Verification (--verify-ag)
# ----------------------------------------------------------------------

def verify_against_reference(generated: Path, reference: Path) -> bool:
    """Diff-check a generated Ag.h5 against the reference. Advisory only."""
    LOG.info("Verifying %s against %s", generated, reference)
    ok = True
    with h5py.File(generated, "r") as fg, h5py.File(reference, "r") as fr:
        # Compare top-level keys (should be identical sets).
        kg, kr = set(fg.keys()), set(fr.keys())
        if kg != kr:
            LOG.warning("  key mismatch: generated=%s reference=%s", sorted(kg), sorted(kr))
            ok = False

        # Metadata attrs
        mg, mr = fg["metadata"].attrs, fr["metadata"].attrs
        for k in ("Z", "element", "method", "core_hole", "oxidation",
                  "electron_configuration"):
            if mg.get(k) != mr.get(k):
                LOG.warning("  metadata[%s]: generated=%r reference=%r",
                            k, mg.get(k), mr.get(k))
                ok = False

        # Per-orbital comparison
        for key in sorted(k for k in kg if k != "metadata"):
            if key not in fr:
                continue
            gg, gr = fg[key], fr[key]
            for attr in ("n", "l", "spin", "orbital", "nq"):
                if gg.attrs[attr] != gr.attrs[attr]:
                    LOG.warning("  /%s attr[%s] mismatch: %r vs %r",
                                key, attr, gg.attrs[attr], gr.attrs[attr])
                    ok = False
            for attr in ("small_component_ratio", "speed"):
                a, b = float(gg.attrs[attr]), float(gr.attrs[attr])
                rel = abs(a - b) / max(abs(b), 1e-12)
                status = "OK" if rel <= 5e-3 else "DIFF"
                LOG.info("  /%s %s: generated=%.6f reference=%.6f rel=%.2e [%s]",
                         key, attr, a, b, rel, status)
            for name in ("r", "density", "density_large", "density_small",
                         "wave_large", "wave_small"):
                sg = gg[name].shape
                sr = gr[name].shape
                delta = abs(sg[0] - sr[0])
                if delta > 5:
                    LOG.info("  /%s %s shape: generated=%s reference=%s (|d|=%d)",
                             key, name, sg, sr, delta)
    LOG.info("Verification result: %s", "PASS" if ok else "FAIL (see warnings)")
    return ok


# ----------------------------------------------------------------------
# Top-level driver
# ----------------------------------------------------------------------

def process_element(Z: int, args) -> bool:
    symbol = element_symbol(Z)
    out_path = Path(args.out) / f"{symbol}.h5"

    if out_path.exists() and not args.force:
        LOG.info("[SKIP] %s (exists at %s)", symbol, out_path)
        return True

    t0 = time.time()
    try:
        atom, config_str = solve_element(
            Z,
            correlation=args.correlation,
            breit=args.breit,
            qed=args.qed,
            max_iter=args.max_iter,
            tol=args.tol,
        )
    except Exception as e:
        LOG.error("[FAIL] %s Z=%d solve error: %s", symbol, Z, e)
        return False

    if not atom.converged:
        LOG.error("[FAIL] %s Z=%d did not converge after %d iterations",
                  symbol, Z, atom.iterations)
        return False

    result = atom.result
    grid = np.asarray(result.waves.grid, dtype=np.float64)

    # Build orbitals lookup by (n, kappa) since MCDHF with SD correlation
    # also returns virtual orbitals, which we must skip.
    orb_by_nk = {(int(o.n), int(o.kappa)): o for o in result.waves.orbitals.values()}

    payloads = []
    for (n, kappa, nq, ref_label) in reference_subshells(Z):
        orb = orb_by_nk.get((n, kappa))
        if orb is None:
            LOG.error("[FAIL] %s: no orbital for (n=%d, kappa=%d, %s) in MCDHF result",
                      symbol, n, kappa, ref_label)
            return False
        payloads.append(build_orbital_payload(orb, nq, grid))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_h5(out_path, Z, symbol, config_str, payloads)

    LOG.info("[OK] %s Z=%d | %d orbitals | conv=%s iters=%d | %.1fs",
             symbol, Z, len(payloads), atom.converged, atom.iterations,
             time.time() - t0)
    return True


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("elements", nargs="*",
                    help="Element symbols, Z ints, or inclusive ranges. "
                         "Supported: 'Ag', '47', '1-10', '1..20', '1:86', "
                         "'H-Ne'. Multiple tokens may be mixed: "
                         "'Ag Cu 26 50-60 La-Lu'.")
    ap.add_argument("--all", action="store_true",
                    help="Iterate Z=1..118 (overrides positional elements)")
    ap.add_argument("--out", default=DEFAULT_OUT,
                    help=f"Output directory (default: {DEFAULT_OUT})")
    ap.add_argument("--correlation", default=None, choices=[None, "s", "sd"],
                    help="Correlation level: None (default, DBSR_HF-style jj-"
                         "averaged DHF), 's' (singles), 'sd' (singles+doubles, "
                         "much slower)")
    ap.add_argument("--breit", action=argparse.BooleanOptionalAction, default=True,
                    help="Include Breit interaction (default: on)")
    ap.add_argument("--qed", action=argparse.BooleanOptionalAction, default=True,
                    help="Include QED corrections (default: on)")
    ap.add_argument("--max-iter", type=int, default=300,
                    help="SCF iteration cap (default: 300 — f-shell and "
                         "anomalous configs like Cu can need >100 iters)")
    ap.add_argument("--tol", type=float, default=1e-6,
                    help="SCF convergence tolerance (default: 1e-6)")
    ap.add_argument("--force", action="store_true",
                    help="Overwrite existing output files")
    ap.add_argument("--verify-ag", action="store_true",
                    help=f"After Ag, diff-check against {REFERENCE_AG}")
    ap.add_argument("-v", "--verbose", action="count", default=0,
                    help="-v for INFO, -vv for DEBUG")
    args = ap.parse_args(argv)

    level = logging.WARNING
    if args.verbose == 1:
        level = logging.INFO
    elif args.verbose >= 2:
        level = logging.DEBUG
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.all:
        Zs = list(range(1, 119))
    elif args.elements:
        Zs = parse_elements(args.elements)
    else:
        ap.error("Provide element tokens or --all")

    fails = 0
    for Z in Zs:
        symbol = element_symbol(Z)
        ok = process_element(Z, args)
        if not ok:
            fails += 1
            continue
        if args.verify_ag and symbol == "Ag":
            try:
                verify_against_reference(
                    Path(args.out) / "Ag.h5",
                    Path(REFERENCE_AG),
                )
            except Exception as e:
                LOG.warning("--verify-ag failed: %s", e)

    n = len(Zs)
    if fails:
        LOG.warning("Summary: %d/%d elements failed", fails, n)
    else:
        LOG.info("Summary: all %d elements succeeded", n)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
