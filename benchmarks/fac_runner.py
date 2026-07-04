"""Thin wrapper to run sfac scripts with the right DYLD_LIBRARY_PATH.

Usage:
    from benchmarks.fac_runner import run_sfac
    stdout, elapsed = run_sfac("my_script.sf", workdir=Path("tmp/"))
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
import time
import struct
from pathlib import Path

import numpy as np

_DEFAULT_SFAC_BIN = Path("/Users/zhangzz/code/ASAP/fac/bin/bin/sfac")
_DEFAULT_DYLD_LIB = "/opt/homebrew/Cellar/gcc/14.2.0_1/lib/gcc/current"


def resolve_sfac_bin() -> Path | None:
    """Return the configured/available ``sfac`` executable, if any."""
    for env_name in ("FAC_SFAC_PATH", "FAC_BIN"):
        value = os.environ.get(env_name)
        if value:
            return Path(value).expanduser()

    bin_dir = os.environ.get("FAC_BIN_DIR")
    if bin_dir:
        return Path(bin_dir).expanduser() / "sfac"

    path_hit = shutil.which("sfac")
    if path_hit:
        return Path(path_hit)

    return _DEFAULT_SFAC_BIN


def require_sfac_bin() -> Path:
    """Return an executable ``sfac`` path or raise a clear configuration error."""
    sfac_bin = resolve_sfac_bin()
    if sfac_bin is not None and sfac_bin.exists() and os.access(sfac_bin, os.X_OK):
        return sfac_bin

    configured = [
        name for name in ("FAC_SFAC_PATH", "FAC_BIN", "FAC_BIN_DIR")
        if os.environ.get(name)
    ]
    configured_msg = f" Checked: {', '.join(configured)}." if configured else ""
    raise FileNotFoundError(
        "sfac binary not found. Set FAC_SFAC_PATH or FAC_BIN to the sfac "
        "executable, set FAC_BIN_DIR to the directory containing sfac, or add "
        f"sfac to PATH.{configured_msg}"
    )


SFAC_BIN = resolve_sfac_bin()
DYLD_LIB = os.environ.get("FAC_DYLD_LIB", _DEFAULT_DYLD_LIB)


def run_sfac(script_path: Path, workdir: Path | None = None, timeout: float = 300.0) -> tuple[str, float, Path]:
    """Run sfac on the given script. Returns (stdout, elapsed_s, workdir)."""
    script_path = Path(script_path).resolve()
    sfac_bin = require_sfac_bin()
    if workdir is None:
        workdir = Path(tempfile.mkdtemp(prefix="fac_bench_"))
    workdir.mkdir(parents=True, exist_ok=True)
    dest = workdir / script_path.name
    if dest.resolve() != script_path.resolve():
        shutil.copy(script_path, dest)

    env = os.environ.copy()
    env["DYLD_LIBRARY_PATH"] = DYLD_LIB + ":" + env.get("DYLD_LIBRARY_PATH", "")

    t0 = time.perf_counter()
    proc = subprocess.run(
        [str(sfac_bin), script_path.name],
        cwd=workdir,
        capture_output=True,
        text=True,
        env=env,
        timeout=timeout,
    )
    t1 = time.perf_counter()
    if proc.returncode != 0:
        raise RuntimeError(f"sfac failed (rc={proc.returncode}):\n{proc.stderr}\n{proc.stdout}")
    return proc.stdout, t1 - t0, workdir


def parse_levels(lev_file: Path) -> list[dict]:
    """Parse a FAC .lev file (ASCII form, as written by PrintTable).

    Returns list of dicts with keys: ILEV, IBASE, ENERGY_eV, P, VNL, TWOJ, config, state.
    """
    rows = []
    in_data = False
    for line in Path(lev_file).read_text().splitlines():
        if line.strip().startswith("ILEV") and "IBASE" in line:
            in_data = True
            continue
        if not in_data:
            continue
        parts = line.strip().split()
        if len(parts) < 6:
            continue
        try:
            ilev = int(parts[0])
            ibase = int(parts[1])
            energy = float(parts[2])
            p = int(parts[3])
            vnl = parts[4]
            twoj = int(parts[5])
        except (ValueError, IndexError):
            continue
        config = " ".join(parts[6:])[:64].strip() if len(parts) > 6 else ""
        rows.append({
            "ILEV": ilev, "IBASE": ibase, "ENERGY_eV": energy,
            "P": p, "VNL": vnl, "TWOJ": twoj, "config": config,
        })
    return rows


def parse_e0(lev_file: Path) -> float | None:
    """Parse the E0 reference from the .lev header (eV)."""
    for line in Path(lev_file).read_text().splitlines():
        if line.startswith("E0"):
            # "E0\t= 0, -3.12294355E+04"
            parts = line.split("=")
            if len(parts) >= 2:
                tail = parts[1].strip()
                # "0, -3.12294355E+04"
                items = [x.strip() for x in tail.split(",")]
                if len(items) >= 2:
                    try:
                        return float(items[1])
                    except ValueError:
                        return None
    return None


def parse_hamiltonian_binary(ham_file: Path) -> dict:
    """Parse a FAC binary Hamiltonian file written by ``Structure(..., hfn, ...)``.

    FAC stores one lower-triangular sparse matrix per symmetry block. Values
    are in Hartree. The returned block matrices are expanded to dense,
    symmetric NumPy arrays because these diagnostics target small benchmark
    blocks.
    """
    data = Path(ham_file).read_bytes()
    off = 0

    def read_i() -> int:
        nonlocal off
        value = struct.unpack_from("i", data, off)[0]
        off += 4
        return value

    def read_d() -> float:
        nonlocal off
        value = struct.unpack_from("d", data, off)[0]
        off += 8
        return value

    ng0 = read_i()
    ng = read_i()
    kg = [read_i() for _ in range(ng)]
    ngp = read_i()
    kgp = [read_i() for _ in range(ngp)]

    blocks = []
    while off < len(data):
        symmetry = read_i()
        dim = read_i()
        if dim <= 0:
            continue
        orig_dim = read_i()
        n_basis = read_i()
        basis = [read_i() for _ in range(n_basis)]
        n_values = read_i()
        h = np.zeros((n_basis, n_basis), dtype=float)
        values = []
        for _ in range(n_values):
            i = read_i()
            j = read_i()
            value = read_d()
            h[i, j] = value
            h[j, i] = value
            values.append((i, j, value))
        blocks.append({
            "symmetry": symmetry,
            "dim": dim,
            "orig_dim": orig_dim,
            "n_basis": n_basis,
            "basis": basis,
            "hamiltonian": h,
            "values": values,
        })

    return {"ng0": ng0, "ng": ng, "kg": kg, "ngp": ngp, "kgp": kgp, "blocks": blocks}




def parse_debug_hamiltonian_log(log_file: Path) -> list[dict]:
    """Parse FAC ``FAC_DEBUG >= DEBUG_STRUCTURE`` Hamiltonian contribution logs.

    A debug-enabled FAC build prints sections beginning with ``HPAIR i j``
    followed by shell descriptors and ``1 electron:`` / ``2 electron:``
    contribution totals.  This parser groups consecutive HPAIR sections into
    Hamiltonian blocks (a new block starts when FAC returns to pair ``0 0``),
    expands each group to a dense symmetric matrix, and preserves the
    shell-resolved contribution details for coefficient audits.

    The values are exactly as printed by FAC.  For scalar Hamiltonian matrix
    elements FAC later applies the Wigner-Eckart factor ``1/sqrt(2J+1)``;
    callers comparing with binary Hamiltonian files should apply that factor
    when appropriate.
    """
    sections: list[dict] = []
    current: dict | None = None
    shell_lines: list[tuple[int, int, int, int, int]] = []
    shell_re = re.compile(r"^\s*(-?\d+)\s+(-?\d+)\s+(-?\d+)\s+(-?\d+)\s+(-?\d+)\s*$")
    contrib_re = re.compile(r"^\s*([12]) electron:\s+([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?)\s*$")

    for line in Path(log_file).read_text(errors="ignore").splitlines():
        if line.startswith("HPAIR "):
            if current is not None:
                sections.append(current)
            _tag, i, j = line.split()[:3]
            current = {"pair": (int(i), int(j)), "one": [], "two": [], "details": []}
            shell_lines = []
            continue
        if current is None:
            continue

        m_shell = shell_re.match(line)
        if m_shell:
            shell_lines.append(tuple(int(x) for x in m_shell.groups()))
            continue

        m_contrib = contrib_re.match(line)
        if m_contrib:
            kind = "one" if m_contrib.group(1) == "1" else "two"
            value = float(m_contrib.group(2))
            n_shells = 2 if kind == "one" else 4
            shells = tuple(shell_lines[-n_shells:])
            current[kind].append(value)
            current["details"].append({"kind": kind, "value": value, "shells": shells})

    if current is not None:
        sections.append(current)

    groups: list[list[dict]] = []
    group: list[dict] = []
    for section in sections:
        if section["pair"] == (0, 0) and group:
            groups.append(group)
            group = []
        group.append(section)
    if group:
        groups.append(group)

    out: list[dict] = []
    for group in groups:
        dim = max(max(section["pair"]) for section in group) + 1
        h = np.zeros((dim, dim), dtype=float)
        for section in group:
            i, j = section["pair"]
            value = float(sum(section["one"]) + sum(section["two"]))
            h[i, j] = value
            h[j, i] = value
        out.append({"dimension": dim, "hamiltonian": h, "sections": group})
    return out


def parse_transitions(tr_file: Path) -> list[dict]:
    """Parse a FAC .tr file written by PrintTable(..., 1).

    Yields dicts with upper/lower level indices, Δ E in eV, gA rate in s⁻¹, gf.

    File format (after headers):
      upper  p_up  lower  p_lo   Delta_E_eV   gf   gA_per_s   gf_alt
    """
    rows = []
    current = None
    for line in Path(tr_file).read_text().splitlines():
        ls = line.strip()
        if ls.startswith("MULTIP"):
            try:
                current = int(ls.split("=")[1].strip().split()[0])
            except (ValueError, IndexError):
                current = None
            continue
        parts = ls.split()
        if len(parts) < 7:
            continue
        try:
            up = int(parts[0])
            p_up = int(parts[1])
            lo = int(parts[2])
            p_lo = int(parts[3])
            delta_e = float(parts[4])
            gf = float(parts[5])
            ga = float(parts[6])
        except (ValueError, IndexError):
            continue
        rows.append({
            "mult": current, "upper": up, "lower": lo,
            "DELTA_E_eV": delta_e, "gf": gf, "gA_per_s": ga,
            "p_up": p_up, "p_lo": p_lo,
        })
    return rows


def parse_rr(rr_file: Path) -> list[dict]:
    """Parse a FAC RR table written by ``PrintTable(..., 1)``.

    Returns one row per bound/free level pair.  ``E_eV`` and
    ``sigma_RR_1e20cm2`` are NumPy arrays on the FAC user electron-energy
    grid.  FAC prints several per-grid columns; the first data column after
    electron energy is the RR cross section used by the parity benchmark.
    """
    lines = Path(rr_file).read_text().splitlines()
    nusr = None
    for i, line in enumerate(lines):
        if line.strip().startswith("NUSR"):
            try:
                nusr = int(line.split("=")[1].strip())
            except (IndexError, ValueError):
                nusr = None
            break
    if nusr is None:
        raise ValueError(f"Could not find NUSR in {rr_file}")

    rows = []
    i = 0
    while i < len(lines):
        parts = lines[i].strip().split()
        if len(parts) == 6:
            try:
                bound = int(parts[0])
                twoJ_b = int(parts[1])
                free = int(parts[2])
                twoJ_f = int(parts[3])
                delta_e = float(parts[4])
                transition_type = int(parts[5])
            except ValueError:
                i += 1
                continue

            param_idx = i + 1
            data_start = i + 2
            data_end = data_start + nusr
            if data_end > len(lines):
                break
            try:
                params = [float(x) for x in lines[param_idx].strip().split()]
                grid = [[float(x) for x in lines[j].strip().split()]
                        for j in range(data_start, data_end)]
            except ValueError:
                i += 1
                continue
            if any(len(g) < 4 for g in grid):
                i += 1
                continue

            arr = np.asarray(grid, dtype=float)
            rows.append({
                "bound": bound,
                "twoJ_b": twoJ_b,
                "free": free,
                "twoJ_f": twoJ_f,
                "delta_e_eV": delta_e,
                "transition_type": transition_type,
                "params": params,
                "E_eV": arr[:, 0],
                "sigma_RR_1e20cm2": arr[:, 1],
            })
            i = data_end
            continue
        i += 1
    return rows


def _parse_nusr(lines: list[str], path: Path) -> int:
    for line in lines:
        if line.strip().startswith("NUSR"):
            try:
                return int(line.split("=")[1].strip())
            except (IndexError, ValueError):
                break
    raise ValueError(f"Could not find NUSR in {path}")


def _parse_grid_table(path: Path, ncols: int) -> list[dict]:
    """Parse FAC CE/CI-style grid tables with transition blocks."""
    lines = Path(path).read_text().splitlines()
    nusr = _parse_nusr(lines, Path(path))
    rows = []
    i = 0
    while i < len(lines):
        parts = lines[i].strip().split()
        if len(parts) == 6:
            try:
                bound = int(parts[0])
                twoJ_b = int(parts[1])
                free = int(parts[2])
                twoJ_f = int(parts[3])
                delta_e = float(parts[4])
                transition_type = int(parts[5])
            except ValueError:
                i += 1
                continue
            param_idx = i + 1
            data_start = i + 2
            data_end = data_start + nusr
            if data_end > len(lines):
                break
            try:
                params = [float(x) for x in lines[param_idx].strip().split()]
                grid = [[float(x) for x in lines[j].strip().split()]
                        for j in range(data_start, data_end)]
            except ValueError:
                i += 1
                continue
            if any(len(g) < ncols for g in grid):
                i += 1
                continue
            arr = np.asarray(grid, dtype=float)
            rows.append({
                "bound": bound,
                "twoJ_b": twoJ_b,
                "free": free,
                "twoJ_f": twoJ_f,
                "DELTA_E_eV": delta_e,
                "transition_type": transition_type,
                "params": params,
                "E_eV": arr[:, 0],
                "sigma_1e20cm2": arr[:, 1],
                "Omega": arr[:, 2] if arr.shape[1] > 2 else np.zeros_like(arr[:, 0]),
            })
            i = data_end
            continue
        i += 1
    return rows


def parse_ce(ce_file: Path) -> list[dict]:
    """Parse FAC electron-impact excitation table."""
    return _parse_grid_table(ce_file, ncols=3)


def parse_ci(ci_file: Path) -> list[dict]:
    """Parse FAC electron-impact ionisation table."""
    rows = _parse_grid_table(ci_file, ncols=3)
    # FAC PrintCITable writes: E_user, strength, sigma.  This differs from
    # PrintCETable, where the second column is the cross section and the third
    # is collision strength.  Keep the common key names for benchmark callers.
    lines = Path(ci_file).read_text().splitlines()
    # Re-parse only the grid rows so the generic transition/header parsing
    # remains in one place.
    nusr = None
    grid_blocks: list[np.ndarray] = []
    i = 0
    while i < len(lines):
        ls = lines[i].strip()
        if ls.startswith("NUSR"):
            try:
                nusr = int(ls.split("=")[1].strip())
            except Exception:
                nusr = None
        parts = ls.split()
        if nusr is not None and len(parts) >= 6:
            try:
                int(parts[0]); int(parts[2]); float(parts[4])
            except ValueError:
                i += 1
                continue
            data_start = i + 2
            data_end = data_start + nusr
            if data_end <= len(lines):
                try:
                    grid = np.asarray(
                        [[float(x) for x in lines[j].strip().split()]
                         for j in range(data_start, data_end)],
                        dtype=float,
                    )
                except ValueError:
                    i += 1
                    continue
                if grid.ndim == 2 and grid.shape[1] >= 3:
                    grid_blocks.append(grid)
                i = data_end
                continue
        i += 1
    for row, grid in zip(rows, grid_blocks):
        row["strength"] = grid[:, 1]
        row["sigma_1e20cm2"] = grid[:, 2]
        row["Omega"] = grid[:, 1]
    return rows


def parse_ai(ai_file: Path) -> list[dict]:
    """Parse a FAC .ai (autoionization) file written by ``PrintTable(..., 1)``.

    Each data row written by FAC has the form::

        bound  2J_bound   free  2J_free   Delta_E_eV   AI_rate_per_s   strength

    where ``bound``/``free`` are FAC ILEV indices for the autoionising and the
    continuum-anchor states.  Returns one dict per row.
    """
    rows: list[dict] = []
    for line in Path(ai_file).read_text().splitlines():
        parts = line.strip().split()
        if len(parts) < 7:
            continue
        try:
            upper = int(parts[0])
            twoj_u = int(parts[1])
            lower = int(parts[2])
            twoj_l = int(parts[3])
            delta_e = float(parts[4])
            rate = float(parts[5])
            strength = float(parts[6])
        except (ValueError, IndexError):
            continue
        rows.append({
            "upper": upper, "lower": lower,
            "TWOJ_upper": twoj_u, "TWOJ_lower": twoj_l,
            "DELTA_E_eV": delta_e,
            "rate_per_s": rate,
            "strength": strength,
        })
    return rows
