"""FAC ASCII I/O for atomx.

Writes and reads FAC's ``PrintTable(..., 1)`` ASCII output for:

* ``LEV``  — atomic energy levels (Type 1)
* ``TR``   — radiative transitions (Type 2)
* ``CE``   — electron-impact collision excitation (Type 3)

The on-disk layout mirrors what ``sfac`` itself emits (key/value header
lines separated by tabs, a column-name banner, then whitespace-separated
data rows).  See Gu (2008) §5 and the FAC source
(``interface.c::SaveLevels``, ``SaveTransition``, ``SaveExcitation``)
for the canonical definitions.

Numbers are written wide enough that a write→read round trip preserves
level energies and A-rates to better than 1e-6 relative.

Readers return lightweight ``list[dict]`` rows; callers that want a
``pandas.DataFrame`` can wrap them with ``pd.DataFrame(rows)``.
"""

from __future__ import annotations

import contextlib
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from atomx.angular.recoupling import l_of_kappa
from atomx.constants import HARTREE_EV, element_symbol

FAC_TYPE_LEV = 1
FAC_TYPE_TR = 2
FAC_TYPE_CE = 3

FAC_VERSION = "FAC 1.1.5"

# Multipole-string ↔ FAC integer code.  FAC encodes E1 = -1, M1 = 1,
# E2 = -2, M2 = 2, … in its binary headers; the ASCII ``MULTIP`` line
# carries the same convention.
_MULTIP_CODE = {
    "E1": -1, "M1": 1, "E2": -2, "M2": 2,
    "E3": -3, "M3": 3, "E4": -4, "M4": 4,
}
_MULTIP_FROM_CODE = {v: k for k, v in _MULTIP_CODE.items()}


# ---------------------------------------------------------------------------
# Header utilities
# ---------------------------------------------------------------------------

def _emit_global_header(
    table_type: int,
    Z: int,
    *,
    n_blocks: int = 1,
    verbose: int = 1,
    tsess: int | None = None,
    e0_eV: float | None = None,
) -> list[str]:
    """Emit the seven-line block that opens every FAC ASCII table."""
    if tsess is None:
        tsess = int(time.time())
    sym = element_symbol(int(Z))
    lines = [
        FAC_VERSION,
        "Endian\t= 0",
        f"TSess\t= {tsess}",
        f"Type\t= {table_type}",
        f"Verbose\t= {verbose}",
        f"{sym} Z\t= {float(Z):5.1f}",
        f"NBlocks\t= {n_blocks}",
    ]
    if table_type == FAC_TYPE_LEV:
        # FAC writes "E0\t= 0, <ground binding eV>".  The leading 0 is
        # the block index; we only ever emit a single block.
        lines.append(f"E0\t= 0, {float(e0_eV or 0.0): .8E}")
    lines.append("")  # blank line before per-block header
    return lines


def _parse_global_header(lines: Iterable[str]) -> dict[str, Any]:
    """Pull ``Z``, ``Type``, ``NBlocks`` and ``E0`` out of the seven-line
    preamble.  Returns whatever it can find; missing keys are ``None``."""
    meta: dict[str, Any] = {"Z": None, "Type": None, "NBlocks": None,
                            "E0_eV": None, "fac_version": None}
    for raw in lines:
        line = raw.rstrip()
        if not line:
            break
        if line.startswith("FAC "):
            meta["fac_version"] = line.strip()
            continue
        if "\t=" not in line and " =" not in line and "=" not in line:
            continue
        if line.startswith("Type"):
            meta["Type"] = int(line.split("=")[1].split()[0])
        elif line.startswith("NBlocks"):
            meta["NBlocks"] = int(line.split("=")[1].split()[0])
        elif line.startswith("E0"):
            # "E0\t= 0, -2.01497850E+04"
            tail = line.split("=", 1)[1]
            parts = [p.strip() for p in tail.split(",")]
            if len(parts) >= 2:
                with contextlib.suppress(ValueError):
                    meta["E0_eV"] = float(parts[1])
        elif " Z\t=" in line or " Z =" in line:
            # "<Sym> Z\t=  26.0"
            with contextlib.suppress(ValueError, IndexError):
                meta["Z"] = int(round(float(line.split("=")[1].strip())))
    return meta


# ---------------------------------------------------------------------------
# LEV (levels)
# ---------------------------------------------------------------------------

_LEV_BANNER = "  ILEV  IBASE    ENERGY       P   VNL   2J"


def _vnl_for_level(level: dict[str, Any]) -> int:
    """Compact ``100*n + l`` encoding FAC uses for the dominant orbital.

    Returns 0 when ``n``/``l`` aren't provided.
    """
    n = level.get("n")
    l = level.get("l")
    if n is None or l is None:
        return 0
    return 100 * int(n) + int(l)


def _format_lev_row(
    ilev: int,
    energy_eV: float,
    parity: int,
    vnl: int,
    twoj: int,
    config: str,
    state: str,
    *,
    ibase: int = -1,
) -> str:
    """Render one ``.lev`` data line in FAC's column layout."""
    # FAC widths (verified against sfac output): ILEV %5d, IBASE %5d,
    # ENERGY %15.8E with a leading space, P %2d, VNL %5d, 2J %4d, then
    # two free-form 64-char wide descriptors.
    cfg = (config or "")[:64].ljust(64)
    sta = (state or "")[:64].ljust(64)
    return (
        f"{ilev:5d}  {ibase:5d}  {energy_eV: .8E} {int(parity):d}"
        f" {vnl:5d}  {twoj:3d} {cfg} {sta}"
    )


def write_lev(
    atom_or_levels,
    path: str | Path,
    *,
    Z: int | None = None,
    e0_eV: float | None = None,
    tsess: int | None = None,
) -> Path:
    """Write FAC LEV (Type 1) ASCII for the bound levels of ``atom``.

    Parameters
    ----------
    atom_or_levels :
        Either a solved :class:`atomx.Atom` (its CI / MCDHF result is
        introspected for level energies, J and parity), or a list of
        dicts with explicit fields::

            {ilev, energy_eV, parity, twoj, config[, state, n, l, ibase]}

        ``parity`` follows FAC's convention (``0`` even, ``1`` odd).
    path :
        Output file (anything :class:`pathlib.Path` accepts).
    Z :
        Nuclear charge.  Required when ``atom_or_levels`` is a list of
        dicts; inferred from the Atom otherwise.
    e0_eV :
        Ground-state binding energy in eV (negative).  If ``None`` and
        an Atom is supplied, taken from ``atom.total_energy``.  Stored
        in the LEV header's ``E0`` field for downstream consumers
        (CHIANTI / FLYCHK) that need the absolute zero.

    Returns the path written.
    """
    path = Path(path)

    if isinstance(atom_or_levels, list):
        levels = list(atom_or_levels)
        if Z is None:
            raise ValueError("Z is required when writing from an explicit level list.")
        e0_eV = e0_eV if e0_eV is not None else 0.0
    else:
        atom = atom_or_levels
        if Z is None:
            Z = atom.Z
        if e0_eV is None and getattr(atom, "total_energy", None) is not None:
            e0_eV = float(atom.total_energy) * HARTREE_EV
        levels = _levels_from_atom(atom)

    if Z is None:
        raise ValueError("Z must be provided or inferable from the Atom.")
    # Number of electrons (best effort).
    nele = _count_electrons(atom_or_levels, fallback=Z if isinstance(atom_or_levels, list) else None)

    lines = _emit_global_header(FAC_TYPE_LEV, int(Z), e0_eV=e0_eV, tsess=tsess)
    lines.append(f"NELE\t= {nele}")
    lines.append(f"NLEV\t= {len(levels)}")
    lines.append(_LEV_BANNER)
    for L in levels:
        lines.append(_format_lev_row(
            ilev=int(L.get("ilev", L.get("ILEV", 0))),
            energy_eV=float(L["energy_eV"]),
            parity=int(L.get("parity", L.get("P", 0))),
            vnl=int(L.get("vnl", _vnl_for_level(L))),
            twoj=int(L.get("twoj", L.get("2J", 0))),
            config=str(L.get("config", "")),
            state=str(L.get("state", L.get("config", ""))),
            ibase=int(L.get("ibase", L.get("IBASE", -1))),
        ))
    lines.append("")
    path.write_text("\n".join(lines))
    return path


def read_lev(path: str | Path) -> list[dict[str, Any]]:
    """Read a FAC LEV ASCII table.

    Returns one dict per level with keys
    ``{ILEV, IBASE, ENERGY_eV, P, VNL, TWOJ, config, state, n, l, Z, E0_eV}``.

    Tolerant of column-padding changes — the parser splits on whitespace
    and consumes the first six numeric fields, then keeps the remaining
    text as a free-form configuration / state descriptor.
    """
    raw = Path(path).read_text().splitlines()
    meta = _parse_global_header(raw)
    rows: list[dict[str, Any]] = []
    in_data = False
    for line in raw:
        s = line.strip()
        if s.startswith("ILEV") and "IBASE" in s:
            in_data = True
            continue
        if not in_data or not s:
            continue
        parts = s.split()
        if len(parts) < 6:
            continue
        try:
            ilev = int(parts[0])
            ibase = int(parts[1])
            energy = float(parts[2])
            p = int(parts[3])
            vnl = int(parts[4])
            twoj = int(parts[5])
        except ValueError:
            continue
        # Position-based config/state extraction would be fragile against
        # FAC version drift, so we split on whitespace and reconstruct.
        if len(parts) >= 7:
            # heuristic: state = last token, config = everything before
            state = parts[-1]
            cfg = " ".join(parts[6:-1]).strip()
        else:
            state, cfg = "", ""
        n = vnl // 100 if vnl else None
        l = (vnl % 100) if vnl else None
        rows.append({
            "ILEV": ilev,
            "IBASE": ibase,
            "ENERGY_eV": energy,
            "P": p,
            "VNL": vnl,
            "TWOJ": twoj,
            "config": cfg,
            "state": state,
            "n": n,
            "l": l,
            "Z": meta["Z"],
            "E0_eV": meta["E0_eV"],
        })
    return rows


# ---------------------------------------------------------------------------
# TR (radiative transitions)
# ---------------------------------------------------------------------------

def _format_tr_row(
    upper: int, twoj_upper: int,
    lower: int, twoj_lower: int,
    deltaE_eV: float, gf: float, A_per_s: float,
    reduced_matrix_element: float,
) -> str:
    return (
        f"{upper:6d} {twoj_upper:2d}  {lower:5d} {twoj_lower:2d}"
        f"  {deltaE_eV: .8E}  {gf: .8E}  {A_per_s: .8E}  {reduced_matrix_element: .8E}"
    )


def write_tra(
    atom_or_transitions,
    path: str | Path,
    *,
    multipole: str = "E1",
    gauge: str = "length",
    Z: int | None = None,
    tsess: int | None = None,
    levels: Sequence[dict] | None = None,
) -> Path:
    """Write FAC TR (Type 2) ASCII for the radiative transitions of ``atom``.

    Parameters
    ----------
    atom_or_transitions :
        Either a solved :class:`atomx.Atom` (which is asked for
        ``atom.transitions(multipoles=[multipole], gauge=gauge)``), or a
        pre-computed iterable of dicts::

            {upper, twoj_upper, lower, twoj_lower,
             deltaE_eV, gf, A_per_s, reduced_matrix_element}

        For MCDHF results we recover ``upper`` / ``lower`` from
        ``TransitionResult.{initial,final}_label`` which look like
        ``"State_3"`` — anything else is treated as orbital labels and
        we fall back to enumerating the labels in encounter order.
    multipole, gauge :
        Forwarded to ``atom.transitions``.  Default E1 / length gauge.
    Z, tsess :
        Override metadata; ``Z`` is required when passing an explicit
        list and not an Atom.
    levels :
        Optional list of dicts (same shape as :func:`read_lev` rows) so
        the writer can look up ``twoj`` from the LEV table when the
        ``TransitionResult`` doesn't carry it.

    Computed rows export the positive magnitude of FAC's reduced amplitude:
    TransitionResult retains line strengths, not coherent matrix-element
    phases. Explicit rows preserve the supplied signed amplitude. Columns
    follow FAC's rate convention: A, not degeneracy-weighted gA.

    Returns the path written.
    """
    path = Path(path)

    is_explicit_rows = isinstance(atom_or_transitions, (list, tuple)) and (
        not atom_or_transitions or isinstance(atom_or_transitions[0], dict)
    )
    if is_explicit_rows:
        rows = list(atom_or_transitions)
        if Z is None:
            raise ValueError("Z is required when writing from an explicit transition list.")
    else:
        atom = atom_or_transitions
        if Z is None:
            Z = int(atom.Z)
        rows = _transitions_from_atom(atom, multipole=multipole, gauge=gauge,
                                       levels=levels)

    if Z is None:
        raise ValueError("Z must be provided or inferable from the Atom.")
    nele = _count_electrons(atom_or_transitions, fallback=int(Z) if is_explicit_rows else None)

    if multipole.upper() not in _MULTIP_CODE:
        raise ValueError(f'unsupported FAC multipole {multipole!r}')
    code = _MULTIP_CODE[multipole.upper()]
    gauge_code = 2 if gauge.lower() in ("length", "babushkin") else 1

    lines = _emit_global_header(FAC_TYPE_TR, int(Z), tsess=tsess)
    lines.append(f"NELE\t= {nele}")
    lines.append(f"NTRANS\t= {len(rows)}")
    lines.append(f"MULTIP\t= {code}")
    lines.append(f"GAUGE\t= {gauge_code}")
    default_mode = 0 if multipole.upper().startswith('M') else 1
    modes = {int(r.get('mode', default_mode)) for r in rows}
    if len(modes) > 1 or not modes.issubset({0, 1}):
        raise ValueError('one FAC transition block requires a common mode 0 or 1')
    lines.append(f"MODE\t= {next(iter(modes), default_mode)}")
    for r in rows:
        lines.append(_format_tr_row(
            upper=int(r["upper"]),
            twoj_upper=int(r.get("twoj_upper", r.get("2J_upper", 0))),
            lower=int(r["lower"]),
            twoj_lower=int(r.get("twoj_lower", r.get("2J_lower", 0))),
            deltaE_eV=float(r["deltaE_eV"]),
            gf=float(r["gf"]),
            A_per_s=float(r["A_per_s"]),
            reduced_matrix_element=float(r["reduced_matrix_element"]),
        ))
    lines.append("")
    path.write_text("\n".join(lines))
    return path


def read_tra(path: str | Path) -> list[dict[str, Any]]:
    """Read a FAC TR ASCII table.

    Returns one dict per transition with keys
    ``{upper, twoj_upper, lower, twoj_lower, DELTA_E_eV, gf, A_per_s,
       reduced_matrix_element, multipole, multipole_code, gauge, mode, Z}``.

    FAC ``dbase.c::PrintTRTable`` prints A (already divided by the upper
    degeneracy) and a signed reduced multipole amplitude in columns 7 and 8.
    Column 8 is not a second gf. Aggregated MULTIP=0 tables carry total gf
    there instead; their reduced_matrix_element is None and aggregate_gf
    retains the value. UTA tables have a different layout and are rejected.
    """
    raw = Path(path).read_text().splitlines()
    meta = _parse_global_header(raw)
    rows: list[dict[str, Any]] = []
    multip_code = None
    gauge_code = None
    mode = None
    seen_data = False
    for line in raw:
        s = line.strip()
        if not s:
            continue
        if s.startswith("MULTIP"):
            try:
                multip_code = int(s.split("=")[1].strip().split()[0])
            except (ValueError, IndexError):
                multip_code = None
            continue
        if s.startswith("GAUGE"):
            try:
                gauge_code = int(s.split("=")[1].strip().split()[0])
            except (ValueError, IndexError):
                gauge_code = None
            continue
        if s.startswith("MODE"):
            mode = int(s.split('=')[1])
            continue
        if s.startswith("NELE") or s.startswith("NTRANS"):
            continue
        parts = s.split()
        if len(parts) < 7:
            continue
        try:
            upper = int(parts[0])
            twoj_u = int(parts[1])
            lower = int(parts[2])
            twoj_l = int(parts[3])
            de = float(parts[4])
            gf = float(parts[5])
            rate = float(parts[6])
            amplitude = float(parts[7]) if len(parts) > 7 else None
        except ValueError:
            continue
        if len(parts) != 8 or multip_code is None or gauge_code is None or mode is None:
            raise ValueError('unsupported or incomplete FAC resolved-transition table')
        seen_data = True
        rows.append({
            "upper": upper,
            "twoj_upper": twoj_u,
            "lower": lower,
            "twoj_lower": twoj_l,
            "DELTA_E_eV": de,
            "gf": gf,
            "A_per_s": rate,
            "reduced_matrix_element": amplitude if multip_code != 0 else None,
            "aggregate_gf": amplitude if multip_code == 0 else None,
            "multipole": _MULTIP_FROM_CODE.get(multip_code, "total" if multip_code == 0 else "unknown"),
            "multipole_code": multip_code,
            "mode": mode,
            "gauge": "length" if (gauge_code or 2) == 2 else "velocity",
            "Z": meta["Z"],
        })
    if not seen_data and not rows:
        return []
    return rows


# ---------------------------------------------------------------------------
# CE (collision excitation)
# ---------------------------------------------------------------------------

def _is_int_token(s: str) -> bool:
    """Cheap test for ``int(s)`` succeeding without a decimal/exponent."""
    if not s:
        return False
    if "." in s or "e" in s or "E" in s:
        return False
    try:
        int(s)
    except ValueError:
        return False
    return True


def _is_ce_transition_header(parts: list[str]) -> bool:
    """Distinguish a CE transition descriptor ``"lo 2J_lo up 2J_up E NRADS"``
    from a 3-column ``(E, omega, sigma)`` data row.

    The first four fields and the sixth are integers; the fifth is a
    float (the threshold energy).  Data rows are exactly 3 floats.
    """
    if len(parts) < 6:
        return False
    if not all(_is_int_token(p) for p in (parts[0], parts[1], parts[2], parts[3], parts[5])):
        return False
    try:
        float(parts[4])
    except ValueError:
        return False
    return True



def write_ce(
    transitions: Sequence[dict[str, Any]],
    path: str | Path,
    *,
    Z: int,
    nele: int | None = None,
    tsess: int | None = None,
    energy_grid_eV: Sequence[float] | None = None,
    te0_eV: float | None = None,
) -> Path:
    """Write FAC CE (Type 3) ASCII for a list of collision-excitation
    transitions.

    Each entry in ``transitions`` is a dict with::

        {upper, twoj_upper, lower, twoj_lower, deltaE_eV,
         energies_eV, collision_strength, sigma_1e20cm2}

    where ``energies_eV``, ``collision_strength`` and ``sigma_1e20cm2``
    are 1-D arrays sampled on a shared electron-energy grid (the usr
    grid).  Per-transition grids are allowed; pass ``energy_grid_eV`` to
    override the common usr/NEGRID listing in the header.

    The current writer emits the simplest FAC variant:
    ``QKMODE = 0``, ``NPARAMS = 0``, ``MSUB = 0``, ``PWTYPE = 0``,
    ``ETYPE = UTYPE = 1`` — i.e. no Born-asymptote fit coefficients, no
    magnetic-sublevel detail, electron energies given on a user grid.
    Variants with non-zero NPARAMS are out of scope (see HYP-141 Sp2).
    """
    path = Path(path)
    rows = list(transitions)
    if not rows:
        raise ValueError("write_ce requires at least one transition.")

    if energy_grid_eV is None:
        energy_grid_eV = np.asarray(rows[0]["energies_eV"], dtype=float)
    energy_grid = np.asarray(energy_grid_eV, dtype=float)
    n_usr = int(energy_grid.size)

    def _de(r: dict[str, Any]) -> float:
        return float(r.get("deltaE_eV", r.get("DELTA_E_eV", 0.0)))

    if te0_eV is None:
        te0_eV = float(np.mean([_de(r) for r in rows]))

    if nele is None:
        nele = 0

    lines = _emit_global_header(FAC_TYPE_CE, Z, tsess=tsess)
    lines.append(f"NELE\t= {nele}")
    lines.append(f"NTRANS\t= {len(rows)}")
    lines.append("QKMODE\t= 0")
    lines.append("NPARAMS\t= 0")
    lines.append("MSUB\t= 0")
    lines.append("PWTYPE\t= 0")
    # te-grid: pin to the per-transition thresholds (NTEGRID = NTRANS,
    # one threshold per transition).  This mirrors what sfac writes for
    # the small Li-like test case.
    lines.append(f"NTEGRID\t= {len(rows)}")
    for r in rows:
        lines.append(f"\t  {_de(r): .8E}")
    lines.append(f"TE0\t= {float(te0_eV): .8E}")
    lines.append("ETYPE\t= 1")
    lines.append(f"NEGRID\t= {n_usr}")
    for e in energy_grid:
        lines.append(f"\t  {float(e): .8E}")
    lines.append("UTYPE\t= 1")
    lines.append(f"NUSR\t= {n_usr}")
    for e in energy_grid:
        lines.append(f"\t  {float(e): .8E}")
    for r in rows:
        twoj_u = int(r.get("twoj_upper", r.get("2J_upper", 0)))
        twoj_l = int(r.get("twoj_lower", r.get("2J_lower", 0)))
        # FAC SaveExcitation header: lower 2J_l upper 2J_u ΔE NRADS
        lines.append(
            f"{int(r['lower']):6d} {twoj_l:2d}  {int(r['upper']):5d} {twoj_u:2d}"
            f"  {_de(r): .8E} 1"
        )
        energies = np.asarray(r["energies_eV"], dtype=float)
        omegas = np.asarray(r["collision_strength"], dtype=float)
        sigmas = np.asarray(r["sigma_1e20cm2"], dtype=float)
        if not (energies.size == omegas.size == sigmas.size):
            raise ValueError(
                "energies_eV, collision_strength, sigma_1e20cm2 must share length."
            )
        for ei, oi, si in zip(energies, omegas, sigmas, strict=False):
            lines.append(f" {float(ei): .8E}  {float(oi): .8E}  {float(si): .8E}")
    lines.append("")
    path.write_text("\n".join(lines))
    return path


def read_ce(path: str | Path) -> list[dict[str, Any]]:
    """Read a FAC CE ASCII table.

    Returns one dict per transition::

        {upper, lower, DELTA_E_eV, twoj_upper, twoj_lower,
         energy_eV: ndarray, collision_strength: ndarray,
         sigma_1e20cm2: ndarray, Z}

    Tolerant of the optional Born-asymptote 4-float coefficient line FAC
    emits between blocks when ``NPARAMS > 0`` (it is silently skipped).
    """
    raw = Path(path).read_text().splitlines()
    meta = _parse_global_header(raw)
    rows: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    energies: list[float] = []
    strengths: list[float] = []
    sigmas: list[float] = []
    nparams = 0
    pending_header = False

    def _close():
        nonlocal current, energies, strengths, sigmas, pending_header
        if current is None or not energies:
            current = None
            energies, strengths, sigmas = [], [], []
            pending_header = False
            return
        current["energy_eV"] = np.asarray(energies, dtype=float)
        current["collision_strength"] = np.asarray(strengths, dtype=float)
        current["sigma_1e20cm2"] = np.asarray(sigmas, dtype=float)
        current["Z"] = meta["Z"]
        rows.append(current)
        current = None
        energies, strengths, sigmas = [], [], []
        pending_header = False

    for raw_line in raw:
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("NUSR"):
            _close()
            continue
        if line.startswith("NPARAMS"):
            try:
                nparams = int(line.split("=")[-1].strip())
            except ValueError:
                nparams = 0
            continue
        # Header / grid lines we don't need.
        if any(line.startswith(k) for k in (
            "FAC", "Endian", "TSess", "Type", "Verbose", "NBlocks", "NELE",
            "NTRANS", "QKMODE", "MSUB", "PWTYPE", "NTEGRID", "TE0", "ETYPE",
            "NEGRID", "UTYPE",
        )) or (" Z" in line and line.split()[0].isalpha()):
            continue
        parts = line.split()
        # Transition descriptor: "lower 2J_lo upper 2J_up E NRADS" (FAC's
        # SaveExcitation convention — see interface.c).  Six fields, the
        # first four and the last are integers, the fifth is a float.
        is_descriptor = len(parts) >= 6 and _is_ce_transition_header(parts)
        if is_descriptor:
            _close()
            lower = int(parts[0])
            twoj_l = int(parts[1])
            upper = int(parts[2])
            twoj_u = int(parts[3])
            dE = float(parts[4])
            current = {
                "upper": upper, "twoj_upper": twoj_u,
                "lower": lower, "twoj_lower": twoj_l,
                "DELTA_E_eV": dE,
            }
            energies, strengths, sigmas = [], [], []
            pending_header = nparams > 0
            continue
        if pending_header:
            pending_header = False
            try:
                _ = [float(p) for p in parts]
                if len(parts) == 4:
                    continue
            except ValueError:
                pass
        if current is not None and len(parts) >= 3:
            try:
                e = float(parts[0])
                strength = float(parts[1])
                sigma = float(parts[2])
            except ValueError:
                continue
            energies.append(e)
            strengths.append(strength)
            sigmas.append(sigma)
    _close()
    return rows


# ---------------------------------------------------------------------------
# Atom → level / transition rows
# ---------------------------------------------------------------------------

def _count_electrons(atom_or_obj, *, fallback: int | None = None) -> int:
    """Best-effort electron count for the ``NELE`` header field."""
    if hasattr(atom_or_obj, "Z") and hasattr(atom_or_obj, "charge"):
        try:
            return int(atom_or_obj.Z) - int(atom_or_obj.charge)
        except Exception:
            pass
    if isinstance(atom_or_obj, list) and atom_or_obj and isinstance(atom_or_obj[0], dict):
        nele = atom_or_obj[0].get("nele")
        if nele is not None:
            return int(nele)
    return int(fallback or 0)


def _levels_from_atom(atom) -> list[dict[str, Any]]:
    """Build a list of LEV-row dicts from a solved Atom.

    Handles three cases:

    * single-state DHF — one row at ΔE = 0
    * CI / MCDHF with explicit ``state_energies`` — one row per state,
      with parity and 2J pulled from the dominant CSF
    * shooting / DFS results that don't expose ``csfs`` — fall back to
      labelling everything ``J = 0`` even parity
    """
    energies_ha = atom.state_energies
    if energies_ha is None or len(energies_ha) == 0:
        # Fall back to a single ground-state row.
        return [{
            "ilev": 0, "energy_eV": 0.0, "parity": 0, "vnl": 0,
            "twoj": 0, "config": str(getattr(atom, "_config_strings", [""])[0]),
            "state": "",
        }]

    e0 = float(energies_ha[0])
    de_eV = (np.asarray(energies_ha, dtype=float) - e0) * HARTREE_EV

    mcdhf = getattr(atom, "_csf_result", None)
    levels: list[dict[str, Any]] = []
    if mcdhf is not None and getattr(mcdhf, "csfs", None) and getattr(mcdhf, "mixing_coefficients", None) is not None:
        csfs = mcdhf.csfs
        mixing = np.asarray(mcdhf.mixing_coefficients)
        n_states = mixing.shape[1] if mixing.ndim == 2 else len(csfs)
        for I in range(n_states):
            dom = int(np.argmax(np.abs(mixing[:, I]))) if mixing.ndim == 2 else 0
            csf = csfs[dom] if dom < len(csfs) else csfs[0]
            twoj = int(getattr(csf, "total_J2", 0))
            par_sign = int(getattr(csf, "parity", 1))
            # FAC: 0 = even, 1 = odd; atomx CSF: +1 = even, -1 = odd.
            parity = 0 if par_sign >= 0 else 1
            label = str(getattr(csf, "label", "") or "")
            # Identify the dominant open-shell orbital for VNL.
            vnl = _vnl_from_csf(csf)
            levels.append({
                "ilev": I,
                "ibase": -1,
                "energy_eV": float(de_eV[I]),
                "parity": parity,
                "vnl": vnl,
                "twoj": twoj,
                "config": label,
                "state": label,
            })
    else:
        cfg = ""
        cfg_attr = getattr(atom, "_config_strings", None)
        if cfg_attr:
            cfg = str(cfg_attr[0])
        for I, dE in enumerate(de_eV):
            levels.append({
                "ilev": I,
                "ibase": -1,
                "energy_eV": float(dE),
                "parity": 0,
                "vnl": 0,
                "twoj": 0,
                "config": cfg,
                "state": cfg,
            })
    return levels


def _vnl_from_csf(csf) -> int:
    """Pick the highest-n, highest-l open-shell orbital from a CSF and
    encode it as ``100 * n + l`` for the LEV ``VNL`` column."""
    best = None
    for sh in getattr(csf, "subshells", []):
        # Closed shells (full j-occupancy) shouldn't drive the label.
        # A subshell without kappa violates the Subshell contract and
        # must fail loudly: defaulting it to 0 would encode l=-1 in VNL.
        j2 = 2 * abs(sh.kappa) - 1
        max_occ = j2 + 1
        occ = float(getattr(sh, "occupation", 0.0))
        if occ >= max_occ - 1e-6:
            # Treat as closed; still keep as fallback in case nothing open.
            cand = (False, int(sh.n), l_of_kappa(sh.kappa))
        else:
            cand = (True, int(sh.n), l_of_kappa(sh.kappa))
        if best is None:
            best = cand
        else:
            # Prefer open shells, then highest n, then highest l.
            if (cand[0], cand[1], cand[2]) > (best[0], best[1], best[2]):
                best = cand
    if best is None:
        return 0
    return 100 * best[1] + best[2]


def _state_index_from_label(label: str, fallback: int) -> int:
    """Parse FAC level index from a TransitionResult label.

    MCDHF labels are ``"State_3"`` → 3.  Anything else returns the
    ``fallback`` (used to enumerate orbital-pair transitions in order).
    """
    if not label:
        return fallback
    if label.startswith("State_"):
        try:
            return int(label.split("_", 1)[1])
        except (ValueError, IndexError):
            return fallback
    return fallback


def _transitions_from_atom(
    atom,
    *,
    multipole: str = "E1",
    gauge: str = "length",
    levels: Sequence[dict] | None = None,
) -> list[dict[str, Any]]:
    """Compute transitions and pack them into write_tra row dicts."""
    from atomx.properties.transitions.rates import _double_factorial
    rank = int(multipole[1:])
    trs = atom.transitions(multipoles=[multipole], gauge=gauge)

    # Build {ilev -> twoj} lookup from either the passed-in level list or
    # from the atom's MCDHF dominant-CSF table.
    twoj_by_state: dict[int, int] = {}
    if levels:
        for L in levels:
            twoj_by_state[int(L.get("ilev", L.get("ILEV", 0)))] = int(L.get("twoj", L.get("2J", 0)))
    else:
        for L in _levels_from_atom(atom):
            twoj_by_state[L["ilev"]] = L["twoj"]

    rows = []
    label_to_idx: dict[str, int] = {}

    def _idx(label: str) -> int:
        if label in label_to_idx:
            return label_to_idx[label]
        fallback = len(label_to_idx)
        idx = _state_index_from_label(label, fallback)
        label_to_idx[label] = idx
        return idx

    for t in trs:
        upper_lbl = t.final_label
        lower_lbl = t.initial_label
        upper = _idx(upper_lbl)
        lower = _idx(lower_lbl)
        twoj_u = twoj_by_state.get(upper, 0)
        twoj_l = twoj_by_state.get(lower, 0)
        rows.append({
            "upper": upper,
            "twoj_upper": twoj_u,
            "lower": lower,
            "twoj_lower": twoj_l,
            "deltaE_eV": float(t.energy_eV),
            "gf": float(t.gf_value),
            "A_per_s": float(t.A_rate_per_s),
            "reduced_matrix_element": float(np.sqrt(t.line_strength * (rank+1)/rank)
                / _double_factorial(2*rank-1)),
            "multipole": t.multipole,
            "mode": 0 if t.photon_approximation == 'finite_wavelength' else 1,
        })
    return rows
