"""B-spline Dirac solves in external local central potentials.

This module is deliberately numerical: callers provide the physical local
potential ``V(r)`` in Hartree, and the routines assemble/solve the B-spline
Dirac eigenproblem in that potential.  FAC-comparison code uses this to test
AtomX radial numerics against FAC's dumped central potential without routing
through AtomX's own SCF potential generation.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Callable

import numpy as np

from atomx.numerics.bspline.basis import prepare_bspline_data
from atomx.numerics.bspline.dirac_assembly import build_dirac_matrices, solve_dirac
from atomx.numerics.bspline.integrals import _assemble_integral
from atomx.numerics.bspline.knots import make_knots


OrbitalKey = tuple[int, int]


def _fac_csi_screening_correction(
    r: np.ndarray,
    screening: np.ndarray,
    *,
    n_total: float,
    ihx: float = -1.0,
    csi: float = 0.55,
) -> np.ndarray:
    """Port FAC's default ``_csi > 0`` screening-charge tail correction.

    FAC applies this in ``radial.c:PotentialHX`` after converting density to a
    screening charge.  For the default ``ihx=-1`` it smoothly drives the local
    screening charge toward ``N-1`` outside the radius where the raw screening
    crosses ``N-(1+csi)``.  This is separate from Latter's final cap and is the
    main FAC-local self-interaction correction for bound orbitals.
    """
    r = np.asarray(r, dtype=float)
    ut = np.asarray(screening, dtype=float).copy()
    if r.ndim != 1 or ut.ndim != 1 or r.shape != ut.shape:
        raise ValueError("r and screening must be matching one-dimensional arrays")
    if r.size < 16 or csi <= 0.0 or ihx >= 0.0 or n_total <= 0.0:
        return ut

    jmax = r.size - 1
    n0 = float(n_total)
    n1 = max(0.0, n0 + float(ihx))
    n2 = max(0.0, n0 + (1.0 + float(csi)) * float(ihx))
    if n2 <= 0.0:
        return ut

    jm = jmax
    for m in range(jmax, 0, -1):
        if ut[m - 1] < ut[m]:
            jm = m
            break
    if jm + 1 <= jmax:
        ut[jm + 1:] = ut[jm]

    i = 0
    for m in range(jm, 0, -1):
        if ut[m] <= n1:
            i = m
            break
    j = 0
    for m in range(i, 0, -1):
        if ut[m] < n2:
            j = m
            break
    if j < 10:
        ut[:jmax + 1] = 0.0
        return ut
    if j >= jm:
        if jm + 1 <= jmax:
            ut[jm + 1:] = ut[jm]
        return ut

    ur = ut[j]
    denom = n0 - ur
    denom1 = n1 - ur
    if denom <= 0.0 or abs(denom1) < 1.0e-14:
        return ut
    a = 0.0
    count = 0
    for m in range(j - 3, j + 4):
        if 0 <= m < jmax and m != j:
            dr = r[m] - r[j]
            if abs(dr) < 1.0e-30:
                continue
            arg = 1.0 - (ut[m] - ur) / denom
            if arg <= 0.0:
                continue
            a += -np.log(arg) / dr
            count += 1
    if count == 0:
        return ut
    a /= count
    b = a * denom / denom1
    r0 = r[j]
    corrected = ut.copy()
    for m in range(j + 1, jmax + 1):
        corrected[m] = ur + denom1 * (1.0 - np.exp(-b * (r[m] - r0)))
    return corrected


def local_central_potential_from_radial_density(
    *,
    Z: int | float,
    rho_rad: np.ndarray,
    gr: np.ndarray,
    grw: np.ndarray,
    n_total: float,
    exchange: str = "fac",
    alpha_x: float = 2.0 / 3.0,
    fac_ihx: float = -1.0,
    fac_hxs: float = 1.0,
    fac_csi: float = 0.55,
    fac_self_interaction_correction: bool = True,
    nuclear_grid: np.ndarray | None = None,
    latter_correction: bool = True,
) -> dict[str, np.ndarray | float]:
    """Build a local central potential from radial density on a B-spline grid.

    ``rho_rad`` is the radial electron density, normalized as
    ``Integral rho_rad dr = N``.  The returned arrays are in Hartree on the
    same quadrature grid.  The exchange term is local-density exchange; for
    ``exchange='fac'``/``'fac-local'`` the coefficient is Gu/FAC's
    ``alpha_xc(Z, N)`` from :mod:`atomx.physics.local_exchange`.  FAC's
    default ``_csi`` self-interaction screening correction is enabled by
    default for FAC exchange and can be disabled for raw component tests.
    """
    from atomx.numerics.bspline.evaluate import compute_yk
    from atomx.physics.local_exchange import fac_gu_exchange, slater_exchange

    Z_f = float(Z)
    rho_rad = np.asarray(rho_rad, dtype=float)
    gr = np.asarray(gr, dtype=float)
    grw = np.asarray(grw, dtype=float)
    if rho_rad.shape != gr.shape or grw.shape != gr.shape:
        raise ValueError("rho_rad, gr, and grw must have matching shapes")
    if not np.isfinite(Z_f) or Z_f <= 0.0:
        raise ValueError("Z must be finite and positive")
    n_total_f = float(n_total)
    if not np.isfinite(n_total_f) or n_total_f < 0.0:
        raise ValueError("n_total must be finite and non-negative")
    if not np.all(np.isfinite(rho_rad)):
        raise ValueError("rho_rad values must be finite")
    if np.any(rho_rad < 0.0):
        raise ValueError("rho_rad values must be non-negative")
    if not np.all(np.isfinite(gr)) or np.any(gr <= 0.0):
        raise ValueError("gr values must be finite and positive")
    if not np.all(np.isfinite(grw)):
        raise ValueError("grw values must be finite")

    nv, n_gauss = gr.shape
    hartree = compute_yk(rho_rad, gr, grw, nv, n_gauss, 0)
    rho_vol = rho_rad / (4.0 * np.pi * gr * gr)
    exchange_l = str(exchange).strip().lower()
    if exchange_l in {"fac", "fac-local"}:
        vx = fac_gu_exchange(
            rho_vol,
            gr,
            Z=Z_f,
            N_total=n_total_f,
            ihx=fac_ihx,
            hxs=fac_hxs,
            slater_alpha=alpha_x,
        )
    elif exchange_l == "slater":
        vx = slater_exchange(rho_vol, gr, alpha=alpha_x)
    else:
        raise ValueError("exchange must be 'fac', 'fac-local', or 'slater'")

    if nuclear_grid is None:
        nuclear = -Z_f / gr
    else:
        nuclear = np.asarray(nuclear_grid, dtype=float)
        if nuclear.shape != gr.shape:
            raise ValueError("nuclear_grid must match gr shape")
        if not np.all(np.isfinite(nuclear)):
            raise ValueError("nuclear_grid values must be finite")
    central = nuclear + hartree + vx
    if fac_self_interaction_correction and exchange_l in {"fac", "fac-local"}:
        flat_r = gr.ravel()
        flat_screening = (gr * (hartree + vx)).ravel()
        order = np.argsort(flat_r)
        corrected = _fac_csi_screening_correction(
            flat_r[order],
            flat_screening[order],
            n_total=n_total_f,
            ihx=fac_ihx,
            csi=fac_csi,
        )
        flat_out = np.empty_like(flat_screening)
        flat_out[order] = corrected
        screening = flat_out.reshape(gr.shape)
        central = nuclear + screening / gr
        vx = central - nuclear - hartree
    if latter_correction:
        rvinf = -(1.0 + Z_f - n_total_f)
        central = np.where(gr * central > rvinf, rvinf / gr, central)
        vx = central - nuclear - hartree
    return {
        "nuclear": nuclear,
        "hartree": hartree,
        "exchange": vx,
        "central": central,
        "n_total": n_total_f,
    }


def _orbital_bound_index(n: int, kappa: int, *, restricted_kinetic_balance: bool) -> int:
    """Return the physical bound-state index for ``(n, kappa)``.

    With RKB the positive-``kappa`` shift used by the older same-order basis is
    not applied; the physical radial node count is simply ``n-l-1``.
    """
    l = kappa if kappa > 0 else -kappa - 1
    idx = int(n) - int(l) - 1
    if kappa > 0 and not restricted_kinetic_balance:
        idx += 1
    return idx


def radial_density_from_orbitals(
    orbitals_dict: Mapping[OrbitalKey, tuple[np.ndarray, np.ndarray]],
    bsp_data: Mapping,
    occupations: Mapping[OrbitalKey, float] | None = None,
) -> tuple[np.ndarray, float]:
    """Evaluate the occupied radial density on a B-spline quadrature grid.

    The returned density is ``sum_q occ_q * (P_q(r)^2 + Q_q(r)^2)`` on
    ``bsp_data['gr']`` and is normalized per ``dr`` rather than per volume.
    This is the density convention expected by
    :func:`local_central_potential_from_radial_density` and by FAC's local
    central-potential construction.
    """
    from atomx.numerics.bspline.evaluate import eval_orbital_on_grid

    gr = np.asarray(bsp_data["gr"], dtype=float)
    nv, _n_gauss = gr.shape
    ns = int(bsp_data["ns"])
    ksp = int(bsp_data.get("ksp", bsp_data.get("ks", 0)))
    ksq = int(bsp_data.get("ksq", bsp_data.get("ks", 0)))
    nsp = int(bsp_data.get("nsp", ns))
    nsq = int(bsp_data.get("nsq", ns))
    bsp_p = bsp_data["bsp_p"]
    bsp_q = bsp_data.get("bsp_q", bsp_p)

    rho_rad = np.zeros_like(gr)
    n_total = 0.0
    occ_map = occupations or {}
    for raw_key, pair in orbitals_dict.items():
        key = (int(raw_key[0]), int(raw_key[1]))
        occ = float(occ_map.get(key, 1.0))
        if not np.isfinite(occ) or occ < 0.0:
            raise ValueError("orbital occupations must be finite and non-negative")
        if occ == 0.0:
            continue
        P, Q = pair
        P_arr = np.asarray(P, dtype=float)
        Q_arr = np.asarray(Q, dtype=float)
        if P_arr.size < nsp or Q_arr.size < nsq:
            raise ValueError("orbital coefficient arrays are shorter than the B-spline basis")
        if not np.all(np.isfinite(P_arr)) or not np.all(np.isfinite(Q_arr)):
            raise ValueError("orbital coefficients must be finite")
        fP = eval_orbital_on_grid(P_arr[:nsp], bsp_p, nv, ksp, nsp)
        fQ = eval_orbital_on_grid(Q_arr[:nsq], bsp_q, nv, ksq, nsq)
        rho_rad += occ * (fP * fP + fQ * fQ)
        n_total += occ
    return rho_rad, float(n_total)


def fac_local_central_potential_from_orbitals(
    *,
    Z: int | float,
    orbitals_dict: Mapping[OrbitalKey, tuple[np.ndarray, np.ndarray]],
    bsp_data: Mapping,
    occupations: Mapping[OrbitalKey, float] | None = None,
    fac_ihx: float = -1.0,
    fac_hxs: float = 1.0,
    fac_csi: float = 0.55,
    fac_self_interaction_correction: bool = True,
    latter_correction: bool = True,
) -> dict[str, np.ndarray | float]:
    """Build FAC's local central potential from occupied B-spline orbitals.

    This is the reusable physics/numerics bridge for FAC-local comparisons:
    orbital coefficients are evaluated on the B-spline quadrature grid, the
    radial density is converted to Gu/FAC local exchange, and the Hartree,
    exchange, nuclear, and total central-potential components are returned in
    Hartree on that same grid.
    """
    rho_rad, n_total = radial_density_from_orbitals(
        orbitals_dict, bsp_data, occupations=occupations
    )
    return local_central_potential_from_radial_density(
        Z=Z,
        rho_rad=rho_rad,
        gr=np.asarray(bsp_data["gr"], dtype=float),
        grw=np.asarray(bsp_data["grw"], dtype=float),
        n_total=n_total,
        exchange="fac",
        fac_ihx=fac_ihx,
        fac_hxs=fac_hxs,
        fac_csi=fac_csi,
        fac_self_interaction_correction=fac_self_interaction_correction,
        latter_correction=latter_correction,
    )


def local_central_interaction_matrices_from_radial_density(
    *,
    Z: int | float,
    rho_rad: np.ndarray,
    bsp_data: Mapping,
    n_total: float,
    exchange: str = "fac",
    alpha_x: float = 2.0 / 3.0,
    fac_ihx: float = -1.0,
    fac_hxs: float = 1.0,
    fac_csi: float = 0.55,
    fac_self_interaction_correction: bool = True,
    nuclear_grid: np.ndarray | None = None,
    latter_correction: bool = True,
) -> dict[str, np.ndarray | float]:
    """Assemble the local central interaction matrix from a radial density.

    The returned ``V_PP``/``V_QQ`` blocks represent ``V_central - V_nuclear``
    in the B-spline Dirac Hamiltonian.  Keeping this in the numerics module
    lets FAC-local SCF and FAC-local diagnostics use the same potential
    construction and matrix projection.
    """
    gr = np.asarray(bsp_data["gr"], dtype=float)
    grw = np.asarray(bsp_data["grw"], dtype=float)
    local = local_central_potential_from_radial_density(
        Z=Z,
        rho_rad=rho_rad,
        gr=gr,
        grw=grw,
        n_total=n_total,
        exchange=exchange,
        alpha_x=alpha_x,
        fac_ihx=fac_ihx,
        fac_hxs=fac_hxs,
        fac_csi=fac_csi,
        fac_self_interaction_correction=fac_self_interaction_correction,
        nuclear_grid=nuclear_grid,
        latter_correction=latter_correction,
    )
    delta_v = np.asarray(local["central"], dtype=float) - np.asarray(
        local["nuclear"], dtype=float
    )
    ns = int(bsp_data["ns"])
    ksp = int(bsp_data.get("ksp", bsp_data.get("ks", 0)))
    ksq = int(bsp_data.get("ksq", bsp_data.get("ks", 0)))
    bsp_p = bsp_data["bsp_p"]
    bsp_q = bsp_data.get("bsp_q", bsp_p)
    V_PP = _assemble_integral(
        int(bsp_data["nv"]), ksp, ksp, ns, bsp_p, bsp_p, grw * delta_v
    )
    V_QQ = _assemble_integral(
        int(bsp_data["nv"]), ksq, ksq, ns, bsp_q, bsp_q, grw * delta_v
    )
    local["V_PP"] = 0.5 * (V_PP + V_PP.T)
    local["V_QQ"] = 0.5 * (V_QQ + V_QQ.T)
    return local


def fac_local_interaction_matrices_from_orbitals(
    *,
    Z: int | float,
    orbitals_dict: Mapping[OrbitalKey, tuple[np.ndarray, np.ndarray]],
    bsp_data: Mapping,
    occupations: Mapping[OrbitalKey, float] | None = None,
    fac_ihx: float = -1.0,
    fac_hxs: float = 1.0,
    fac_csi: float = 0.55,
    fac_self_interaction_correction: bool = True,
    latter_correction: bool = True,
) -> dict[str, np.ndarray | float]:
    """Assemble FAC-local central-potential matrices from occupied orbitals."""
    rho_rad, n_total = radial_density_from_orbitals(
        orbitals_dict, bsp_data, occupations=occupations
    )
    return local_central_interaction_matrices_from_radial_density(
        Z=Z,
        rho_rad=rho_rad,
        bsp_data=bsp_data,
        n_total=n_total,
        exchange="fac",
        fac_ihx=fac_ihx,
        fac_hxs=fac_hxs,
        fac_csi=fac_csi,
        fac_self_interaction_correction=fac_self_interaction_correction,
        latter_correction=latter_correction,
    )


def solve_orbitals_in_local_potential(
    *,
    Z: int | float,
    potential_r: Iterable[float],
    potential_v: Iterable[float],
    orbital_keys: Iterable[OrbitalKey],
    ns: int,
    ks: int,
    r_min: float | None = None,
    r_max: float | None = None,
    bsp_data: Mapping | None = None,
    nuclear_potential_func: Callable[[float], float] | None = None,
) -> tuple[dict[OrbitalKey, tuple[np.ndarray, np.ndarray]], dict, dict[OrbitalKey, float]]:
    """Solve bound Dirac orbitals in a supplied local central potential.

    Parameters
    ----------
    Z
        Nuclear charge used by the one-electron Dirac assembly.
    potential_r, potential_v
        Radial table for the full local central potential ``V(r)`` in Hartree.
        The table is linearly interpolated onto the B-spline quadrature grid.
    orbital_keys
        Iterable of ``(n, kappa)`` orbitals to extract from each solved
        ``kappa`` symmetry.
    ns, ks, r_min, r_max, bsp_data
        B-spline basis controls.  Supplying ``bsp_data`` reuses an existing
        prepared basis; otherwise a kinetic-balance basis is constructed.
    nuclear_potential_func
        Optional nuclear potential already included by ``build_dirac_matrices``.
        When omitted, the point potential ``-Z/r`` is assumed.

    Returns
    -------
    orbitals, bsp_data, energies
        ``orbitals`` maps ``(n, kappa)`` to ``(P, Q)`` coefficient arrays.
        ``energies`` maps the same key to orbital energy in Hartree.
    """
    Z_f = float(Z)
    if not np.isfinite(Z_f) or Z_f <= 0.0:
        raise ValueError("Z must be finite and positive")
    keys = tuple((int(n), int(kappa)) for n, kappa in orbital_keys)
    if not keys:
        return {}, dict(bsp_data or {}), {}
    for n, kappa in keys:
        if n <= 0:
            raise ValueError("orbital principal quantum numbers must be positive")
        if kappa == 0:
            raise ValueError("orbital kappa values must be non-zero")
        l = kappa if kappa > 0 else -kappa - 1
        if n <= l:
            raise ValueError("orbital keys must satisfy n > l")

    pot_r = np.asarray(tuple(potential_r), dtype=float)
    pot_v = np.asarray(tuple(potential_v), dtype=float)
    if pot_r.shape != pot_v.shape:
        raise ValueError("potential_r and potential_v must have matching shape")
    if pot_r.ndim != 1:
        raise ValueError("potential_r and potential_v must be one-dimensional")
    if pot_r.size == 0:
        raise ValueError("potential table must not be empty")
    if not np.all(np.isfinite(pot_r)):
        raise ValueError("potential_r values must be finite")
    if not np.all(np.isfinite(pot_v)):
        raise ValueError("potential_v values must be finite")
    if np.any(pot_r <= 0.0):
        raise ValueError("potential_r values must be positive")
    order = np.argsort(pot_r)
    pot_r = pot_r[order]
    pot_v = pot_v[order]
    if np.any(np.diff(pot_r) <= 0.0):
        raise ValueError("potential_r values must be strictly increasing")

    if bsp_data is None:
        if r_min is None:
            r_min = 0.05 / Z_f
        if r_max is None:
            r_max = max(40.0, 60.0 / Z_f)
        bsp_data = prepare_bspline_data(
            make_knots(int(ns), int(ks), float(r_min), float(r_max)),
            int(ns),
            int(ks),
            kinetic_balance=True,
        )
    else:
        bsp_data = dict(bsp_data)

    gr = np.asarray(bsp_data["gr"], dtype=float)
    grw = np.asarray(bsp_data["grw"], dtype=float)
    nv = int(bsp_data["nv"])
    ns_full = int(bsp_data["ns"])
    ksp = int(bsp_data.get("ksp", ks))
    ksq = int(bsp_data.get("ksq", ks))
    restricted_kinetic_balance = ksp != ksq

    r_min_grid = float(np.min(gr))
    r_max_grid = float(np.max(gr))
    tol = 8.0 * np.finfo(float).eps * max(1.0, abs(r_min_grid), abs(r_max_grid))
    if pot_r[0] > r_min_grid + tol or pot_r[-1] < r_max_grid - tol:
        raise ValueError(
            "potential table must cover the B-spline quadrature grid "
            f"[{r_min_grid:.8e}, {r_max_grid:.8e}], got "
            f"[{pot_r[0]:.8e}, {pot_r[-1]:.8e}]"
        )
    V_full = np.interp(gr, pot_r, pot_v)
    if nuclear_potential_func is None:
        V_nuc = -Z_f / gr
    else:
        V_nuc = np.empty_like(gr)
        for iv in range(gr.shape[0]):
            for ig in range(gr.shape[1]):
                value = float(nuclear_potential_func(float(gr[iv, ig])))
                if not np.isfinite(value):
                    raise ValueError("nuclear_potential_func must return finite values")
                V_nuc[iv, ig] = value
    delta_v = V_full - V_nuc

    Vp = _assemble_integral(
        nv, ksp, ksp, ns_full, bsp_data["bsp_p"], bsp_data["bsp_p"], grw * delta_v
    )
    Vq = _assemble_integral(
        nv, ksq, ksq, ns_full, bsp_data["bsp_q"], bsp_data["bsp_q"], grw * delta_v
    )
    Vp = 0.5 * (Vp + Vp.T)
    Vq = 0.5 * (Vq + Vq.T)

    orbitals: dict[OrbitalKey, tuple[np.ndarray, np.ndarray]] = {}
    energies: dict[OrbitalKey, float] = {}
    for kappa in sorted({kappa for _n, kappa in keys}):
        H, S = build_dirac_matrices(ns_full, kappa, Z_f, bsp_data, nuclear_potential_func)
        H[:ns_full, :ns_full] += Vp
        H[ns_full:, ns_full:] += Vq
        result = solve_dirac(
            H,
            S,
            ns_full,
            kappa,
            nsp=bsp_data.get("nsp"),
            nsq=bsp_data.get("nsq"),
        )
        for n, key_kappa in keys:
            if key_kappa != kappa:
                continue
            idx = _orbital_bound_index(
                n, kappa, restricted_kinetic_balance=restricted_kinetic_balance
            )
            if 0 <= idx < result.n_bound:
                P, Q, E = result.get_orbital(idx)
                key = (n, kappa)
                orbitals[key] = (P, Q)
                energies[key] = float(E)
    missing = sorted(set(keys) - set(orbitals), key=lambda item: (item[0], item[1]))
    if missing:
        raise ValueError(f"requested local-potential orbitals were not found: {missing}")
    return orbitals, dict(bsp_data), energies


__all__ = [
    "fac_local_central_potential_from_orbitals",
    "fac_local_interaction_matrices_from_orbitals",
    "local_central_interaction_matrices_from_radial_density",
    "local_central_potential_from_radial_density",
    "radial_density_from_orbitals",
    "solve_orbitals_in_local_potential",
]
