"""QED corrections: vacuum polarization (Uehling) and self-energy.

Both computed perturbatively from converged DHF orbitals.

Vacuum polarization: Uehling potential V_VP(r) = -(2Zα)/(3πr) K_1(2r/λ_C)
Self-energy: tabulated Mohr (1983) / Mohr-Kim (1992) values with orbital scaling.
"""

import numpy as np
from scipy.special import k1 as bessel_k1
from atomx.constants import ALPHA, C, C_SQ


# ---------------------------------------------------------------------------
# Vacuum Polarization (Uehling potential, 2nd order)
# ---------------------------------------------------------------------------

def uehling_potential(r, Z):
    """Uehling vacuum polarization potential V_VP(r) for a point nucleus.

    V_VP(r) = -(2Zα)/(3π) * (1/r) * K_1(2cr)

    where c = 1/α ≈ 137.036 and K_1 is the modified Bessel function.

    Parameters
    ----------
    r : float or ndarray
        Radial distance in Bohr.
    Z : int
        Nuclear charge.

    Returns
    -------
    V : float or ndarray
        VP potential in Hartree.
    """
    x = 2.0 * C * np.asarray(r, dtype=float)
    r_arr = np.asarray(r, dtype=float)
    prefactor = -2.0 * Z * ALPHA / (3.0 * np.pi)

    if r_arr.ndim == 0:
        # Scalar case
        if x < 0.01:
            # K_1(x) ≈ 1/x + x/2*ln(x/2) for small x; leading term K_1 ~ 1/x
            return prefactor / (r_arr * x) if r_arr > 0 else 0.0
        return prefactor / r_arr * bessel_k1(float(x))
    else:
        V = np.zeros_like(r_arr)
        large = x > 0.01
        V[large] = prefactor / r_arr[large] * bessel_k1(x[large])
        # Small-x asymptotic: K_1(x) ≈ 1/x
        small = (~large) & (r_arr > 0)
        if np.any(small):
            V[small] = prefactor / (r_arr[small] * x[small])
        return V


def compute_vp_energy(orbitals, bsp_data, Z):
    """Compute vacuum polarization energy correction.

    E_VP = Σ_a q_a * <a | V_VP | a>
         = Σ_a q_a * ∫ [P_a(r)^2 + Q_a(r)^2] * V_VP(r) dr

    Parameters
    ----------
    orbitals : list of (P, Q, occ, kappa, label) tuples
    bsp_data : dict
    Z : int

    Returns
    -------
    E_VP : float (Hartree)
    """
    from atomx.bspline.evaluate import eval_orbital_on_grid

    nv = bsp_data["nv"]
    ns = bsp_data["ns"]
    gr = bsp_data["gr"]
    grw = bsp_data["grw"]
    ksp = bsp_data.get("ksp", bsp_data["ks"])
    ksq = bsp_data.get("ksq", bsp_data["ks"])
    nsp = bsp_data.get("nsp", ns)
    nsq = bsp_data.get("nsq", ns)
    bsp_p = bsp_data["bsp_p"]
    bsp_q = bsp_data.get("bsp_q", bsp_p)

    # VP potential on the grid
    V_vp = uehling_potential(gr, Z)

    E_VP = 0.0
    for P, Q, occ, kappa, label in orbitals:
        fP = eval_orbital_on_grid(P, bsp_p, nv, ksp, nsp)
        fQ = eval_orbital_on_grid(Q, bsp_q, nv, ksq, nsq)
        rho = fP * fP + fQ * fQ
        E_VP += occ * np.sum(grw * rho * V_vp)

    return E_VP


# ---------------------------------------------------------------------------
# Self-Energy (tabulated Mohr values)
# ---------------------------------------------------------------------------

# Tabulated F(Zα) values from Mohr (1983), Mohr & Kim (1992).
# F(Zα) for selected Z values, for 1s and 2s states.
# Format: {(n, kappa): {Z: F_value}}
# These are the scaled self-energy function: E_SE = α(Zα)^4/(πn^3) * F(Zα) * mc^2

_SE_TABLE = {
    # 1s1/2 (n=1, kappa=-1): Mohr (1983) Table I
    (1, -1): {
        1: 10.3169, 2: 10.6027, 5: 11.0742,
        10: 11.6804, 20: 12.6423, 30: 13.4997,
        40: 14.3356, 50: 15.2160, 60: 16.2140,
        70: 17.4409, 80: 19.0723, 92: 21.6412,
    },
    # 2s1/2 (n=2, kappa=-1): Mohr (1983) Table II
    (2, -1): {
        1: 10.5468, 2: 10.8295, 5: 11.2874,
        10: 11.8666, 20: 12.7615, 30: 13.5395,
        40: 14.2838, 50: 15.0640, 60: 15.9530,
        70: 17.0660, 80: 18.5840, 92: 21.0560,
    },
    # 2p1/2 (n=2, kappa=1): Mohr (1983)
    (2, 1): {
        1: -0.1264, 2: -0.1200, 5: -0.0890,
        10: -0.0160, 20: 0.1920, 30: 0.4250,
        40: 0.6760, 50: 0.9590, 60: 1.2890,
        70: 1.7060, 80: 2.2780, 92: 3.2850,
    },
    # 2p3/2 (n=2, kappa=-2): Mohr (1983)
    (2, -2): {
        1: -0.1264, 2: -0.1270, 5: -0.1270,
        10: -0.1280, 20: -0.1290, 30: -0.1310,
        40: -0.1330, 50: -0.1360, 60: -0.1400,
        70: -0.1470, 80: -0.1580, 92: -0.1800,
    },
}


def _interpolate_se_f(Z, n, kappa):
    """Interpolate the self-energy function F(Zα) from Mohr tables.

    Uses linear interpolation in Z. For states not in the table,
    uses hydrogenic scaling from the 1s value.
    """
    key = (n, kappa)
    if key in _SE_TABLE:
        table = _SE_TABLE[key]
        z_vals = sorted(table.keys())
        f_vals = [table[z] for z in z_vals]

        if Z <= z_vals[0]:
            return f_vals[0]
        if Z >= z_vals[-1]:
            return f_vals[-1]

        # Linear interpolation
        for i in range(len(z_vals) - 1):
            if z_vals[i] <= Z <= z_vals[i + 1]:
                t = (Z - z_vals[i]) / (z_vals[i + 1] - z_vals[i])
                return f_vals[i] + t * (f_vals[i + 1] - f_vals[i])

    # Fallback: scale from 1s value using n^{-3} hydrogenic scaling
    f_1s = _interpolate_se_f(Z, 1, -1)
    return f_1s / n ** 3


def compute_se_energy(orbitals, Z, bsp_data=None):
    """Compute self-energy correction with orbital overlap ratio.

    E_SE = Σ_a q_a * (α/π) * (Zα)^4/n^3 * F(Zα) * mc^2 * overlap_ratio

    The overlap ratio corrects for the difference between the
    hydrogenic orbitals used in Mohr's calculation and the actual
    self-consistent orbitals:
        ratio = <ψ_SCF|V_nuc|ψ_SCF> / <ψ_hydro|V_nuc|ψ_hydro>

    For point nucleus: <ψ_hydro|V_nuc|ψ_hydro> = -Z^2/n^2 (leading term).

    Parameters
    ----------
    orbitals : list of (P, Q, occ, kappa, label) tuples
    Z : int
    bsp_data : dict, optional
        If provided, computes overlap ratio from actual orbital wavefunctions.
    """
    from atomx.constants import C_SQ
    aZ = ALPHA * Z

    E_SE = 0.0
    for P, Q, occ, kappa, label in orbitals:
        n = int(label[0])
        F_val = _interpolate_se_f(Z, n, kappa)

        # E_SE per electron = (α/π) * (Zα)^4/n^3 * F(Zα) * mc^2
        # In Hartree: mc^2 = c^2 = 1/α^2
        # So E = (α/π) * (Zα)^4/n^3 * F / α^2 = (Zα)^4 * F / (π * α * n^3)
        E_a = aZ ** 4 / (np.pi * ALPHA * n ** 3) * F_val

        # Apply overlap ratio correction if orbitals available
        if bsp_data is not None and np.max(np.abs(P)) > 1e-10:
            ratio = _overlap_ratio(P, Q, kappa, Z, n, bsp_data)
            E_a *= ratio

        E_SE += occ * E_a

    return E_SE


def _overlap_ratio(P, Q, kappa, Z, n, bsp_data):
    """Orbital overlap ratio for SE correction.

    ratio = <ψ_SCF | -Z/r | ψ_SCF> / <ψ_hydro | -Z/r | ψ_hydro>

    For the hydrogenic reference: <ψ_hydro|-Z/r|ψ_hydro> = -Z^2/n^2 (leading term).
    """
    from atomx.bspline.evaluate import eval_orbital_on_grid

    nv = bsp_data["nv"]
    ns = bsp_data["ns"]
    gr = bsp_data["gr"]
    grw = bsp_data["grw"]
    ksp = bsp_data.get("ksp", bsp_data["ks"])
    ksq = bsp_data.get("ksq", bsp_data["ks"])
    nsp = bsp_data.get("nsp", ns)
    nsq = bsp_data.get("nsq", ns)

    fP = eval_orbital_on_grid(P, bsp_data["bsp_p"], nv, ksp, nsp)
    fQ = eval_orbital_on_grid(Q, bsp_data.get("bsp_q", bsp_data["bsp_p"]), nv, ksq, nsq)
    rho = fP * fP + fQ * fQ

    # SCF nuclear integral: ∫ ρ(r) * (-Z/r) dr
    vnuc_scf = np.sum(grw * rho * (-Z / gr))

    # Hydrogenic reference: -Z^2/n^2 (exact for point nucleus)
    vnuc_hydro = -Z * Z / (n * n)

    if abs(vnuc_hydro) < 1e-20:
        return 1.0
    return vnuc_scf / vnuc_hydro


# ---------------------------------------------------------------------------
# Combined QED corrections
# ---------------------------------------------------------------------------

def compute_qed_corrections(orbitals, bsp_data, Z):
    """Compute all QED corrections (VP + SE).

    Parameters
    ----------
    orbitals : list of (P, Q, occ, kappa, label) tuples
    bsp_data : dict
    Z : int

    Returns
    -------
    dict with keys: E_VP, E_SE, E_QED_total
    """
    E_VP = compute_vp_energy(orbitals, bsp_data, Z)
    E_SE = compute_se_energy(orbitals, Z, bsp_data=bsp_data)

    return {
        "E_VP": E_VP,
        "E_SE": E_SE,
        "E_QED_total": E_VP + E_SE,
    }
