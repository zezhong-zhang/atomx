"""Mohr tabulated self-energy backend.

F(Zα) values from:
    P. J. Mohr, "Self-energy correction to one-electron energy levels in a
    strong Coulomb field", Phys. Rev. A 46, 4421 (1992)
    P. J. Mohr, "Quantum electrodynamics of high-Z few-electron atoms",
    Phys. Rev. A 26, 2338 (1982).

F is tabulated for 1s1/2, 2s1/2, 2p1/2, 2p3/2 on a grid of Z values
{1, 2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 92} and linearly interpolated.
An orbital-overlap ratio corrects for the deviation of the actual
SCF orbital from the hydrogenic reference used in Mohr's calculation.

Orbitals outside the explicit table (n ≥ 3, higher-l) fall back to the
hydrogenic n^-3 scaling of the 1s1/2 value — this is the limit of what the
Mohr table can deliver.  Use the ``welton`` or ``shabaev`` backends for
systematically better coverage.
"""

from __future__ import annotations

import numpy as np

from atomx.constants import ALPHA
from atomx.corrections.qedmod.base import SEBackend, SEResult, register_backend


# ---------------------------------------------------------------------------
# Tabulated F(Zα) values
# ---------------------------------------------------------------------------

# Format: {(n, kappa): {Z: F_value}}
# Defining relation: E_SE = α(Zα)^4/(πn^3) * F(Zα) * mc^2

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
    """Interpolate F(Zα) from the Mohr tables.

    Linear in Z; hydrogenic n^-3 fallback for orbitals outside the table.
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

        for i in range(len(z_vals) - 1):
            if z_vals[i] <= Z <= z_vals[i + 1]:
                t = (Z - z_vals[i]) / (z_vals[i + 1] - z_vals[i])
                return f_vals[i] + t * (f_vals[i + 1] - f_vals[i])

    # Fallback: hydrogenic n^-3 scaling of the 1s value
    f_1s = _interpolate_se_f(Z, 1, -1)
    return f_1s / n ** 3


def _overlap_ratio(P, Q, kappa, Z, n, bsp_data):
    """⟨ψ_SCF|-Z/r|ψ_SCF⟩ / ⟨ψ_hydro|-Z/r|ψ_hydro⟩ with the hydrogenic value
    taken as -Z^2/n^2 (point-nucleus leading term)."""
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
    fQ = eval_orbital_on_grid(
        Q, bsp_data.get("bsp_q", bsp_data["bsp_p"]), nv, ksq, nsq
    )
    rho = fP * fP + fQ * fQ

    vnuc_scf = np.sum(grw * rho * (-Z / gr))
    vnuc_hydro = -Z * Z / (n * n)

    if abs(vnuc_hydro) < 1e-20:
        return 1.0
    return vnuc_scf / vnuc_hydro


# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------

@register_backend
class MohrBackend(SEBackend):
    """Tabulated Mohr F(Zα) with orbital-overlap screening."""

    name = "mohr"
    description = "Tabulated Mohr (1982/1992) F(Zα), hydrogenic n^-3 fallback."

    def compute(self, orbitals, bsp_data, Z) -> SEResult:
        aZ = ALPHA * Z
        use_overlap = bsp_data is not None

        per_orbital: dict = {}
        E_SE = 0.0
        for P, Q, occ, kappa, label in orbitals:
            n = int(label[0])
            F_val = _interpolate_se_f(Z, n, kappa)

            # E_SE (per electron) = (α/π)(Zα)^4/n^3 × F × mc²
            # In Hartree with mc² = 1/α²  ⇒  (Zα)^4 × F / (π α n^3)
            E_a = aZ ** 4 / (np.pi * ALPHA * n ** 3) * F_val

            if use_overlap and np.max(np.abs(P)) > 1e-10:
                ratio = _overlap_ratio(P, Q, kappa, Z, n, bsp_data)
                E_a *= ratio

            per_orbital[label] = E_a
            E_SE += occ * E_a

        return SEResult(E_SE=E_SE, per_orbital=per_orbital, method=self.name)
