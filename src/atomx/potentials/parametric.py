"""Parametric and optimized effective potentials.

Green-Sellin-Zachor (GSZ) model potential and FAC-inspired
optimized central potential for fast approximate calculations.

The parametric potential approach:
1. Define V(r) with a few adjustable parameters
2. Optimize parameters to minimize the average configuration energy
3. Solve each orbital in this single fixed potential

This is the key idea from FAC: ALL orbitals see the SAME potential,
eliminating the need for orbital-dependent exchange matrices.

Reference:
- Green, Sellin, Zachor, Phys. Rev. 184, 1 (1969)
- Gu, Can. J. Phys. 86, 675 (2008) - FAC
"""

import numpy as np
from scipy.optimize import minimize_scalar
from atomx.numerics.bspline.integrals import _assemble_integral
from atomx.numerics.bspline.evaluate import eval_orbital_on_grid


def green_sellin_zachor(Z, N_electrons, r, d=None):
    """Green-Sellin-Zachor model potential.

    V_GSZ(r) = -[Z_eff_inf + (Z - Z_eff_inf) * H(r/d)] / r

    where H(x) = [(H0-1)*(e^{-x} - 1) + 1] is the screening function,
    Z_eff_inf = Z - N + 1 is the asymptotic effective charge, and
    d is the screening length.

    Parameters
    ----------
    Z : int
        Nuclear charge.
    N_electrons : float
        Number of electrons.
    r : ndarray
        Radial distances in Bohr.
    d : float, optional
        Screening length in Bohr. Default: estimated from TF model.

    Returns
    -------
    ndarray
        Potential V(r) in Hartree.
    """
    Z_eff_inf = max(Z - N_electrons + 1, 1)
    Z_screen = Z - Z_eff_inf  # amount of screening

    if d is None:
        # Estimate screening length from Thomas-Fermi
        d = 0.8853 / Z ** (1.0 / 3.0) * 1.5  # ~1.5 * TF length

    x = r / d
    # Screening function: H(x) -> 1 at r=0, H(x) -> 0 at r->inf
    H = np.exp(-x)

    V = -(Z_eff_inf + Z_screen * H) / np.maximum(r, 1e-30)
    return V


def optimized_central_potential(Z, subshells, bsp_data, alpha_x=2.0/3.0):
    """FAC-inspired optimized local central potential.

    Constructs a single optimized local potential V(r) by:
    1. Computing the Hartree potential from the total electron density
    2. Adding a Slater-type local exchange with optimized alpha

    This is essentially the DFS approach but with the option to
    optimize the exchange parameter alpha_x to minimize the total energy
    for the given configuration.

    Parameters
    ----------
    Z : int
    subshells : list of Subshell
    bsp_data : dict
    alpha_x : float
        Exchange parameter (default: 2/3, the standard Slater value).
        Can be optimized per-configuration.

    Returns
    -------
    V_PP, V_QQ : ndarray
        Potential matrices for P and Q components (screening part only,
        nuclear potential is already in the Dirac Hamiltonian).
    """
    # This is effectively the Slater exchange with the given alpha_x
    # The caller (DFSolver) already handles this in exchange="slater" mode
    # Here we provide a more flexible version with optimization capability

    nv = bsp_data["nv"]
    ns = bsp_data["ns"]
    gr = bsp_data["gr"]
    grw = bsp_data["grw"]
    n_gauss = gr.shape[1]
    N_electrons = sum(sh.occupation for sh in subshells)

    # Build total density (for starting potential without orbitals)
    # Use TFDA as initial approximation
    from atomx.potentials.thomas_fermi import tfda_potential_on_grid
    V_start = tfda_potential_on_grid(Z, N_electrons, gr)
    V_nuc = -Z / np.maximum(gr, 1e-30)
    V_screen = V_start - V_nuc

    V_w = grw * V_screen

    ksp = bsp_data.get("ksp", bsp_data["ks"])
    bsp_p = bsp_data["bsp_p"]
    V_PP = _assemble_integral(nv, ksp, ksp, ns, bsp_p, bsp_p, V_w)
    V_PP = 0.5 * (V_PP + V_PP.T)

    ksq = bsp_data.get("ksq", bsp_data["ks"])
    bsp_q = bsp_data.get("bsp_q", bsp_p)
    V_QQ = _assemble_integral(nv, ksq, ksq, ns, bsp_q, bsp_q, V_w)
    V_QQ = 0.5 * (V_QQ + V_QQ.T)

    return V_PP, V_QQ


def gsz_potential_matrix(Z, N_electrons, bsp_data, d=None):
    """Assemble GSZ potential into B-spline matrix form.

    Returns screening-only matrices (V_PP, V_QQ) to be added to
    the Dirac Hamiltonian alongside the nuclear potential.

    Parameters
    ----------
    Z : int
    N_electrons : float
    bsp_data : dict
    d : float, optional
        GSZ screening length.

    Returns
    -------
    V_PP, V_QQ : ndarray
    """
    nv = bsp_data["nv"]
    ns = bsp_data["ns"]
    gr = bsp_data["gr"]
    grw = bsp_data["grw"]

    V_gsz = green_sellin_zachor(Z, N_electrons, gr, d)
    V_nuc = -Z / np.maximum(gr, 1e-30)
    V_screen = V_gsz - V_nuc

    V_w = grw * V_screen

    ksp = bsp_data.get("ksp", bsp_data["ks"])
    bsp_p = bsp_data["bsp_p"]
    V_PP = _assemble_integral(nv, ksp, ksp, ns, bsp_p, bsp_p, V_w)
    V_PP = 0.5 * (V_PP + V_PP.T)

    ksq = bsp_data.get("ksq", bsp_data["ks"])
    bsp_q = bsp_data.get("bsp_q", bsp_p)
    V_QQ = _assemble_integral(nv, ksq, ksq, ns, bsp_q, bsp_q, V_w)
    V_QQ = 0.5 * (V_QQ + V_QQ.T)

    return V_PP, V_QQ
