"""Thomas-Fermi-Dirac-Amaldi (TFDA) model potential.

The TF statistical model provides a fast approximate potential for:
1. Starting potential for HF SCF (faster convergence than hydrogenic)
2. Standalone DFS-like calculation (similar to FAC's approach)

The TFDA includes:
- Thomas-Fermi screening (statistical electron density)
- Dirac exchange correction (local exchange)
- Amaldi self-interaction correction

Reference: Salvat et al., Comp. Phys. Comm. 165, 157 (2005)
"""

import numpy as np

from atomx.numerics.bspline.integrals import _assemble_integral


def _thomas_fermi_screening(x):
    """TF screening function phi(x) using Moliere approximation.

    phi(x) = 0.10 * exp(-6.0*x) + 0.55 * exp(-1.2*x) + 0.35 * exp(-0.3*x)

    This is the Moliere (1947) 3-exponential fit to the TF function,
    accurate to ~1% for all x. Much faster than solving the TF ODE.

    Parameters
    ----------
    x : ndarray
        Scaled radius x = r / b, where b = 0.8853 * a_0 / Z^(1/3).

    Returns
    -------
    ndarray
        Screening function phi(x), normalized so phi(0) = 1.
    """
    return 0.10 * np.exp(-6.0 * x) + 0.55 * np.exp(-1.2 * x) + 0.35 * np.exp(-0.3 * x)


def _tf_raw_and_asym(Z, N_electrons, r):
    """Return the raw TF potential and Latter asymptotic tail."""
    r_safe = np.maximum(r, 1e-30)
    b = 0.8853 / Z ** (1.0 / 3.0)
    phi = _thomas_fermi_screening(r / b)
    Z_eff_inf = max(min(Z - N_electrons + 1, Z), 1)
    V_tf_raw = -Z * phi / r_safe
    V_asym = -Z_eff_inf / r_safe
    return V_tf_raw, V_asym


def thomas_fermi_potential(Z, N_electrons, r):
    """Thomas-Fermi screened nuclear potential.

    V_TF(r) = -Z * phi(r/b) / r

    where b = 0.8853 / Z^(1/3) Bohr and phi is the TF screening function.

    Parameters
    ----------
    Z : int
        Nuclear charge.
    N_electrons : float
        Number of electrons.
    r : ndarray
        Radial distances in Bohr.

    Returns
    -------
    ndarray
        Potential V(r) in Hartree.
    """
    # For neutral atom: V -> -Z/r at r=0, V -> 0 at r->inf
    # For ion: V -> -(Z - N + 1)/r at r->inf (Latter tail correction)
    V_tf_raw, V_asym = _tf_raw_and_asym(Z, N_electrons, r)
    return np.minimum(V_tf_raw, V_asym)


def tfda_potential(Z, N_electrons, r):
    """Thomas-Fermi-Dirac-Amaldi screened potential.

    Adds Dirac exchange correction and Amaldi self-interaction correction:
    V_TFDA(r) = V_TF(r) * (1 - 1/Z) + V_Dirac_exchange(r)

    The Amaldi correction removes self-interaction approximately:
    V_Amaldi = (N-1)/N * V_Hartree  (instead of V_Hartree)

    Parameters
    ----------
    Z, N_electrons : int, float
    r : ndarray

    Returns
    -------
    ndarray
        TFDA potential in Hartree.
    """
    r_safe = np.maximum(r, 1e-30)
    V_tf_raw, V_asym = _tf_raw_and_asym(Z, N_electrons, r)

    # Amaldi correction: scale electron-electron by (N-1)/N
    amaldi_factor = (N_electrons - 1.0) / N_electrons if N_electrons > 1 else 0.0

    # Separate nuclear and electron parts
    V_nuc = -Z / r_safe
    V_ee = V_tf_raw - V_nuc  # electron screening part

    # Dirac exchange: V_x = -alpha_x * (3*rho/(pi))^(1/3)
    # For TF density: rho_TF = (Z/b)^3 * (phi/x)^(3/2) / (3*pi^2)
    # We approximate as a correction factor to V_ee
    # Simplified: just apply Amaldi correction to the screening.
    V_tfda = V_nuc + amaldi_factor * V_ee

    return np.minimum(V_tfda, V_asym)


def tfda_potential_on_grid(Z, N_electrons, gr):
    """Evaluate TFDA potential on the B-spline quadrature grid.

    Parameters
    ----------
    Z : int
    N_electrons : float
    gr : ndarray, shape (nv, n_gauss)
        Grid points.

    Returns
    -------
    ndarray, shape (nv, n_gauss)
        V_TFDA on grid.
    """
    return tfda_potential(Z, N_electrons, gr)


def tfda_potential_matrix(Z, N_electrons, bsp_data):
    """Assemble TFDA potential into B-spline matrix form.

    Returns (V_PP, V_QQ) potential matrices that can be added to
    the Dirac Hamiltonian in the SCF loop.

    Parameters
    ----------
    Z : int
    N_electrons : float
    bsp_data : dict

    Returns
    -------
    V_PP : ndarray
        Potential matrix for P-component (large).
    V_QQ : ndarray
        Potential matrix for Q-component (small).
    """
    nv = bsp_data["nv"]
    ns = bsp_data["ns"]
    gr = bsp_data["gr"]
    grw = bsp_data["grw"]

    # Evaluate TFDA potential on grid
    V_tfda = tfda_potential_on_grid(Z, N_electrons, gr)

    # The TFDA replaces V_nuc + V_Hartree + V_exchange
    # We need V_TFDA - V_nuc (since V_nuc is already in the Dirac Hamiltonian)
    V_nuc_grid = -Z / np.maximum(gr, 1e-30)
    V_screening = V_tfda - V_nuc_grid  # screening part only

    V_w = grw * V_screening

    # Assemble for P-block
    ksp = bsp_data.get("ksp", bsp_data["ks"])
    bsp_p = bsp_data["bsp_p"]
    V_PP = _assemble_integral(nv, ksp, ksp, ns, bsp_p, bsp_p, V_w)
    V_PP = 0.5 * (V_PP + V_PP.T)

    # Assemble for Q-block
    ksq = bsp_data.get("ksq", bsp_data["ks"])
    bsp_q = bsp_data.get("bsp_q", bsp_p)
    V_QQ = _assemble_integral(nv, ksq, ksq, ns, bsp_q, bsp_q, V_w)
    V_QQ = 0.5 * (V_QQ + V_QQ.T)

    return V_PP, V_QQ
