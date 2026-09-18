"""Cowan-style least-squares fit of Slater-integral scale factors.

The fit model (linear in the parameters — each level energy is a
*linear combination* of ab-initio Slater integrals):

    E_i^{calc}  =  E_0  +  Σ_p  s_p · A_{i, p} · P_p^{abinitio}      (1)

with unknowns ``(s_1, …, s_M, E_0)``.  :class:`SlaterFitResult`
reports the fitted scale factors, their 1 σ uncertainties, χ² / dof,
and the per-level residuals.

Since the model is linear we solve the weighted least-squares problem
directly via :func:`scipy.optimize.lsq_linear` — no iterative
minimisation needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import numpy as np
from scipy.optimize import lsq_linear


@dataclass
class SlaterFitResult:
    """Result of a Cowan-style Slater-parameter fit.

    Attributes
    ----------
    scale_factors : dict
        ``{parameter_label: s_p}`` — fitted scale factors.  Typical
        Cowan values are 0.85–0.95 for F^k, 0.70–0.85 for G^k, 0.95–
        1.00 for ζ_nl.
    scale_uncertainties : dict
        1 σ uncertainties on the scale factors from the fit covariance.
    E_offset_cm1 : float
        Overall energy offset (E_0), cm⁻¹.
    chisq : float
        χ² value at the minimum.
    dof : int
        Degrees of freedom = n_levels − n_parameters − 1.
    rms_residual_cm1 : float
        RMS residual of the fit in cm⁻¹.
    residuals_cm1 : np.ndarray
        Per-level residual ``E_calc − E_exp``, cm⁻¹.
    level_predictions_cm1 : np.ndarray
        Fitted level energies.
    """

    scale_factors: Dict[str, float]
    scale_uncertainties: Dict[str, float]
    E_offset_cm1: float
    chisq: float
    dof: int
    rms_residual_cm1: float
    residuals_cm1: np.ndarray
    level_predictions_cm1: np.ndarray


def fit_slater_parameters(
    experimental_energies_cm1: Sequence[float],
    angular_matrix: np.ndarray,
    abinitio_parameters: Sequence[float],
    parameter_labels: Sequence[str],
    *,
    uncertainties_cm1: Optional[Sequence[float]] = None,
    fit_offset: bool = True,
) -> SlaterFitResult:
    """Fit Cowan-style scale factors to a set of experimental levels.

    Parameters
    ----------
    experimental_energies_cm1 : sequence of float, length n_levels
        Measured level energies (cm⁻¹).
    angular_matrix : np.ndarray, shape (n_levels, n_parameters)
        Angular coefficient matrix ``A_{i, p}`` — typically from a
        diagonalisation of the model Hamiltonian in the LS-coupled
        basis (Cowan's RCE or equivalent).  Units of cm⁻¹.
    abinitio_parameters : sequence of float, length n_parameters
        Ab-initio values of the Slater integrals ``P_p^{abinitio}``
        (Hartree-Fock or Dirac-Fock calculation).  Units of cm⁻¹.
    parameter_labels : sequence of str
        Human-readable labels: ``("F^2(3d,3d)", "G^1(3p,3d)", ...)``.
    uncertainties_cm1 : sequence of float, optional
        Per-level experimental uncertainties σ_i in cm⁻¹.  Default:
        1 cm⁻¹ for every level (uniform weighting).
    fit_offset : bool, default True
        Whether to fit an additional overall energy offset ``E_0``.

    Returns
    -------
    :class:`SlaterFitResult`
    """
    E_exp = np.asarray(experimental_energies_cm1, dtype=float)
    A = np.asarray(angular_matrix, dtype=float)
    P_ab = np.asarray(abinitio_parameters, dtype=float)
    n_levels, n_params = A.shape

    if len(E_exp) != n_levels:
        raise ValueError(
            f"Length mismatch: {len(E_exp)} experimental energies vs "
            f"{n_levels} rows in angular_matrix."
        )
    if len(P_ab) != n_params:
        raise ValueError(
            f"Length mismatch: {len(P_ab)} ab-initio parameters vs "
            f"{n_params} columns in angular_matrix."
        )
    if len(parameter_labels) != n_params:
        raise ValueError("parameter_labels length must match n_params.")

    sigmas = (
        np.ones(n_levels)
        if uncertainties_cm1 is None
        else np.asarray(uncertainties_cm1, dtype=float)
    )
    if np.any(sigmas <= 0):
        raise ValueError("Uncertainties must be positive.")

    # Design matrix: each column (for parameter p) is A[:, p] * P_ab[p]
    # Final column (optional): ones vector for E_offset.
    X_cols = [A[:, p] * P_ab[p] for p in range(n_params)]
    if fit_offset:
        X_cols.append(np.ones(n_levels))
    X = np.column_stack(X_cols)     # shape (n_levels, n_params[+1])

    # Weighted problem:  (X / σ) x = E_exp / σ
    W = 1.0 / sigmas[:, None]
    Xw = X * W
    yw = E_exp / sigmas

    # Unconstrained linear least-squares
    sol = lsq_linear(Xw, yw, bounds=(-np.inf, np.inf))
    x_hat = sol.x      # shape (n_params[+1],)

    # Predictions and residuals
    y_calc = X @ x_hat
    residuals = y_calc - E_exp
    chisq = float(np.sum((residuals / sigmas) ** 2))
    dof = n_levels - len(x_hat)
    rms = float(np.sqrt(np.mean(residuals ** 2)))

    # Covariance: (X^T W^2 X)^(-1) × χ²_reduced
    try:
        XtWX = Xw.T @ Xw
        cov = np.linalg.inv(XtWX)
        if dof > 0:
            cov *= chisq / dof
        uncertainties = np.sqrt(np.diag(cov))
    except np.linalg.LinAlgError:
        uncertainties = np.full(len(x_hat), np.nan)

    # Pack results
    scale_factors = {
        label: float(x_hat[i]) for i, label in enumerate(parameter_labels)
    }
    scale_uncertainties = {
        label: float(uncertainties[i])
        for i, label in enumerate(parameter_labels)
    }

    E_offset = float(x_hat[-1]) if fit_offset else 0.0

    return SlaterFitResult(
        scale_factors=scale_factors,
        scale_uncertainties=scale_uncertainties,
        E_offset_cm1=E_offset,
        chisq=chisq,
        dof=dof,
        rms_residual_cm1=rms,
        residuals_cm1=residuals,
        level_predictions_cm1=y_calc,
    )
