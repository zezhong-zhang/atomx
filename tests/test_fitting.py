"""Tests for Cowan-style Slater-parameter fitting."""

from __future__ import annotations

import numpy as np
import pytest

from atomx.fitting import SlaterFitResult, fit_slater_parameters


class TestSlaterFit:
    def test_exact_one_parameter_fit(self):
        """Single parameter, single level: trivially solvable."""
        # E_exp = 10000 cm⁻¹, P_ab = 5000, A = 1.0 → s = 2.0
        result = fit_slater_parameters(
            experimental_energies_cm1=[10000.0],
            angular_matrix=np.array([[1.0]]),
            abinitio_parameters=[5000.0],
            parameter_labels=["F2"],
            fit_offset=False,
        )
        assert result.scale_factors["F2"] == pytest.approx(2.0, rel=1e-10)
        assert result.rms_residual_cm1 == pytest.approx(0.0, abs=1e-10)

    def test_recovers_known_scale_factors(self):
        """Known scale factors: fit should recover them exactly when
        the data are consistent."""
        # Two parameters, five levels, angular coefficients chosen to make
        # the system well-conditioned
        A = np.array([
            [1.0, 0.0],
            [0.0, 1.0],
            [0.5, 0.5],
            [1.0, 0.5],
            [0.3, 0.8],
        ])
        P_ab = np.array([1000.0, 800.0])
        s_true = np.array([0.85, 0.92])
        # Without offset:  E_i = Σ_p s_p A_ip P_p
        E_exp = A @ (s_true * P_ab)

        result = fit_slater_parameters(
            experimental_energies_cm1=E_exp,
            angular_matrix=A,
            abinitio_parameters=P_ab,
            parameter_labels=["F2(3d,3d)", "G1(3p,3d)"],
            fit_offset=False,
        )
        assert result.scale_factors["F2(3d,3d)"] == pytest.approx(0.85, abs=1e-10)
        assert result.scale_factors["G1(3p,3d)"] == pytest.approx(0.92, abs=1e-10)
        assert result.rms_residual_cm1 == pytest.approx(0.0, abs=1e-6)

    def test_noisy_fit_has_positive_chisq(self):
        """Noisy data → χ² > 0, recovered scales close to truth."""
        rng = np.random.default_rng(42)
        n_levels = 20
        A = rng.standard_normal(size=(n_levels, 3))
        P_ab = np.array([500.0, 300.0, 200.0])
        s_true = np.array([0.85, 0.9, 0.95])
        E_true = A @ (s_true * P_ab) + 1000.0   # with offset
        noise = rng.normal(0.0, 5.0, size=n_levels)
        E_exp = E_true + noise

        result = fit_slater_parameters(
            experimental_energies_cm1=E_exp,
            angular_matrix=A,
            abinitio_parameters=P_ab,
            parameter_labels=["F2", "F4", "G1"],
            uncertainties_cm1=[5.0] * n_levels,
            fit_offset=True,
        )
        # χ² ≈ DOF for well-fitting model
        assert 0.5 < result.chisq / result.dof < 2.5
        # Scale factors close to truth (a few %, depends on conditioning)
        assert result.scale_factors["F2"] == pytest.approx(0.85, abs=0.05)
        assert result.scale_factors["G1"] == pytest.approx(0.95, abs=0.05)
        # RMS a few cm⁻¹
        assert result.rms_residual_cm1 < 10.0

    def test_rejects_shape_mismatch(self):
        with pytest.raises(ValueError, match="Length mismatch"):
            fit_slater_parameters(
                experimental_energies_cm1=[1.0, 2.0],
                angular_matrix=np.array([[1.0]]),  # 1 level, but 2 supplied
                abinitio_parameters=[1.0],
                parameter_labels=["F"],
            )

    def test_rejects_negative_uncertainty(self):
        with pytest.raises(ValueError, match="positive"):
            fit_slater_parameters(
                experimental_energies_cm1=[1000.0],
                angular_matrix=np.array([[1.0]]),
                abinitio_parameters=[500.0],
                parameter_labels=["F"],
                uncertainties_cm1=[-1.0],
                fit_offset=False,
            )

    def test_parameter_labels_length_must_match(self):
        with pytest.raises(ValueError, match="parameter_labels"):
            fit_slater_parameters(
                experimental_energies_cm1=[1000.0, 2000.0],
                angular_matrix=np.array([[1.0], [1.0]]),
                abinitio_parameters=[500.0],
                parameter_labels=["F", "G"],     # 2 labels but 1 param
                fit_offset=False,
            )

    def test_returns_proper_dataclass(self):
        result = fit_slater_parameters(
            experimental_energies_cm1=[1000.0, 2000.0],
            angular_matrix=np.array([[1.0, 0.5], [0.5, 1.0]]),
            abinitio_parameters=[800.0, 1200.0],
            parameter_labels=["F", "G"],
            fit_offset=False,
        )
        assert isinstance(result, SlaterFitResult)
        assert "F" in result.scale_factors
        assert "G" in result.scale_factors
        assert len(result.residuals_cm1) == 2
        assert len(result.level_predictions_cm1) == 2
