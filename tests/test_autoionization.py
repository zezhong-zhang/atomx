"""Tests for Auger / DR machinery."""

from __future__ import annotations

import os
import subprocess
import sys

import numpy as np
import pytest

from atomx.scattering.autoionization import (
    AutoionizationState,
    DRRateCoefficient,
    auger_rate_j_resolved,
    dr_calculate,
    dr_rate_coefficient,
    dr_strengths,
)
from atomx.scattering.dielectronic_recombination.enumerate import (
    enumerate_dr_resonances,
)

# ---------------------------------------------------------------------------
# AutoionizationState
# ---------------------------------------------------------------------------

class TestAutoionizationState:
    def test_basic_fields(self):
        d = AutoionizationState(
            label="2p^5 3d^2",
            E_resonance_eV=60.0, J=2.5,
            gamma_auger_per_s=1e14, a_radiative_per_s=1e13,
        )
        assert d.g == 2 * 2.5 + 1
        assert d.total_width_per_s == pytest.approx(1.1e14, rel=1e-12)
        # Fluorescence yield ~ 1/11
        assert d.fluorescence_yield == pytest.approx(1 / 11, rel=1e-6)

    def test_zero_widths_gives_zero_yield(self):
        d = AutoionizationState(
            label="empty", E_resonance_eV=10.0, J=0.5,
            gamma_auger_per_s=0, a_radiative_per_s=0,
        )
        assert d.fluorescence_yield == 0.0


def test_auger_rate_j_resolved_forwards_J_res(monkeypatch):
    import atomx.scattering.autoionization as autoion

    captured = {}

    def fake_coarse(result, **kwargs):
        captured.update(kwargs)
        return {"Gamma_per_s": 0.0}

    monkeypatch.setattr(autoion, "_coarse_auger_rate", fake_coarse)
    row = auger_rate_j_resolved(
        object(),
        J_res=1.5,
        J_final=0.5,
        hole_label="2p1/2",
        source_label="3s1/2",
        source2_label="3p1/2",
        resonance_energy_eV=100.0,
        state_index=7,
    )

    assert row == {"Gamma_per_s": 0.0}
    assert captured["J_res"] == 1.5
    assert captured["J_final"] == 0.5
    assert captured["state_index"] == 7


# ---------------------------------------------------------------------------
# DR rate coefficient — Burgess formula
# ---------------------------------------------------------------------------

class TestDRRateCoefficient:
    def test_single_resonance_magnitude(self):
        """Single resonance: α(T) ∝ T^{-3/2} · exp(−E/T) · Γ_Aug Γ_rad/Γ_tot.

        Test a simple case: E=10 eV, Γ_A=Γ_r=1e13, J=0.5, g_i=1:
            ω = 2/(2·1) = 1,  branch = 0.5e13 s⁻¹
        At T=10 eV:  α = 6.603e-24 · 10^{-3/2} · 1 · 5e12 · e^{-1}
                       ≈ 6.603e-24 · 0.0316 · 5e12 · 0.368
                       ≈ 3.84e-13  cm³/s
        """
        d = AutoionizationState(
            label="X", E_resonance_eV=10.0, J=0.5,
            gamma_auger_per_s=1e13, a_radiative_per_s=1e13,
        )
        rc = dr_rate_coefficient([d], T_eV=10.0, initial_state_g=1.0)
        assert rc.alpha_cm3_s[0] == pytest.approx(3.84e-13, rel=0.02)

    def test_vectorised_over_T(self):
        d = AutoionizationState(
            label="X", E_resonance_eV=10.0, J=0.5,
            gamma_auger_per_s=1e13, a_radiative_per_s=1e13,
        )
        T = np.array([5.0, 10.0, 20.0, 50.0, 100.0])
        rc = dr_rate_coefficient([d], T, initial_state_g=1.0)
        assert rc.alpha_cm3_s.shape == T.shape
        # All positive
        assert np.all(rc.alpha_cm3_s > 0)

    def test_DR_peaks_near_kT_equal_E(self):
        """α^DR(T) as a function of T for a single resonance peaks
        where d/dT [T^{−3/2} · exp(−E/T)] = 0  ⇒  T = 2E/3."""
        d = AutoionizationState(
            label="X", E_resonance_eV=30.0, J=0.5,
            gamma_auger_per_s=1e13, a_radiative_per_s=1e13,
        )
        T = np.linspace(1.0, 200.0, 2001)
        rc = dr_rate_coefficient([d], T)
        idx_peak = int(np.argmax(rc.alpha_cm3_s))
        T_peak = T[idx_peak]
        assert T_peak == pytest.approx(2.0 * 30.0 / 3.0, abs=0.5)

    def test_negative_T_raises(self):
        d = AutoionizationState(
            label="X", E_resonance_eV=10.0, J=0.5,
            gamma_auger_per_s=1e13, a_radiative_per_s=1e13,
        )
        with pytest.raises(ValueError, match="positive"):
            dr_rate_coefficient([d], T_eV=-5.0)

    def test_multi_resonance_sums_linearly(self):
        d1 = AutoionizationState(
            label="d1", E_resonance_eV=10.0, J=0.5,
            gamma_auger_per_s=1e13, a_radiative_per_s=1e13,
        )
        d2 = AutoionizationState(
            label="d2", E_resonance_eV=20.0, J=0.5,
            gamma_auger_per_s=2e13, a_radiative_per_s=1e13,
        )
        rc_both = dr_rate_coefficient([d1, d2], T_eV=15.0)
        rc_1 = dr_rate_coefficient([d1], T_eV=15.0)
        rc_2 = dr_rate_coefficient([d2], T_eV=15.0)
        assert rc_both.alpha_cm3_s[0] == pytest.approx(
            rc_1.alpha_cm3_s[0] + rc_2.alpha_cm3_s[0], rel=1e-12,
        )

    def test_per_resonance_breakdown(self):
        d1 = AutoionizationState(
            label="d1", E_resonance_eV=10.0, J=0.5,
            gamma_auger_per_s=1e13, a_radiative_per_s=1e13,
        )
        d2 = AutoionizationState(
            label="d2", E_resonance_eV=20.0, J=0.5,
            gamma_auger_per_s=2e13, a_radiative_per_s=1e13,
        )
        rc = dr_rate_coefficient([d1, d2], T_eV=[5.0, 10.0, 20.0])
        assert "d1" in rc.per_resonance
        assert "d2" in rc.per_resonance

    def test_at_T_interpolation(self):
        d = AutoionizationState(
            label="X", E_resonance_eV=10.0, J=0.5,
            gamma_auger_per_s=1e13, a_radiative_per_s=1e13,
        )
        T = np.linspace(1.0, 100.0, 1001)
        rc = dr_rate_coefficient([d], T)
        # Interpolated value at 10 eV matches the tabulated one
        assert rc.at_T(10.0) == pytest.approx(rc.alpha_cm3_s[np.argmin(np.abs(T - 10.0))],
                                              rel=1e-6)

    def test_zero_width_resonance_ignored(self):
        d_ok = AutoionizationState(
            label="ok", E_resonance_eV=10.0, J=0.5,
            gamma_auger_per_s=1e13, a_radiative_per_s=1e13,
        )
        d_bad = AutoionizationState(
            label="bad", E_resonance_eV=10.0, J=0.5,
            gamma_auger_per_s=0.0, a_radiative_per_s=1e13,
        )
        rc_only_ok = dr_rate_coefficient([d_ok], T_eV=10.0)
        rc_both = dr_rate_coefficient([d_ok, d_bad], T_eV=10.0)
        assert rc_only_ok.alpha_cm3_s[0] == pytest.approx(rc_both.alpha_cm3_s[0], rel=1e-12)


class _FakeIon:
    charge = 16
    converged = True
    _config_strings = ["1s2 2s2 2p6"]
    result = None
    orbital_energies = {
        "1s1/2": -250.0,
        "2s1/2": -50.0,
        "2p1/2": -40.0,
        "2p3/2": -38.0,
    }


class _SolvedFakeIon(_FakeIon):
    _result = object()


class TestDRDriver:
    def test_dr_direct_import_paths_are_acyclic(self):
        env = os.environ.copy()
        env["PYTHONPATH"] = "src"
        code = (
            "from atomx.scattering.dielectronic_recombination.solver "
            "import dr_calculate; "
            "from atomx.scattering.dielectronic_recombination.widths "
            "import make_abinitio_auger_rate_fn"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=os.getcwd(),
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr

    def test_dr_calculate_defaults_to_manifold_states(self, monkeypatch):
        """No hooks -> the determinant-CI manifold pipeline is the default."""
        from atomx.scattering.dielectronic_recombination import solver

        calls = {}
        canned = [
            AutoionizationState(
                label="2p3/2^-1 3s 3p (manifold)",
                E_resonance_eV=250.0,
                J=1.5,
                gamma_auger_per_s=3.0e12,
                a_radiative_per_s=4.0e11,
            )
        ]

        def fake_manifold_states(atom):
            calls["manifold_atom"] = atom
            return canned

        monkeypatch.setattr(solver, "_dr33_manifold_states", fake_manifold_states)

        atom = _SolvedFakeIon()
        rc = solver.dr_calculate(
            atom,
            "1s2 2s2 2p6 3s1",
            T_eV=300.0,
        )

        assert calls["manifold_atom"] is atom
        assert rc.resonances[0].gamma_auger_per_s == pytest.approx(3.0e12)
        assert rc.resonances[0].a_radiative_per_s == pytest.approx(4.0e11)

    def test_dr_calculate_without_hooks_raises_for_unsupported_input(self):
        """Non-Ne-like / unsolved inputs must fail loudly, not fall back to
        the old hydrogenic-width placeholders (rms 19610 vs FAC)."""
        with pytest.raises(NotImplementedError, match="manifold"):
            dr_calculate(
                _FakeIon(),
                "1s2 2s2 2p6 3s1",
                n_max=3,
                l_max=2,
                T_eV=300.0,
            )

    def test_dr_calculate_hook_path_defaults_missing_hook_to_abinitio(
        self, monkeypatch,
    ):
        """With one explicit hook + a solved result, the other hook defaults
        to the coarse ab-initio width factory (legacy hook path)."""
        from atomx.scattering.dielectronic_recombination import solver

        calls = {}

        def fake_radiative_factory(atom):
            calls["radiative_factory_atom"] = atom

            def radiative(_spec, _energy_eV):
                return 4.0e11, {"fake-e1": 4.0e11}

            return radiative

        def fake_build_autoionization_states(specs, atom, **kwargs):
            spec = specs[0]
            auger = kwargs["auger_rate_fn"]
            radiative = kwargs["radiative_rate_fn"]
            a_rad, channels = radiative(spec, 25.0)
            return [
                AutoionizationState(
                    label=spec.label,
                    E_resonance_eV=25.0,
                    J=spec.J,
                    gamma_auger_per_s=auger(spec, 25.0),
                    a_radiative_per_s=a_rad,
                    channels=channels,
                )
            ]

        monkeypatch.setattr(
            solver, "make_abinitio_radiative_rate_fn", fake_radiative_factory,
        )
        monkeypatch.setattr(
            solver, "build_autoionization_states", fake_build_autoionization_states,
        )

        atom = _SolvedFakeIon()
        rc = solver.dr_calculate(
            atom,
            "1s2 2s2 2p6 3s1",
            n_max=3,
            l_max=2,
            T_eV=300.0,
            auger_rate_fn=lambda _spec, _energy_eV: 3.0e12,
        )

        assert calls["radiative_factory_atom"] is atom
        assert rc.resonances[0].gamma_auger_per_s == pytest.approx(3.0e12)
        assert rc.resonances[0].a_radiative_per_s == pytest.approx(4.0e11)
        assert rc.resonances[0].channels == {"fake-e1": 4.0e11}

    def test_enumerates_kll_like_resonances(self):
        specs = enumerate_dr_resonances(
            _FakeIon(),
            "1s2 2s2 2p6 3s1",
            n_max=3,
            l_max=2,
        )
        assert specs
        labels = [s.label for s in specs]
        assert any("2p1/2^-1" in label or "2p3/2^-1" in label for label in labels)
        assert any(s.core_from.n == 2 for s in specs)
        assert all(s.intermediate_csf is not None for s in specs)
        assert all(s.intermediate_csf.total_J2 == int(round(2 * s.J)) for s in specs)

    def test_enumerated_intermediates_are_j_resolved_for_hyp171_transition_sum(self):
        """HYP-171 landed: radiative sums can filter explicit-J E1 branches."""
        specs = enumerate_dr_resonances(
            _FakeIon(),
            "1s2 2s2 2p6 3s1",
            n_max=3,
            l_max=1,
        )
        assert specs
        for spec in specs[:10]:
            assert spec.intermediate_csf.total_J2 >= 0
            assert spec.intermediate_csf.parity in (-1, 1)
            assert "J=" in spec.intermediate_csf.label

    def test_dr_calculate_returns_rate_and_resonance_table(self):
        rc = dr_calculate(
            _FakeIon(),
            "1s2 2s2 2p6 3s1",
            n_max=3,
            l_max=2,
            T_eV=[100.0, 300.0, 1000.0],
            auger_rate_fn=lambda _spec, _e: 1.0e13,
            radiative_rate_fn=lambda _spec, _e: 2.0e12,
        )
        assert isinstance(rc, DRRateCoefficient)
        assert rc.alpha_cm3_s.shape == (3,)
        assert len(rc.resonances) > 0
        assert np.all(rc.alpha_cm3_s >= 0.0)

        rows = dr_strengths(rc)
        assert len(rows) == len(rc.resonances)
        assert rows[0]["alpha_cm3_s"].shape == (3,)
        assert "gamma_auger_per_s" in rows[0]

    def test_dr_calculate_accepts_width_hooks(self):
        def auger(spec, energy_eV):
            return 1.0e13

        def rad(spec, energy_eV):
            return 2.0e12, {"mock": 2.0e12}

        rc = dr_calculate(
            _FakeIon(),
            "1s2 2s2 2p6 3s1",
            n_max=3,
            l_max=2,
            T_eV=300.0,
            auger_rate_fn=auger,
            radiative_rate_fn=rad,
        )
        assert len(rc.resonances) > 0
        assert all(d.gamma_auger_per_s == pytest.approx(1.0e13) for d in rc.resonances)
        assert all(d.a_radiative_per_s == pytest.approx(2.0e12) for d in rc.resonances)
