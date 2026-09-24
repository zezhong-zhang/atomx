"""Tests for the pint-based units layer.

Covers:
* Availability metadata (``pint_available``, ``codata_version``).
* Converters (``to_hartree``, ``to_bohr``, ``to_kelvin``, ``to_MHz``).
* Quantity factories.
* ``.total_energy_as``, ``.orbital_energy_as`` on ``SCFResult``.
* ``.A_as``, ``.B_as`` on ``HyperfineResult``.
* ``.total_as``, ``.NMS_as``, ``.SMS_as``, ``.FS_as`` on
  ``IsotopeShiftResult``.
"""

from __future__ import annotations

import importlib.util

import pytest

#: ``pint`` is an optional extra (``units``).  The guard is a module-level
#: ``skipif`` (not an import-time ``importorskip``): every node in this file is
#: the units layer's only coverage, and a bare import-time skip reports it as
#: one anonymous line.  ``pint>=0.21`` is in the ``dev`` extra, so a fresh
#: ``pip install -e .[dev]`` (or ``.[units]``) runs all 32 nodes below.
HAS_PINT = importlib.util.find_spec("pint") is not None
pytestmark = pytest.mark.skipif(
    not HAS_PINT,
    reason="the optional `units` extra is not installed (pint>=0.21); "
           "install with `pip install -e .[dev]`",
)

from atomx import codata_version, pint_available, ureg  # noqa: E402
from atomx.units import (  # noqa: E402
    GHz,
    Rydberg,
    cm_inv,
    energy_quantity,
    eV,
    frequency_quantity,
    length_quantity,
    to_bohr,
    to_hartree,
    to_kelvin,
    to_MHz,
)

# ---------------------------------------------------------------------------
# Availability + metadata
# ---------------------------------------------------------------------------

class TestAvailability:
    def test_pint_available(self):
        assert pint_available() is True

    def test_codata_default_is_2022(self):
        assert codata_version() == "2022"

    def test_ureg_is_shared(self):
        from atomx import ureg as ureg_top
        from atomx.units import ureg as ureg_mod
        assert ureg_mod is ureg_top


# ---------------------------------------------------------------------------
# Converters: float + Quantity input
# ---------------------------------------------------------------------------

class TestToHartree:
    def test_float_passthrough(self):
        assert to_hartree(1.5) == 1.5

    def test_rydberg_half_hartree(self):
        assert to_hartree(1.0 * Rydberg) == pytest.approx(0.5, rel=1e-14)

    def test_eV_conversion(self):
        # 27.2114 eV ≈ 1 Ha
        assert to_hartree(27.211386245981 * eV) == pytest.approx(1.0, rel=1e-9)

    def test_cm_inv_via_spectroscopy(self):
        # 219474.6 cm^-1 ≈ 1 Ha
        assert to_hartree(219474.6313632 * cm_inv) == pytest.approx(1.0, rel=1e-9)

    def test_joule_conversion(self):
        assert to_hartree(4.35974e-18 * ureg.joule) == pytest.approx(1.0, rel=1e-4)


class TestToBohr:
    def test_float_passthrough(self):
        assert to_bohr(1.5) == 1.5

    def test_angstrom_to_bohr(self):
        # 1 Å ≈ 1.8897 Bohr
        assert to_bohr(1.0 * ureg.angstrom) == pytest.approx(1.88972613, rel=1e-8)

    def test_nm_to_bohr(self):
        assert to_bohr(10.0 * ureg.nanometer) == pytest.approx(188.972613, rel=1e-6)


class TestToMHz:
    def test_GHz_to_MHz(self):
        assert to_MHz(5.0 * GHz) == pytest.approx(5000.0, rel=1e-14)

    def test_spectroscopy_from_eV(self):
        # 1 eV ≈ 241.8 THz = 2.418e8 MHz
        mhz = to_MHz(1.0 * eV)
        assert mhz == pytest.approx(2.41799e8, rel=1e-4)


class TestToKelvin:
    def test_direct_kelvin(self):
        assert to_kelvin(300.0 * ureg.kelvin) == pytest.approx(300.0)

    def test_energy_via_kB(self):
        # 1 eV → k_B T with T = 11604.5 K
        T = to_kelvin(1.0 * eV)
        assert pytest.approx(11604.525, rel=1e-4) == T


# ---------------------------------------------------------------------------
# Quantity factories
# ---------------------------------------------------------------------------

class TestQuantityFactories:
    def test_energy_quantity(self):
        q = energy_quantity(2.5)
        assert q.magnitude == 2.5
        assert q.units == ureg.hartree

    def test_length_quantity(self):
        q = length_quantity(1.0)
        assert q.to(ureg.meter).magnitude == pytest.approx(
            5.29177210544e-11, rel=1e-10)

    def test_frequency_quantity(self):
        q = frequency_quantity(1.0e6)
        assert q.to("MHz").magnitude == pytest.approx(1.0, rel=1e-14)


# ---------------------------------------------------------------------------
# SCFResult unit-aware methods
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def h_1s_result():
    from atomx.methods.dirac_fock import DFSolver, Subshell
    shells = [Subshell(n=1, kappa=-1, occupation=1)]
    solver = DFSolver(Z=1, subshells=shells, ns=60, ks=9, exchange="hf")
    return solver.solve(max_iter=50, tol=1e-8)


class TestSCFResultUnits:
    def test_total_energy_as_hartree(self, h_1s_result):
        q = h_1s_result.total_energy_as("hartree")
        assert q.magnitude == pytest.approx(-0.5, abs=1e-4)
        assert q.units == ureg.hartree

    def test_total_energy_as_eV(self, h_1s_result):
        q = h_1s_result.total_energy_as("eV")
        # -0.5 Ha ≈ -13.6 eV
        assert q.magnitude == pytest.approx(-13.6057, rel=1e-4)

    def test_total_energy_via_spectroscopy_cm_inv(self, h_1s_result):
        q = h_1s_result.total_energy_as("cm^-1")
        # 0.5 Ha × 219474.6 cm^-1/Ha ≈ 109737
        assert q.magnitude == pytest.approx(-109737.315, rel=1e-3)

    def test_orbital_energy_as(self, h_1s_result):
        q = h_1s_result.orbital_energy_as("1s1/2", "eV")
        assert q.magnitude == pytest.approx(-13.6057, rel=1e-4)

    def test_orbital_unknown_label_raises(self, h_1s_result):
        with pytest.raises(KeyError):
            h_1s_result.orbital_energy_as("nope", "eV")


# ---------------------------------------------------------------------------
# HyperfineResult unit-aware methods
# ---------------------------------------------------------------------------

class TestHyperfineResultUnits:
    def test_A_as_Hz(self, h_1s_result):
        hf = h_1s_result.hyperfine_constants()
        q = hf.A_as("1s1/2", "Hz")
        # ~1.42e9 Hz for H 1s
        assert q.magnitude == pytest.approx(1.42e9, rel=5e-3)

    def test_A_as_cm_inv(self, h_1s_result):
        hf = h_1s_result.hyperfine_constants()
        q = hf.A_as("1s1/2", "cm^-1")
        # 1420 MHz = 0.0474 cm^-1
        assert q.magnitude == pytest.approx(0.0474, rel=1e-2)

    def test_A_as_MHz_default(self, h_1s_result):
        hf = h_1s_result.hyperfine_constants()
        q = hf.A_as("1s1/2")
        assert q.units == ureg.megahertz

    def test_A_unknown_label(self, h_1s_result):
        hf = h_1s_result.hyperfine_constants()
        with pytest.raises(KeyError):
            hf.A_as("bogus")

    def test_B_as_zero_when_missing(self, h_1s_result):
        hf = h_1s_result.hyperfine_constants()
        # H 1s has no B entry (j=1/2); should give a zero quantity
        q = hf.B_as("1s1/2", "MHz")
        assert q.magnitude == 0.0


# ---------------------------------------------------------------------------
# IsotopeShiftResult unit-aware methods
# ---------------------------------------------------------------------------

class TestIsotopeShiftUnits:
    def test_total_as_THz(self, h_1s_result):
        iso = h_1s_result.isotope_shift((1, 1), (1, 2))
        q = iso.total_as("THz")
        # Mostly NMS, expected ~ +0.895 THz (H level sits higher than D:
        # E_NMS = +(m_e/M)·⟨T_e⟩ raises the lighter-isotope level more)
        assert q.magnitude == pytest.approx(+0.895, abs=0.01)

    def test_NMS_as_GHz(self, h_1s_result):
        iso = h_1s_result.isotope_shift((1, 1), (1, 2))
        q = iso.NMS_as("GHz")
        # ~ +895 GHz, sign per the 6f59415 NMS convention
        assert q.magnitude == pytest.approx(+895.4, abs=1.0)

    def test_SMS_default_is_zero(self, h_1s_result):
        iso = h_1s_result.isotope_shift((1, 1), (1, 2))
        assert iso.SMS_as("MHz").magnitude == 0.0

    def test_FS_as_kHz(self, h_1s_result):
        iso = h_1s_result.isotope_shift((1, 1), (1, 2))
        q = iso.FS_as("kHz")
        # ~ -6000 kHz for H→D
        assert abs(q.magnitude) < 50000
