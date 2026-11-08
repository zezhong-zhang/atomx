"""Division of labor: DFT computes the environment, atomx computes the
many-body physics that DFT fundamentally cannot see.

Demonstrated on NiO d8 (ABACUS DFT+U converged on this host):
- DFT+U sees 5 single-particle d levels spanning 0.08 eV
- atomx resolves 45 many-body states spanning 5.27 eV
- the 5.2 eV gap is the multiplet physics DFT is blind to
"""
import numpy as np
import pytest

from atomx.embedding.division_of_labor import (
    dft_vs_many_body,
    f_shell_dimension,
)

DFTU_H_D = np.array([
    [0.6513, -0.0015, -0.0015, 0., -0.003],
    [-0.0015, 0.7259, -0.0053, -0.0026, 0.0053],
    [-0.0015, -0.0053, 0.7259, 0.0026, 0.0053],
    [0., -0.0026, 0.0026, 0.6513, 0.],
    [-0.003, 0.0053, 0.0053, 0., 0.7259],
])


@pytest.fixture(scope="module")
def comparison():
    return dft_vs_many_body(
        DFTU_H_D,
        slater_eV={"F0": 0.409, "F2": 7.9521, "F4": 4.9387},
        zeta_d_eV=0.083,
    )


def test_dft_sees_five_levels(comparison):
    assert len(comparison["dft_single_particle_eV"]) == 5
    assert comparison["dft_span_eV"] < 0.1  # only the CF splitting


def test_atomx_resolves_many_body_manifold(comparison):
    assert comparison["many_body_states"] == 45  # C(10,8) for d8
    assert comparison["many_body_span_eV"] > 5.0  # full multiplet


def test_physics_gap_is_the_point(comparison):
    """The 5.2 eV gap IS the many-body physics DFT cannot compute —
    this is why atomx exists in the pipeline."""
    gap = comparison["physics_gap_eV"]
    assert gap > 5.0
    assert gap > 50 * comparison["dft_span_eV"]  # DFT misses >99%


def test_division_of_labor_documented(comparison):
    div = comparison["division"]
    assert "crystal-field" in " ".join(div["abacus_computes"])
    assert "multiplet" in " ".join(div["atomx_computes"])
    assert "RIXS" in " ".join(div["atomx_computes"])


def test_f_shell_dimensions():
    """f-shell: the gap is even larger (thousands of states for f^n)."""
    assert f_shell_dimension(7) == 3432  # C(14,7)
    assert f_shell_dimension(2) == 91
    assert f_shell_dimension(12) == 91  # particle-hole
    assert f_shell_dimension(1) == 14
