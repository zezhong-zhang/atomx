"""EDM enhancement-factor closed-form scaling tests.

Classification note (audit S4, findings 9/18/23): ``scaling.py`` is a
one-line closed form

    R_e = 10 α² Z³ n_eff / (J(J+1)(2L+1))

so the Z³ / p-to-s / Z⁴ ratio tests below cannot fail for a physics
reason — they are *formula pins* (frozen self-regression, marked
``snapshot``).  The falsifiable content is the comparison with the
external values:

* Liu & Kelly, *Phys. Rev. A* **45**, R4210 (1992): all-order MBPT for Cs
  gives ``R_e ≈ 120`` (the module documents factor-of-2 accuracy);
* Dzuba & Flambaum, *Phys. Rev. A* **80**, 062509 (2009): Schiff-moment
  enhancement systematics, ``K_S(Hg) ≈ 4e-17 cm``.

Measured at HEAD: ``R_e(Cs) = 224.4456`` (1.87x Liu-Kelly, inside the
documented factor of 2) and ``K_S(Hg) = 4.3624e-17 cm`` (9 % above 4e-17).
The old gates (``30 < R_e < 500``, ``1e-18 < K_S < 1e-16``) could not see
the 2.8x error the shipped docstring carried (it printed "R_e ≈ 80" while
the code returned 224.4).
"""

import pytest

from atomx import Atom
from atomx.properties.edm import (
    edm_summary,
    electron_edm_enhancement_sandars,
    schiff_moment_enhancement_scaling,
)

#: Liu & Kelly (1992) all-order MBPT enhancement for Cs.
R_E_CS_LIU_KELLY = 120.0
#: Dzuba & Flambaum (2009) Schiff-moment enhancement for Hg, cm.
K_S_HG_DZUBA_FLAMBAUM = 4e-17


def test_R_e_cs_like_within_factor_two_of_liu_kelly():
    """Cs ²S₁/₂ (Z=55, n_eff=1.9) is within the documented factor of 2 of
    the Liu-Kelly all-order MBPT value 120.

    Measured 224.4456 (= 1.87 × 120, i.e. inside factor 2 but not the
    "≈80" the module docstring used to print).
    """
    R_e = electron_edm_enhancement_sandars(Z=55, n_eff=1.9, J=0.5, L=0)
    assert R_E_CS_LIU_KELLY / 2.0 < R_e < 2.0 * R_E_CS_LIU_KELLY, (
        f"R_e(Cs) = {R_e:.3f}, outside the documented factor-of-2 window "
        f"[{R_E_CS_LIU_KELLY / 2}, {2 * R_E_CS_LIU_KELLY}] around Liu-Kelly "
        f"{R_E_CS_LIU_KELLY}")


@pytest.mark.snapshot
def test_R_e_cs_like_frozen_value():
    """Frozen atomx value of the closed form at the shipped Cs example.

    This is the gate that distinguishes the code (224.4456) from the
    docstring's old "R_e ≈ 80" (the number the J(J+1) → 2J+1 denominator
    confusion produces: 224.4 × 0.75/2 = 84.2).  No external principle
    fixes this literal — only atomx's own arithmetic — hence ``snapshot``.
    """
    R_e = electron_edm_enhancement_sandars(Z=55, n_eff=1.9, J=0.5, L=0)
    assert pytest.approx(224.4456, rel=1e-4) == R_e
    # And it must NOT be the docstring's old (wrong) worked example.
    assert abs(R_e - 80.0) / 80.0 > 1.0, (
        f"R_e(Cs) = {R_e:.3f} is back at the old docstring value ~80")


@pytest.mark.snapshot
def test_R_e_formula_pin_z_cubed_scaling():
    """Formula pin: ``scaling.py`` divides by Z³ (a re-derivation of it)."""
    R1 = electron_edm_enhancement_sandars(Z=10, n_eff=2.0)
    R2 = electron_edm_enhancement_sandars(Z=20, n_eff=2.0)
    assert pytest.approx(8.0, rel=1e-12) == R2 / R1


@pytest.mark.snapshot
def test_R_e_formula_pin_p_state_third():
    """Formula pin: the ``1/(2L+1)`` denominator gives exactly 1/3 for L=1."""
    R_s = electron_edm_enhancement_sandars(Z=55, n_eff=1.9, J=0.5, L=0)
    R_p = electron_edm_enhancement_sandars(Z=55, n_eff=1.9, J=0.5, L=1)
    assert R_p / R_s == pytest.approx(1.0 / 3.0, rel=1e-12)


def test_R_e_invalid():
    with pytest.raises(ValueError):
        electron_edm_enhancement_sandars(Z=0)
    with pytest.raises(ValueError):
        electron_edm_enhancement_sandars(Z=10, n_eff=0)
    with pytest.raises(ValueError):
        electron_edm_enhancement_sandars(Z=10, J=0)
    with pytest.raises(ValueError):
        electron_edm_enhancement_sandars(Z=10, L=-1)


@pytest.mark.snapshot
def test_K_S_formula_pin_z_fourth_scaling():
    """Formula pin: ``K_S ∝ Z²(αZ)²`` = Z⁴ in the closed form."""
    K_lo = schiff_moment_enhancement_scaling(Z=20, A=40, valence="s")
    K_hi = schiff_moment_enhancement_scaling(Z=80, A=200, valence="s")
    ratio = K_hi / K_lo
    expected_z4 = (80.0 / 20.0) ** 4
    assert ratio == pytest.approx(expected_z4, rel=1e-12)


def test_K_S_hg_matches_dzuba_flambaum_scale():
    """Hg (Z=80, A=199) K_S against the Dzuba-Flambaum ~4e-17 cm scale.

    Measured 4.3624e-17 cm, 9 % above the named value (the old gate
    ``1e-18 < K < 1e-16`` admitted a 100x error).
    """
    K = schiff_moment_enhancement_scaling(Z=80, A=199, valence="closed")
    assert pytest.approx(K_S_HG_DZUBA_FLAMBAUM, rel=0.2) == K, (
        f"K_S(Hg) = {K:.4e} cm vs Dzuba-Flambaum "
        f"{K_S_HG_DZUBA_FLAMBAUM:.0e} cm")
    # The Xe < Hg < Ra ordering the module claims.
    K_xe = schiff_moment_enhancement_scaling(Z=54, A=131, valence="closed")
    K_ra = schiff_moment_enhancement_scaling(Z=88, A=226, valence="closed")
    assert K_xe < K < K_ra


def test_K_S_invalid_valence():
    with pytest.raises(ValueError):
        schiff_moment_enhancement_scaling(Z=80, A=199, valence="banana")


def test_K_S_invalid_Z_A():
    with pytest.raises(ValueError):
        schiff_moment_enhancement_scaling(Z=0, A=10)
    with pytest.raises(ValueError):
        schiff_moment_enhancement_scaling(Z=10, A=0)


def test_compute_edm_enhancement_high_level_api():
    """High-level compute_edm_enhancement accepts a solved Atom."""
    from atomx.methods.edm import compute_edm_enhancement

    atom = Atom(Z=3, config="1s2 2s1", engine="bspline", corrections="none")
    atom.solve()
    result = compute_edm_enhancement(atom, "2s1/2")
    assert hasattr(result, "R_electron_edm")
    assert hasattr(result, "per_orbital")


def test_edm_summary_scaling_path():
    """``edm_summary`` returns the inputs *and* the computed scaling values.

    Replaces the old key-existence-only test: the keys are asserted
    together with the values they carry (measured R_e = 224.4456,
    K_S = 1.9491e-16 cm at Z=55, A=133, n_eff=1.9).
    """
    res = edm_summary(Z=55, A=133, n_eff=1.9, J=0.5, L=0)
    assert set(res) >= {
        "Z", "A", "n_eff", "J", "L", "valence",
        "R_electron_edm", "K_schiff_cm",
    }
    assert (res["Z"], res["A"], res["n_eff"], res["J"], res["L"]) == (
        55, 133, 1.9, 0.5, 0)
    assert pytest.approx(224.4456, rel=1e-4) == res["R_electron_edm"]
    assert pytest.approx(1.94913e-16, rel=1e-4) == res["K_schiff_cm"]
