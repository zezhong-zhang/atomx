"""Unit tests for the FAC-style asymptotic-potential fix (mode 0).

Realistic local-exchange SCF output shape:
  rv(r) = r·V(r)  with
    rv(0) ≈ -Z                           (nuclear dominance)
    rv(r) negative, rising towards 0     (exponentially-screened tail)
    rv(r → ∞) → 0                        (local exchange decays too fast)

Latter's clip pulls the too-weak tail down to rv_inf = -(Z-N+1); the
smooth envelope blends the interior into that asymptote.
"""

import numpy as np
import pytest

from atomx.numerics.local_dfs.asymptotic import apply_fac_tail_fix


def _realistic_rv(r: np.ndarray, Z: float) -> np.ndarray:
    """A plausible local-exchange r·V(r): -Z at r=0, → 0 at r→∞."""
    return -Z * np.exp(-r / 1.5)


def _running_charge(r: np.ndarray, N: float) -> np.ndarray:
    """Synthetic q(r) = N·(1 - e^{-r}) that reaches N at large r."""
    return N * (1.0 - np.exp(-r))


class TestLatterOnly:
    """With ihx = 0 the smooth envelope is disabled — only Latter runs."""

    def test_neutral_atom_tail_clamps_to_minus_one(self):
        r = np.geomspace(1e-5, 100.0, 2000)
        rv = _realistic_rv(r, Z=10)
        q = _running_charge(r, N=10)
        rv_fixed = apply_fac_tail_fix(
            r=r, rv=rv, running_charge=q, Z=10, N=10, ihx=0.0,
        )
        # Outer tail: was → 0; now must sit at or below rv_inf = -1.
        assert rv_fixed[-1] == pytest.approx(-1.0, abs=1e-9)
        # No point is less-binding than -1 (rv_fixed ≤ -1 everywhere
        # where the raw rv is too weak).
        too_weak = rv > -1.0
        assert np.all(rv_fixed[too_weak] == pytest.approx(-1.0, abs=1e-12))

    def test_positive_ion_tail_clamps_to_minus_residual(self):
        # Fe^{16+} (Ne-like): Z-N+1 = 26-10+1 = 17 → rv_inf = -17
        r = np.geomspace(1e-5, 100.0, 2000)
        rv = _realistic_rv(r, Z=26)
        q = _running_charge(r, N=10)
        rv_fixed = apply_fac_tail_fix(
            r=r, rv=rv, running_charge=q, Z=26, N=10, ihx=0.0,
        )
        assert rv_fixed[-1] == pytest.approx(-17.0, abs=1e-9)
        too_weak = rv > -17.0
        assert np.all(rv_fixed[too_weak] == pytest.approx(-17.0, abs=1e-12))

    def test_interior_unchanged(self):
        r = np.geomspace(1e-5, 100.0, 2000)
        rv = _realistic_rv(r, Z=10)
        q = _running_charge(r, N=10)
        rv_fixed = apply_fac_tail_fix(
            r=r, rv=rv, running_charge=q, Z=10, N=10, ihx=0.0,
        )
        # Interior (r < 1.0): rv strongly binding (< -1), Latter leaves it.
        inner = r < 1.0
        assert np.allclose(rv_fixed[inner], rv[inner], rtol=0.0, atol=1e-12)


class TestSmoothEnvelope:
    """With ihx < 0 the exponential envelope smoothly joins rv to rv_inf."""

    def test_envelope_reaches_asymptote(self):
        r = np.geomspace(1e-5, 100.0, 3000)
        rv = _realistic_rv(r, Z=10)
        q = _running_charge(r, N=10)
        rv_fixed = apply_fac_tail_fix(
            r=r, rv=rv, running_charge=q, Z=10, N=10,
            ihx=-0.05, csi=0.55,
        )
        # rv(∞) → rv_inf = -1
        assert rv_fixed[-1] == pytest.approx(-1.0, abs=0.01)

    def test_envelope_is_more_binding_than_raw_tail(self):
        """In the outer region the envelope should be ≤ (more binding
        than) the raw rv, since raw rv decays to 0 while envelope
        decays to rv_inf < 0."""
        r = np.geomspace(1e-5, 100.0, 3000)
        rv = _realistic_rv(r, Z=10)
        q = _running_charge(r, N=10)
        rv_fixed = apply_fac_tail_fix(
            r=r, rv=rv, running_charge=q, Z=10, N=10,
            ihx=-0.05, csi=0.55,
        )
        # Outer region beyond typical ionic radius
        outer = r > 3.0
        # rv_fixed ≤ rv_raw up to the floor rv_inf.
        assert np.all(rv_fixed[outer] <= rv[outer] + 1e-9)

    def test_interior_unchanged_with_envelope(self):
        r = np.geomspace(1e-5, 100.0, 3000)
        rv = _realistic_rv(r, Z=10)
        q = _running_charge(r, N=10)
        rv_fixed = apply_fac_tail_fix(
            r=r, rv=rv, running_charge=q, Z=10, N=10,
            ihx=-0.05, csi=0.55,
        )
        # Deep interior (r < 0.5): nuclear dominance, rv ≈ -Z.
        inner = r < 0.1
        # Both rv and rv_fixed should be close to -Z here.
        assert np.all(np.abs(rv_fixed[inner] - rv[inner]) < 0.5)


class TestEdgeCases:
    def test_hydrogen_like_no_clip_needed(self):
        """Z=10 N=1 → rv_inf = -10, same as rv(0); Latter never fires."""
        r = np.geomspace(1e-5, 100.0, 1000)
        rv = np.full_like(r, -10.0)     # exact Coulomb
        q = _running_charge(r, N=1)
        rv_fixed = apply_fac_tail_fix(
            r=r, rv=rv, running_charge=q, Z=10, N=1, ihx=-0.05, csi=0.55,
        )
        assert np.allclose(rv_fixed, rv, rtol=0.0, atol=1e-12)

    def test_grid_shape_preserved(self):
        r = np.geomspace(1e-5, 100.0, 777)
        rv = _realistic_rv(r, Z=10)
        q = _running_charge(r, N=10)
        out = apply_fac_tail_fix(
            r=r, rv=rv, running_charge=q, Z=10, N=10, ihx=-0.05, csi=0.55,
        )
        assert out.shape == r.shape

    def test_disabled_envelope_when_N_too_small(self):
        """N ≤ 1 → envelope disabled, only Latter."""
        r = np.geomspace(1e-5, 100.0, 1000)
        rv = _realistic_rv(r, Z=5)
        q = _running_charge(r, N=1)
        out = apply_fac_tail_fix(
            r=r, rv=rv, running_charge=q, Z=5, N=1, ihx=-0.05, csi=0.55,
        )
        # rv_inf = -(5-1+1) = -5; all too-weak points clip to -5
        # (here: nothing clips because rv is always ≤ -5? Let's check.)
        assert out[-1] == pytest.approx(min(rv[-1], -5.0), abs=1e-9)
