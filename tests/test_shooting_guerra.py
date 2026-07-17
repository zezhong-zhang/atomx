"""Guerra pair-weight seam of ShootingHFSolver (jj_configs kwarg).

Three gates:

(a) ``jj_configs=None`` is BIT-IDENTICAL to the pre-seam statistical
    path — pinned against total/orbital energies captured on the
    pre-change tree (same machine, pure-numpy elementwise kernels, so
    the pins are exact-equality reproducible; any bit change means the
    default statistical arithmetic was perturbed).

(b) A single JJConfig whose occupations equal the statistical
    jj-average reduces to the statistical products:
    ``WW(a,b)/q_a = q_a q_b / q_a`` differs from ``q_b`` only by float
    rounding, so totals must agree to <= 1e-12 relative (measured
    4e-16 for C).

(c) With genuine two-CSF weights the ci-vs-statistical energy DELTA
    matches the B-spline ``DFSolver`` given the same jj_configs.  The
    B-spline delta is basis-converged (identical to 12 digits from
    ns=50 to ns=113); at shooting tol=1e-10 the deltas agree to
    ~2.6e-7 relative (gate 1e-4, target was 1e-3).  Both engines must
    also move in the same direction.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from atomx.config.jj_coupling import JJConfig, generate_jj_average  # noqa: E402
from atomx.methods.dirac_fock import DFSolver, Subshell  # noqa: E402
from atomx.numerics.shooting.hf_shooting import ShootingHFSolver  # noqa: E402


def _statistical_subshells(config_str):
    subshells, _ = generate_jj_average(config_str, averaging="statistical")
    return [sh for sh in subshells if sh.occupation > 1e-10]


# ----------------------------------------------------------------------
# (a) jj_configs=None: bit-identical to the pre-seam statistical path
# ----------------------------------------------------------------------
#: Captured from the pre-change tree (defaults: n_grid=1000, r_max=75,
#: point nucleus, solve() defaults). Exact equality on purpose.
_BASELINE = {
    "Ne": ("1s2 2s2 2p6", 10, -128.69197743882302, {
        "1s1/2": -32.81746674512012,
        "2s1/2": -1.9358533149080215,
        "2p1/2": -0.8528341378011031,
        "2p3/2": -0.8482719942102857,
    }),
    "Co": ("1s2 2s2 2p6 3s2 3p6 3d7 4s2", 27, -1392.0024688180542, None),
}


@pytest.mark.parametrize("name", list(_BASELINE))
def test_no_jj_configs_bit_identical(name):
    cfg, Z, e_ref, oe_ref = _BASELINE[name]
    res = ShootingHFSolver(
        Z=Z, subshells=_statistical_subshells(cfg), jj_configs=None,
    ).solve()
    assert res.converged
    assert res.total_energy == e_ref, (
        f"{name}: statistical path changed bit-level: "
        f"{res.total_energy!r} != {e_ref!r}")
    if oe_ref is not None:
        for label, eps in oe_ref.items():
            assert res.orbital_energies[label] == eps, (
                f"{name} {label}: {res.orbital_energies[label]!r} "
                f"!= {eps!r}")


# ----------------------------------------------------------------------
# (b) single config == statistical occupations -> statistical products
# ----------------------------------------------------------------------
def test_single_config_matches_statistical():
    subshells = _statistical_subshells("1s2 2s2 2p2")
    single = [JJConfig(
        subshells=[Subshell(n=s.n, kappa=s.kappa, occupation=s.occupation)
                   for s in subshells],
        weight=1.0)]
    r_stat = ShootingHFSolver(Z=6, subshells=subshells).solve()
    r_one = ShootingHFSolver(Z=6, subshells=subshells,
                             jj_configs=single).solve()
    assert r_one.converged
    rel = abs(r_one.total_energy - r_stat.total_energy) / abs(
        r_stat.total_energy)
    assert rel <= 1e-12, (
        f"single-config Guerra vs statistical: rel {rel:.2e}")
    for label, eps in r_stat.orbital_energies.items():
        rel_o = abs(r_one.orbital_energies[label] - eps) / abs(eps)
        assert rel_o <= 1e-10, f"{label}: rel {rel_o:.2e}"


# ----------------------------------------------------------------------
# (c) two-CSF weights: delta agrees with DFSolver on the same jj_configs
# ----------------------------------------------------------------------
def _carbon_two_csf():
    """C 2p^2 mixture: 0.5 (2p-)^1(2p+)^1 + 0.5 (2p+)^2.

    Keeps the 2p-/2p+ cross terms nonzero (WW(2p-, 2p+) = 0.5), unlike
    a (2p-)^2 / (2p+)^2 mixture where they vanish.
    """
    def mk(qm, qp):
        return [Subshell(1, -1, 2.0), Subshell(2, -1, 2.0),
                Subshell(2, 1, float(qm)), Subshell(2, -2, float(qp))]
    subshells = mk(0.5, 1.5)  # weighted-average occupations
    configs = [JJConfig(subshells=mk(1, 1), weight=0.5),
               JJConfig(subshells=mk(0, 2), weight=0.5)]
    return subshells, configs


@pytest.fixture(scope="module")
def two_csf_deltas():
    subshells, configs = _carbon_two_csf()
    # Shooting tol is on the per-sweep energy CHANGE with 0.5 damping;
    # the default 1e-7 cap leaves ~2e-5 Ha residual, which is exactly
    # the size of the delta discrepancy being gated — tighten it.
    s_stat = ShootingHFSolver(Z=6, subshells=subshells).solve(
        max_iter=200, tol=1e-10)
    s_ci = ShootingHFSolver(Z=6, subshells=subshells,
                            jj_configs=configs).solve(
        max_iter=200, tol=1e-10)
    b_stat = DFSolver(Z=6, subshells=subshells, exchange="hf",
                      nuclear="point").solve(max_iter=200, tol=1e-9)
    b_ci = DFSolver(Z=6, subshells=subshells, exchange="hf",
                    nuclear="point", jj_configs=configs).solve(
        max_iter=200, tol=1e-9)
    for r in (s_stat, s_ci, b_stat, b_ci):
        assert r.converged
    return (s_ci.total_energy - s_stat.total_energy,
            float(b_ci.total_energy) - float(b_stat.total_energy))


def test_two_csf_delta_same_direction(two_csf_deltas):
    d_shoot, d_bspline = two_csf_deltas
    # Non-vacuous: the mixture must actually move the energy.
    assert abs(d_bspline) > 1e-4
    assert d_shoot * d_bspline > 0.0


def test_two_csf_delta_matches_bspline(two_csf_deltas):
    d_shoot, d_bspline = two_csf_deltas
    rel = abs(d_shoot - d_bspline) / abs(d_bspline)
    assert rel <= 1e-4, (
        f"ci-vs-statistical delta: shooting {d_shoot!r} vs "
        f"bspline {d_bspline!r} (rel {rel:.2e})")


# ----------------------------------------------------------------------
# seam validation
# ----------------------------------------------------------------------
def test_misaligned_jj_configs_rejected():
    subshells = _statistical_subshells("1s2 2s2 2p2")
    bad = [JJConfig(subshells=list(reversed(subshells)), weight=1.0)]
    with pytest.raises(ValueError, match="align positionally"):
        ShootingHFSolver(Z=6, subshells=subshells, jj_configs=bad)


def test_zero_occupation_rejected_with_jj_configs():
    subshells = [Subshell(1, -1, 2.0), Subshell(2, -1, 0.0)]
    cfgs = [JJConfig(subshells=[Subshell(1, -1, 2.0),
                                Subshell(2, -1, 0.0)], weight=1.0)]
    with pytest.raises(ValueError, match="q > 0"):
        ShootingHFSolver(Z=4, subshells=subshells, jj_configs=cfgs)
