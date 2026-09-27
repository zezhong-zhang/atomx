"""Regression guards for the CI-memo structural reuse.

Implements the two P0 fixes from
``docs/HPC_NUMERICS_RESEARCH_2026-09-27.md`` §2.2-2.3 (measured together
at -36 % wall time on Fe XVII j-resolved MCDHF, energies bit-identical):

* the ``("h0", ns, kappa, Z)`` one-body Dirac blocks are orbital
  independent, so ``build_ci_matrix`` carries them across its entry reset
  instead of rebuilding them ~5x per CI construction; and
* the orbital-metric orthonormality audit runs on the first build of each
  radial workspace and then every ``_METRIC_CHECK_EVERY`` builds, not on
  every build (``ATOMX_CI_METRIC_CHECK_EVERY=1`` restores per-build).

The guards pin the *semantics* (when blocks are reused vs rebuilt, when
the tripwire fires), not timings.
"""

from __future__ import annotations

import numpy as np
import pytest

import atomx.methods.mcdhf.matrix as matrix
import tests.test_ci_matrix_vs_mscheme_reference as _ref
from atomx.methods.mcdhf.matrix import build_ci_matrix


def _h0_entries():
    return {
        k: v
        for k, v in matrix._SLATER_MEMO.items()
        if isinstance(k, tuple) and k[0] == "h0"
    }


def _same_objects(a, b):
    """Identity comparison for memo dicts (values are arrays — == is elementwise)."""
    return (set(a) == set(b)) and all(a[k] is b[k] for k in a)


@pytest.fixture(scope="module")
def cap():
    """A solved Be ground state with the exact arguments ci_solve saw."""
    result, _csfs, captured = _ref._run_capturing_ci(
        ["1s2 2s2"], radial_optimization="fac_frozen"
    )
    assert result.converged
    return captured


@pytest.fixture()
def fresh_memo():
    """Isolate each test from module-global memo/cadence state."""
    matrix._reset_memo()
    matrix._METRIC_CHECK_STATE["context"] = None
    matrix._METRIC_CHECK_STATE["count"] = 0
    yield
    matrix._reset_memo()
    matrix._METRIC_CHECK_STATE["context"] = None
    matrix._METRIC_CHECK_STATE["count"] = 0


def _build(captured, **overrides):
    args = dict(
        csfs=captured["csfs"],
        orbitals_dict=captured["orbitals_dict"],
        orbital_basis=None,
        bsp_data=captured["bsp_data"],
        Z=captured["Z"],
        ns=captured["ns"],
        nuclear_potential=captured.get("nuclear_potential"),
        convol_cache=captured.get("convol_cache"),
    )
    args.update(overrides)
    return build_ci_matrix(**args)


class TestH0Carry:
    def test_h0_blocks_survive_entry_reset_same_workspace(self, cap, fresh_memo):
        h1 = _build(cap)
        h0_after_first = _h0_entries()
        assert h0_after_first, "first build must populate the h0 memo"
        h2 = _build(cap)  # entry reset runs here
        assert _same_objects(_h0_entries(), h0_after_first)  # carried
        np.testing.assert_array_equal(h1, h2)

    def test_full_reset_rebuilds_deterministically(self, cap, fresh_memo):
        h1 = _build(cap)
        h0_first = _h0_entries()
        matrix._reset_memo()  # simulate an outer-iteration boundary
        h2 = _build(cap)
        assert not _same_objects(_h0_entries(), h0_first)  # rebuilt
        np.testing.assert_array_equal(h1, h2)  # determinism: identical values

    def test_new_workspace_rebuilds(self, cap, fresh_memo):
        _build(cap)
        h0_first = _h0_entries()
        twin = {
            k: (v.copy() if isinstance(v, np.ndarray) else v)
            for k, v in cap["bsp_data"].items()
        }
        _build(cap, bsp_data=twin)
        assert not _same_objects(_h0_entries(), h0_first)

    def test_different_nuclear_potential_rebuilds(self, cap, fresh_memo):
        _build(cap)
        h0_first = _h0_entries()
        other_potential = lambda r: -4.0 / np.sqrt(r * r + 1e-6)  # noqa: E731
        assert other_potential is not cap.get("nuclear_potential")
        _build(cap, nuclear_potential=other_potential)
        assert not _same_objects(_h0_entries(), h0_first)


class TestMetricCheckCadence:
    def test_cadence_gate(self, fresh_memo, monkeypatch):
        monkeypatch.setattr(matrix, "_METRIC_CHECK_EVERY", 3)
        ws = object()
        assert matrix._metric_check_due(ws) is True  # first build: always
        assert [matrix._metric_check_due(ws) for _ in range(5)] == [
            False,
            False,
            True,
            False,
            False,
        ]
        assert matrix._metric_check_due(object()) is True  # new workspace

    def test_every_one_restores_per_build_checking(self, fresh_memo, monkeypatch):
        monkeypatch.setattr(matrix, "_METRIC_CHECK_EVERY", 1)
        ws = object()
        assert all(matrix._metric_check_due(ws) for _ in range(4))

    def test_every_zero_checks_first_build_only(self, fresh_memo, monkeypatch):
        monkeypatch.setattr(matrix, "_METRIC_CHECK_EVERY", 0)
        ws = object()
        assert matrix._metric_check_due(ws) is True
        assert not any(matrix._metric_check_due(ws) for _ in range(3))
        assert matrix._metric_check_due(object()) is True

    def test_tripwire_fires_on_first_build_of_workspace(self, cap, fresh_memo):
        """A corrupted orbital set is rejected even under the cadence."""
        bad_orbitals = dict(cap["orbitals_dict"])
        key = next(iter(bad_orbitals))
        p, q = bad_orbitals[key]
        bad_orbitals[key] = (1.5 * p, 1.5 * q)  # breaks normalization
        with pytest.raises(ValueError, match="orthonormal"):
            _build(cap, orbitals_dict=bad_orbitals)
