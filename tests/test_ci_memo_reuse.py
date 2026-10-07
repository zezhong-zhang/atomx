"""Regression guards for the CI-memo structural reuse.

Implements the two P0 fixes from
``docs/HPC_NUMERICS_RESEARCH_2026-09-27.md`` §2.2-2.3 (measured together
at -36 % wall time on Fe XVII j-resolved MCDHF, energies bit-identical):

* the ``("h0", ns, kappa, Z)`` one-body Dirac blocks are orbital
  independent, so ``build_ci_matrix`` carries them across its entry reset
  instead of rebuilding them ~5x per CI construction; and
* the orbital-metric orthonormality audit is memoised per radial
  workspace on a CONTENT fingerprint of the audited orbitals: it runs on
  the first build of a workspace and again whenever the orbital byte
  content changes, and is skipped only for byte-identical orbitals
  already validated in the same workspace context.  There is no cadence
  and no environment knob.

The guards pin the *semantics* (when blocks are reused vs rebuilt, when
the tripwire fires), not timings.
"""

from __future__ import annotations

import numpy as np
import pytest

import atomx.physics.csf.matrix as matrix
import tests.test_ci_matrix_vs_mscheme_reference as _ref
from atomx.physics.csf.matrix import build_ci_matrix


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
    """Isolate each test from module-global memo/audit state."""
    matrix._reset_memo()
    matrix._METRIC_CHECK_STATE["context"] = None
    matrix._METRIC_CHECK_STATE["fingerprint"] = None
    yield
    matrix._reset_memo()
    matrix._METRIC_CHECK_STATE["context"] = None
    matrix._METRIC_CHECK_STATE["fingerprint"] = None


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
    def test_live_evaluators_keep_their_nuclear_model(self, cap, fresh_memo):
        """Delayed h_D evaluation must not inherit a later build's nucleus."""
        args = (cap['orbitals_dict'], cap['bsp_data'], cap['Z'], cap['ns'])
        point = matrix._radial_evaluator(*args)
        smooth = matrix._radial_evaluator(*args,
            nuclear_potential=lambda r: -cap['Z']/np.sqrt(r*r+1e-4))
        key = ('h', 1, -1, 1)
        # Neither closure has evaluated h yet; another context is active.
        hs = smooth(key)
        hp = point(key)
        expected = matrix._radial_evaluator(*args)(key)
        assert hp == expected
        assert abs(hp-hs) > 1e-3

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


    def test_mutated_nuclear_callable_invalidates_h0(self, cap, fresh_memo):
        shift = [0.0]
        def potential(r):
            return -cap["Z"] / r + shift[0]
        before = _build(cap, nuclear_potential=potential)
        cached = _h0_entries()
        shift[0] = 0.01
        after = _build(cap, nuclear_potential=potential)
        assert not _same_objects(cached, _h0_entries())
        # Four electrons: a constant shift in V adds exactly 4*shift.
        np.testing.assert_allclose(after-before, 0.04*np.eye(len(before)),
                                   atol=2e-12, rtol=0)


class TestMetricCheckMemo:
    def test_workspace_content_change_invalidates_audit_and_h0(self, cap, fresh_memo):
        """An in-place metric change must not reuse the earlier pass or H0."""
        workspace = {
            k: (v.copy() if isinstance(v, np.ndarray) else v)
            for k, v in cap["bsp_data"].items()
        }
        _build(cap, bsp_data=workspace)
        original = _h0_entries()
        workspace["grw"] *= 1.5
        for _ in range(2):
            with pytest.raises(ValueError, match="orthonormal"):
                _build(cap, bsp_data=workspace)
        assert not _same_objects(_h0_entries(), original)

    @staticmethod
    def _spy(monkeypatch):
        """Count real audit invocations through build_ci_matrix."""
        calls = []
        real = matrix._check_orbital_metric

        def spy(*args, **kwargs):
            calls.append(args)
            return real(*args, **kwargs)

        monkeypatch.setattr(matrix, "_check_orbital_metric", spy)
        return calls

    def test_same_content_audits_once(self, cap, fresh_memo, monkeypatch):
        calls = self._spy(monkeypatch)
        _build(cap)
        _build(cap)
        _build(cap)
        assert len(calls) == 1

    def test_changed_content_reaudits(self, cap, fresh_memo, monkeypatch):
        calls = self._spy(monkeypatch)
        _build(cap)
        assert len(calls) == 1
        tweaked = {
            k: (p.copy(), q.copy()) for k, (p, q) in cap["orbitals_dict"].items()
        }
        key = next(iter(tweaked))
        p, q = tweaked[key]
        # A sub-tolerance perturbation: the audit must RUN (content changed)
        # and PASS (defect ~2e-12 << 1e-7).
        tweaked[key] = (p * (1.0 + 1e-12), q)
        _build(cap, orbitals_dict=tweaked)
        assert len(calls) == 2

    def test_new_workspace_reaudits(self, cap, fresh_memo, monkeypatch):
        calls = self._spy(monkeypatch)
        _build(cap)
        twin = {
            k: (v.copy() if isinstance(v, np.ndarray) else v)
            for k, v in cap["bsp_data"].items()
        }
        _build(cap, bsp_data=twin)
        assert len(calls) == 2

    def test_failed_audit_is_not_memoised(self, cap, fresh_memo, monkeypatch):
        """A rejected orbital set is re-audited on every later attempt."""
        calls = self._spy(monkeypatch)
        bad = dict(cap["orbitals_dict"])
        key = next(iter(bad))
        p, q = bad[key]
        bad[key] = (1.5 * p, 1.5 * q)  # breaks normalization
        for _ in range(2):
            with pytest.raises(ValueError, match="orthonormal"):
                _build(cap, orbitals_dict=bad)
        assert len(calls) == 2

    def test_fingerprint_tracks_content_not_identity(self):
        a = np.arange(8.0)
        orbs = {(1, -1): (a, a.copy())}
        keys = [(1, -1)]
        fp1 = matrix._orbital_metric_fingerprint(orbs, keys, 4, 8)
        clone = {(1, -1): (a.copy(), a.copy())}
        assert matrix._orbital_metric_fingerprint(clone, keys, 4, 8) == fp1
        clone[(1, -1)][0][3] += 1e-9
        assert matrix._orbital_metric_fingerprint(clone, keys, 4, 8) != fp1

    def test_tripwire_fires_on_first_build_of_workspace(self, cap, fresh_memo):
        """A corrupted orbital set is rejected on its first build."""
        bad_orbitals = dict(cap["orbitals_dict"])
        key = next(iter(bad_orbitals))
        p, q = bad_orbitals[key]
        bad_orbitals[key] = (1.5 * p, 1.5 * q)  # breaks normalization
        with pytest.raises(ValueError, match="orthonormal"):
            _build(cap, orbitals_dict=bad_orbitals)
