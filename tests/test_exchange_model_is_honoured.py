"""The requested exchange model must be used, or refused — never ignored.

Found 2026-09-19 while cross-verifying against DBSR_HF: the multi-configuration
path (`configs=[...]` or `correlation=...`) built its `MCDHFSolver` without an
`exchange` argument, so `Atom(configs=[...], exchange="slater")` returned a
**Hartree-Fock** energy while `provenance` reported `'exchange': 'slater'`.
Measured before the fix: `hf`, `slater` and `gu` all gave -128.691926 Ha for
`configs=["1s2 2s2 2p6"]`, where the single-configuration path gives
-128.691926 (hf) and -128.623437 (slater).

The MCDHF/CI solver implements non-local HF exchange only, so the fix is to
refuse the unsupported combination loudly rather than report a model that did
not run.  The single-configuration path honours all three and is unaffected.
"""

from __future__ import annotations

import pytest

from atomx import Atom


class TestExchangeIsHonoured:
    @pytest.mark.parametrize("exchange", ["slater", "gu"])
    def test_single_config_path_honours_local_exchange(self, exchange):
        """The path that *does* implement them must still work."""
        a = Atom(
                 Z=10,
                 engine="bspline",
                 exchange=exchange,
                 config="1s2 2s2 2p6",
                 corrections="none",
            )
        a.solve()
        b = Atom(
                 Z=10,
                 engine="bspline",
                 exchange="hf",
                 config="1s2 2s2 2p6",
                 corrections="none",
            )
        b.solve()
        assert a.total_energy != b.total_energy, (
            "slater/gu must differ from hf on the single-configuration path"
        )
        assert a.provenance["exchange"] == exchange

    @pytest.mark.parametrize("exchange", ["slater", "gu"])
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"config": ["1s2 2s2 2p6"]},
            {"config": "1s2 2s2 2p6", "correlation": "s"},
        ],
    )
    def test_mcdhf_path_refuses_local_exchange(self, exchange, kwargs):
        """No silent HF fallback with a provenance that claims otherwise."""
        with pytest.raises(ValueError, match="not implemented for multi-configuration"):
            Atom(
                 Z=10,
                 engine="bspline",
                 exchange=exchange,
                 corrections="none",
                 **kwargs,
            ).solve()

    def test_mcdhf_path_still_accepts_hf(self):
        a = Atom(
                 Z=10,
                 engine="bspline",
                 exchange="hf",
                 config=["1s2 2s2 2p6"],
                 corrections="none",
            )
        a.solve()
        assert a.provenance["exchange"] == "hf"
        b = Atom(
                 Z=10,
                 engine="bspline",
                 exchange="hf",
                 config="1s2 2s2 2p6",
                 corrections="none",
            )
        b.solve()
        assert a.total_energy == pytest.approx(b.total_energy, rel=1e-12)

    def test_provenance_never_claims_an_unused_model(self):
        """Whatever runs, `provenance` must describe it."""
        for kwargs in ({"config": "1s2 2s2 2p6"},
                       {"config": ["1s2 2s2 2p6"]}):
            for exchange in ("hf", "slater"):
                try:
                    a = Atom(
                             Z=10,
                             engine="bspline",
                             exchange=exchange,
                             corrections="none",
                             **kwargs,
                        )
                    a.solve()
                except ValueError:
                    continue  # refused: nothing was run, nothing claimed
                assert a.provenance["exchange"] == exchange
