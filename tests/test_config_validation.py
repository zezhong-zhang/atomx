"""Regression tests for central config-string validation (HYP-1778).

Invalid configuration strings (malformed shell tokens, ``n=0``,
over-occupied shells) must be rejected at parse/entry time with a clear
``ValueError`` — previously they crashed deep in the SCF/QED kernels with
cryptic ``KeyError`` / ``ZeroDivisionError`` / ``max() arg is empty``.
"""

import re

import pytest

from atomx import Atom
from atomx.methods.mcdhf.config import (
    _parse_config_string,
    count_j_resolved_csfs,
    enumerate_jj_csfs,
    validate_config_string,
)


class TestCentralParser:
    """Each rejection class raises a clear ValueError at parse time."""

    def test_invalid_shell_letter(self):
        # Was: KeyError: 'x' from l_map[token[1]] in enumerate_jj_csfs.
        with pytest.raises(ValueError, match="invalid configuration token '2x2'"):
            validate_config_string("1s2 2x2")

    def test_incomplete_token_missing_occupation(self):
        # Was: ValueError: invalid literal for int() with base 10: ''.
        with pytest.raises(ValueError, match="invalid configuration token '2s'"):
            validate_config_string("1s2 2s")

    def test_trailing_garbage_token(self):
        with pytest.raises(ValueError, match="invalid configuration token 'junk'"):
            validate_config_string("1s2 2s2 junk")

    def test_n_zero_rejected(self):
        # Was: ZeroDivisionError deep in the QED self-energy (mohr.py).
        with pytest.raises(ValueError, match="n must be >= 1"):
            validate_config_string("0s2")

    def test_overoccupied_shell_rejected(self):
        # Was: ValueError: max() arg is an empty sequence in dirac_fock.py.
        with pytest.raises(ValueError, match="exceeds the Pauli capacity"):
            validate_config_string("1s10")

    def test_overoccupied_p_shell_rejected(self):
        with pytest.raises(ValueError, match=re.escape("2(2l+1) = 6")):
            validate_config_string("2p7")

    def test_parse_config_string_uses_central_validation(self):
        with pytest.raises(ValueError, match="invalid configuration token"):
            _parse_config_string("1s2 2x2")
        with pytest.raises(ValueError, match="n must be >= 1"):
            _parse_config_string("0s2")
        with pytest.raises(ValueError, match="exceeds the Pauli capacity"):
            _parse_config_string("1s10")


class TestEnumerateJjCsfsValidation:
    """enumerate_jj_csfs reuses the central parser instead of raw slicing."""

    def test_invalid_shell_letter(self):
        with pytest.raises(ValueError, match="invalid configuration token '2x2'"):
            enumerate_jj_csfs("1s2 2x2")

    def test_incomplete_token(self):
        with pytest.raises(ValueError, match="invalid configuration token '2s'"):
            enumerate_jj_csfs("1s2 2s")

    def test_trailing_garbage(self):
        with pytest.raises(ValueError, match="invalid configuration token 'junk'"):
            enumerate_jj_csfs("1s2 2s2 junk")

    def test_n_zero(self):
        with pytest.raises(ValueError, match="n must be >= 1"):
            enumerate_jj_csfs("0s2")

    def test_overoccupied(self):
        with pytest.raises(ValueError, match="exceeds the Pauli capacity"):
            enumerate_jj_csfs("1s10")

    def test_count_j_resolved_csfs_validates(self):
        with pytest.raises(ValueError, match="invalid configuration token"):
            count_j_resolved_csfs("1s2 2x2")


class TestAtomEntryValidation:
    """Atom(...) rejects invalid configs at construction, before solve()."""

    def test_invalid_shell_letter_configs(self):
        with pytest.raises(ValueError, match="invalid configuration token '2x2'"):
            Atom(Z=4, configs=["1s2 2x2"])

    def test_incomplete_token_configs(self):
        with pytest.raises(ValueError, match="invalid configuration token '2s'"):
            Atom(Z=4, configs=["1s2 2s"])

    def test_trailing_garbage_configs(self):
        with pytest.raises(ValueError, match="invalid configuration token 'junk'"):
            Atom(Z=4, configs=["1s2 2s2 junk"])

    def test_n_zero_configs(self):
        with pytest.raises(ValueError, match="n must be >= 1"):
            Atom(Z=4, configs=["0s2"])

    def test_overoccupied_configs(self):
        with pytest.raises(ValueError, match="exceeds the Pauli capacity"):
            Atom(Z=4, configs=["1s10"])

    def test_single_config_kwarg_validated(self):
        with pytest.raises(ValueError, match="invalid configuration token"):
            Atom(Z=4, config="1s2 2x2")

    def test_invalid_config_among_valid_ones(self):
        # One bad config in a multi-config list must not slip through.
        with pytest.raises(ValueError, match="invalid configuration token"):
            Atom(Z=4, configs=["1s2 2s2", "1s2 2s2 junk"])


class TestValidConfigsUnchanged:
    """Legal physical inputs keep working exactly as before."""

    @pytest.mark.parametrize("config", [
        "1s2 2s2 2p6",
        "1s2 2s2 2p5 3d1",
        "10d3",          # multi-digit n
        "7i1",           # high-l letter (l up to 'o' supported)
        "2p6",           # shell at full Pauli capacity is legal
        "1s1",
    ])
    def test_valid_configs_parse(self, config):
        validate_config_string(config)
        assert _parse_config_string(config)
        assert enumerate_jj_csfs(config)

    def test_closed_shell_still_single_j0_csf(self):
        csfs = enumerate_jj_csfs("1s2 2s2 2p6")
        assert len(csfs) == 1
        assert csfs[0].total_J2 == 0
        assert csfs[0].parity == 1

    def test_valid_atom_constructs(self):
        atom = Atom(Z=4, configs=["1s2 2s2", "1s2 2s1 2p1"])
        assert atom is not None
