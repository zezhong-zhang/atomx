"""Regression tests for central config-string validation (HYP-1778).

Invalid configuration strings (malformed shell tokens, ``n=0``,
over-occupied shells) are rejected at parse/entry time with a clear
``ValueError`` naming the token — previously they crashed deep in the
SCF/QED kernels with cryptic ``KeyError`` / ``ZeroDivisionError`` /
``max() arg is an empty sequence``.  Ported from PR #82, whose own
parser duplicate is not landed: the assertions here are
implementation-agnostic (token and capacity numbers in the message),
and implicit unit occupations (``"2s"`` == ``"2s1"``) are VALID per the
shared parser's reviewed semantics (review finding M2).
"""

import pytest

from atomx import Atom
from atomx.config.parsing import validate_config_string
from atomx.methods.mcdhf.config import (
    count_j_resolved_csfs,
    enumerate_jj_csfs,
)


class TestValidateConfigString:
    """Each rejection class raises a clear ValueError at parse time."""

    @pytest.mark.parametrize("bad", ["1s2 2x2", "1s2 2s2 junk", "2x6", "junk"])
    def test_malformed_token_names_token(self, bad):
        # Was: KeyError: 'x' from l_map[token[1]] deep in enumeration.
        with pytest.raises(ValueError) as excinfo:
            validate_config_string(bad)
        assert bad.split()[-1] in str(excinfo.value)

    def test_n_zero_rejected(self):
        # Was: ZeroDivisionError deep in the QED self-energy (mohr.py).
        with pytest.raises(ValueError, match="n must be >= 1"):
            validate_config_string("0s2")

    @pytest.mark.parametrize(
        "config_str, capacity",
        [("1s10", "2"), ("2p7", "6"), ("1s2 2s9", "2"), ("3d11", "10")],
    )
    def test_overoccupied_shell_names_capacity(self, config_str, capacity):
        # Was: ValueError: max() arg is an empty sequence in dirac_fock.py.
        with pytest.raises(ValueError) as excinfo:
            validate_config_string(config_str)
        msg = str(excinfo.value)
        assert config_str.split()[-1] in msg
        assert capacity in msg

    @pytest.mark.parametrize("good", ["1s2 2s2 2p6", "10s1", "11s", "7i1"])
    def test_valid_strings_pass(self, good):
        # Implicit unit occupation ("11s" == "11s1") is deliberate M2
        # behaviour — #82's regex rejected it; the shared parser accepts it.
        validate_config_string(good)

    def test_none_passes(self):
        validate_config_string(None)


class TestEnumerationPathsUseSharedValidation:
    """enumerate_jj_csfs / count_j_resolved_csfs share the validated parser."""

    def test_enumerate_multi_digit_n(self):
        # Was: KeyError: '0' on the single-char slice of "10s1".
        csfs = enumerate_jj_csfs("1s2 10s1")
        assert len(csfs) == 1

    def test_enumerate_malformed_names_token(self):
        with pytest.raises(ValueError) as excinfo:
            enumerate_jj_csfs("1s2 2x2")
        assert "2x2" in str(excinfo.value)

    def test_count_j_resolved_multi_digit_n_matches_low_n(self):
        # count_j_resolved_csfs kept raw single-char slicing after the
        # audit migrated enumerate_jj_csfs; it now shares the parser.
        assert count_j_resolved_csfs("1s2 10p1") == count_j_resolved_csfs("1s2 2p1")

    def test_count_j_resolved_overoccupied_rejected(self):
        with pytest.raises(ValueError) as excinfo:
            count_j_resolved_csfs("2p7")
        assert "2p7" in str(excinfo.value)


class TestAtomEntryValidation:
    """Atom.__init__ rejects invalid config strings (fail fast, HYP-1778)."""

    @pytest.mark.parametrize(
        "bad", ["1s2 2x6", "0s2", "1s2 2s9", "1s2 2s2 junk"]
    )
    def test_invalid_config_rejected_at_entry(self, bad):
        with pytest.raises(ValueError) as excinfo:
            Atom(Z=8, config=bad)
        assert bad.split()[-1] in str(excinfo.value)

    def test_valid_config_unaffected(self):
        atom = Atom(Z=8, config="1s2 2s2 2p6")
        assert atom._config_strings == ["1s2 2s2 2p6"]

    def test_implicit_occupation_unaffected(self):
        atom = Atom(Z=11, config="1s2 2s2 2p6 11s")
        assert atom._config_strings == ["1s2 2s2 2p6 11s"]
