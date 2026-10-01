"""Regression tests for atomx.config.jj_coupling.generate_jj_average (HYP-1785).

The parser previously split each token by single characters
(``n = int(token[0])``, ``l_char = token[1]``), which mis-parsed principal
quantum numbers n >= 10 (Rydberg configurations) and crashed with a cryptic
``KeyError``. These tests pin multi-digit-n parsing, clear validation
errors for malformed tokens, and unchanged behaviour for legacy inputs.

The implementation under test is the shared parser
(:func:`atomx.config.parsing.parse_nr_shells`, review finding M2), which
supersedes HYP-1785's inline regex: it additionally accepts an implicit
unit occupation (``"11s"`` == ``"11s1"``), so the "missing occupation"
case moved from the malformed list to its own test below.  The error
contract pinned here is implementation-agnostic — a ``ValueError`` that
names the offending token, not a ``KeyError``.
"""

import re

import pytest

from atomx.config.csf import _parse_config_string
from atomx.config.jj_coupling import generate_jj_average


def _avg_occ(subshells):
    return {(sh.n, sh.kappa): sh.occupation for sh in subshells}


def test_multi_digit_n_s_shell():
    """n >= 10 s-shell parses with correct n and occupation."""
    subshells, configs = generate_jj_average("10s1", averaging="statistical")
    assert _avg_occ(subshells) == {(10, -1): 1.0}
    assert len(configs) == 1
    assert configs[0].weight == pytest.approx(1.0)


def test_multi_digit_n_p_shell_statistical_average():
    """n >= 10 p^2 splits across j=1/2 (kappa=+1) and j=3/2 (kappa=-2).

    Integer distributions of q=2 over (2, 4) capacities: (0,2), (1,1),
    (2,0) with statistical weights C(2,qm)*C(4,qp) = 6, 8, 1, giving
    average occupations 2/3 and 4/3.
    """
    subshells, configs = generate_jj_average("11p2", averaging="statistical")
    occ = _avg_occ(subshells)
    assert occ[(11, 1)] == pytest.approx(2.0 / 3.0)
    assert occ[(11, -2)] == pytest.approx(4.0 / 3.0)
    assert sum(c.weight for c in configs) == pytest.approx(1.0)


def test_multi_digit_n_mixed_config():
    """Mixed configuration with multi-digit n parses every token."""
    subshells, _ = generate_jj_average("1s2 10s1 12d3", averaging="equal")
    occ = _avg_occ(subshells)
    assert occ[(1, -1)] == 2.0
    assert occ[(10, -1)] == 1.0
    # equal averaging of d^3 over (4, 6) capacities: distributions
    # (0,3),(1,2),(2,1),(3,0) each with weight 1 -> averages 1.5, 1.5.
    assert occ[(12, 2)] == pytest.approx(1.5)
    assert occ[(12, -3)] == pytest.approx(1.5)


def _nl(subshell):
    l = subshell.kappa if subshell.kappa > 0 else -subshell.kappa - 1
    return (subshell.n, l)


def test_parity_with_central_parser():
    """Parsed (n, l) shells agree with the central MCDHF parser.

    The two parsers split occupations between the j = l±1/2 subshells
    differently by design (central fills j- first; jj-average distributes
    statistically), so only the resolved non-relativistic shells match.
    """
    for config_str in ["10s1", "11p2", "1s2 2s2 2p2", "12d3"]:
        jj_subshells, _ = generate_jj_average(config_str)
        jj_nl = {_nl(sh) for sh in jj_subshells if sh.occupation > 0}
        central_nl = {
            _nl(sh) for sh in _parse_config_string(config_str) if sh.occupation > 0
        }
        assert jj_nl == central_nl


def test_legacy_config_unchanged():
    """Existing single-digit-n inputs keep their previous behaviour."""
    subshells, _ = generate_jj_average("1s2 2s2 2p2", averaging="statistical")
    occ = _avg_occ(subshells)
    assert occ[(1, -1)] == 2.0
    assert occ[(2, -1)] == 2.0
    assert occ[(2, 1)] == pytest.approx(2.0 / 3.0)
    assert occ[(2, -2)] == pytest.approx(4.0 / 3.0)


def test_implicit_unit_occupation():
    """A missing occupation means 1 (shared-parser M2 behaviour)."""
    subshells, _ = generate_jj_average("10s1 11s", averaging="statistical")
    assert _avg_occ(subshells) == {(10, -1): 1.0, (11, -1): 1.0}


@pytest.mark.parametrize(
    "bad_token",
    ["1s2x",     # trailing garbage
     "xs1",      # non-numeric n
     "10x2",     # unknown angular momentum letter
     "1S2",      # case-sensitive: uppercase not accepted
     "1s2.5",    # non-integer occupation
     ],
)
def test_malformed_token_raises_clear_value_error(bad_token):
    """Malformed tokens raise ValueError naming the token, not KeyError."""
    with pytest.raises(ValueError, match=re.escape(bad_token)):
        generate_jj_average(f"1s2 {bad_token}")


# ---------------------------------------------------------------------------
# Over-occupied shells (HYP-1790).  Ported from PR #86, whose code change is
# subsumed by the shared parser's capacity check: the contract pinned here
# is a ValueError naming the shell and its electron capacity — not the
# silent all-zeros occupation vector the distribution loop used to return
# (which even dropped the legal companion shells, and crashed Atom.solve
# with a cryptic IndexError downstream).
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "config_str",
    ["2p7",        # one electron over the p-shell capacity
     "1s2 2p9",    # HYP-1790 headline: silently dropped ALL electrons before
     "3d11",       # one electron over the d-shell capacity
     "2s3",        # s-shell over-occupation also gets a clear message
     ],
)
def test_over_occupied_shell_raises_clear_value_error(config_str):
    """Over-occupied shells raise ValueError naming the shell and bounds."""
    with pytest.raises(ValueError) as excinfo:
        generate_jj_average(config_str)
    msg = str(excinfo.value)
    bad_token = config_str.split()[-1]
    assert bad_token in msg
    # the capacity number appears in the message (6 for p, 10 for d, 2 for s)
    capacity = {"2p7": "6", "1s2 2p9": "6", "3d11": "10", "2s3": "2"}[config_str]
    assert capacity in msg


def test_atom_solve_rejects_over_occupied_config():
    """Atom surfaces the validation error, not a cryptic IndexError.

    Since HYP-1778 the rejection fires at ``Atom(...)`` construction
    (central entry validation), before any solver machinery runs.
    """
    from atomx import Atom

    with pytest.raises(ValueError, match="2p9"):
        Atom(Z=8, config="1s2 2p9")


@pytest.mark.parametrize(
    "config_str",
    ["2p6",        # closed p shell: exactly at capacity
     "3d10",       # closed d shell
     "1s2 2s2 2p2",
     ],
)
def test_shell_at_capacity_still_accepted(config_str):
    """Shells filled exactly to capacity remain valid."""
    subshells, _ = generate_jj_average(config_str)
    assert sum(sh.occupation for sh in subshells) > 0
