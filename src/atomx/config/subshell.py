"""Shared relativistic subshell identity, independent of radial solvers."""
from dataclasses import dataclass

from atomx.angular.recoupling import l_of_kappa, twoj_of_kappa

#: Spectroscopic letters indexed by orbital angular momentum ``l``
#: (``s, p, d, f, g, h, i, k, ...``).  Canonical alphabet for the whole
#: package: every label formatter uses it and
#: :data:`atomx.config.parsing.L_MAP` is built from it, so formatters and
#: parsers always agree.  Truncated copies (``"spdfgh"``,
#: ``"spdfghiklmno"``) are prefixes of this table and agree for the ``l``
#: values they reach; formatters fall back to ``f"l{l}"`` beyond it.
L_LETTERS = "spdfghiklmnoqrtuvwxyz"


def l_letter(l: int) -> str:
    """Spectroscopic letter for ``l``; ``f"l{l}"`` beyond :data:`L_LETTERS`."""
    return L_LETTERS[l] if 0 <= l < len(L_LETTERS) else f"l{l}"


@dataclass
class Subshell:
    """An atomic subshell (n, kappa) with occupation.

    Supports fractional occupations for averaged monoconfiguration mode
    (Guerra Eq 10: ϖᵢ = Σⱼ cⱼ² nᵢʲ).
    """
    n: int
    kappa: int
    occupation: float  # float to support fractional (Guerra averaged monoconfig)
    label: str = ""

    def __post_init__(self):
        l = l_of_kappa(self.kappa)
        j2 = twoj_of_kappa(self.kappa)
        max_occ = j2 + 1
        if self.occupation > max_occ + 0.01:
            raise ValueError(f"Occupation {self.occupation} exceeds max {max_occ} for kappa={self.kappa}")
        if not self.label:
            self.label = f"{self.n}{l_letter(l)}{j2}/2"

    @property
    def l(self):  # noqa: E743 — `l` is the orbital angular-momentum quantum
        # number and part of the public API (`Subshell.l`), as in the FAC/
        # DBSR_HF sources this module ports.
        return l_of_kappa(self.kappa)

    @property
    def j2(self):
        return twoj_of_kappa(self.kappa)
