"""Shared relativistic subshell identity, independent of radial solvers."""
from dataclasses import dataclass


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
        l = self.kappa if self.kappa > 0 else -self.kappa - 1
        j2 = 2 * abs(self.kappa) - 1
        max_occ = j2 + 1
        if self.occupation > max_occ + 0.01:
            raise ValueError(f"Occupation {self.occupation} exceeds max {max_occ} for kappa={self.kappa}")
        if not self.label:
            l_char = "spdfghiklmnoqrtuvwxyz"[l] if l < 21 else f"l{l}"
            self.label = f"{self.n}{l_char}{j2}/2"

    @property
    def l(self):  # noqa: E743 — `l` is the orbital angular-momentum quantum
        # number and part of the public API (`Subshell.l`), as in the FAC/
        # DBSR_HF sources this module ports.
        return self.kappa if self.kappa > 0 else -self.kappa - 1

    @property
    def j2(self):
        return 2 * abs(self.kappa) - 1
