"""Theory-ladder abstraction for AtomX.

A :class:`Ladder` is an ordered sequence of :class:`atomx.methods.Method`
rungs that an :class:`atomx.Atom` can be solved against, one rung at a
time, accumulating a :class:`LadderRun` with per-rung outcomes.

Quick start::

    from atomx import Atom, Ladder

    ne = Atom(Z=10)
    run = ne.climb(Ladder.standard())
    run.show(unit="eV")

The standard ladder spans non-relativistic Hartree-Fock-Slater up to
Dirac-Fock + Breit + QED; pass ``through=DiracFock(breit=True)`` to stop
early, or hand a custom list to ``Ladder([...])``.

Failures (e.g. perturbative Breit-Pauli on a single-electron system)
are recorded on the :class:`RungOutcome` rather than raised — the
rest of the ladder still runs.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

from atomx.constants import HARTREE_CM1, HARTREE_EV
from atomx.methods.base import Method
from atomx.methods.dirac_fock import DiracFock
from atomx.methods.hartree_fock_slater import HartreeFockSlater
from atomx.methods.hydrogenic import HydrogenicDirac, HydrogenicSchrodinger
from atomx.methods.breit_pauli import BreitPauli
from atomx.types import Exchange, TheoryLevel

# ---------------------------------------------------------------------------
# Outcome objects
# ---------------------------------------------------------------------------

@dataclass
class RungOutcome:
    """Result of running a single rung against an :class:`atomx.Atom`."""

    rung: Method
    name: str
    theory_level: TheoryLevel
    result: Optional[object] = None
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        """True iff this rung produced a usable result."""
        return self.result is not None and self.error is None

    @property
    def total_energy(self) -> Optional[float]:
        """Total energy in Hartree, or ``None`` if the rung failed."""
        if not self.ok:
            return None
        return float(self.result.total_energy)


@dataclass
class LadderRun:
    """Outcome of climbing a :class:`Ladder` against an atom.

    Iterating yields :class:`RungOutcome` objects in ladder order.
    """

    outcomes: List[RungOutcome] = field(default_factory=list)

    def __iter__(self):
        return iter(self.outcomes)

    def __len__(self):
        return len(self.outcomes)

    def __getitem__(self, key):
        return self.outcomes[key]

    @property
    def results(self) -> List[object]:
        """Successful results, in ladder order.  Failed rungs are skipped."""
        return [o.result for o in self.outcomes if o.ok]

    @property
    def failed(self) -> List[RungOutcome]:
        """Failed outcomes, in ladder order."""
        return [o for o in self.outcomes if not o.ok]

    def to_table(self, unit: str = "Ha") -> List[dict]:
        """Per-rung dict rows suitable for :func:`pandas.DataFrame`.

        Each row carries ``rung``, ``theory_level``, ``ok``,
        ``E_total(<unit>)``, ``delta_prev``, and (if the rung failed)
        ``error``.
        """
        conv = _conv(unit)
        rows: List[dict] = []
        prev_E: Optional[float] = None
        e_key = f"E_total({unit})"
        for o in self.outcomes:
            row: dict = {
                "rung": o.name,
                "theory_level": o.theory_level.name,
                "ok": o.ok,
            }
            if o.ok:
                E = o.total_energy * conv
                row[e_key] = E
                row["delta_prev"] = (E - prev_E) if prev_E is not None else None
                prev_E = E
            else:
                row[e_key] = None
                row["delta_prev"] = None
                row["error"] = o.error
            rows.append(row)
        return rows

    def show(self, unit: str = "eV", file=None) -> None:
        """Pretty-print the ladder with adjacent diffs.

        Failed rungs are rendered with ``(failed)`` and a one-line error
        instead of an energy.
        """
        out = file or sys.stdout
        if not self.outcomes:
            out.write("(empty ladder)\n")
            return
        conv = _conv(unit)
        name_w = max(20, max(len(o.name) for o in self.outcomes))
        e_w = 14
        d_w = 14
        header = (
            f"  {'rung':<{name_w}}"
            f"  {('E_total (' + unit + ')'):>{e_w}}"
            f"  {'Δ from prev':>{d_w}}"
            f"  what this rung adds"
        )
        out.write(header + "\n")
        out.write("  " + "-" * (len(header) - 2) + "\n")
        prev: Optional[RungOutcome] = None
        for o in self.outcomes:
            if o.ok:
                E = o.total_energy * conv
                if prev is not None and prev.ok:
                    dE = (o.total_energy - prev.total_energy) * conv
                    delta_str = f"{dE:>+{d_w}.4f}"
                    adds = _adds_over(o.rung, prev.rung)
                else:
                    delta_str = f"{'—':>{d_w}}"
                    adds = "—"
                out.write(
                    f"  {o.name:<{name_w}}"
                    f"  {E:>{e_w}.4f}"
                    f"  {delta_str}"
                    f"  {adds}\n"
                )
                prev = o
            else:
                out.write(
                    f"  {o.name:<{name_w}}"
                    f"  {'(failed)':>{e_w}}"
                    f"  {'':>{d_w}}"
                    f"  {o.error}\n"
                )


# ---------------------------------------------------------------------------
# Ladder
# ---------------------------------------------------------------------------

@dataclass
class Ladder:
    """Ordered sequence of theory rungs to solve an atom against.

    Use the named factories for canonical progressions, or pass a
    custom list of :class:`atomx.methods.Method` objects.
    """

    rungs: Sequence[Method]
    name: str = "custom"

    def __iter__(self):
        return iter(self.rungs)

    def __len__(self):
        return len(self.rungs)

    def __getitem__(self, key):
        return self.rungs[key]

    @classmethod
    def standard(cls) -> "Ladder":
        """Canonical progression from Schrödinger to Dirac-Fock + QED.

        Each step adds one well-defined physical effect on top of the
        previous one (with the caveat that ``HartreeFockSlater`` here is
        currently HF-Slater under the hood — see issue HYP-13).
        """
        return cls(
            rungs=[
                HydrogenicSchrodinger(),
                HydrogenicDirac(),
                HartreeFockSlater(),
                BreitPauli(),
                DiracFock(),
                DiracFock(breit=True),
                DiracFock(breit=True, qed=True),
            ],
            name="standard",
        )

    @classmethod
    def relativistic(cls) -> "Ladder":
        """Relativistic-only ladder — useful for heavy atoms where the
        non-relativistic rungs are not informative."""
        return cls(
            rungs=[
                HydrogenicDirac(),
                DiracFock(exchange=Exchange.SLATER),
                DiracFock(),
                DiracFock(breit=True),
                DiracFock(breit=True, qed=True),
            ],
            name="relativistic",
        )

    @classmethod
    def coulomb_only(cls) -> "Ladder":
        """Ladder that stops before Breit and QED — clean isolation of
        relativistic effects from electron-electron Breit retardation."""
        return cls(
            rungs=[
                HydrogenicSchrodinger(),
                HartreeFockSlater(),
                BreitPauli(),
                DiracFock(),
            ],
            name="coulomb_only",
        )

    def through(self, stop: Method) -> "Ladder":
        """Sub-ladder up to (and including) the rung matching ``stop``.

        Matching uses ``_rung_eq`` (class identity + the physics-defining
        kwargs ``breit``, ``qed``, ``exchange`` for ``DiracFock``).  If no
        rung in the ladder matches ``stop``, ``stop`` is appended.
        """
        kept: List[Method] = []
        for r in self.rungs:
            kept.append(r)
            if _rung_eq(r, stop):
                return Ladder(rungs=kept, name=f"{self.name}[:{_rung_label(stop)}]")
        kept.append(stop)
        return Ladder(rungs=kept, name=f"{self.name}+{_rung_label(stop)}")

    def run(self, atom, sampling=None) -> LadderRun:
        """Climb the ladder against ``atom``.

        Each rung is solved via ``atom.solve(rung, sampling=sampling)``;
        exceptions are recorded on the corresponding :class:`RungOutcome`
        rather than raised, so a single failure does not abort the
        whole climb.
        """
        outcomes: List[RungOutcome] = []
        for rung in self.rungs:
            try:
                result = atom.solve(rung, sampling=sampling)
                if result is None or not hasattr(result, "total_energy"):
                    # ``Atom.solve()`` (no method) returns ``self``; pull
                    # the actual result off the atom.
                    result = getattr(atom, "_result", None) or getattr(atom, "result", None)
                outcomes.append(RungOutcome(
                    rung=rung,
                    name=_rung_label(rung),
                    theory_level=rung.theory_level,
                    result=result,
                ))
            except Exception as e:  # noqa: BLE001 — recorded, not silenced
                outcomes.append(RungOutcome(
                    rung=rung,
                    name=_rung_label(rung),
                    theory_level=rung.theory_level,
                    error=f"{type(e).__name__}: {e}",
                ))
        return LadderRun(outcomes=outcomes)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _conv(unit: str) -> float:
    """Hartree-to-``unit`` conversion factor."""
    u = unit.lower()
    if u in ("ha", "hartree"):
        return 1.0
    if u == "ev":
        return HARTREE_EV
    if u in ("cm-1", "cm^-1", "wavenumber"):
        return HARTREE_CM1
    if u in ("ry", "rydberg"):
        return 2.0
    raise ValueError(f"Unknown energy unit {unit!r}")


def _rung_label(rung: Method) -> str:
    """Display name for a rung — short and unambiguous."""
    cls_name = rung.__class__.__name__
    if isinstance(rung, DiracFock):
        if rung.exchange == Exchange.SLATER:
            return "DiracFockSlater"
        extras: List[str] = []
        if rung.breit:
            extras.append("breit")
        if rung.qed:
            extras.append("qed")
        if extras:
            return f"DiracFock+{'+'.join(extras)}"
        return "DiracFock"
    return cls_name


def _rung_eq(a: Method, b: Method) -> bool:
    """Two rungs match iff same class + same physics-defining kwargs."""
    if type(a) is not type(b):
        return False
    if isinstance(a, DiracFock):
        return (
            a.breit == b.breit
            and a.qed == b.qed
            and a.exchange == b.exchange
        )
    return a.theory_level == b.theory_level


# Keyed by ``_rung_label(rung)`` of the *current* rung — describes what
# the current rung adds over a generic predecessor.  ``show`` falls back
# to the rung's own label when no entry exists.
_RUNG_CONTRIBUTION = {
    "HydrogenicSchrodinger": "exact 1-electron, no e–e interaction",
    "HydrogenicDirac": "fine structure (Sommerfeld)",
    "HartreeFockSlater": "mean-field e–e + (local Slater) exchange",
    "BreitPauli": "perturbative relativity (mass-vel + Darwin + L·S)",
    "DiracFockSlater": "variational relativity, local Slater exchange",
    "DiracFock": "variational relativity, non-local HF exchange",
    "DiracFock+breit": "Breit interaction (magnetic + retardation)",
    "DiracFock+qed": "QED corrections (Uehling VP + SE)",
    "DiracFock+breit+qed": "Breit + QED (Uehling VP + SE)",
    "MCDHF": "electron correlation via multi-configuration mixing",
}


def _adds_over(rung: Method, prev_rung: Method) -> str:
    """Plain-English string for what ``rung`` adds over ``prev_rung``."""
    label = _rung_label(rung)
    return _RUNG_CONTRIBUTION.get(label, label)


__all__ = ["Ladder", "LadderRun", "RungOutcome"]
