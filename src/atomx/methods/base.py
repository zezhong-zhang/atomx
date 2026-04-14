"""Base class for theory-level methods."""

from __future__ import annotations

from dataclasses import dataclass

from atomx.types import NuclearModel, TheoryLevel


@dataclass
class Method:
    """Base class for all atomic theory methods.

    Subclasses define the physics level of a calculation.
    Pass a method object to ``Atom.solve(method)`` to run the calculation.

    Parameters
    ----------
    nuclear : NuclearModel
        Nuclear charge distribution model.  Default: Fermi distribution.
    """
    nuclear: NuclearModel = NuclearModel.FERMI

    @property
    def theory_level(self) -> TheoryLevel:
        raise NotImplementedError

    @property
    def name(self) -> str:
        return self.__class__.__name__
