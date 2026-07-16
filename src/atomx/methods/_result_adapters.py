"""Shared input adapters for method-layer entry points.

Home for small converters that coerce high-level user-facing objects
(:class:`~atomx.Atom`) into the method-layer result types they wrap.
"""

from __future__ import annotations


def to_scf_result(atom_or_result, *, caller: str):
    """Coerce an :class:`~atomx.Atom` or ``SCFResult`` to an SCFResult.

    ``Atom`` is matched by duck-typing (it carries an ``_result``
    attribute, ``None`` until ``.solve()`` runs) so that this module
    never imports :mod:`atomx.atom` and stays free of circular imports.

    Parameters
    ----------
    atom_or_result
        A (possibly unsolved) ``atomx.Atom`` or a converged ``SCFResult``.
    caller
        Name of the public entry point, used in error messages
        (e.g. ``"compute_pnc()"``).
    """
    # atomx.Atom high-level wrapper: solve on demand.
    if hasattr(atom_or_result, "_result"):
        if atom_or_result._result is None:
            if not hasattr(atom_or_result, "solve"):
                raise TypeError(
                    f"{caller} expects a solved atomx.Atom or an "
                    f"SCFResult, got {type(atom_or_result).__name__!r}."
                )
            atom_or_result.solve()
        return atom_or_result._result
    # Bare converged SCFResult.
    if hasattr(atom_or_result, "converged") and hasattr(atom_or_result, "_bsp_data"):
        return atom_or_result
    raise TypeError(
        f"{caller} expects a solved atomx.Atom or an SCFResult, "
        f"got {type(atom_or_result).__name__!r}."
    )
