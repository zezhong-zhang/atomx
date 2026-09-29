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
        An explicitly solved ``atomx.Atom`` or a converged ``SCFResult``.
    caller
        Name of the public entry point, used in error messages
        (e.g. ``"compute_pnc()"``).
    """
    # Coercion never starts a calculation or chooses its physical model.
    if hasattr(atom_or_result, "_result"):
        if atom_or_result._result is None:
            raise ValueError(f"{caller} requires a solved Atom; call .solve() explicitly first.")
        atom_or_result = atom_or_result._result
    if hasattr(atom_or_result, "converged") and not atom_or_result.converged:
        raise ValueError(f"{caller} requires a converged result.")
    # Bare converged SCFResult.
    if hasattr(atom_or_result, "converged") and hasattr(atom_or_result, "_bsp_data"):
        return atom_or_result
    raise TypeError(
        f"{caller} expects a solved atomx.Atom or an SCFResult, "
        f"got {type(atom_or_result).__name__!r}."
    )
