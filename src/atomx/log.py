"""Calculation logging for AtomX.

Generates a human-readable log file for each calculation, recording
the physics setup, SCF convergence history, and final results.

Usage:
    from atomx.log import CalculationLog

    log = CalculationLog("ne_dhf.log")
    log.header(Z=10, config="1s2 2s2 2p6", method="DiracFock")
    log.scf_iteration(1, energy=-128.5, delta_E=1.2, residual=0.03)
    ...
    log.converged(result)
    log.close()

Or use the high-level API — ``Atom.solve()`` writes a log automatically
when ``log=True`` or ``log="path/to/file.log"`` is passed.
"""

from __future__ import annotations

import datetime
from numbers import Real
from typing import Optional

from atomx.constants import HARTREE_CM1, HARTREE_EV


class CalculationLog:
    """Write a structured calculation log file.

    The log is designed to be read by physicists: it records the physical
    setup, basis set parameters, SCF convergence, and final orbital energies
    in a format familiar from GRASP, MCDFGME, and DBSR_HF output.
    """

    def __init__(self, path: Optional[str] = None, verbose: bool = True):
        """
        Parameters
        ----------
        path : str or None
            Log file path.  If None, log to stdout only.
        verbose : bool
            If True, also print log lines to stdout.
        """
        self._path = path
        # The handle deliberately outlives this call (written in _write,
        # closed in close()); a `with` block would close it immediately.
        self._file = open(path, "w") if path else None  # noqa: SIM115
        self._verbose = verbose
        self._lines = []

    def _write(self, line: str = ""):
        self._lines.append(line)
        if self._file:
            self._file.write(line + "\n")
            self._file.flush()
        if self._verbose:
            print(line)  # noqa: T201

    def header(self, Z: int, config: str, method: str, **params):
        """Write calculation header."""
        self._write("=" * 70)
        self._write("  AtomX Calculation Log")
        self._write(f"  {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self._write("=" * 70)
        self._write(f"  Element:       Z = {Z}")
        self._write(f"  Configuration: {config}")
        self._write(f"  Method:        {method}")
        for key, val in params.items():
            self._write(f"  {key:15s}: {val}")
        self._write("-" * 70)

    def basis_info(self, bsp_data: dict):
        """Log basis set information."""
        ks = bsp_data.get("ks", "?")
        nv = bsp_data.get("nv", "?")
        nsp = bsp_data.get("nsp", "?")
        nsq = bsp_data.get("nsq", "?")
        ksp = bsp_data.get("ksp", ks)
        r_min = bsp_data["gr"][0, 0] if "gr" in bsp_data else "?"
        r_max = bsp_data["gr"][-1, -1] if "gr" in bsp_data else "?"
        kinetic_balance = ksp != ks
        self._write("  B-spline basis:")
        self._write(f"    Order:     ksp={ksp} (large), ksq={ks} (small)")
        self._write(f"    Splines:   nsp={nsp} (large), nsq={nsq} (small)")
        self._write(f"    Intervals: nv={nv}")
        self._write(f"    Range:     [{r_min:.2e}, {r_max:.1f}] Bohr")
        if kinetic_balance:
            self._write("    Kinetic balance: Yes (ksp=ks-1 eliminates spurious states)")
        else:
            self._write("    Kinetic balance: No (same order for P and Q)")
        self._write("-" * 70)

    def scf_start(self):
        """Log SCF iteration header."""
        self._write(f"  {'Iter':>4s}  {'Total Energy (Ha)':>20s}  "
                     f"{'Delta E (Ha)':>14s}  {'Max dP':>10s}")
        self._write(f"  {'----':>4s}  {'------------------':>20s}  "
                     f"{'------------':>14s}  {'------':>10s}")

    def scf_iteration(self, iteration: int, energy: float,
                      delta_E: float = 0.0, max_dp: float = 0.0):
        """Log one SCF iteration."""
        self._write(f"  {iteration:4d}  {energy:20.12f}  "
                     f"{delta_E:14.2e}  {max_dp:10.2e}")

    def converged(self, result):
        """Log convergence summary and final results."""
        self._write("-" * 70)
        if result.converged:
            self._write(f"  SCF CONVERGED in {result.iterations} iterations")
        else:
            self._write(f"  *** SCF NOT CONVERGED after {result.iterations} "
                         f"iterations ***")
        self._write()
        self._write(f"  Total energy:  {result.total_energy:20.12f} Ha")
        self._write(f"                 {result.total_energy * HARTREE_EV:20.6f} eV")
        self._write()

        # Energy breakdown
        bd = getattr(result, 'energy_breakdown',
                     getattr(result, 'energy_components', {}))
        if bd:
            self._write("  Energy components:")
            for key, val in sorted(bd.items()):
                if isinstance(val, Real) and abs(val) > 1e-15:
                    self._write(f"    {key:20s}: {val:18.10f} Ha"
                                 f"  ({val * HARTREE_CM1:14.2f} cm-1)")
                elif not isinstance(val, Real):
                    self._write(f"    {key:20s}: {val}")
            self._write()

        # Orbital energies
        self._write("  Orbital energies:")
        self._write(f"    {'Orbital':>10s}  {'Energy (Ha)':>16s}  "
                     f"{'Energy (eV)':>14s}  {'Energy (cm-1)':>14s}")
        for label, E in sorted(result.orbital_energies.items(),
                                key=lambda x: x[1]):
            self._write(f"    {label:>10s}  {E:16.10f}  "
                         f"{E * HARTREE_EV:14.4f}  {E * HARTREE_CM1:14.1f}")
        self._write("=" * 70)

    def message(self, msg: str):
        """Write a free-form message."""
        self._write(f"  {msg}")

    def close(self):
        """Close the log file."""
        if self._file:
            self._file.close()
            self._file = None
