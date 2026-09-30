"""MCDHF entry point for the shared variational CSF equations."""
from atomx.methods.csf.solver import VariationalCSFResult, VariationalCSFSolver


class MCDHFResult(VariationalCSFResult):
    """Results for an explicitly expanded configuration space."""


class MCDHFSolver(VariationalCSFSolver):
    """Optimize amplitudes and orbitals in an explicit configuration space."""

    result_type = MCDHFResult
    method_name = "MCDHF"
