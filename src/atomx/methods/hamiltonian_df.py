"""Monoconfiguration Dirac–Fock with Hamiltonian intermediate coupling.

Guerra et al., *Relativistic calculations of screening parameters and atomic
radii of neutral atoms* (2016 revision), Eqs. (4)–(10): diagonalize H within
one nonrelativistic configuration, optimize its radial orbitals against
c†Hc, and report occupations sum_j |c_j|² n_ij. The shared variational
kernel retains two-body CSF interference; occupations alone do not define
the orbital equations. No excited configurations are generated here.
"""
from atomx.config.csf import enumerate_jj_csfs
from atomx.methods.csf.solver import VariationalCSFResult, VariationalCSFSolver


class HamiltonianDFResult(VariationalCSFResult):
    """DF intermediate-coupling states within one configuration."""

    @property
    def theory_level(self):
        from atomx.types import TheoryLevel
        return TheoryLevel.DIRAC_HARTREE_FOCK

    @property
    def method_name(self):
        return "DHF"


class HamiltonianDFSolver(VariationalCSFSolver):
    """Ground-state DF with variational mixing inside a single configuration.

    Exactly degenerate ground eigenstates are averaged equally. Radial
    variation currently uses Dirac–Coulomb; Breit, QED and recoil follow
    the explicitly recorded post-orbital correction stages of the shared
    state solver. These corrections are not claimed to be variational.
    """

    result_type = HamiltonianDFResult
    method_name = "DHF"
    initializer_sweeps = 1
    # Direct source equations follow the physical radial ordering. Equal
    # P/Q orders introduce a spurious positive-branch root for kappa > 0.
    # Keep the balanced representation also when the user specifies ns.
    initializer_kinetic_balance = True

    def __init__(self, Z, configuration, n_grid=1000, **kwargs):
        if not isinstance(configuration, str):
            raise TypeError("Hamiltonian DF requires one configuration string")
        if kwargs.get("radial_optimization", "mcdhf") != "mcdhf":
            raise ValueError("Hamiltonian DF requires variational orbital optimization")
        if kwargs.get("per_block_orbital_sets", False):
            raise ValueError("Hamiltonian DF uses one ground-state orbital set")
        if kwargs.get("shooting_space") is not None:
            raise ValueError("Hamiltonian DF uses direct radial equations; use Sampling.n_grid "
                             "for shooting refinement. ShootingSpace is an MCDHF control.")
        if kwargs.get("orbital_preconditioner", "spectral") != "spectral":
            raise ValueError("orbital_preconditioner controls the MCDHF optimizer, not direct DF")
        if "variational_projector" in kwargs:
            raise ValueError("Hamiltonian DF follows the self-consistent electron branch")
        self.configuration = configuration
        if kwargs.get("engine", "bspline") == "shooting":
            self.n_grid = n_grid
        super().__init__(Z=Z, csfs=enumerate_jj_csfs(configuration), **kwargs)
        self.variational_projector = "self_consistent"
        self.shooting_space = None

    def _solve_shooting_state(self, *, max_iter, tolerance):
        from atomx.methods.csf.direct_shooting import solve_direct_shooting
        return solve_direct_shooting(self, max_iter=max_iter, tolerance=tolerance,
                                     orbital_tolerance=self.orbital_tolerance)

    def _optimize_bspline_state(self, functional, orbitals_dict, dhf, dhf_result,
                                *, max_iter, tolerance, energy_tolerance):
        from atomx.methods.csf.direct_bspline import solve_direct_bspline
        return solve_direct_bspline(functional, orbitals_dict,
                                    max_iter=max_iter, tolerance=tolerance,
                                    orbital_tolerance=self.orbital_tolerance)

    def solve(self, **kwargs):
        from dataclasses import replace
        kwargs.setdefault("scf_max_iter", kwargs.get("max_iter", 30))
        result = super().solve(**kwargs)
        result.calculation_spec = replace(result.calculation_spec, state={
            **result.calculation_spec.state,
            "configuration": self.configuration,
            "objective": "ground_state_intermediate_coupling",
            "configuration_expansion": False,
        })
        return result
