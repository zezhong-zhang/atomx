"""Fixed-Hamiltonian SCF budget control for the nightly Na EDM fixture."""
import json

from atomx.methods.dirac_fock import DiracFock, Exchange, Subshell

shells = [Subshell(1, -1, 2), Subshell(2, -1, 2), Subshell(2, 1, 2),
          Subshell(2, -2, 4), Subshell(3, -1, 1)]
for budget in (40, 200):
    result = DiracFock(exchange=Exchange.HARTREE_FOCK).solve(
        Z=11, subshells=shells, max_iter=budget)
    print(json.dumps(dict(max_iter=budget, converged=bool(result.converged),
                          iterations=result.iterations, residual=float(result.residual),
                          energy=float(result.total_energy),
                          overlap=result.energy_components["max_same_kappa_overlap"])),
          flush=True)
