"""Probe: is the assembler's one-body Brillouin-consistent with the DF solve?
<2s|h|4s> should vanish if h carries the same mean field the orbitals
were optimized in."""
import re
from atomx.methods.dirac_fock import DFSolver, Subshell
import atomx.physics.csf.matrix as MAT

extra = [(2, 1), (2, -2), (3, -1), (3, 1), (3, -2), (3, 2), (3, -3),
         (4, -1), (4, 1), (4, -2), (4, 2), (4, -3), (4, -4),
         (5, -1), (5, 1), (5, -2), (5, 2), (5, -3), (5, -4)]
dhf = DFSolver(Z=4, subshells=[Subshell(n=1, kappa=-1, occupation=2.0),
                                Subshell(n=2, kappa=-1, occupation=2.0)] +
               [Subshell(n=nk[0], kappa=nk[1], occupation=1e-6)
                for nk in extra], ns=50, ks=9, starting_potential="tfda")
r = dhf.solve()
od = {(n, k): (P, Q) for (n, k), (P, Q, _o) in r.orbitals_by_subshell().items()}
for a, b in [((2, -1), (4, -1)), ((2, -1), (3, -1)), ((2, -1), (5, -1)),
             ((2, -1), (2, -1)), ((4, -1), (4, -1))]:
    Pa, Qa = od[a]; Pb, Qb = od[b]
    h = MAT._h_one_body(Pa, Qa, Pb, Qb, a[1], 4, dhf.bsp_data, 50)
    n = MAT._norm_sq(Pb, Qb, b[1], 4, dhf.bsp_data, 50)
    print(f"<{a}|h|{b}> = {h / n:+.6e} Ha")
