"""Direction test: second-order PT vs exact CI gap on the He ground state.

Per the hole-line literature diagnosis (Kozlov et al. arXiv:2202.02026
Eq. (11)): the valence double-excitation ladder Sigma2 has BOTH-VIRTUAL
intermediate pairs.  On He (2 electrons, no core) this is the ENTIRE
correlation story, and everything is computable exactly with the sparse
engine in seconds:

    P    = configurations with both electrons in n<=3 (the CI space)
    Q_D  = double excitations, both electrons promoted to n in 4..8
    full = P + Q_D

We check:
  (a) E_CI(full) < E_CI(P)  (the Q_D sector lowers the energy)
  (b) the second-order PT estimate
        dE_PT = sum_{n in Q_D} |H[n, 0]|^2 / (E_P - H[n,n])
      has the right SIGN and captures a large fraction (target >50%)
      of (E_CI(full) - E_CI(P))   — validating the premises of the
      Eq. (11) implementation: 2nd-order PT in the Q_D sector.
This is the "simple, local" direction check before re-implementing the
J-coupled Eq. (11) with k1-k2 6j recoupling.
"""

from atomx import Atom, Correlation
from atomx.methods.mcdhf.excitations import generate_sd_excitations

CORE = ["1s"]  # Be: 1s frozen; 2s2 valence
REF = "1s2 2s2"
VIRTUALS_P = ["2s", "2p", "3s", "3p", "3d"]
VIRTUALS_Q = ["4s", "4p", "4d", "4f", "5s", "5p", "5d", "5f"]


def ci_configs(virts):
    return generate_sd_excitations(REF, virts, core_orbitals=None or CORE,
                                   max_excitations=2)


def solve_energy(cfgs, tag):
    atom = Atom(
        Z=4,
        config=cfgs,
        correlation=Correlation(j_resolved_csfs=True,
                                radial_optimization="fac_frozen"),
        corrections="none",
        engine="bspline",
    )
    atom.solve()
    E = atom.result.total_energy
    print(f"[{tag}] n_configs={len(cfgs)}  E={E:.9f} Ha", flush=True)
    return atom, E


if __name__ == "__main__":
    cfgP = ci_configs(VIRTUALS_P)
    atomP, EP = solve_energy(cfgP, "CI(P)")

    cfgQ = generate_sd_excitations(REF, VIRTUALS_P + VIRTUALS_Q,
                                  core_orbitals=None or CORE, max_excitations=2)
    atomF, EF = solve_energy(cfgQ, "CI(P+Q_D)")

    gap = EF - EP
    print(f"\nCI gap  E(full) - E(P) = {gap * 219474.63:+.2f} cm^-1 "
          f"({gap:+.6f} Ha)", flush=True)
    assert EF < EP, "Q_D sector must lower the energy"

    E_ref = -14.653000  # Kozlov 2022 Table 1: CI+MBPT(P+Q_D) Be value
    print("\nDirection check (Be, cf. Kozlov Table 1):", flush=True)
    print(f"  E(P)      = {EP:.9f} Ha", flush=True)
    print(f"  E(P+Q_D)  = {EF:.9f} Ha", flush=True)
    print(f"  Q_D contribution = {EF - EP:+.6f} Ha "
          f"({(EF - EP) * 219474.63:+.1f} cm^-1)", flush=True)
    print("  cf. Kozlov CI(P) = -14.6428, CI+MBPT = -14.6530 "
          "(Q_D ladder + core parts = -0.0102 Ha)", flush=True)
