"""Declared tiny native atomic radial producer; complete multiplets only."""

from atomx.methods.dirac_fock import DFSolver, Subshell
from atomx.methods.mbpt import build_virtual_basis
from atomx.physics.csf.spinor_integrals import build_spinor_integrals


def build(core_electrons=4, virtual_s_count=2):
    core = DFSolver(Z=4, subshells=[Subshell(1, -1, 2)], ns=30, ks=7).solve()
    od = {(1, -1): core.orbitals_by_subshell()[1, -1][:2]}
    for kappa in (-1, 1, -2):
        _, states = build_virtual_basis(
            core, kappa, e_max=50, potential_mode="VN", orthogonal_to_occupied=True
        )
        for index, state in enumerate(states[: virtual_s_count if kappa == -1 else 1]):
            od[index + 2, kappa] = state[:2]
    all_virtuals = [key for key in od if key != (1, -1)]
    common = dict(nuclear_potential=core._nuclear_potential, representation="packed")
    dc = build_spinor_integrals(
        od,
        core._bsp_data,
        4,
        30,
        occupied_keys=[(1, -1)],
        virtual_keys=all_virtuals,
        **common,
    )
    frequencies = {
        label[:2]: float(dc.fock_Ha[i, i].real) for i, label in enumerate(dc.labels)
    }
    occupied = [(1, -1)] if core_electrons == 2 else [(1, -1), (2, -1)]
    virtual = [key for key in od if key not in occupied]
    integrals = build_spinor_integrals(
        od,
        core._bsp_data,
        4,
        30,
        occupied_keys=occupied,
        virtual_keys=virtual,
        include_breit=True,
        breit_orbital_energies=frequencies,
        **common,
    )
    return core, od, integrals
