"""F5 verdict: does from_atom still map emission backwards / set g=1?"""
import sys
import numpy as np
from types import SimpleNamespace

from atomx.constants import HARTREE_EV
from atomx.plasma.crm import CollisionalRadiativeModel


class StubCSF:
    def __init__(self, two_j):
        self.total_J2 = two_j


def make_stub_atom_ci():
    # 3 CI states: E = 0, 1, 2 Ha ; J = 0, 1, 2  -> g = 1, 3, 5
    # transition rows use initial=LOWER, final=UPPER (the convention the
    # old review reported); we also add one row with the opposite order.
    result = SimpleNamespace(
        mixing_coefficients=np.eye(3),
        csfs=[StubCSF(0), StubCSF(2), StubCSF(4)],
    )
    trs = [
        SimpleNamespace(initial_label="State_0", final_label="State_1",
                        A_rate_per_s=1.0e9, gf_value=0.5),
        SimpleNamespace(initial_label="State_2", final_label="State_1",
                        A_rate_per_s=2.0e9, gf_value=0.7),  # reversed order
    ]
    atom = SimpleNamespace(
        _csf_result=result,
        state_energies=[0.0, 1.0, 2.0],
        transitions=lambda multipoles: trs,
        Z=2, charge=0, _config_strings=["1s2", "1s1 2p1", "1s1 3d1"],
    )
    return atom


def main():
    atom = make_stub_atom_ci()
    crm = CollisionalRadiativeModel.from_atom(atom)
    print("levels:", [(l.label, l.energy_eV, l.g) for l in crm.levels])
    print("transitions:", [(t.upper, t.lower, t.A_rate) for t in crm.transitions])

    # checks
    gs = [l.g for l in crm.levels]
    assert gs == [1, 3, 5], f"g wrong: {gs}"
    for t in crm.transitions:
        eu, el = crm.levels[t.upper].energy_eV, crm.levels[t.lower].energy_eV
        assert eu > el, f"upper/lower not energy-ordered: {t}"

    R = crm._build_rate_matrix(0.0, 10.0)
    # ground-state column: no outgoing rate at ne=0
    outgoing_ground = R[1:, 0]
    print("outgoing rates from ground at ne=0:", outgoing_ground)
    assert np.all(outgoing_ground == 0.0), "ground state decays spontaneously!"

    pop = crm.steady_state(0.0, 10.0)
    print("steady state at ne=0:", pop)
    assert pop[0] == 1.0

    # orbital branch
    shells = [SimpleNamespace(label="1s", kappa=-1),
              SimpleNamespace(label="2p", kappa=-2)]
    atom2 = SimpleNamespace(
        _csf_result=None,
        _result=SimpleNamespace(subshells=shells),
        orbital_energies={"1s": -2.0, "2p": -1.0},
        transitions=lambda mp: [],
        Z=4, charge=2, _config_strings=None,
    )
    crm2 = CollisionalRadiativeModel.from_atom(atom2)
    print("orbital levels:", [(l.label, l.g) for l in crm2.levels])
    assert [l.g for l in crm2.levels] == [2, 4]
    print("P1-fromatom: F5 FIXED (energy-sorted direction, g=2J+1 / 2|kappa|)")


if __name__ == "__main__":
    main()
