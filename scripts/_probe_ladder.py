import re

from atomx.methods.dirac_fock import DFSolver, Subshell
from atomx.methods.mbpt_sigma2 import build_sigma2_ladder


def kap(l):
    m = re.fullmatch(r"(\d+)([spdfgh])(\d+)/(\d+)", l)
    n = int(m.group(1))
    ll = "spdfgh".index(m.group(2))
    num = int(m.group(3))
    den = int(m.group(4))
    j2 = int(round(2 * num / den))
    return n, (-(ll + 1) if j2 == 2 * ll + 1 else ll)


def build():
    extra = [(2, 1), (2, -2), (3, -1), (3, 1), (3, -2), (3, 2), (3, -3),
              (4, -1), (4, 1), (4, -2), (4, 2), (4, -3), (4, -4),
              (5, -1), (5, 1), (5, -2), (5, 2), (5, -3), (5, -4)]
    subs = [Subshell(n=1, kappa=-1, occupation=2.0),
            Subshell(n=2, kappa=-1, occupation=2.0)]
    subs += [Subshell(n=nk[0], kappa=nk[1], occupation=1e-6)
             for nk in extra]
    dhf = DFSolver(Z=4, subshells=subs, ns=50, ks=9,
                   starting_potential="tfda")
    r = dhf.solve()
    od = {(n, k): (P, Q) for (n, k), (P, Q, _o) in
          r.orbitals_by_subshell().items()}
    eps = {kap(l): e for l, e in r.orbital_energies.items()}
    return dhf, r, od, eps


if __name__ == "__main__":
    dhf, r, od, eps = build()
    VAL = [k for v in ["2s", "2p", "3s", "3p", "3d"]
           for k in [kap(v + "1/2"), kap(v + "3/2")] if k in od]
    Q = [k for v in ["4s", "4p", "4d", "4f", "5s", "5p", "5d", "5f"]
         for k in [kap(v + "1/2"), kap(v + "3/2")] if k in od and k in eps]
    from atomx.methods.mbpt_sigma2 import build_sigma2_ladder
    lad = build_sigma2_ladder(od, dhf.bsp_data, 4, 50,
                               virt_keys=Q, valence_keys=VAL, eps=eps)
    up, dn = lad.index[(2, -1, 1)], lad.index[(2, -1, -1)]
    print("VAL:", sorted(set(VAL)))
    print("Q:", sorted(set(Q)))
    print("spinors[up]:", lad.spinors[up], "spinors[dn]:", lad.spinors[dn])
    print("delta:", lad(up, dn, up, dn))
    # instrument: monkeypatch-free manual walk of the same math
    # (check bucket 0 pair list and G1 row norm by re-deriving)
    b0 = [(mi, ni) for mi in range(len(lad.spinors))
           for ni in range(len(lad.spinors))
           if lad.spinors[mi][:2] in [k for k in Q]
           and lad.spinors[ni][:2] in [k for k in Q]
           and lad.spinors[mi][2] + lad.spinors[ni][2] == 0
           and mi != ni]
    print("manual bucket-0 pair count:", len(b0))
