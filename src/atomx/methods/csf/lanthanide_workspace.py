"""Frozen [Xe] statistical-4f workspace shared by validation drivers."""

Xe = [
    (1, -1),
    (2, -1),
    (2, 1),
    (2, -2),
    (3, -1),
    (3, 1),
    (3, -2),
    (3, 2),
    (3, -3),
    (4, -1),
    (4, 1),
    (4, -2),
    (4, 2),
    (4, -3),
    (5, -1),
    (5, 1),
    (5, -2),
]


def build_lanthanide_workspace(Z, nval, ns=50):
    from atomx.config.csf import enumerate_jj_csfs_many
    from atomx.methods.dirac_fock import DFSolver, Subshell
    from atomx.methods.mcdhf.excitations import generate_sd_excitations
    from atomx.physics.csf.sparse_ci import patterns_from_csfs

    def cap(k):
        return 2 * k if k > 0 else -2 * k

    subs = [Subshell(n=n, kappa=k, occupation=float(cap(k))) for n, k in Xe]
    subs.append(Subshell(n=4, kappa=3, occupation=nval * 6 / 14))
    subs.append(Subshell(n=4, kappa=-4, occupation=nval * 8 / 14))
    dhf = DFSolver(Z=Z, subshells=subs, ns=ns, ks=9, starting_potential="tfda")
    r = dhf.solve()
    od = {(n, k): (P, Q) for (n, k), (P, Q, _o) in r.orbitals_by_subshell().items()}
    REF = "1s2 2s2 2p6 3s2 3p6 3d10 4s2 4p6 4d10 5s2 5p6 4f" + str(nval)
    cfgs = generate_sd_excitations(REF, ["4f"], core_orbitals=None, max_excitations=0)
    pats = patterns_from_csfs(enumerate_jj_csfs_many(cfgs))
    pats = [p for p in pats if all((n, k) in od for (n, k, _q) in p)]
    return od, dhf.bsp_data, pats
