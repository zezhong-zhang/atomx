"""Executable diagnostic probes for the 2026-09-23 physics/numerics review.

Run each mode in a fresh process from the repository root. Some modes replace
functions in memory to isolate an algebraic error or measure a cache; they do
not edit library files. See the companion report for interpretation and scope.
"""

import json
import sys
import time

import numpy as np
from scipy.integrate import quad

from atomx import Atom, Basis
from atomx.constants import HARTREE_EV


def emit(label, value):
    print(
        label,
        json.dumps(
            value, default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x)
        ),
        flush=True,
    )


if sys.argv[1] == "rr":
    from atomx.methods.hydrogenic_cross_sections import milne_rr_cross_section
    from atomx.scattering.models import ContinuumModel, PotentialModel
    from atomx.scattering.radiative_recombination import rr_cross_section

    a = Atom(2, config="1s1", engine="bspline", corrections="none")
    a.solve()
    E = np.array([1.0, 10.0, 100.0])
    ref = milne_rr_cross_section(2, 1, E / HARTREE_EV)
    for model in [
        None,
        ContinuumModel.shooting(potential=PotentialModel.bare(), n_grid=4000),
    ]:
        t = time.perf_counter()
        rr = rr_cross_section(a.result, E, continuum_model=model, multipoles=["E1"])
        emit(
            "rr",
            dict(
                model=str(model),
                E_eV=E,
                ratio_to_stobbe=rr.sigma_total_cm2 / ref,
                time=time.perf_counter() - t,
            ),
        )

elif sys.argv[1] == "crm":
    import atomx.plasma.crm as c

    I = 13.6
    for T in [0.01, 1.0, 13.6, 136.0, 1360.0]:
        # Milne + Maxwellian, independently integrated in outgoing E/T.
        s0 = c._hydrogenic_sigma0_cm2(I)
        me = 9.1093837139e-28
        ev = 1.602176634e-12
        speed = 2.99792458e10
        pref = (8 / (np.pi * me)) ** 0.5 * (T * ev) ** -1.5 * ev**2
        integral = quad(
            lambda x, s0=s0, T=T, ev=ev, me=me, speed=speed: (
                s0 * I**3 / (I + T * x) * ev / (2 * me * speed**2) * np.exp(-x) * T
            ),
            0,
            100,
            epsabs=1e-40,
        )[0]
        spontaneous = pref * integral
        got = c.radiative_recombination_rate_cm3_s(I, T)
        emit(
            "crm_rr",
            dict(T=T, code=got, spontaneous=spontaneous, ratio=got / spontaneous),
        )
    a = Atom(2, config=["1s2", "1s1 2p1"], corrections="none")
    a.solve()
    trs = a.transitions(["E1"])
    emit(
        "transitions",
        [
            dict(initial=t.initial_label, final=t.final_label, A=t.A_rate_per_s)
            for t in trs
        ],
    )
    crm = c.CollisionalRadiativeModel.from_atom(a)
    emit("crm_levels", [vars(l) for l in crm.levels])
    emit("crm_transitions", [vars(t) for t in crm.transitions])
    R = crm._build_rate_matrix(0.0, 1.0)
    emit("crm_zero_density", dict(pop=crm.steady_state(0.0, 1.0), rates=R))

elif sys.argv[1] == "grid":
    for cfg in ["1s1", ["1s1"]]:
        for basis in [Basis(), Basis(grid_hi=0.005, grid_he=0.1, r_max=30.0)]:
            a = Atom(
                92,
                config=cfg,
                nuclear="point",
                corrections="none",
                engine="bspline",
                basis=basis,
            )
            a.solve()
            b = a.result._bsp_data
            emit(
                "grid",
                dict(
                    config=cfg,
                    basis=str(basis),
                    energy=a.total_energy,
                    ns=b["ns"],
                    last_knot=b["t"][-1],
                    converged=a.result.converged,
                ),
            )

elif sys.argv[1] == "occupations":
    for cfg in [["1s2", "1s1 2p1"], ["1s2 2s2", "1s2 2p2"]]:
        a = Atom(2 if "2s2" not in cfg[0] else 4, config=cfg, corrections="none")
        a.solve()
        emit(
            "occupations",
            dict(
                config=cfg,
                occupations=str(a.result.occupations),
                total=sum(a.result.occupations.values()),
            ),
        )

elif sys.argv[1] == "perf_rr":
    import atomx.numerics.bspline.continuum as cont
    from atomx.scattering.radiative_recombination import rr_cross_section

    a = Atom(2, config="1s1", engine="bspline", corrections="none")
    a.solve()
    E = np.geomspace(1.0, 500.0, 80)
    original = cont.linalg.eigh
    cache = {}
    counts = [0, 0]
    mode = [False]

    def eigen(*args, **kwargs):
        counts[0] += 1
        key = tuple((v.shape, v.tobytes()) for v in args)
        if mode[0] and key in cache:
            return cache[key]
        counts[1] += 1
        result = original(*args, **kwargs)
        if mode[0]:
            cache[key] = result
        return result

    cont.linalg.eigh = eigen
    outputs = []
    for cached in [False, True, False, True, False, True]:
        cache.clear()
        mode[0] = cached
        counts[:] = [0, 0]
        t = time.perf_counter()
        r = rr_cross_section(a.result, E, multipoles=["E1"])
        outputs.append(r.sigma_total_a0_sq)
        emit(
            "rr_perf",
            dict(
                cached=cached,
                time=time.perf_counter() - t,
                calls=counts[0],
                diagonalizations=counts[1],
                ns=a.result._bsp_data["ns"],
            ),
        )
    emit("rr_cache_equal", all(np.array_equal(outputs[0], x) for x in outputs[1:]))

elif sys.argv[1] == "davidson":
    from types import SimpleNamespace

    import atomx.methods.ci as ci
    import atomx.physics.csf.matrix as matrix

    matrix.build_ci_matrix = lambda *args, **kwargs: np.diag([1.0, 2.0, 3.0, 4.0])
    calls = []
    original = ci.linalg.eigh

    def eigen(H, *args, **kwargs):
        calls.append(H.shape)
        return original(H, *args, **kwargs)

    ci.linalg.eigh = eigen
    csfs = [SimpleNamespace(total_J2=j, parity=1) for j in [0, 0, 2, 2]]
    r = ci.ci_solve(csfs, {}, [], {}, 1, 1, n_states=1, solver="davidson")
    emit(
        "davidson_dispatch",
        dict(
            provenance=r.provenance,
            dense_shapes=calls,
            returned_states=len(r.state_energies),
        ),
    )

elif sys.argv[1] == "cap":
    import resource

    from atomx.angular.csf_hamiltonian import (
        ExactCISpaceError,
        block_terms,
        pattern_layout,
    )

    p = ((4, -4, 4), (4, 3, 3), (5, -5, 5))
    t = time.perf_counter()
    before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    try:
        block_terms((p,), 0, max_determinants=1)
    except ExactCISpaceError as e:
        emit("cap_error", str(e))
    emit(
        "cap_cost",
        dict(
            time=time.perf_counter() - t,
            sectors_materialized=len(pattern_layout(p).by_m2),
            peak_rss_increase_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            - before,
        ),
    )

elif sys.argv[1] == "orthogonality":
    from atomx.methods.dirac_fock import DFSolver
    from atomx.physics.csf.matrix import build_ci_matrix
    from atomx.numerics.bspline.dirac_assembly import build_dirac_matrices

    original_solve = DFSolver.solve
    inner = []

    def tracked(self, *args, **kwargs):
        r = original_solve(self, *args, **kwargs)
        inner.append(
            dict(
                converged=r.converged,
                overlap=r.energy_components.get("max_same_kappa_overlap"),
                warning=r.energy_components.get("orthogonality_warning"),
            )
        )
        return r

    DFSolver.solve = tracked
    a = Atom(
        2,
        config=["1s2", "1s1 2s1"],
        engine="bspline",
        nuclear="point",
        corrections="none",
    )
    a.solve(tol=1e-10)
    res = a.result
    bsp = res._bsp_data
    orbs = res.orbitals_dict
    _, S = build_dirac_matrices(bsp["ns"], -1, 2, bsp)
    v = np.array([np.concatenate(orbs[k]) for k in [(1, -1), (2, -1)]])
    emit(
        "orbital_gram",
        dict(
            gram=v @ S @ v.T,
            converged=res.converged,
            energy=res.total_energy,
            last_inner=inner[-1],
        ),
    )
    gs = v.copy()
    gs[1] -= (gs[0] @ S @ gs[1]) / (gs[0] @ S @ gs[0]) * gs[0]
    gs[1] /= np.sqrt(gs[1] @ S @ gs[1])
    ns = bsp["ns"]
    gd = {
        k: (gs[i, :ns].copy(), gs[i, ns:].copy())
        for i, k in enumerate([(1, -1), (2, -1)])
    }
    GH = build_ci_matrix(res.csfs, gd, tuple(gd), bsp, 2, ns)
    emit("gram_schmidt_same_span", float(np.linalg.eigvalsh(GH)[0]))

elif sys.argv[1] == "lines":
    from atomx.angular.coefficients import reduced_C
    from atomx.angular.wigner import wigner_3j
    from atomx.angular.csf_hamiltonian import (
        _remap,
        _union_offsets,
        csf_matrix,
        csf_signature,
        pattern_layout,
    )
    from atomx.properties.transitions.multipole import radial_eL
    from atomx.properties.transitions.rates import einstein_A_eL

    a = Atom(2, config=["1s2", "1s1 2p1"], corrections="none", nuclear="point")
    a.solve()
    res = a.result
    sigs = [csf_signature(c) for c in res.csfs]
    offsets = dict(_union_offsets(tuple(s.pattern for s in sigs)))
    spinors = {}
    for (n, k), off in offsets.items():
        for ii, m in enumerate(range(-(2 * abs(k) - 1), 2 * abs(k), 2)):
            spinors[off + ii] = (n, k, m)

    def vector(s):
        layout = pattern_layout(s.pattern)
        mat = csf_matrix([s], s.pattern, offsets)
        return {
            _remap(m, layout, offsets): float(mat[i, 0])
            for i, m in enumerate(layout.by_m2[s.total_J2])
        }

    vectors = [vector(s) for s in sigs]
    # q=+1 absorption from J=M=0 to J=M=1. Construct sum_ab d_ab a+_a a_b.
    onebody = {}
    for ia, (na, ka, ma) in spinors.items():
        for ib, (nb, kb, mb) in spinors.items():
            if ma - mb != 2:
                continue
            tj = 2 * abs(ka) - 1
            ang = (
                (-1.0) ** ((tj - ma) // 2)
                * wigner_3j(tj, 2, 2 * abs(kb) - 1, -ma, 2, mb)
                * reduced_C(ka, kb, 1)
            )
            if abs(ang) < 1e-14:
                continue
            Pa, Qa = res.orbitals_dict[(na, ka)]
            Pb, Qb = res.orbitals_dict[(nb, kb)]
            onebody[(ia, ib)] = ang * radial_eL(Pa, Qa, Pb, Qb, res.bsp_data, 1)

    def me(vf, vi):
        total = 0.0
        for det, amp in vi.items():
            for (ia, ib), d in onebody.items():
                if not (det >> ib) & 1:
                    continue
                phase = (-1.0) ** ((det & ((1 << ib) - 1)).bit_count())
                removed = det ^ (1 << ib)
                if (removed >> ia) & 1:
                    continue
                phase *= (-1.0) ** ((removed & ((1 << ia) - 1)).bit_count())
                total += vf.get(removed | (1 << ia), 0.0) * amp * phase * d
        return total

    D = np.zeros((len(sigs), len(sigs)))
    for f, sf in enumerate(sigs):
        for i, si in enumerate(sigs):
            if sf.total_J2 == 2 and si.total_J2 == 0:
                D[f, i] = me(vectors[f], vectors[i])
    state_D = res.mixing_coefficients.T @ D @ res.mixing_coefficients
    for tr in a.transitions(["E1"]):
        i = int(tr.initial_label.split("_")[1])
        f = int(tr.final_label.split("_")[1])
        S = 3 * state_D[f, i] ** 2
        emit(
            "line_oracle",
            dict(
                initial=i,
                final=f,
                code_S=tr.line_strength,
                exact_S_same_orbitals=S,
                code_A=tr.A_rate_per_s,
                exact_A_same_orbitals=einstein_A_eL(tr.energy_diff_hartree, S, 1, 2),
            ),
        )

elif sys.argv[1] == "reciprocity":
    from atomx.scattering.r_matrix.cc_outer import (
        scattering_matrices_and_cross_sections,
    )
    from atomx.scattering.r_matrix.channel_table import Channel
    from atomx.scattering.r_matrix.target_states import TargetState

    s = TargetState("1s", 1, 0, 2.0, -2.0, lambda r: r)
    p = TargetState("2p", 2, 1, 2.0, -0.5, lambda r: r)
    channels = [Channel(0, s, 1, -2.0), Channel(1, p, 0, -0.5)]
    K = np.array([[0.1, 0.2], [0.2, 0.3]])
    # Both channels open at E=0: E-epsilon_i = (2, .5) Ha.
    k = np.array([2.0, 1.0])
    S, T, sigma, omega = scattering_matrices_and_cross_sections(
        channels, np.array([0, 1]), k, K, 1, 0.0
    )
    emit(
        "reciprocity",
        dict(
            omega=omega,
            ratio=omega[0, 1] / omega[1, 0],
            S_symmetry_error=float(np.max(np.abs(S - S.T))),
            S_unitarity_error=float(np.max(np.abs(S.conj().T @ S - np.eye(2)))),
        ),
    )

elif sys.argv[1] == "matching":
    from atomx.scattering.r_matrix.cc_outer import match_K_matrix
    from atomx.scattering.r_matrix.channel_table import Channel
    from atomx.scattering.r_matrix.target_states import TargetState

    target = TargetState("neutral", 1, 0, 1.0, 0.0, lambda r: r)
    channel = Channel(0, target, 0, 0.0)
    k = 1.0
    radius = 30.0
    for delta in [0.1, 0.3]:
        # Exact outer solution u=sin(kr+delta); R=u/(a*u').
        R = np.array([[np.tan(k * radius + delta) / (radius * k)]])
        m = match_K_matrix(
            0.5,
            [channel],
            R,
            radius,
            compute_cross_sections=True,
            L_total=0,
            S_total=0.0,
        )
        emit(
            "matching",
            dict(
                delta=delta,
                expected_K=np.tan(delta),
                code_K=m.K_matrix,
                code_phase=m.eigenphases,
            ),
        )
    # Isolate the algebra from the independent Coulomb-function approximation.
    import atomx.scattering.r_matrix.cc_outer as outer

    outer.coulomb_FG = lambda l, eta, rho: (
        np.sin(rho),
        np.cos(rho),
        np.cos(rho),
        -np.sin(rho),
    )
    for delta in [0.1, 0.3]:
        R = np.array([[np.tan(k * radius + delta) / (radius * k)]])
        m = match_K_matrix(0.5, [channel], R, radius)
        emit(
            "matching_exact_FG",
            dict(
                delta=delta,
                expected_K=np.tan(delta),
                code_K=m.K_matrix,
                code_phase=m.eigenphases,
            ),
        )
