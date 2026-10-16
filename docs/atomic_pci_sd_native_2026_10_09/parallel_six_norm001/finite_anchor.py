"""Fresh complete signed/magnetic fermion anchor for the consuming helper."""

from types import SimpleNamespace


def run(norm):
    import numpy as np
    from test_reduced_monovalent_sd import expanded

    from atomx.methods.core_sd import plan_core_sd
    from atomx.methods.monovalent_sd import FiniteMonovalentSD, _core_terms
    from atomx.methods.reduced_monovalent_sd import plan_reduced_monovalent_sd
    from atomx.numerics.fermion import apply_terms

    occupied = ((1, -1),)
    virtual = (
        (2, -1),
        (3, -1),
        (2, 1),
        (2, -2),
        (3, 2),
        (3, -3),
        (4, 3),
        (4, -4),
        (5, 4),
        (5, -5),
    )
    feature = dict(
        auxiliary_nodes=20,
        quadrature_nodes=(2, 3),
        same_cell_edges=12,
        coefficient_sizes=(4, 4),
        component_orders=(3, 3),
        payload_estimate_bytes=10000,
    )
    cp = plan_core_sd(occupied, virtual, feature)
    vp = plan_reduced_monovalent_sd(cp, virtual)
    labels = tuple(
        (n, k, m) for n, k in occupied + virtual for m in norm.projections(k)
    )
    integrals = SimpleNamespace(labels=labels, noccupied=2)
    core = SimpleNamespace(plan=cp, integrals=integrals)
    problem = SimpleNamespace(core=core, plan=vp)
    rng = np.random.default_rng(719230)
    crho = rng.normal(size=cp["raw_amplitudes"]) * 0.013
    vt = {key: rng.normal(size=vp["amplitudes"][key]) * 0.017 for key in virtual}
    terms = _core_terms(core, crho)
    singles, doubles = (
        tuple(t for t in terms if len(t[1]) == degree) for degree in (2, 4)
    )

    def square(state):
        return sum(value**2 for value in state.values())

    core_double = square(apply_terms(doubles, {3: 1.0}))
    rows, maxima = [], {}
    for key in virtual:
        for m in norm.projections(key[1]):
            index = labels.index((*key, m))
            finite = SimpleNamespace(integrals=integrals, valence_index=index)
            finite.excitations = FiniteMonovalentSD._excitations(finite)
            attached = {3 | (1 << index): 1.0}
            ts, td = apply_terms(singles, attached), apply_terms(doubles, attached)
            for sign in (1, -1):
                packed = sign * vt[key]
                value = expanded(problem, key, packed, finite)
                valence = FiniteMonovalentSD._valence_terms(finite, value)
                vs = apply_terms(tuple(t for t in valence if len(t[1]) == 2), attached)
                vd = apply_terms(tuple(t for t in valence if len(t[1]) == 4), attached)
                full = apply_terms(terms + valence, attached)
                mask = next(iter(attached))
                full[mask] = full.get(mask, 0) + 1
                expected = dict(
                    valence_singles=square(vs),
                    valence_doubles=square(vd),
                    core_singles_unblocked=square(ts),
                    core_singles_valence_doubles_interference=2
                    * sum(v * vd.get(k, 0) for k, v in ts.items()),
                    core_doubles_total=core_double,
                    core_doubles_pauli_blocked=core_double - square(td),
                    core_doubles_unblocked=square(td),
                    norm_squared=square(full),
                )
                actual = norm.norm_components(cp, vp, crho, packed, key, twoM=m)
                errors = {name: abs(actual[name] - v) for name, v in expected.items()}
                if max(errors.values()) > 1e-11:
                    raise AssertionError("fresh complete signed finite anchor failed")
                for name, error in errors.items():
                    maxima[name] = max(maxima.get(name, 0), error)
                rows.append(
                    dict(
                        key=key,
                        twoM=m,
                        sign=sign,
                        expected=expected,
                        actual={name: actual[name] for name in expected},
                        errors=errors,
                    )
                )
    return dict(
        status="pass",
        cases=len(rows),
        component_checks=8 * len(rows),
        maxima=maxima,
        rows=rows,
        seed=719230,
        scope="Fresh abstract orthonormal complete l4 fermion metric anchor; no Hamiltonian/root/atomic spectrum calculation",
    )
