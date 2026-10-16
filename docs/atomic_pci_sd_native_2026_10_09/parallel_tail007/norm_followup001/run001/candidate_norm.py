"""Private real J-reduced linearized SD wave norm, including core spectators.

The physical state is (1+T_core+S_v)|Phi_v>, not exp(T+S)|Phi_v>.
Small complete magnetic angular Gram matrices contract unchanged radial rho
blocks. No magnetic quartic amplitude tensor, determinant expansion, amplitude
cutoff, inferred Euclidean metric or normalized property is supplied here.
"""

from functools import lru_cache

import numpy as np

from atomx.angular.sd_rho import rho_coefficient


def projections(kappa):
    return range(-2 * abs(kappa) + 1, 2 * abs(kappa), 2)


@lru_cache(maxsize=4096)
def grams(km, kn, ki, kj, left, right, fixed_role, fixed_m):
    """Complete angular sums; role is unrestricted, first hole, first particle."""
    ff, rr = [], []
    for mm in projections(km):
        if fixed_role == "particle" and mm != fixed_m:
            continue
        for mi in projections(ki):
            if fixed_role == "hole" and mi != fixed_m:
                continue
            for mj in projections(kj):
                mn = mi + mj - mm
                if mn not in projections(kn):
                    continue
                a, b, i, j = (1, km, mm), (1, kn, mn), (1, ki, mi), (1, kj, mj)
                ff.append([rho_coefficient(k, a, b, i, j) for k in left])
                rr.append([rho_coefficient(k, b, a, i, j) for k in right])
    f = np.asarray(ff, dtype=float).reshape(len(ff), len(left))
    r = np.asarray(rr, dtype=float).reshape(len(rr), len(right))
    result = f.T @ f, r.T @ r, f.T @ r
    for array in result:
        array.flags.writeable = False
    return result


def decode(plan, vector, *, valence_key=None):
    """Zero-copy Fortran radial block views with complete coordinate accounting."""
    values = np.asarray(vector)
    expected_count = (
        plan["raw_amplitudes"]
        if valence_key is None
        else plan["amplitudes"][valence_key]
    )
    if values.shape != (expected_count,) or values.dtype != np.float64:
        raise ValueError("complete finite real FP64 packed amplitudes required")
    # A fixed 1 Mi-element boolean tile, even for the full28M core vector.
    # Check every tile, including the last; no coordinate is omitted.
    for begin in range(0, expected_count, 1024**2):
        if not np.all(np.isfinite(values[begin : begin + 1024**2])):
            raise ValueError("complete finite real FP64 packed amplitudes required")
    virtual = set(plan["virtual_keys"])
    groups = tuple(tuple(k for k in g if k in virtual) for g in plan["groups"])
    tables, single, cursor = {}, {}, 0
    if valence_key is None:
        positions = {key: i for i, g in enumerate(plan["groups"]) for key in g}
        for a in plan["occupied_keys"]:
            for m in groups[positions[a]]:
                single[m, a] = values[cursor]
                cursor += 1
    else:
        for m in plan["single_keys"][valence_key]:
            single[m] = values[cursor]
            cursor += 1
    rows = plan["channels"] if valence_key is None else plan["channels"][valence_key]
    for row in rows:
        im, jn, rank = row[-3:]
        count = len(groups[im]) * len(groups[jn])
        view = values[cursor : cursor + count].reshape(
            (len(groups[im]), len(groups[jn])), order="F"
        )
        tables.setdefault(row[:-1], {})[rank] = view
        cursor += count
    if cursor != len(values):
        raise ValueError("complete packed radial coordinate identity differs")
    return groups, single, tables


def stack(block):
    ranks = tuple(sorted(block))
    return ranks, np.stack([block[k] for k in ranks])


def quadratic(a, b, gram):
    return float(np.einsum("k...,l...,kl->", a, b, gram, optimize=True))


def norm_components(
    core_plan,
    valence_plan,
    core_rho,
    valence_rho,
    key,
    *,
    twoM=None,
    memory_bytes=64 * 1024**2,
):
    """Return physical real scalar wave-norm terms; every radial channel retained.

    Source/state binding remains the caller's duty; the private finite probe and
    complete-Yb consumer bind the actual plans, amplitudes and source hashes.
    Norm alone does not certify an eigenstate, transition operator or resonance.
    """
    key = tuple(key)
    twoM = 2 * abs(key[1]) - 1 if twoM is None else twoM
    if (
        isinstance(twoM, (bool, np.bool_))
        or not isinstance(twoM, (int, np.integer))
        or twoM not in projections(key[1])
    ):
        raise ValueError("actual valence magnetic projection required")
    resource_plan = plan_scratch(core_plan, valence_plan, key)
    extra = resource_plan['required_bytes']
    angular_grams = BoundGrams(grams.__wrapped__)
    if (
        isinstance(memory_bytes, bool)
        or not isinstance(memory_bytes, int)
        or extra > memory_bytes
    ):
        raise MemoryError("complete physical norm extra scratch exceeds cap")
    groups, tc, core_tables = decode(core_plan, core_rho)
    packed_plan = dict(
        valence_plan, groups=core_plan["groups"], virtual_keys=core_plan["virtual_keys"]
    )
    vg, sv, valence_tables = decode(packed_plan, valence_rho, valence_key=key)
    if vg != groups:
        raise ValueError("core and valence radial groups differ")
    occupied = core_plan["occupied_keys"]
    if key not in core_plan["virtual_keys"]:
        raise ValueError("reference radial orbital absent from complete virtual pool")
    iv = next(i for i, group in enumerate(groups) if key in group)
    pv = groups[iv].index(key)
    valence_double = core_double = blocked_double = interference = 0.0

    for (b, im, jn), block in valence_tables.items():
        ranks, u = stack(block)
        reverse_ranks, reverse = (
            stack(valence_tables.get((b, jn, im), {}))
            if (b, jn, im) in valence_tables
            else ((), np.empty((0, len(groups[jn]), len(groups[im]))))
        )
        dd, _, cross = angular_grams(
            groups[im][0][1],
            groups[jn][0][1],
            key[1],
            occupied[b][1],
            ranks,
            reverse_ranks,
            "hole",
            twoM,
        )
        valence_double += quadratic(u, u, dd) - quadratic(
            u, reverse.transpose(0, 2, 1), cross
        )

    for (a, b, im, jn), block in core_tables.items():
        ranks, u = stack(block)
        reverse_key = (a, b, jn, im)
        reverse_ranks, reverse = (
            stack(core_tables[reverse_key])
            if reverse_key in core_tables
            else ((), np.empty((0, len(groups[jn]), len(groups[im]))))
        )
        if a == b:
            # Equal radial holes still have distinct magnetic projections.
            # Full particle AND hole antisymmetrization gives the symmetric
            # radial combination here; a direct redundant input need not have
            # particle-exchange symmetry before physical decoding.
            if reverse_ranks != ranks:
                raise ValueError("equal-hole complete reverse channels required")
            u = 0.5 * (u + reverse.transpose(0, 2, 1))
            reverse = u.transpose(0, 2, 1)
        dd, _, cross = angular_grams(
            groups[im][0][1],
            groups[jn][0][1],
            occupied[a][1],
            occupied[b][1],
            ranks,
            reverse_ranks,
            "none",
            0,
        )
        weight = 0.5 if a == b else 1.0
        core_double += weight * (
            quadratic(u, u, dd) - quadratic(u, reverse.transpose(0, 2, 1), cross)
        )

    pairs = {(a, b) for a, b, _, _ in core_tables}
    for a, b in pairs:
        for im, group in enumerate(groups):
            left = core_tables.get((a, b, iv, im), {})
            right = core_tables.get((a, b, im, iv), {})
            if not left and not right:
                continue
            lr = tuple(sorted(left))
            rr = tuple(sorted(right))
            u = (
                np.stack([left[k][pv, :] for k in lr])
                if lr
                else np.empty((0, len(group)))
            )
            r = (
                np.stack([right[k][:, pv] for k in rr])
                if rr
                else np.empty((0, len(group)))
            )
            if a == b:
                if lr != rr:
                    raise ValueError("equal-hole complete reverse channels required")
                u = 0.5 * (u + r)
                r = u
            dd, ee, cross = angular_grams(
                key[1],
                group[0][1],
                occupied[a][1],
                occupied[b][1],
                lr,
                rr,
                "particle",
                twoM,
            )
            blocked_double += (0.5 if a == b else 1.0) * (
                quadratic(u, u, dd) + quadratic(r, r, ee) - 2 * quadratic(u, r, cross)
            )

    core_single = sum(
        (2 * abs(a[1]) - (m == key)) * value**2 for (m, a), value in tc.items()
    )
    for b, a in enumerate(occupied):
        for im, group in enumerate(groups):
            if not group or group[0][1] != a[1]:
                continue
            left = valence_tables.get((b, iv, im), {})
            right = valence_tables.get((b, im, iv), {})
            for mpos, m in enumerate(group):
                signed = 0.0
                for mb in projections(a[1]):
                    vlabel, mlabel, blabel = (*key, twoM), (*m, mb), (*a, mb)
                    signed += sum(
                        rho_coefficient(k, vlabel, mlabel, vlabel, blabel)
                        * table[pv, mpos]
                        for k, table in left.items()
                    )
                    signed -= sum(
                        rho_coefficient(k, mlabel, vlabel, vlabel, blabel)
                        * table[mpos, pv]
                        for k, table in right.items()
                    )
                interference += 2 * tc[m, a] * signed
    result = dict(
        reference=1.0,
        valence_singles=float(sum(value**2 for value in sv.values())),
        valence_doubles=valence_double,
        core_singles_unblocked=float(core_single),
        core_singles_valence_doubles_interference=interference,
        core_doubles_total=core_double,
        core_doubles_pauli_blocked=blocked_double,
        core_doubles_unblocked=core_double - blocked_double,
        extra_scratch_bound_bytes=extra,
        resource_plan=resource_plan,
        angular_cache=dict(hits=angular_grams.hits, misses=angular_grams.misses, evictions=angular_grams.evictions, peak_array_bytes=angular_grams.peak_bytes),
    )
    norm_squared = sum(
        result[k]
        for k in (
            "reference",
            "valence_singles",
            "valence_doubles",
            "core_singles_unblocked",
            "core_singles_valence_doubles_interference",
            "core_doubles_unblocked",
        )
    )
    if not np.isfinite(norm_squared) or norm_squared < 1 - 1e-11:
        raise ValueError("physical wave norm positivity gate failed")
    result["norm_squared"] = norm_squared
    result["wavefunction_norm"] = float(np.sqrt(norm_squared))
    result["scope"] = (
        "private real linearized SD wave norm only; no property/eigenstate/resonance certificate"
    )
    return result
