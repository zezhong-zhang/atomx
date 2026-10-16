"""Private real J-reduced linearized SD wave norm, including core spectators.

The physical state is (1+T_core+S_v)|Phi_v>, not exp(T+S)|Phi_v>.
Small complete magnetic angular Gram matrices contract unchanged radial rho
blocks. No magnetic quartic amplitude tensor, determinant expansion, amplitude
cutoff, inferred Euclidean metric or normalized property is supplied here.
"""

from functools import lru_cache

import numpy as np

from atomx.angular.sd_rho import rho_coefficient


"""Isolated owned Gram-cache candidate; unchanged angular/norm equations."""

from collections import OrderedDict

CACHE_ENTRIES = 256
CACHE_ARRAY_BYTES = 4 * 1024**2


class BoundGrams:
    """Per-consumer cache with measured array residency and explicit limits."""

    def __init__(
        self, uncached, *, max_entries=CACHE_ENTRIES, max_bytes=CACHE_ARRAY_BYTES
    ):
        if any(
            isinstance(limit, bool) or not isinstance(limit, int) or limit < 1
            for limit in (max_entries, max_bytes)
        ):
            raise ValueError(
                "positive integer cache entry and array-byte caps required"
            )
        self.uncached = uncached
        self.max_entries, self.max_bytes = max_entries, max_bytes
        self.cache = OrderedDict()
        self.resident_bytes = self.peak_bytes = self.hits = self.misses = (
            self.evictions
        ) = 0

    def __call__(self, *key):
        if key in self.cache:
            self.hits += 1
            self.cache.move_to_end(key)
            return self.cache[key]
        self.misses += 1
        value = self.uncached(*key)
        payload = sum(array.nbytes for array in value)
        if payload > self.max_bytes:
            raise MemoryError("complete Gram matrix exceeds declared cache allowance")
        while self.cache and (
            len(self.cache) >= self.max_entries
            or self.resident_bytes + payload > self.max_bytes
        ):
            _, old = self.cache.popitem(last=False)
            self.resident_bytes -= sum(array.nbytes for array in old)
            self.evictions += 1
        self.cache[key] = value
        self.resident_bytes += payload
        self.peak_bytes = max(self.peak_bytes, self.resident_bytes)
        return value


def plan_scratch(core_plan, valence_plan, key):
    """Separate complete views/metadata, bounded cache and one quartet work."""
    d = core_plan["dimensions"]
    maxrank = d["KK"] + 1
    virtual = set(core_plan["virtual_keys"])
    maxradial = max(sum(k in virtual for k in group) for group in core_plan["groups"])
    maxmag = 2 * max(
        abs(k) for _, k in core_plan["occupied_keys"] + core_plan["virtual_keys"]
    )
    singles = sum(
        sum(key in virtual for key in group)
        for occupied in core_plan["occupied_keys"]
        for group in core_plan["groups"]
        if occupied in group
    ) + len(valence_plan["single_keys"][key])
    # Every decoded rank/channel is a zero-copy radial view, with Python
    # key/dict/view metadata reserved explicitly. No array cube is cached.
    view_metadata = 512 * (
        len(core_plan["channels"]) + len(valence_plan["channels"][key])
    )
    single_metadata = 256 * singles
    group_metadata = 256 * (
        len(virtual) + len(core_plan["occupied_keys"]) + len(core_plan["groups"])
    )
    cache = CACHE_ARRAY_BYTES + 1024 * CACHE_ENTRIES
    # All complete magnetic rows for one ordered quartet, including Python
    # coefficient lists, FP64 conversion and all three signed Gram outputs.
    # Include a complete newly constructed Gram triple and a previous
    # triple still held in the norm frame while replacement is evaluated.
    angular = maxmag**3 * (256 + 128 * maxrank) + 48 * maxrank**2
    radial = 64 * maxrank * maxradial**2
    required = (
        2 * 1024**2
        + view_metadata
        + single_metadata
        + group_metadata
        + cache
        + angular
        + radial
    )
    return dict(
        required_bytes=required,
        view_metadata_bytes=view_metadata,
        single_metadata_bytes=single_metadata,
        group_metadata_bytes=group_metadata,
        angular_cache_array_cap_bytes=CACHE_ARRAY_BYTES,
        angular_cache_metadata_bound_bytes=1024 * CACHE_ENTRIES,
        angular_quartet_transient_bound_bytes=angular,
        radial_quartet_transient_bound_bytes=radial,
        cache_entries=CACHE_ENTRIES,
        maximum_rank_count=maxrank,
        maximum_radial_group_count=maxradial,
        maximum_magnetic_multiplet_count=maxmag,
        cached_Gram_array_bound_bytes=min(
            CACHE_ARRAY_BYTES, CACHE_ENTRIES * 24 * maxrank**2
        ),
        scope="allocated norm payload and conservative Python view/cache metadata; excludes interpreter/Numba/shared Wigner-cache runtime RSS",
    )



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
