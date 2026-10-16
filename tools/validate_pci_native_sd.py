#!/usr/bin/env python3
"""Live original-pCI vs independent finite-Fock commutator control."""

import argparse
import ctypes
import hashlib
import itertools
import json
import time
from pathlib import Path

import numpy as np
from build_pci_sd_backend import build

from atomx.angular.recoupling import l_of_kappa, twoj_of_kappa
from atomx.angular.wigner import wigner_3j
from atomx.methods.core_sd import CoreSD
from atomx.methods.dirac_fock import DFSolver, Subshell
from atomx.methods.linearized_sd import solve as dense_sd_solve
from atomx.methods.mbpt import build_virtual_basis
from atomx.numerics.bspline.coulomb_features import CellCoulombFeatures
from atomx.physics.csf.spinor_integrals import build_spinor_integrals

REFERENCE = Path("/home/zzhang/code/atomx-references/2026-10-09-ci-fscc")


def partial(kappa):
    return -2 * kappa - 1 if kappa < 0 else 2 * kappa


def kappa_from_partial(index):
    return -(index + 1) // 2 if index % 2 else index // 2


def channels(occupied, lmax, kind):
    total = 0
    for i, (_, ki) in enumerate(occupied):
        for _, kj in occupied[: i + 1]:
            for km in map(kappa_from_partial, range(1, 2 * lmax + 2)):
                for kn in map(kappa_from_partial, range(1, 2 * lmax + 2)):
                    a, b, c, d = (km, ki, kn, kj) if kind != "x2" else (km, kn, ki, kj)
                    low = max(abs(abs(a) - abs(b)), abs(abs(c) - abs(d)))
                    high = min(abs(a) + abs(b) - 1, abs(c) + abs(d) - 1)
                    for rank in range(low, high + 1):
                        parity = (
                            l_of_kappa(a) + l_of_kappa(b) + rank,
                            l_of_kappa(c) + l_of_kappa(d) + rank,
                        )
                        total += (
                            int(parity[0] % 2 == parity[1] % 2)
                            if kind == "rho"
                            else int(not parity[0] % 2 and not parity[1] % 2)
                        )
    return total


def angular(rank, a, b, c, d):
    ja, jb, jc, jd = (twoj_of_kappa(x[1]) for x in (a, b, c, d))
    ma, mb, mc, md = (x[2] for x in (a, b, c, d))
    if ma + mb != mc + md:
        return 0.0
    q = (ma - mc) // 2
    return (
        (-1)
        ** ((ja - ma + jb - mb) // 2 + q + rank + sum(abs(x[1]) for x in (a, b, c, d)))
        * wigner_3j(ja, 2 * rank, jc, -ma, 2 * q, mc)
        * wigner_3j(jb, 2 * rank, jd, -mb, -2 * q, md)
    )


def apply(mask, operations):
    coefficient = 1
    for bit, create in reversed(operations):
        if bool(mask & (1 << bit)) == create:
            return None, 0
        coefficient *= (-1) ** ((mask & ((1 << bit) - 1)).bit_count())
        mask ^= 1 << bit
    return mask, coefficient


def main(
    directory,
    *,
    case="be2",
    virtual_count=1,
    rotate=False,
    lmax=1,
    kinetic_balance=False,
):
    directory.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    if virtual_count < 1 or lmax < 0:
        raise ValueError(
            "positive radial count and nonnegative partial-wave bound required"
        )
    root = Path(__file__).resolve().parents[1]
    (directory / "contract.json").write_text(
        json.dumps(
            dict(
                case=case,
                virtual_count_per_kappa=virtual_count,
                lmax=lmax,
                kinetic_balance=kinetic_balance,
                rotate=rotate,
                fixed="native closed core; explicit finite anchor pool; full noncanonical DC",
                hypothesis="original reduced diagrams and shared quadrature reproduce independent signed commutator, including high partial waves and unequal P/Q orders",
                gate=1e-11,
                stopping="reject whole overbudget pool or any failed signed equation/energy/amplitude gate",
                scope="finite independent equation anchor, not a complete virtual band or experiment",
                source_sha256={
                    str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in [Path(__file__).resolve()]
                    + sorted((root / "src/atomx").rglob("*.py"))
                },
            ),
            indent=2,
        )
        + "\n"
    )
    Z = 10 if case == "ne10" else 4
    occupied = [(1, -1)] if case == "be2" else [(1, -1), (2, -1)]
    if case == "ne10":
        occupied += [(2, 1), (2, -2)]
    core = DFSolver(
        Z=Z,
        subshells=[Subshell(n, k, 2 * abs(k)) for n, k in occupied],
        ns=30,
        ks=7,
        kinetic_balance=kinetic_balance,
    ).solve()
    orbitals = {key: core.orbitals_by_subshell()[key][:2] for key in occupied}
    virtual = []
    for k in map(kappa_from_partial, range(1, 2 * lmax + 2)):
        _, vs = build_virtual_basis(
            core, k, e_max=50, potential_mode="VN", orthogonal_to_occupied=True
        )
        start = max([n for n, ka in occupied if ka == k] + [l_of_kappa(k), 1]) + 1
        for v in range(virtual_count):
            key = (start + v, k)
            virtual.append(key)
            orbitals[key] = vs[v][:2]
    if rotate:
        rng_rotation = np.random.default_rng(314159)
        for group in (occupied, virtual):
            for k in sorted({a[1] for a in group}):
                selected = [a for a in group if a[1] == k]
                u = np.linalg.qr(
                    rng_rotation.normal(size=(len(selected), len(selected)))
                )[0]
                pcols, qcols = (
                    np.column_stack([orbitals[a][i] for a in selected]) @ u
                    for i in (0, 1)
                )
                for i, a in enumerate(selected):
                    orbitals[a] = (pcols[:, i], qcols[:, i])
    features = CellCoulombFeatures(core._bsp_data)
    arrays = {f"P_{n}_{k}": value[0] for (n, k), value in orbitals.items()}
    arrays.update({f"Q_{n}_{k}": value[1] for (n, k), value in orbitals.items()})
    arrays.update(
        {
            f"workspace_{k}": value
            for k, value in core._bsp_data.items()
            if isinstance(value, (np.ndarray, int, float, bool))
        }
    )
    potential = core._nuclear_potential
    arrays["nuclear_on_outer_gauss"] = (
        np.vectorize(potential, otypes=[float])(core._bsp_data["gr"])
        if callable(potential)
        else -Z / core._bsp_data["gr"]
        if potential is None
        else np.asarray(potential)
    )
    np.savez_compressed(directory / "native_inputs.npz", **arrays)
    ints = build_spinor_integrals(
        orbitals,
        core._bsp_data,
        Z,
        30,
        occupied_keys=occupied,
        virtual_keys=virtual,
        nuclear_potential=core._nuclear_potential,
        representation="on_demand",
    )
    keys = occupied + virtual
    lmax = max(l_of_kappa(k) for _, k in keys)
    grouped = {
        k: [a for a in keys if a[1] == k]
        for k in map(kappa_from_partial, range(1, 2 * lmax + 2))
    }
    limits = [len(a) for a in grouped.values()]
    slots = {key: grouped[key[1]].index(key) for key in keys}
    nxx = max(limits)
    nhf = features.identity["auxiliary_nodes"]
    dims = dict(
        NHF=nhf,
        NX=max(nxx, 2 * lmax + 1),
        NS=len(occupied),
        KX1=2 * lmax + 1,
        NXX=nxx,
        NK=2 * lmax + 1,
        NN=len(occupied),
        KK=2 * max(abs(k) for _, k in keys) - 1,
        NCH=channels(occupied, lmax, "rho"),
        NXK=max(channels(occupied, lmax, "x1"), channels(occupied, lmax, "x2")),
        NV=features.identity["quadrature_nodes"][0],
        NG=features.identity["quadrature_nodes"][1],
        NE=features.identity["same_cell_edges"],
    )
    # Conservative complete static/scratch bound before compilation/allocation.
    workspace = 8 * (
        2 * nhf * dims["NX"] * dims["KX1"]
        + 2 * nhf * dims["NS"]
        + 2 * nxx * nxx * (dims["NCH"] + dims["NXK"])
        + nhf * dims["NX"] ** 2
        + 4 * nhf * dims["NX"]
        + 2 * nhf * nxx * (dims["KK"] + 1) * dims["KK"]
        + (2 * dims["NV"] * dims["NG"] + dims["NE"]) * (dims["KK"] + 1)
    )
    if workspace > 2 * 1024**3:
        raise MemoryError("complete first diagram control exceeds declared 2 GiB cap")
    (directory / "plan.json").write_text(
        json.dumps(
            dict(
                dimensions=dims,
                workspace_bytes=workspace,
                feature_identity=dict(features.identity),
                occupied=occupied,
                virtual=virtual,
                case=case,
                rotation="complete occupied/virtual same-kappa QR, seed314159"
                if rotate
                else None,
            ),
            indent=2,
        )
        + "\n"
    )
    build(
        REFERENCE / "pCI",
        directory / "backend",
        dims,
        REFERENCE / "gfortran",
        runtime_directory=REFERENCE / "toolchain/usr/lib/x86_64-linux-gnu",
    )
    backend = ctypes.CDLL(str(directory / "backend/native_core_sd.so"))
    float_array = np.ctypeslib.ndpointer(dtype=np.float64, flags="F_CONTIGUOUS")
    integer_array = np.ctypeslib.ndpointer(dtype=np.int32, flags="F_CONTIGUOUS")

    def call(name, arguments):
        function = getattr(backend, name)
        function.argtypes = [
            integer_array if a.dtype == np.int32 else float_array for a in arguments
        ]
        function.restype = None
        function(*arguments)

    p, q = (np.zeros((nhf, dims["NX"], dims["KX1"]), order="F") for _ in range(2))
    eps = np.zeros((dims["NX"], dims["KX1"]), order="F")
    for key in keys:
        value = features.sample(*orbitals[key])
        s = slots[key]
        ind = partial(key[1]) - 1
        split = int(features.outer_indices[1, 0, 0])
        p[:split, s, ind] = value[:split]
        q[split:, s, ind] = value[split:]
        mag = ints.labels.index((*key, -twoj_of_kappa(key[1])))
        eps[s, ind] = ints.fock_Ha[mag, mag].real
    ranks = [features.rank_arrays(k) for k in range(dims["KK"] + 1)]
    rw, fw = (
        np.asfortranarray(np.stack([r[i] for r in ranks], axis=2)) for i in (0, 1)
    )
    ew = np.asfortranarray(np.stack([r[2] for r in ranks], axis=1))

    def integer(a):
        return np.asfortranarray(a, dtype=np.int32)

    lengths = np.zeros(dims["NX"], dtype=np.int32)
    lengths[: len(limits)] = limits
    call(
        "atomx_sd_init",
        [
            p,
            q,
            eps,
            integer([slots[a] + 1 for a in occupied]),
            integer([a[1] for a in occupied]),
            lengths,
            np.asfortranarray(features.radii),
            integer(features.outer_indices + 1),
            integer(features.edge_left + 1),
            integer(features.edge_right + 1),
            rw,
            fw,
            ew,
        ],
    )
    metadata = np.zeros((5, dims["NCH"]), dtype=np.int32, order="F")
    initial, source = (np.zeros((nxx, nxx, dims["NCH"]), order="F") for _ in range(2))
    singles = np.zeros((nxx, len(occupied)), order="F")
    counts = np.zeros(3, dtype=np.int32)
    call("atomx_sd_info", [metadata, initial, source, singles, counts])
    lookup = {tuple(c): i for i, c in enumerate(metadata.T)}

    def rho(td, a, b, c, d):
        ia, ib = occupied.index(c[:2]) + 1, occupied.index(d[:2]) + 1
        if ia < ib:
            a, b, c, d = b, a, d, c
            ia, ib = ib, ia
        result = 0.0
        for rank in range(dims["KK"] + 1):
            channel = lookup.get((ia, ib, partial(a[1]), partial(b[1]), rank))
            if channel is not None:
                result += (
                    angular(rank, a, b, c, d) * td[slots[a[:2]], slots[b[:2]], channel]
                )
        return result

    labels = ints.labels
    n = len(labels)
    o = ints.noccupied
    ref = (1 << o) - 1
    # The SD determinant projection is exact for this linked residual:
    # H|Phi> has rank<=2 and T cannot lower excitation rank. No omitted
    # higher-rank determinant can contribute to Q_SD[H,T]|Phi>.
    all_masks = {ref}
    for rank in (1, 2):
        for holes in itertools.combinations(range(o), rank):
            for particles in itertools.combinations(range(o, n), rank):
                if sum(labels[a][2] for a in holes) != sum(
                    labels[a][2] for a in particles
                ):
                    continue
                if (sum(l_of_kappa(labels[a][1]) for a in holes + particles)) % 2:
                    continue
                all_masks.add(
                    ref ^ sum(1 << i for i in holes) ^ sum(1 << a for a in particles)
                )
    masks = tuple(sorted(all_masks))
    if 8 * 8 * len(masks) ** 2 > 128 * 1024**2:
        raise MemoryError("complete independent SD commutator control exceeds 128 MiB")
    index = {m: i for i, m in enumerate(masks)}
    pairs = list(itertools.combinations(range(n), 2))
    g_values = {a + b: ints.eri(*a, *b).real for a in pairs for b in pairs}

    def operator(one, two):
        result = np.zeros((len(masks), len(masks)))
        for col, mask in enumerate(masks):
            for a, b in itertools.product(range(n), repeat=2):
                target, sign = apply(mask, ((a, True), (b, False)))
                if sign and target in index:
                    result[index[target], col] += sign * one[a, b]
            for a, b in pairs:
                for i, j in pairs:
                    value = two(a, b, i, j)
                    if value == 0:
                        continue
                    target, sign = apply(
                        mask, ((a, True), (b, True), (j, False), (i, False))
                    )
                    if sign and target in index:
                        result[index[target], col] += sign * value
        return result

    def generator(ts, td):
        t1 = np.zeros((n, n))
        for a in range(o, n):
            for i in range(o):
                if labels[a][1:] == labels[i][1:]:
                    t1[a, i] = ts[slots[labels[a][:2]], occupied.index(labels[i][:2])]

        def t2(a, b, i, j):
            if min(a, b) < o or max(i, j) >= o:
                return 0.0
            return rho(td, labels[a], labels[b], labels[i], labels[j]) - rho(
                td, labels[b], labels[a], labels[i], labels[j]
            )

        return operator(t1, t2)

    # Canonical calibration uses the native two-body integrals and explicitly
    # altered one-body operator. Native physical Fock is tested separately.
    canonical = np.diag(np.diag(ints.fock_Ha.real))
    h = canonical.copy()
    for a, b in itertools.product(range(n), repeat=2):
        h[a, b] -= sum(ints.eri(a, i, b, i).real for i in range(o))
    full = operator(h, lambda a, b, i, j: g_values[a, b, i, j])
    rng = np.random.default_rng(20261009)
    td = np.zeros_like(initial)
    ts = np.zeros_like(singles)
    for c, (_ia, _ib, im, inn, _rank) in enumerate(metadata.T):
        km, kn = kappa_from_partial(im), kappa_from_partial(inn)
        for m, a in enumerate(grouped[km]):
            for nn, b in enumerate(grouped[kn]):
                if a in virtual and b in virtual:
                    td[m, nn, c] = rng.normal() * 0.01
    # Joint particle/core exchange symmetry for the same radial core pair.
    for c, (ia, ib, im, inn, rank) in enumerate(metadata.T):
        if ia == ib:
            partner = lookup[ia, ib, inn, im, rank]
            if c <= partner:
                value = 0.5 * (td[:, :, c] + td[:, :, partner].T)
                td[:, :, c] = value
                td[:, :, partner] = value.T
    for i, key in enumerate(occupied):
        for a in virtual:
            if a[1] == key[1]:
                ts[slots[a], i] = rng.normal() * 0.01
    rs, rd = np.zeros_like(ts), np.zeros_like(td)
    call("atomx_sd_action", [ts, td, rs, rd])
    ds = np.zeros_like(ts)
    dd = np.zeros_like(td)
    for c, (ia, ib, im, inn, _rank) in enumerate(metadata.T):
        dd[:, :, c] = (
            eps[slots[occupied[ia - 1]], partial(occupied[ia - 1][1]) - 1]
            + eps[slots[occupied[ib - 1]], partial(occupied[ib - 1][1]) - 1]
            - eps[:nxx, im - 1, None]
            - eps[None, :nxx, inn - 1]
        ) * td[:, :, c]
    for i, key in enumerate(occupied):
        ds[:, i] = (
            eps[slots[key], partial(key[1]) - 1] - eps[:nxx, partial(key[1]) - 1]
        ) * ts[:, i]
    reference = np.eye(len(masks))[:, index[ref]]
    t = generator(ts, td)
    independent = (full + full @ t - t @ full) @ reference
    reduced = -generator(ds - rs, dd - source - rd) @ reference
    independent[index[ref]] = 0.0
    energy = np.zeros(1)
    call("atomx_sd_energy", [ts, td, energy])
    fock = ints.fock_Ha.real
    radial_f = {}
    for a in keys:
        for b in keys:
            if a[1] == b[1]:
                m = -twoj_of_kappa(a[1])
                radial_f[a, b] = fock[labels.index((*a, m)), labels.index((*b, m))]

    def value(td, a, b, i, j, rank):
        im, inn = partial(a[1]), partial(b[1])
        if i < j:
            a, b, i, j = b, a, j, i
            im, inn = inn, im
        c = lookup.get((i + 1, j + 1, im, inn, rank))
        return 0.0 if c is None else td[slots[a], slots[b], c]

    extra_s = np.zeros_like(ts)
    extra_d = np.zeros_like(td)
    for c, (ia, ib, im, inn, rank) in enumerate(metadata.T):
        i, j = ia - 1, ib - 1
        oi, oj = occupied[i], occupied[j]
        for a in grouped[kappa_from_partial(im)]:
            if a not in virtual:
                continue
            for b in grouped[kappa_from_partial(inn)]:
                if b not in virtual:
                    continue
                res = 0.0
                for x in grouped[a[1]]:
                    if x in virtual and x != a:
                        res += radial_f[a, x] * value(td, x, b, i, j, rank)
                for x in grouped[b[1]]:
                    if x in virtual and x != b:
                        res += radial_f[b, x] * value(td, a, x, i, j, rank)
                for ii, x in enumerate(occupied):
                    if x[1] == oi[1] and ii != i:
                        res -= radial_f[x, oi] * value(td, a, b, ii, j, rank)
                    if x[1] == oj[1] and ii != j:
                        res -= radial_f[x, oj] * value(td, a, b, i, ii, rank)
                extra_d[slots[a], slots[b], c] = res
    for i, oi in enumerate(occupied):
        mi = -twoj_of_kappa(oi[1])
        for a in grouped[oi[1]]:
            if a not in virtual:
                continue
            res = radial_f[a, oi]
            for x in grouped[a[1]]:
                if x in virtual and x != a:
                    res += radial_f[a, x] * ts[slots[x], i]
            for j, oj in enumerate(occupied):
                if oj[1] == oi[1] and j != i:
                    res -= radial_f[oj, oi] * ts[slots[a], j]
                for b in grouped[oj[1]]:
                    if b not in virtual:
                        continue
                    for mj in range(-twoj_of_kappa(oj[1]), twoj_of_kappa(oj[1]) + 1, 2):
                        res += radial_f[oj, b] * (
                            rho(td, (*a, mi), (*b, mj), (*oi, mi), (*oj, mj))
                            - rho(td, (*b, mj), (*a, mi), (*oi, mi), (*oj, mj))
                        )
            extra_s[slots[a], i] = res
    physical = operator(ints.one_body_Ha.real, lambda a, b, i, j: g_values[a, b, i, j])
    native_independent = (physical + physical @ t - t @ physical) @ reference
    native_independent[index[ref]] = 0.0
    native_reduced = (
        -generator(ds - rs - extra_s, dd - source - rd - extra_d) @ reference
    )
    singles_energy = sum(
        2 * abs(oi[1]) * radial_f[oi, a] * ts[slots[a], i]
        for i, oi in enumerate(occupied)
        for a in virtual
        if a[1] == oi[1]
    )
    radial_blocks = np.zeros((dims["NX"], dims["NX"], dims["KX1"]), order="F")
    for (a, b), f in radial_f.items():
        radial_blocks[slots[a], slots[b], partial(a[1]) - 1] = f
    alpha, beta = (
        np.zeros((dims["NK"], dims["NK"], dims["KK"] + 1), order="F") for _ in range(2)
    )
    for im in range(1, dims["NK"] + 1):
        ki = kappa_from_partial(im)
        mi = -twoj_of_kappa(ki)
        for inn in range(1, dims["NK"] + 1):
            kj = kappa_from_partial(inn)
            for rank in range(dims["KK"] + 1):
                for mj in range(-twoj_of_kappa(kj), twoj_of_kappa(kj) + 1, 2):
                    a, b = (1, ki, mi), (1, kj, mj)
                    alpha[im - 1, inn - 1, rank] += angular(rank, a, b, a, b)
                    beta[im - 1, inn - 1, rank] += angular(rank, b, a, a, b)
    call("atomx_sd_set_fock", [radial_blocks, alpha, beta])
    fortran_s, fortran_d = np.zeros_like(ts), np.zeros_like(td)
    call("atomx_sd_fock_action", [ts, td, fortran_s, fortran_d])
    fortran_energy = np.zeros(1)
    call("atomx_sd_native_energy", [ts, td, fortran_energy])
    problem = CoreSD(
        orbitals,
        core._bsp_data,
        Z,
        30,
        occupied_keys=occupied,
        virtual_keys=virtual,
        backend_directory=directory / "backend",
        nuclear_potential=core._nuclear_potential,
    )

    def pack(s, d):
        parts = []
        for i, a in enumerate(occupied):
            g = grouped[a[1]]
            nocc = sum(x in occupied for x in g)
            parts.append(s[nocc : len(g), i])
        for c, (_, _, im, inn, _) in enumerate(metadata.T):
            gm, gn = grouped[kappa_from_partial(im)], grouped[kappa_from_partial(inn)]
            beginm = sum(x in occupied for x in gm)
            beginn = sum(x in occupied for x in gn)
            parts.append(d[beginm : len(gm), beginn : len(gn), c].ravel(order="F"))
        return np.concatenate(parts)

    def unpack(v):
        s, d = np.zeros_like(ts), np.zeros_like(td)
        offset = 0
        for i, a in enumerate(occupied):
            g = grouped[a[1]]
            begin = sum(x in occupied for x in g)
            count = len(g) - begin
            s[begin : len(g), i] = v[offset : offset + count]
            offset += count
        for c, (_, _, im, inn, _) in enumerate(metadata.T):
            gm, gn = grouped[kappa_from_partial(im)], grouped[kappa_from_partial(inn)]
            bm = sum(x in occupied for x in gm)
            bn = sum(x in occupied for x in gn)
            shape = (len(gm) - bm, len(gn) - bn)
            count = shape[0] * shape[1]
            d[bm : len(gm), bn : len(gn), c] = v[offset : offset + count].reshape(
                shape, order="F"
            )
            offset += count
        return s, d

    action_error = float(
        np.max(
            abs(
                problem.action(pack(ts, td))
                - problem.source
                - pack(ds - rs - extra_s, dd - source - rd - extra_d)
            )
        )
    )
    solution = problem.solve(residual_tolerance_Ha=1e-11)
    ordered = [ref] + [m for m in masks if m != ref]
    indices = [index[m] for m in ordered]
    dense = dense_sd_solve(
        physical[np.ix_(indices, indices)],
        tuple(((m, 1.0),) for m in ordered),
        ref,
        reference_identity="independent finite magnetic SD commutator, native DC",
        residual_tolerance_Ha=1e-11,
    )
    ys = generator(*unpack(solution.amplitudes)) @ reference
    solution_error = float(np.max(abs(ys[indices[1:]] - dense.amplitudes)))
    np.savez_compressed(
        directory / "solved.npz",
        raw=solution.amplitudes,
        independent=dense.amplitudes,
        physical=physical,
        masks=masks,
        source=problem.source,
        denominators=problem.denominators,
    )
    evidence = dict(
        case=case,
        virtual_count_per_kappa=virtual_count,
        requested_lmax=lmax,
        kinetic_balance=kinetic_balance,
        rotation=rotate,
        determinants=len(masks),
        dimensions=dims,
        counts=counts.tolist(),
        workspace_bytes=workspace,
        canonical_commutator_max_Ha=float(np.max(abs(independent - reduced))),
        energy_pci_Ha=float(energy[0]),
        energy_independent_Ha=float(reference @ full @ t @ reference),
        native_noncanonical_commutator_max_Ha=float(
            np.max(abs(native_independent - native_reduced))
        ),
        native_energy_pci_Ha=float(energy[0] + singles_energy),
        native_energy_independent_Ha=float(reference @ physical @ t @ reference),
        native_fock_fortran_vs_python_Ha=float(
            max(np.max(abs(extra_s - fortran_s)), np.max(abs(extra_d - fortran_d)))
        ),
        native_energy_fortran_Ha=float(fortran_energy[0]),
        vector_action_error_Ha=action_error,
        solved_physical_amplitudes_error=solution_error,
        native_core_correlation_Ha=solution.correlation_energy_Ha,
        independent_correlation_Ha=dense.correlation_energy_Ha,
        solve_residual_max_Ha=solution.residual_max_Ha,
        solve_matvecs=solution.matrix_vector_products,
        solve_wall_seconds=solution.wall_seconds,
        resources=solution.provenance["resources"],
        wall_seconds=time.perf_counter() - started,
        approximation="canonical calibration and complete native noncanonical DC Hamiltonian",
        independent_projection="complete fixed-M0/even SD determinants; exact linked SD residual",
    )
    np.savez_compressed(
        directory / "control.npz",
        metadata=metadata,
        source=source,
        initial=initial,
        ts=ts,
        td=td,
        rs=rs,
        rd=rd,
        independent=independent,
        reduced=reduced,
        eps=eps,
        full=full,
        extra_s=extra_s,
        extra_d=extra_d,
        native_independent=native_independent,
        native_reduced=native_reduced,
    )
    (directory / "result.json").write_text(json.dumps(evidence, indent=2) + "\n")
    print(json.dumps(evidence, indent=2), flush=True)
    if (
        max(
            evidence["canonical_commutator_max_Ha"],
            evidence["native_noncanonical_commutator_max_Ha"],
            evidence["native_fock_fortran_vs_python_Ha"],
            abs(energy[0] - evidence["energy_independent_Ha"]),
            abs(fortran_energy[0] - evidence["native_energy_independent_Ha"]),
            action_error,
            solution_error,
            abs(solution.correlation_energy_Ha - dense.correlation_energy_Ha),
        )
        > 1e-11
    ):
        raise AssertionError("predeclared signed commutator/energy gate failed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--case", choices=("be2", "be4", "ne10"), default="be2")
    parser.add_argument("--virtual-count", type=int, default=1)
    parser.add_argument("--rotate", action="store_true")
    parser.add_argument("--lmax", type=int, default=1)
    parser.add_argument("--kinetic-balance", action="store_true")
    args = parser.parse_args()
    main(
        args.directory,
        case=args.case,
        virtual_count=args.virtual_count,
        rotate=args.rotate,
        lmax=args.lmax,
        kinetic_balance=args.kinetic_balance,
    )
