"""Complete finite noncanonical Sigma2 Fov primitive, independent signed gates."""

import ctypes
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OWNER = Path(__file__).resolve().parent
REF = Path("/home/zzhang/code/atomx-references/2026-10-09-ci-fscc")
sys.path.insert(0, str(ROOT / "src"))

from build_sigma2_fock_control import build_sigma2_fock  # noqa: E402
from finite_valence_fixture import prepare  # noqa: E402
from general_rho_control import expand_valence  # noqa: E402
from probe_sigma2_control003 import valence_channels  # noqa: E402
from sigma2_fock_oracle import action, fermion_projection  # noqa: E402
from sigma2_magnetic_oracle import expand_sigma2  # noqa: E402

from atomx.angular.recoupling import l_of_kappa  # noqa: E402
from atomx.angular.sd_rho import kappa_from_partial  # noqa: E402


def main():
    here = OWNER / "sigma2_fock001"
    here.mkdir(exist_ok=False)
    archive = here / "source"
    archive.mkdir()
    names = (
        Path(__file__).name,
        "sigma2_fock_oracle.py",
        "sigma2_fock_control.f90",
        "build_sigma2_fock_control.py",
        "build_sigma2_control.py",
        "sigma2_control.f90",
        "build_valence_control.py",
        "valence_control.f90",
        "finite_valence_fixture.py",
        "general_rho_control.py",
        "probe_sigma2_control003.py",
        "sigma2_magnetic_oracle.py",
    )
    for name in names:
        shutil.copy2(OWNER / name, archive / name)
    (here / "source_sha256.json").write_text(
        json.dumps(
            {
                name: hashlib.sha256((archive / name).read_bytes()).hexdigest()
                for name in names
            },
            indent=2,
        )
        + "\n"
    )
    occupied = ((1, -1), (2, -1))
    virtual = ((3, -1), (4, -1), (2, 1), (3, 1), (2, -2), (3, -2))
    rows = {key: valence_channels(occupied, key[1], 1, "rho") for key in virtual}
    vw, cn = [], []
    for iv, v in enumerate(virtual):
        for iw, w in enumerate(virtual):
            for im in range(3):
                km = kappa_from_partial(im + 1)
                for jn in range(3):
                    kn = kappa_from_partial(jn + 1)
                    for k in range(
                        max(abs(abs(km) - abs(v[1])), abs(abs(kn) - abs(w[1]))),
                        min(abs(km) + abs(v[1]) - 1, abs(kn) + abs(w[1]) - 1) + 1,
                    ):
                        if (l_of_kappa(km) + l_of_kappa(v[1])) % 2 == (
                            l_of_kappa(kn) + l_of_kappa(w[1])
                        ) % 2:
                            vw.append((iv, iw, im, jn, k))
                for c, (_, kc) in enumerate(occupied):
                    for k in range(
                        max(abs(abs(kc) - abs(v[1])), abs(abs(km) - abs(w[1]))),
                        min(abs(kc) + abs(v[1]) - 1, abs(km) + abs(w[1]) - 1) + 1,
                    ):
                        if (
                            not (l_of_kappa(kc) + l_of_kappa(v[1]) + k) % 2
                            and not (l_of_kappa(km) + l_of_kappa(w[1]) + k) % 2
                        ):
                            cn.append((iv, iw, c, im, k))
    sizes = dict(
        NZ=6,
        NXV=1,
        NVH=1,
        NVH2=sum(map(len, rows.values())),
        NVWH=len(vw) + 1,
        NXCN=len(cn) + 1,
        NXV2=sum(
            max(
                len(valence_channels(occupied, key[1], 1, kind))
                for kind in ("x1", "x2")
            )
            for key in virtual
        )
        + 1,
    )

    def builder(source, directory, dimensions, compiler, *, nxv, nvh, runtime):
        sizes.update(NXV=nxv, NVH=nvh)
        # Complete image plus all control outputs and determinant vectors.
        extra = 8 * (
            dimensions["NXX"] ** 2 * 12 * sum(sizes.values())
            + 20**4 * 12
            + 120**2 * 12
            + dimensions["NHF"] * 40 * 6
        )
        if extra + 82 * 1024**2 > 512 * 1024**2:
            raise MemoryError("complete finite noncanonical screening exceeds cap")
        (here / "resource.json").write_text(
            json.dumps(
                dict(
                    extra_payload_bytes=extra,
                    cap_bytes=512 * 1024**2,
                    load=os.getloadavg(),
                    finite_core_preflight_bound_bytes=82 * 1024**2,
                    sizes=sizes,
                    performance="correctness only",
                ),
                indent=2,
            )
            + "\n"
        )
        return build_sigma2_fock(
            source, directory, dimensions, compiler, sizes=sizes, runtime=runtime
        )

    core, _, plan, actual_rows, _, _, problem, fock = prepare(
        here, REF, backend_builder=builder
    )
    if actual_rows != rows or tuple(plan["virtual_keys"]) != virtual:
        raise AssertionError("declared complete channel/state identity differs")
    labels = problem.integrals.labels
    nc, n = problem.integrals.noccupied, len(labels)
    eps = np.diag(fock).copy()
    energies = eps + 0.075
    d = plan["dimensions"]
    rng = np.random.default_rng(2357921)
    sigmas = np.zeros((n, n, nc, n))
    primes = np.zeros_like(sigmas)
    radial, radial_prime = {}, {}
    for key in virtual:
        value = np.zeros((d["NXX"], d["NXX"], len(rows[key])), order="F")
        prime = np.zeros_like(value, order="F")
        for c, (_, im, jn, _) in enumerate(rows[key]):
            for m, mk in enumerate(plan["groups"][im]):
                for nn, nk in enumerate(plan["groups"][jn]):
                    if mk in virtual and nk in virtual:
                        value[m, nn, c], prime[m, nn, c] = rng.normal(size=2) * 0.01
        radial[key], radial_prime[key] = value, prime
        for mj in range(-2 * abs(key[1]) + 1, 2 * abs(key[1]), 2):
            col = labels.index((*key, mj))
            sigmas[:, :, :, col] = expand_valence(problem, plan, rows[key], value, col)
            primes[:, :, :, col] = expand_valence(problem, plan, rows[key], prime, col)
    ptr = np.ctypeslib.ndpointer(dtype=np.float64, flags="F_CONTIGUOUS")
    iptr = np.ctypeslib.ndpointer(dtype=np.int32, flags="F_CONTIGUOUS")
    keys = np.asfortranarray(np.asarray(virtual).T, dtype=np.int32)
    val_energies = np.asfortranarray(
        [energies[labels.index((*k, -2 * abs(k[1]) + 1))] for k in virtual]
    )
    init = problem._backend.atomx_vw_init
    init.argtypes = [ctypes.c_int, iptr, ptr]
    init.restype = None
    init(len(virtual), keys, val_energies)
    load = problem._backend.atomx_vw_load
    load.argtypes = [ctypes.c_int, ctypes.c_int, iptr, ptr, ptr]
    load.restype = None
    for i, key in enumerate(virtual):
        channels = np.asfortranarray(np.asarray(rows[key]).T + 1, dtype=np.int32)
        channels[3] -= 1
        load(i + 1, len(rows[key]), channels, radial[key], radial_prime[key])
    count = np.zeros(1, dtype=np.int32)
    channels = np.zeros((5, sizes["NVWH"]), dtype=np.int32, order="F")
    result = np.zeros((d["NXX"], d["NXX"], sizes["NVWH"] + 1), order="F")
    derivative = np.zeros_like(result, order="F")
    problem._call("atomx_vw_fock", [count, channels, result, derivative])
    full_rows = channels[:, : count[0]].T - np.asarray([1, 1, 1, 1, 0])
    native = expand_sigma2(labels, plan, virtual, full_rows, result)
    native_prime = expand_sigma2(labels, plan, virtual, full_rows, derivative)
    independent = action(fock, eps, sigmas, energies, nc)
    independent_prime = action(fock, eps, sigmas, energies, nc, primes=primes)
    signed_error = float(np.max(abs(native - independent)))
    derivative_error = float(np.max(abs(native_prime - independent_prime)))
    pairs, fermion, reverse = fermion_projection(fock, eps, sigmas, energies, nc)
    physical = np.asarray(
        [
            [
                independent[m - nc, nn - nc, a - nc, b - nc]
                - independent[nn - nc, m - nc, a - nc, b - nc]
                for a, b in pairs
            ]
            for m, nn in pairs
        ]
    )
    fermion_error = float(np.max(abs(fermion - physical)))
    np.savez_compressed(
        here / "complete_vectors.npz",
        sigmas=sigmas,
        primes=primes,
        fock=fock,
        eps=eps,
        energies=energies,
        labels=np.asarray(labels),
        native=native,
        native_prime=native_prime,
        independent=independent,
        independent_prime=independent_prime,
        fermion=fermion,
        physical=physical,
        reverse=reverse,
        pairs=np.asarray(pairs),
    )
    record = dict(
        full_signed_coordinates=int(native.size),
        physical_pair_coordinates=int(physical.size),
        signed_error_Ha=signed_error,
        joint_response_error_Ha=derivative_error,
        fermion_error_Ha=fermion_error,
        reverse_projection_Ha=float(np.max(abs(reverse))),
        core_identity=problem.provenance,
        seed=2357921,
        status="pass"
        if max(signed_error, derivative_error, fermion_error) <= 1e-11
        else "fail",
    )
    (here / "comparison.json").write_text(json.dumps(record, indent=2) + "\n")
    if record["status"] != "pass" or np.max(abs(reverse)) != 0:
        raise AssertionError(
            "complete noncanonical screening signed/fermion gate failed"
        )
    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
