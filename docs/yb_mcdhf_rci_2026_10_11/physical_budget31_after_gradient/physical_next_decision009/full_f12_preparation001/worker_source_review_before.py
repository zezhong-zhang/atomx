"""SOURCE by default; one sealed full B+4f12 symmetry block and exact A restriction.

Default preparation uses JSON/integer metadata only. --execute requires a new
root seal and completed same-source small-space qualification. No public full
CIResult, tracker or retained snapshot is fabricated from selected roots.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import importlib.util
import json
import os
import resource
import signal
import sys
import time
import traceback
import zipfile
from dataclasses import asdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "pyproject.toml").is_file())
BASE = HERE.parent.parent
MATCH = HERE.parent / "matched001"
THREADS = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMBA_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS")


def read(path):
    return json.loads(Path(path).read_text())


def plain(value):
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    return plain(value.tolist()) if hasattr(value, "tolist") else value


def write(path, value):
    temp = path.with_suffix(path.suffix + ".part")
    temp.write_text(json.dumps(plain(value), indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(1024**2), b""):
            h.update(part)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(plain(value), sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def reference_metadata():
    path = BASE / "physical_next_decision006/execution001/run.py"
    spec = importlib.util.spec_from_file_location("bound_frozen_preparation", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.metadata()


def source_check(c):
    for item in c["metadata_inputs"].values():
        if sha(ROOT / item["path"]) != item["sha256"]:
            raise ValueError("metadata changed: " + item["path"])
    source = read(MATCH / "current_source_identity.json")
    mapping = source["all_atomx_python_sha256"]
    if digest(mapping) != c["source_map_sha256"]:
        raise ValueError("source manifest mismatch")
    if {str(p.relative_to(ROOT)) for p in (ROOT / "src/atomx").rglob("*.py")} != set(mapping):
        raise ValueError("source inventory changed")
    for path, expected in mapping.items():
        if sha(ROOT / path) != expected:
            raise ValueError("dirty source changed: " + path)
    if sha(__file__) != c["driver_sha256"]:
        raise ValueError("worker driver changed")


def prepare():
    # Existing metadata guards include complete immutable f043 inputs/schema;
    # their science branch is not entered and no NPZ is opened here.
    bc, _paths, _source, _identity, _base = reference_metadata()
    plan = read(HERE / "plan.json")
    source = read(MATCH / "current_source_identity.json")
    mc = read(MATCH / "contract.json")
    fingerprint = digest(source["all_atomx_python_sha256"])
    if fingerprint != mc["source_map_sha256"]:
        raise ValueError("same-source matched qualification manifest mismatch")
    metadata = {name: spec for name, spec in bc["inputs"].items() if spec["kind"] == "metadata"}
    for name, path in (("models_complete", HERE / "complete_models.json"),
                       ("physical_plan", HERE / "plan.json"),
                       ("matched_contract", MATCH / "contract.json"),
                       ("current_source_identity", MATCH / "current_source_identity.json")):
        metadata[name] = dict(path=str(path.relative_to(ROOT)), sha256=sha(path))
    c = dict(status="SOURCE_READY_FULL_4F12_NOT_EXECUTED_OR_ADMITTED",
             driver_sha256=sha(__file__), source_map_sha256=fingerprint,
             metadata_inputs=metadata,
             arrays=bc["arrays"], array_inputs={k: v for k, v in bc["inputs"].items()
                                               if v["kind"] == "array"},
             fixed=bc["fixed"], UIDs=bc["UIDs"],
             angular_census=plan["angular_census"],
             resources=plan["resources_proposed"], solver=plan["solver"],
             gates=dict(old_subspace_selected_action_maxabs_Ha=1e-8,
                        Gram_maxabs=1e-10, radial_Gram_maxabs=1e-10,
                        no_pair_metric_norm2=1e-12, coordinate_wave_maxabs=1e-12,
                        native_reference_binding_exact=True,
                        same_symmetry_observed_two_root_gap_Ha_min=1e-6,
                        strict_signed_reference_overlap_squared_min=0.5,
                        finite_full_operator_L2_residual_Ha_max=1e-8,
                        signed_bilinear_symmetry_Ha_max=1e-8),
             decoded_input_bytes_cap=256*1024**2,
             caps=dict(H_build=1, numerical_determinant_CSR=1, angular_coefficient_table=0,
                       radial_evaluator=1, radial_value_calls=384706,
                       raw_hS=7, Davidson=2, native_verify=1,
                       block_projection=1, exact_block_matrix=1,
                       operator_columns_per_arm=1000),
             required_matched_status_prefix="PASS", public_full_owner_qualified=False,
             control_reference="Ownedaccepted13 signedC, original531physicalUID ancestry in004",
             stop="First failure writes sharedSTOP and preserves partial; remaining workers reject. No retries/model/cap/gate changes.")
    write(HERE / "worker_contract.json", c)
    source_check(c)
    print(json.dumps(dict(status=c["status"], science_calls=0,
                          current_source_files=len(source["all_atomx_python_sha256"]),
                          worker_sha256=sha(__file__), contract_sha256=sha(HERE/"worker_contract.json"))))


def execute(seal_path):
    c, seal = read(HERE / "worker_contract.json"), read(seal_path)
    source_check(c)
    sector, cpu = tuple(seal["sector"]), seal["CPU"]
    expected = dict(execution_authorized_by_root=True, once_only=True,
                    worker_sha256=sha(__file__), contract_sha256=sha(HERE/"worker_contract.json"),
                    source_map_sha256=c["source_map_sha256"], sector=list(sector), CPU=cpu,
                    resources=c["resources"],
                    matched_result_sha256=sha(MATCH/"actual/result.json"))
    if seal != expected:
        raise ValueError("exact new root seal required")
    matched = read(MATCH/"actual/result.json")
    if (not matched["status"].startswith("PASS")
            or matched["source_map_sha256"] != c["source_map_sha256"]
            or not matched["checks"] or not all(x["PASS"] for x in matched["checks"])):
        raise ValueError("completed same-source matched qualification required")
    cap = c["resources"]
    if (sector not in ((1,1),(3,1),(5,1),(1,-1),(3,-1))
            or cpu not in cap["sector_CPUs"] or os.sched_getaffinity(0) != {cpu}
            or os.path.abspath(sys.executable) != "/data/code/atomx/.venv/bin/python"
            or any(os.environ.get(name) != "1" for name in THREADS)):
        raise ValueError("worker CPU/thread/interpreter/sector guard")
    available = next(int(line.split()[1])*1024 for line in Path("/proc/meminfo").read_text().splitlines()
                     if line.startswith("MemAvailable:"))
    if os.getloadavg()[0] > cap["host_load1_max"] or available < cap["host_MemAvailable_GiB_min"]*1024**3:
        raise ValueError("host load/memory guard")
    stop = HERE / "STOP"
    if stop.exists():
        raise ValueError("another sector already failed")
    out = HERE / (f"actual_J{sector[0]}_" + ("even" if sector[1] == 1 else "odd"))
    out.mkdir()
    resource.setrlimit(resource.RLIMIT_AS, (cap["AS_bytes"], cap["AS_bytes"]))
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("1200s hard stop")))
    signal.alarm(cap["inner_seconds"])
    start = time.monotonic()
    counts, checks = collections.Counter(), []
    report = dict(status="RUNNING", sector=sector, CPU=cpu, counts=counts, checks=checks,
                  source_map_sha256=c["source_map_sha256"],
                  qualified_current_source_match_sha256=seal["matched_result_sha256"],
                  input_physics=c["fixed"], stage="admission", selected={},
                  full_orbital_response_qualified=False, experimental_accuracy_qualified=False,
                  public_partial_tracker_or_snapshot_qualified=False, controlled_timing=False,
                  numerical_root_isolation_is_not_rigorous_omitted_spectrum_certificate=True)

    def checkpoint():
        report.update(counts=dict(counts), elapsed_seconds=time.monotonic()-start,
                      peak_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
        write(out / "partial.json", report)

    def check(name, value, limit=None):
        ok = bool(value) if limit is None else bool(value <= limit)
        checks.append(dict(name=name, observed=plain(value), limit=limit, PASS=ok))
        if not ok:
            raise RuntimeError("gate failed: " + name)

    def entry(label):
        counts[label] += 1
        if counts[label] > c["caps"][label]:
            raise RuntimeError("entry cap exceeded: " + label)

    try:
        for role, spec in c["array_inputs"].items():
            if sha(ROOT/spec["path"]) != spec["sha256"]:
                raise ValueError("whole array file changed: " + role)
        sys.path.insert(0, str(ROOT / "src"))
        import numpy as np
        from scipy.sparse import issparse
        from scipy.sparse.linalg import LinearOperator

        from atomx.angular import csf_hamiltonian as angular
        from atomx.angular import mscheme
        from atomx.config.csf import enumerate_jj_csfs
        from atomx.methods.ci import davidson_eigh
        from atomx.numerics.bspline import orbital_optimization as opt
        from atomx.physics.csf import matrix
        from atomx.physics.csf.effective_one_body import CIReferenceBinding
        from atomx.potentials.nuclear import make_nuclear_potential

        def observe(module, name, label):
            original = getattr(module, name)
            def wrapped(*args, **kw):
                entry(label)
                return original(*args, **kw)
            setattr(module, name, wrapped)
        for module, name, label_ in ((matrix, "build_dirac_matrices", "raw_hS"),
                                     (matrix, "exact_block_matrix", "exact_block_matrix"),
                                     (angular, "csf_block_matrix", "block_projection"),
                                     (angular, "_verify_cached", "native_verify")):
            observe(module, name, label_)

        allowed = {(row["twoj"], row["q"]) for row in c["angular_census"]}
        highest = mscheme.highest_weight_states
        check("cold_highest_weight_cache", highest.cache_info().currsize == 0)
        highest_seen = set()
        def guarded_highest(j, q):
            if (j,q) not in allowed:
                raise ValueError("undeclared subshell angular case")
            highest_seen.add((j,q))
            return highest(j,q)
        mscheme.highest_weight_states = guarded_highest
        original_null = mscheme.null_space
        def null(*args, **kw):
            counts["null_space"] += 1
            if counts["null_space"] > sum(row["nullspace_calls"] for row in c["angular_census"]):
                raise RuntimeError("cold angular nullspace cap")
            return original_null(*args, **kw)
        mscheme.null_space = null
        def forbidden(*_args, **_kw):
            counts["forbidden_table"] += 1
            raise RuntimeError("full angular coefficient table forbidden")
        angular._coefficient_table = forbidden
        original_basis = angular._block_basis
        label = f"2J={sector[0]} parity={sector[1]}"
        models = read(HERE / "complete_models.json")["models"]
        plan = read(HERE / "plan.json")
        b_expected = models["B"]["sectors"][label]
        original_connected = angular._connected_pairs
        def connected(dets, **kw):
            if kw.get("triangular", False):
                raise ValueError("undeclared radial symmetry optimization")
            for row, col in original_connected(dets, **kw):
                counts["structural_pairs"] += 1
                if counts["structural_pairs"] > plan["memory_per_B_sector"][label]["ordered_structural_pairs"]:
                    raise RuntimeError("structural pair cap")
                if counts["structural_pairs"] % 1_000_000 == 0:
                    if stop.exists():
                        raise RuntimeError("another sector failed")
                    checkpoint()
                yield row, col
        angular._connected_pairs = connected
        def basis(*args, **kw):
            value = original_basis(*args, **kw)
            patterns, dets, slices, labels = value
            check("complete native pattern identity", digest(patterns) == b_expected["patterns_sha256"])
            check("full native raw dimension", len(dets) == b_expected["raw_M_determinants"])
            check("69 electrons", all(mask.bit_count() == 69 for mask in dets))
            frozen = dets[0]
            for mask in dets:
                frozen &= mask
            check("common frozen spinors", frozen.bit_count() == 46)
            report["raw_determinant_sha256"] = digest(dets)
            report["patterns_sha256"] = digest(patterns)
            return patterns, dets, slices, labels
        angular._block_basis = basis
        original_stream = angular._streamed_determinant_matrix
        def streamed(*args, **kw):
            entry("numerical_determinant_CSR")
            value = original_stream(*args, **kw)
            check("full structural pair census", counts["structural_pairs"] == plan[
                "memory_per_B_sector"][label]["ordered_structural_pairs"])
            check("stored numeric CSR", issparse(value) and value.format == "csr")
            check("finite full Hdet", np.isfinite(value.data).all())
            check("Hdet structural nnz bound", value.nnz <= counts["structural_pairs"])
            report["Hdet"] = dict(shape=value.shape, nnz=value.nnz,
                                  bytes=value.data.nbytes+value.indices.nbytes+value.indptr.nbytes)
            checkpoint()
            return value
        angular._streamed_determinant_matrix = streamed
        original_radial = matrix._radial_evaluator
        def radial(*args, **kw):
            entry("radial_evaluator")
            evaluate = original_radial(*args, **kw)
            def evaluate_bound(key):
                entry("radial_value_calls")
                return evaluate(key)
            return evaluate_bound
        matrix._radial_evaluator = radial
        decoded = 0
        def archive(role):
            nonlocal decoded
            spec = c["arrays"][role]
            path = ROOT / c["array_inputs"][role]["path"]
            with zipfile.ZipFile(path) as z:
                if len(z.infolist()) > 100 or sum(x.file_size for x in z.infolist()) > spec["uncompressed_cap"]:
                    raise ValueError("archive header cap")
            result = {}
            with np.load(path, allow_pickle=False) as z:
                for key, field in spec["fields"].items():
                    value = z[key]
                    shapes = [field["shape"]] if "shape" in field else field["shape_choices"]
                    if value.dtype.str != field["dtype"] or list(value.shape) not in shapes or not np.isfinite(value).all():
                        raise ValueError("declared array schema changed: "+role+"/"+key)
                    decoded += value.nbytes
                    if decoded > c["decoded_input_bytes_cap"]:
                        raise ValueError("decoded input byte cap")
                    result[key] = value.copy()
            counts["NPZ_load"] += 1
            return result
        dense = {arm: archive(arm+"_R0") for arm in ("A", "B")}
        geometry, sm, cp = (archive(role) for role in ("workspace", "saved_S", "checkpoint_arrays"))
        data = {key[6:]: value.item() if value.ndim == 0 else value
                for key, value in geometry.items() if key.startswith("basis_")}
        keys = tuple(map(tuple, dense["B"]["keys"].tolist()))
        waves = {key:(p,q) for key,p,q in zip(keys,dense["B"]["P"],dense["B"]["Q"],strict=True)}
        check("same full A/B radials", all(np.array_equal(dense["A"][key],dense["B"][key])
                                          for key in ("keys","P","Q")))
        nucleus = make_nuclear_potential(70,"fermi",A=175)
        binding = asdict(CIReferenceBinding.capture(
            [(*waves[key],key[1]) for key in sorted(keys)],{},data,70,123,nucleus))
        check("original nuclear/radial binding", plain(binding) == read(ROOT / c[
            "metadata_inputs"]["checkpoint"]["path"])["owner"]["radial_binding"])
        layout = [(tuple(map(tuple,row["keys"])),cp["projector_"+str(i)],tuple(row["shape"]),slice(*row["slice"]))
                  for i,row in enumerate(read(ROOT / c["metadata_inputs"]["layout"]["path"]))]
        inverse = tuple(tuple(cp["inverse_blocks_"+str(i)]) for i in range(7))
        check("original P/S metric identity", opt._residual_metric_sha256(
            tuple(layout),cp["inverse_diagonal"],inverse) == c["fixed"]["metric_sha256"])
        for group,p,shape,sl in layout:
            ss=sm["s_"+str(group[0][1])]
            cc=np.column_stack([np.concatenate(waves[key]) for key in group])
            yy=cp["x"][sl].reshape(shape)
            check("radial metric Gram "+str(group),float(np.max(abs(cc.T@ss@cc-np.eye(len(group))))),1e-10)
            check("original coordinate wave "+str(group),float(np.max(abs(cc-p@yy))),1e-12)
            leak=cc-p@(p.T@ss@cc)
            check("no-pair "+str(group),float(np.max(np.einsum("ij,ij->j",leak,ss@leak))),1e-12)
        report.update(stage="native_CSF_enumeration", radial_binding=binding)
        checkpoint()
        full, nr_owners = [], []
        parent_count = 0
        for row in models["B"]["NR_configurations"]:
            parent=enumerate_jj_csfs(row["configuration"])
            parent_count+=len(parent)
            selected=[csf for csf in parent if (csf.total_J2,csf.parity)==sector]
            full.extend(selected)
            nr_owners.extend([row["NR_index"]]*len(selected))
        check("full CSF count",len(full)==b_expected["CSFs"])
        check("all angular subshell cases",highest_seen==allowed)
        check("cold highest-weight calls",counts["null_space"]==sum(row["nullspace_calls"] for row in c["angular_census"]))
        signatures=[json.dumps(plain((s.subshells,s.coupling,csf.total_J2,csf.parity)),separators=(",",":"))
                    for csf in full for s in (angular.csf_signature(csf),)]
        check("unique full signed signatures",len(set(signatures))==len(signatures))
        index={signature:i for i,signature in enumerate(signatures)}
        a_indices=np.asarray([i for i,n in enumerate(nr_owners) if n<91 or n>=100],dtype=int)
        check("full A nested CSF count",len(a_indices)==models["A"]["sectors"][label]["CSFs"])
        a_patterns=sorted({angular.csf_signature(full[int(i)]).pattern for i in a_indices})
        check("full A pattern identity",digest(a_patterns)==models["A"]["sectors"][label]["patterns_sha256"])
        maps=read(ROOT/c["metadata_inputs"]["basis_maps"]["path"])
        old_info={}
        for arm,model in (("A","baseline"),("B","expanded")):
            old_sigs=maps[model]["selected_signatures"]
            old_rows=[i for i,s in enumerate(old_sigs) if tuple(json.loads(s)[2:])==sector]
            new_rows=np.asarray([index[old_sigs[i]] for i in old_rows],dtype=int)
            target=next(row for row in read(ROOT/c["metadata_inputs"][arm+"_R0_record"]["path"])["rows"]
                        if tuple(row["UID"][:2])==sector)
            ref=dense[arm]["mixing"][old_rows,target["column"]]
            check(arm+" accepted13 reference norm",abs(float(ref@ref)-1),1e-10)
            reference=np.zeros(len(full))
            reference[new_rows]=ref/np.linalg.norm(ref)
            old_info[arm]=dict(old_rows=old_rows,new_rows=new_rows,reference=reference,target=target)
        write(out/"basis.json",dict(B_signatures=signatures,A_nested_indices=a_indices,
                                     B_signatures_sha256=digest(signatures),parent_all_J_CSFs=parent_count,
                                     ownership="Full declaredCSF basis, not a full CI eigensystem"))
        report.update(stage="native_full_B_H",CSFs=len(full),A_CSFs=len(a_indices),
                      B_basis_sha256=sha(out/"basis.json"))
        checkpoint()
        entry("H_build")
        H=matrix.build_ci_matrix(full,waves,keys,data,70,123,nuclear_potential=nucleus,
                                 max_determinants=92715,include_breit=False,as_operator=True,
                                 angular_storage="streamed")
        check("one full symmetry operator",isinstance(H,LinearOperator) and H.shape==(len(full),len(full)) and len(H.blocks)==1)
        rng=np.random.default_rng(20261014+sector[0]+10*sector[1])
        u,v=rng.normal(size=(2,len(full)))
        u/=np.linalg.norm(u)
        v/=np.linalg.norm(v)
        check("signed full bilinear symmetry",abs(float(u@(H@v)-v@(H@u))),1e-8)
        for arm in ("A","B"):
            if stop.exists():
                raise RuntimeError("another sector failed")
            selection=a_indices if arm=="A" else np.arange(len(full))
            diag=H.diagonal()[selection]
            def apply(value,selection=selection,arm=arm):
                columns=1 if value.ndim==1 else value.shape[1]
                counts["operator_columns_"+arm]+=columns
                if counts["operator_columns_"+arm]>c["caps"]["operator_columns_per_arm"]:
                    raise RuntimeError("operator columns cap")
                padded=np.zeros((len(full),)+value.shape[1:])
                padded[selection]=value
                return (H@padded)[selection]
            operator=LinearOperator((len(selection),len(selection)),matvec=apply,matmat=apply,rmatvec=apply,dtype=float)
            operator.diagonal=lambda diag=diag:diag.copy()
            old=old_info[arm]
            original_action=H@old["reference"]
            rows=old["old_rows"]
            refsmall=old["reference"][old["new_rows"]]
            original_expected=dense[arm]["H"][np.ix_(rows,rows)]@refsmall
            check(arm+" exact old reference submatrix action",float(np.max(abs(original_action[old["new_rows"]]-original_expected))),1e-8)
            report.update(stage="native_Davidson_"+arm)
            checkpoint()
            entry("Davidson")
            E,C,iterations=davidson_eigh(operator,n_states=2,tol=c["solver"]["tol_L2_Ha"],
                                        max_iter=c["solver"]["max_iter"])
            residual=np.linalg.norm(operator@C-C*E,axis=0)
            check(arm+" direct full-space residual",float(np.max(residual)),1e-8)
            check(arm+" selected Gram",float(np.max(abs(C.T@C-np.eye(2)))),1e-10)
            overlap=old["reference"][selection]@C
            weights=overlap**2
            chosen=int(np.argmax(weights))
            check(arm+" strict accepted13 reference majority",float(weights[chosen])>0.5)
            gap=float(abs(E[1]-E[0]))
            check(arm+" observed selected root gap",gap>c["gates"]["same_symmetry_observed_two_root_gap_Ha_min"])
            np.savez_compressed(out/(arm+"_selected.npz"),energies=E,mixing=C,
                                B_nested_indices=selection,accepted13_reference=old["reference"][selection],
                                residual_L2_Ha=residual,signed_overlap=overlap)
            report["selected"][arm]=dict(UID=old["target"]["UID"],energy_Ha=float(E[chosen]),
                                         accepted13_energy_Ha=old["target"]["energy_Ha"],
                                         energy_shift_Ha=float(E[chosen]-old["target"]["energy_Ha"]),
                                         selected_local_root=chosen,energy_pair_Ha=E,
                                         signed_overlap=overlap,overlap_squared=weights,
                                         observed_two_root_gap_Ha=gap,residual_L2_Ha=residual,
                                         iterations=iterations,full_CSF_rows=len(selection),
                                         selected_arrays_sha256=sha(out/(arm+"_selected.npz")),
                                         matching="Numerical isolated-root transport from ownedaccepted13; Bessel omission bound conditional on exacteigenstate. No rigorous omittedgapcertificate.")
            checkpoint()
        source_check(c)
        for name,module in tuple(sys.modules.items()):
            if name.startswith("atomx") and getattr(module,"__file__",None):
                path=Path(module.__file__).resolve()
                if path.suffix==".py" and not path.is_relative_to(ROOT/"src/atomx"):
                    raise ValueError("wrong imported Atomx origin")
        report.update(status="PASS_full_4f12_fixed_R0_sector_numerical_selected_roots",
                      stage="complete",whole_physical_class_energy_calculated=True,
                      no_radial_relaxation=True,coefficient_table_calls=counts["forbidden_table"])
    except BaseException as error:
        report.update(status="FAILED_preserved",error=repr(error),traceback=traceback.format_exc())
        stop.touch(exist_ok=True)
    finally:
        signal.alarm(0)
        checkpoint()
        write(out/"result.json",report)
    print(json.dumps(dict(status=report["status"],sector=sector,checks=len(checks),
                          passed=sum(row["PASS"] for row in checks),seconds=report["elapsed_seconds"],
                          peak_RSS_MiB=report["peak_RSS_MiB"])))
    return 0 if report["status"].startswith("PASS") else 1


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--execute",type=Path)
    args=parser.parse_args()
    if args.execute:
        raise SystemExit(execute(args.execute))
    prepare()
