"""PRIVATE native31 signed-gradient adapter preparation, not qualified code.

Only standard-library metadata work is available without a numerical factory.
H/CI/tracking/state weights/native constraints stay on their public CPU paths.
This adapter consumes the independently bound full normalized signed RDM; it
never constructs, filters, rescales or diagonalizes that physical density.
"""

import hashlib
import importlib.util
import json
import math
import sys
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path

ENGINE_PATH = Path(__file__).absolute().parent.parent / "private_batch_preparation001/batch_executor.py"
ENGINE_SHA256 = "37e30d4b02dddcc37ee35036b941b178de14e8f7d1fb8c5f448ea19920af67ae"
PUBLIC_MODULES = ("atomx.methods.csf.variational", "atomx.numerics.bspline.convol",
                  "atomx.physics.csf.radial_density", "atomx.physics.csf.matrix",
                  "atomx.methods.ci")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"))
                          .encode()).hexdigest()


def tuple_key(value):
    return tuple(tuple_key(v) for v in value) if isinstance(value, (tuple, list)) else value


def quantum_plain(value):
    """Canonical existing integer CSF metadata; no physical coercion."""
    if isinstance(value, (tuple, list)):
        return tuple(quantum_plain(v) for v in value)
    if isinstance(value, Integral) and not isinstance(value, bool):
        return int(value)
    raise ValueError("signed basis metadata requires exact integer quantum fields")


def frozen_engine():
    """Load one literal frozen source file, never AST/text fragments."""
    if hashlib.sha256(ENGINE_PATH.read_bytes()).hexdigest() != ENGINE_SHA256:
        raise ValueError("frozen batch executor source changed")
    name = "private_native_coulomb_batch_37e30d4b"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, ENGINE_PATH)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    module = sys.modules[name]
    if Path(module.__file__).absolute() != ENGINE_PATH:
        raise ValueError("batch executor namespace collision")
    return module


@dataclass(frozen=True, slots=True)
class SignedLayout:
    """Immutable coefficient order and original derivative incidence order."""

    keys: tuple
    items: tuple
    terms: tuple
    weights_sha256: str
    source_union_sha256: str

    @property
    def jobs(self):
        return tuple(job for job, _ in self.terms)


def collect_signed_layout(keys, rows):
    """Exact public source/term order; no abs cutoff or zero/sign selection."""
    keys = tuple(keys)
    if len(keys) != 31 or keys != tuple(sorted(set(keys))):
        raise ValueError("complete ordered native31 keys required")
    items, seen = [], set()
    for key, weight in rows:
        key = tuple_key(key)
        if (type(key) is not tuple or not key or key in seen
                or type(weight) is not float or not math.isfinite(weight)):
            raise ValueError("duplicate key or non-native binary64 signed coefficient")
        if key[0] == "h":
            if (len(key) != 4 or any(type(v) is not int for v in key[1:])
                    or (key[1], key[2]) not in keys or (key[3], key[2]) not in keys):
                raise ValueError("unsupported one-body radial key")
        elif key[0] == "r":
            if (len(key) != 6 or type(key[1]) is not int or not 0 <= key[1] <= 6
                    or any(v not in keys for v in key[2:])):
                raise ValueError("unsupported Coulomb radial key")
        else:
            raise ValueError("only complete h/r Dirac-Coulomb signed density supported")
        seen.add(key)
        items.append((key, weight))
    if not items:
        raise ValueError("full signed density cannot be empty")
    gradient_terms = {}
    for key, weight in items:
        if key[0] != "h":
            _, rank, a, b, c, d = key
            for left, right, source_a, source_b in ((a, b, c, d), (c, d, a, b)):
                if left not in keys and right not in keys:
                    continue
                pair = (rank, *sorted((source_a, source_b)))
                gradient_terms.setdefault(pair, []).append((left, right, weight))
    terms = tuple((job, tuple(values)) for job, values in gradient_terms.items())
    jobs = tuple(job for job, _ in terms)
    return SignedLayout(keys, tuple(items), terms,
                        digest(tuple((k, w.hex()) for k, w in items)), digest(jobs))


def plan_from_layout(layout, *, source_sha256, workspace_sha256, ns, orders,
                     extents, chunk_sources=128):
    engine = frozen_engine()
    return engine.BatchPlan(source_sha256, workspace_sha256, layout.source_union_sha256,
                            ns, orders, extents, layout.keys, layout.jobs, chunk_sources)


@dataclass(frozen=True, slots=True)
class NativeBinding:
    """Declared external proof identities; it is not an eigenpair certificate."""

    source_sha256: str
    public_file_sha256: tuple
    basis_signatures_sha256: str
    objective_sha256: str
    original_metric_sha256: str
    native_reference_result_sha256: str

    def __post_init__(self):
        engine = frozen_engine()
        for value in (self.source_sha256, self.basis_signatures_sha256,
                      self.objective_sha256, self.original_metric_sha256,
                      self.native_reference_result_sha256):
            engine._sha(value)
        if (type(self.public_file_sha256) is not tuple
                or tuple(row[0] for row in self.public_file_sha256) != PUBLIC_MODULES):
            raise ValueError("complete ordered public source module identities required")
        for row in self.public_file_sha256:
            if type(row) is not tuple or len(row) != 2:
                raise ValueError("immutable public source identity row required")
            engine._sha(row[1])


def validate_loaded_sources(binding):
    for name, expected in binding.public_file_sha256:
        module = sys.modules.get(name)
        if (module is None or not getattr(module, "__file__", None)
                or hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() != expected):
            raise ValueError("loaded public numerical source differs from declared epoch")


class NativeBatchGradient:
    """Retain one geometry and source layout; new support requires a new owner.

    Numerical construction/gradient is deferred until root exact qualification.
    The retained H owner validates actual physical inputs, not its eigenpairs
    or arbitrary caller weights. Complete CI residual/Gram/degenerate span,
    UID/.2, exact normalization and signed-ledger matching remain external live
    consumer gates. No energies or constraint/native metric code is changed.
    """

    def __init__(self, functional, orbitals, weights, ci_result, *, binding,
                 backend, threads=1, expected_weights_sha256):
        if type(binding) is not NativeBinding:
            raise ValueError("complete declared native proof binding required")
        if (backend not in ("cpu", "cuda") or type(threads) is not int
                or not 1 <= threads <= 64 or (backend == "cuda" and threads != 1)):
            raise ValueError("explicit supported backend/worker scope required")
        layout = collect_signed_layout(functional.keys, weights.items())
        if layout.weights_sha256 != expected_weights_sha256:
            raise ValueError("independently bound full signed weight ledger mismatch")
        validate_loaded_sources(binding)
        import numpy as np

        from atomx.methods.csf.variational import CSFOrbitalFunctional
        from atomx.numerics.bspline.convol import _density_sym

        if type(functional) is not CSFOrbitalFunctional or functional.Z != 70 or functional.ns != 123:
            raise ValueError("exact native scalar Z70/ns123 functional required")
        self.functional, self.binding = functional, binding
        self.np, self.layout = np, layout
        self.engine = frozen_engine()
        self.orders = (functional.data.get("ksp", functional.data["ks"]),
                       functional.data.get("ksq", functional.data["ks"]))
        if self.orders[0] != self.orders[1] and not {"nsp", "nsq"} <= set(functional.data):
            raise ValueError("unequal RKB requires explicit component extents")
        self.extents = (functional.data.get("nsp", functional.ns),
                        functional.data.get("nsq", functional.ns))
        self._check_functional(ci_result, orbitals)
        owner = ci_result.hamiltonian_snapshot
        self.workspace_sha256 = self._workspace_identity(owner)
        self.plan = plan_from_layout(layout, source_sha256=binding.source_sha256,
                                     workspace_sha256=self.workspace_sha256, ns=functional.ns,
                                     orders=self.orders, extents=self.extents)
        factors = {(rank, component): functional.cache.get_coulomb_factors(rank, component)
                   for rank in self.plan.ranks for component in self.engine.COMPONENTS}
        self.executor = self.engine.PrivateBatchExecutor(self.plan, factors,
                                                         backend=backend, threads=threads)
        self._density_sym = _density_sym
        self.counters = {"gradient": 0, "density_band_requests": 0,
                         "factor_requests": len(factors), "band_expansions": 0,
                         "signed_term_incidence": 0, "pair_actions": 0,
                         "one_body_actions": 0, "H_owner_validation": 1}

    def _check_functional(self, ci_result, orbitals):
        from atomx.angular.csf_hamiltonian import csf_signature

        functional = self.functional
        if set(orbitals) != set(self.layout.keys):
            raise ValueError("complete same native31 orbital set required")
        for pair in orbitals.values():
            if (len(pair) != 2 or any(value.shape != (functional.ns,)
                    or value.dtype != self.np.dtype(self.np.float64)
                    or not self.np.isfinite(value).all() for value in pair)):
                raise ValueError("complete finite real native binary64 radial vectors required")
            for value, extent in zip(pair, self.extents, strict=True):
                if self.np.any(value[extent:] != 0.0):
                    raise ValueError("nonzero structural RKB wave tail; no clipping")
        for matrices, shape in ((functional.one_body, (2 * functional.ns, 2 * functional.ns)),
                                (functional.metric, (2 * functional.ns, 2 * functional.ns))):
            if set(matrices) != {k[1] for k in self.layout.keys} or any(
                    value.shape != shape or value.dtype != self.np.dtype(self.np.float64)
                    or not self.np.isfinite(value).all() for value in matrices.values()):
                raise ValueError("complete finite native nuclear h/S blocks required")
        signatures = tuple(csf_signature(c) for c in functional.csfs)
        encoded = tuple(json.dumps(quantum_plain((s.subshells, s.coupling, c.total_J2, c.parity)),
                                   separators=(",", ":"))
                        for s, c in zip(signatures, functional.csfs, strict=True))
        if (len(signatures) != 1595 or {c.n_electrons for c in functional.csfs} != {69}
                or digest(encoded) != self.binding.basis_signatures_sha256
                or signatures != functional.radial_density.signatures
                or functional.keys != self.layout.keys
                or functional.level_tracking is None
                or functional.level_tracking.state_average != functional.state_average):
            raise ValueError("stale complete signed basis/functional/target declaration")
        # No arbitrary H, no public-build call and no eigenpair certification.
        ci_result.hamiltonian_for(functional.csfs, orbitals, functional.keys,
                                  functional.data, functional.Z, functional.ns,
                                  nuclear_potential=functional.nuclear_potential,
                                  max_determinants=functional.max_determinants)

    def _workspace_identity(self, owner):
        functional = self.functional
        hashes = tuple((k, hashlib.sha256(functional.one_body[k].tobytes()).hexdigest(),
                        hashlib.sha256(functional.metric[k].tobytes()).hexdigest())
                       for k in sorted(functional.one_body))
        return digest((owner.radial_binding.geometry, owner.radial_binding.nuclear,
                       functional.Z, functional.ns, hashes))

    def gradient_from_weights(self, orbitals, weights, ci_result, *, expected_weights_sha256,
                              record_diagnostics=True):
        """Only replace source-band execution; return original full raw g."""
        np, functional = self.np, self.functional
        if not isinstance(record_diagnostics, (bool, np.bool_)):
            raise TypeError("record_diagnostics must be boolean")
        current = collect_signed_layout(functional.keys, weights.items())
        if (current.jobs != self.plan.jobs or current.weights_sha256 != expected_weights_sha256):
            raise ValueError("changed complete source support/order or unbound signed table")
        validate_loaded_sources(self.binding)
        self._check_functional(ci_result, orbitals)
        self.counters["H_owner_validation"] += 1
        owner = ci_result.hamiltonian_snapshot
        if self._workspace_identity(owner) != self.workspace_sha256:
            raise ValueError("workspace/nuclear/one-body/metric changed; rebuild owner")
        from atomx.numerics.bspline.convol import _banded_to_full_add

        densities = {}
        for pair in self.plan.pairs:
            for right in range(2):
                densities[(pair, right)] = self._density_sym(
                    orbitals[pair[0]][right], orbitals[pair[1]][right], functional.ns, self.orders[right])
        self.counters["density_band_requests"] += len(densities)
        trial_sha256 = digest((owner.input_sha256, owner.matrix_sha256, current.weights_sha256,
                               self.binding.objective_sha256, self.binding.original_metric_sha256))
        snapshot = self.engine.DensitySnapshot.from_public_bands(
            self.plan, densities, trial_sha256=trial_sha256)
        # The literal public h updates, then pair actions and signed additions.
        if record_diagnostics:
            functional.last_weights = weights
        selected = functional.keys
        gradient = {key: np.zeros(2 * functional.ns) for key in selected}
        if record_diagnostics:
            functional.last_occupations = {key: max(0., weights.get(("h", key[0], key[1], key[0]), 0.))
                                           for key in functional.keys}
        vectors = {key: np.concatenate(orbitals[key]) for key in functional.keys}
        one_body_actions = {}
        for key, weight in current.items:
            if key[0] == "h":
                _, na, kappa, nb = key
                a, b = (na, kappa), (nb, kappa)
                h = functional.one_body[kappa]
                if a in gradient:
                    if b not in one_body_actions:
                        one_body_actions[b] = h @ vectors[b]
                        self.counters["one_body_actions"] += 1
                    gradient[a] += weight * one_body_actions[b]
                if b in gradient:
                    if a not in one_body_actions:
                        one_body_actions[a] = h @ vectors[a]
                        self.counters["one_body_actions"] += 1
                    gradient[b] += weight * one_body_actions[a]
        gradient_terms = dict(current.terms)
        for source_jobs, packets in self.executor.chunks(snapshot):
            for source_index, job in enumerate(source_jobs):
                output = [np.zeros((functional.ns, functional.ns)),
                          np.zeros((functional.ns, functional.ns))]
                for left in range(2):
                    for right in range(2):
                        _banded_to_full_add(output[left], packets[source_index, 2 * left + right],
                                            functional.ns, self.orders[left], 1.0)
                        self.counters["band_expansions"] += 1
                pp, qq = output
                actions = {}
                for left, right, weight in gradient_terms[job]:
                    self.counters["signed_term_incidence"] += 1
                    if left in gradient:
                        if right not in actions:
                            pr, qr = orbitals[right]
                            actions[right] = pp @ pr, qq @ qr
                            self.counters["pair_actions"] += 1
                        gp, gq = actions[right]
                        gradient[left][:functional.ns] += weight * gp
                        gradient[left][functional.ns:] += weight * gq
                    if right in gradient:
                        if left not in actions:
                            pl, ql = orbitals[left]
                            actions[left] = pp @ pl, qq @ ql
                            self.counters["pair_actions"] += 1
                        gp, gq = actions[left]
                        gradient[right][:functional.ns] += weight * gp
                        gradient[right][functional.ns:] += weight * gq
        if record_diagnostics:
            functional.last_gradient = gradient
        self.counters["gradient"] += 1
        return gradient
