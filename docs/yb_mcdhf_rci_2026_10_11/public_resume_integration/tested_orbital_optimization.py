"""Constrained minimization of a supplied radial state functional."""
import hashlib
import time
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np


def _array(value, name, ndim):
    """Own immutable float64 bytes, not a caller-writable read-only view."""
    value = np.asarray(value)
    if value.dtype != np.dtype(float) or value.ndim != ndim or not np.isfinite(value).all():
        raise ValueError(f"{name} requires finite float64 rank-{ndim} data")
    return np.frombuffer(value.tobytes(order="C"), dtype=float).reshape(value.shape)


def _hash_value(digest, value):
    def frame(kind, payload):
        encoded = kind.encode() + b":" + str(len(payload)).encode() + b":" + payload
        digest.update(encoded)

    if isinstance(value, np.ndarray):
        if value.dtype.hasobject or not np.isfinite(value).all():
            raise ValueError("optimizer geometry requires finite numerical arrays")
        frame("array", repr((value.shape, value.dtype.str)).encode())
        frame("bytes", value.tobytes(order="C"))
    elif isinstance(value, Mapping):
        frame("mapping", str(len(value)).encode())
        for key, item in sorted(value.items()):
            _hash_value(digest, key)
            _hash_value(digest, item)
    elif isinstance(value, (tuple, list)):
        frame(type(value).__name__, str(len(value)).encode())
        for item in value:
            _hash_value(digest, item)
    elif isinstance(value, (str, int, float, bool, np.number, type(None))):
        if isinstance(value, (float, np.number)) and not np.isfinite(value):
            raise ValueError("optimizer geometry requires finite scalars")
        frame(type(value).__name__, repr(value).encode())
    else:
        raise TypeError(f"unsupported optimizer geometry entry {type(value).__name__}")


def _physical_geometry_sha256(functional):
    digest = hashlib.sha256()
    _hash_value(digest, (functional.Z, functional.ns, tuple(functional.keys), functional.data,
                         functional.one_body, functional.metric))
    nuclear = functional.nuclear_potential
    sampled = (None if nuclear is None else np.asarray(
        [nuclear(r) for r in functional.data["gr"].ravel()], dtype=float))
    _hash_value(digest, sampled)
    return digest.hexdigest()


def _residual_metric_sha256(layout, inverse_diagonal, inverse_blocks):
    # Original numerical epoch used lists for keys; retain its exact digest.
    digest = hashlib.sha256()
    for keys, projector, shape, _ in layout:
        digest.update(repr((list(keys), shape)).encode())
        digest.update(np.ascontiguousarray(projector).tobytes())
    digest.update(inverse_diagonal.tobytes())
    for blocks in inverse_blocks:
        for block in blocks:
            digest.update(np.ascontiguousarray(block).tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class _OptimizationGeometry:
    keys: tuple
    layout: tuple
    inverse_diagonal: np.ndarray
    inverse_blocks: tuple
    projector_operators: tuple
    metrics: tuple
    projector_source: str
    occupation_floor: float
    physical_sha256: str
    residual_sha256: str
    payload_sha256: str


@dataclass(frozen=True)
class OrbitalOptimizationState:
    """Immutable accepted coordinates in a validated original metric.

    Import archived arrays through from_arrays. The owned float64 bytes bind
    complete same-kappa positive subspaces, workspace, nucleus and S geometry.
    No CI, gradient, secant pairs or convergence flag is stored.
    objective_identity is the caller's scientific identity declaration,
    not an inferred certification of its physical target.
    """

    _geometry: _OptimizationGeometry
    coordinates: np.ndarray
    objective_identity: str

    def __post_init__(self):
        if not isinstance(self._geometry, _OptimizationGeometry):
            raise TypeError("optimizer state requires validated geometry")
        if not isinstance(self.objective_identity, str) or not self.objective_identity:
            raise ValueError("objective_identity requires a nonempty scientific identity string")
        point = _array(self.coordinates, "coordinates", 1)
        if point.shape != self._geometry.inverse_diagonal.shape:
            raise ValueError("coordinates do not match the complete layout")
        for _, _, shape, where in self.layout:
            y = point[where].reshape(shape)
            if np.linalg.norm(y.T @ y - np.eye(shape[1]), ord=np.inf) > 1e-10:
                raise ValueError("coordinate point is not orthonormal; no polar repair is performed")
        for (_, projector, shape, where), (_, metric) in zip(
                self.layout, self._geometry.metrics, strict=True):
            coefficients = projector @ point[where].reshape(shape)
            if np.linalg.norm(coefficients.T @ metric @ coefficients - np.eye(shape[1]), ord=np.inf) > 1e-10:
                raise ValueError("physical radial point is not S-orthonormal")
        object.__setattr__(self, "coordinates", point)

    @property
    def layout(self):
        return self._geometry.layout

    @property
    def inverse_diagonal(self):
        return self._geometry.inverse_diagonal

    @property
    def inverse_blocks(self):
        return self._geometry.inverse_blocks

    @property
    def projector_source(self):
        return self._geometry.projector_source

    @property
    def occupation_floor(self):
        return self._geometry.occupation_floor

    @property
    def residual_metric_sha256(self):
        return self._geometry.residual_sha256

    @property
    def geometry_sha256(self):
        return self._geometry.payload_sha256

    @property
    def state_sha256(self):
        digest = hashlib.sha256(self.geometry_sha256.encode())
        _hash_value(digest, (self.coordinates, self.objective_identity))
        return digest.hexdigest()

    @classmethod
    def from_arrays(cls, functional, *, layout, coordinates, inverse_diagonal,
                    inverse_blocks=(), projector_source="nuclear",
                    objective_identity, projector_operators=None, occupation_floor=.05):
        """Validate original arrays without reconstructing Fock fields.

        Layout entries are (ordered_keys, projector, shape, contiguous_slice).
        Fock spaces require actual complete original projector operators
        keyed by kappa; inverse blocks cannot recover those operators.
        A validation eigensolve anchors the positive subspace but never
        replaces saved projectors or coordinates.
        """
        geometry = _seal_geometry(
            functional, layout, inverse_diagonal, inverse_blocks,
            projector_source, projector_operators, occupation_floor, verify_positive=True)
        return cls(geometry, coordinates, objective_identity)

    def validate(self, functional):
        """Reject changed caller geometry before evaluating its objective."""
        if tuple(functional.keys) != self._geometry.keys:
            raise ValueError("optimizer state ordered orbital keys do not match functional")
        if _physical_geometry_sha256(functional) != self._geometry.physical_sha256:
            raise ValueError("optimizer state workspace/nucleus/S geometry mismatch")
        geometry = self._geometry
        residual = _residual_metric_sha256(self.layout, self.inverse_diagonal, self.inverse_blocks)
        digest = hashlib.sha256((geometry.physical_sha256 + residual).encode())
        _hash_value(digest, (geometry.projector_operators, geometry.metrics,
                            geometry.projector_source, geometry.occupation_floor))
        if residual != geometry.residual_sha256 or digest.hexdigest() != geometry.payload_sha256:
            raise ValueError("optimizer state immutable geometry payload mismatch")


def _seal_geometry(functional, layout, inverse_diagonal, inverse_blocks,
                   projector_source, projector_operators, occupation_floor, *, verify_positive):
    from atomx.numerics.bspline.dirac_assembly import solve_dirac
    from atomx.numerics.bspline.orbital_subspace import radial_projector_start

    if not isinstance(layout, (tuple, list)) or not isinstance(inverse_blocks, (tuple, list)):
        raise ValueError("layout and inverse_blocks require nested sequences")
    if projector_source not in ("nuclear", "fock"):
        raise ValueError("projector_source must be nuclear or fock")
    if not np.isfinite(occupation_floor) or occupation_floor <= 0:
        raise ValueError("occupation_floor must be finite and positive")
    groups = {}
    for key in functional.keys:
        groups.setdefault(key[1], []).append(key)
    expected = [tuple(keys) for _, keys in sorted(groups.items())]
    if len(layout) != len(expected):
        raise ValueError("layout requires one complete group per kappa")
    owned, operators, metrics = [], [], []
    cursor = 0
    for entry, keys_expected in zip(layout, expected, strict=True):
        keys, projector, shape, where = entry
        keys, shape = tuple(tuple(key) for key in keys), tuple(shape)
        if keys != keys_expected or len(set(keys)) != len(keys):
            raise ValueError("layout ordered keys require one complete same-kappa group")
        if (len(shape) != 2 or any(isinstance(v, (bool, np.bool_))
                or not isinstance(v, (int, np.integer)) or v < 1 for v in shape)):
            raise ValueError("layout shape requires two positive integer extents")
        extent = int(np.prod(shape))
        if (not isinstance(where, slice) or where.step is not None
                or where.start != cursor or where.stop != cursor + extent):
            raise ValueError("layout slices must be contiguous and nonoverlapping")
        projector = _array(projector, "projector", 2)
        if projector.shape != (2 * functional.ns, shape[0]) or shape[1] != len(keys):
            raise ValueError("projector/layout shapes do not match physical spinors")
        kappa = keys[0][1]
        metric = _array(functional.metric[kappa], "physical S metric", 2)
        ns = functional.ns
        nsp, nsq = functional.data.get("nsp", ns), functional.data.get("nsq", ns)
        if any(isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer))
               or v < 1 or v > ns for v in (nsp, nsq)):
            raise ValueError("active large/small component extents must be positive integers within ns")
        if metric.shape != (2 * ns, 2 * ns):
            raise ValueError("physical S metric shape does not match spinors")
        active = np.r_[np.arange(nsp), ns + np.arange(nsq)]
        inactive = np.setdiff1d(np.arange(2 * ns), active)
        if np.any(metric[inactive]) or np.any(metric[:, inactive]) or np.any(projector[inactive]):
            raise ValueError("inactive padded metric/projector rows must be exactly zero")
        if (metric.shape != (2 * ns, 2 * ns) or not np.isfinite(metric).all()
                or not np.allclose(metric, metric.T, rtol=0, atol=1e-12)
                or np.linalg.eigvalsh(metric[np.ix_(active, active)]).min() <= 0):
            raise ValueError("physical S metric requires symmetric positive active geometry")
        if np.linalg.norm(projector.T @ metric @ projector - np.eye(shape[0]), ord=np.inf) > 1e-10:
            raise ValueError("saved projector is not S-orthonormal")
        if projector_source == "nuclear":
            operator = functional.one_body[kappa]
            if projector_operators is not None and not np.array_equal(projector_operators[kappa], operator):
                raise ValueError("nuclear projector operator differs from actual nuclear one-body")
        else:
            if projector_operators is None or kappa not in projector_operators:
                raise ValueError("Fock state requires explicit original projector operators")
            operator = projector_operators[kappa]
        operator = _array(operator, "projector operator", 2)
        if operator.shape != metric.shape or not np.allclose(operator, operator.T, rtol=0, atol=1e-12):
            raise ValueError("projector operator requires the complete real symmetric matrix")
        if verify_positive:
            eigen = solve_dirac(operator, metric, ns, kappa, nsp=nsp, nsq=nsq)
            first = radial_projector_start(kappa, min(key[0] for key in keys), functional.data)
            positive = eigen.coefficients[:, first:]
            if positive.shape != projector.shape:
                raise ValueError("projector does not span the complete selected positive space")
            reconstructed = positive @ (positive.T @ metric @ projector)
            if np.linalg.norm(reconstructed - projector) > 1e-10 * max(1., np.linalg.norm(projector)):
                raise ValueError("projector disagrees with declared no-pair positive space")
        owned.append((keys, projector, shape, slice(cursor, cursor + extent)))
        operators.append((kappa, operator))
        metrics.append((kappa, metric))
        cursor += extent
    diagonal = _array(inverse_diagonal, "inverse diagonal", 1)
    if diagonal.shape != (cursor,) or np.any(diagonal <= 0):
        raise ValueError("inverse diagonal requires positive complete coordinate entries")
    blocks_out = []
    if len(inverse_blocks) not in (0, len(owned)):
        raise ValueError("inverse blocks do not match the complete layout")
    for entry, blocks in zip(owned, inverse_blocks, strict=bool(inverse_blocks)):
        keys, _, shape, _ = entry
        if len(blocks) != len(keys):
            raise ValueError("inverse blocks require one matrix per ordered orbital")
        values = []
        for block in blocks:
            block = _array(block, "inverse block", 2)
            if (block.shape != (shape[0], shape[0])
                    or np.linalg.norm(block - block.T, ord=np.inf) > 1e-12 * max(1., np.linalg.norm(block, ord=np.inf))
                    or np.linalg.eigvalsh(block).min() <= 0):
                raise ValueError("inverse block requires symmetric positive complete geometry")
            values.append(block)
        blocks_out.append(tuple(values))
    owned, blocks_out = tuple(owned), tuple(blocks_out)
    physical = _physical_geometry_sha256(functional)
    residual = _residual_metric_sha256(owned, diagonal, blocks_out)
    digest = hashlib.sha256((physical + residual).encode())
    _hash_value(digest, (tuple(operators), tuple(metrics), projector_source, float(occupation_floor)))
    return _OptimizationGeometry(tuple(functional.keys), owned, diagonal, blocks_out,
                                 tuple(operators), tuple(metrics), projector_source, float(occupation_floor),
                                 physical, residual, digest.hexdigest())


def _polar_retraction(y):
    """Evaluate the polar map without reconstructing a nearly unit matrix.

    A convergent binomial series for ``(I + D)**(-1/2) - I`` avoids
    roundoff from arbitrary eigenvectors of a nearly degenerate Gram matrix.
    Wide precision is used only for these small Gram/correction matrices;
    the stored coordinates and physical objective retain their usual dtype.
    """
    gram = y.T @ y
    if np.linalg.norm(gram - np.eye(gram.shape[0]), ord=np.inf) < .01:
        wide = y.astype(np.longdouble)
        defect = wide.T @ wide - np.eye(gram.shape[0], dtype=np.longdouble)
        power = defect.copy()
        coefficient = -.5
        correction = coefficient * power
        # With ||D||_inf < .01, 24 terms give a remainder below 1e-48.
        # Stop earlier when the next term is below wide relative precision.
        for order in range(2, 25):
            power = power @ defect
            coefficient *= -(order - .5) / order
            term = coefficient * power
            correction += term
            if np.max(np.abs(term)) <= np.finfo(np.longdouble).eps * max(
                    np.max(np.abs(correction)), np.finfo(np.longdouble).tiny):
                break
        return np.asarray(wide + wide @ correction, dtype=y.dtype)
    values, vectors = np.linalg.eigh(gram)
    if values.min() <= 1e-14:
        raise ValueError("rank-deficient orbital trial")
    return y @ ((vectors / np.sqrt(values)) @ vectors.T)


def optimize_orbitals(functional, orbitals=None, *, max_iter=160, tolerance=1e-6,
                      energy_tolerance=1e-10, preconditioner_matrices=None,
                      preconditioner_factory=None, refresh_patience=15,
                      refresh_min_ratio=0.25, max_refreshes=4,
                      projector_source="nuclear", occupation_floor=0.05,
                      progress_callback=None, orbital_preconditioner="spectral",
                      history_size=10, energy_difference=None,
                      linear_energy_increments=False, resume_state=None,
                      objective_identity=None, allow_objective_change=False,
                      checkpoint_callback=None, max_wall_time=None, return_state=False):
    """Minimize the correlated energy in a fixed positive-energy subspace.

    A nuclear Dirac projector fixes the finite-basis no-pair Hamiltonian.
    Every trial uses a metric-orthonormal set (polar retraction); an Armijo
    line search accepts only energy-lowering steps. The derivative is the
    full CI Hellmann–Feynman derivative, not a configuration average.
    Limited-memory secant updates and orbital-energy preconditioning avoid
    forming the orbital Hessian.

    ``history_size`` bounds retained secant pairs. Increasing it can help
    larger correlation spaces without changing the energy functional or
    convergence tests; the default retains the previous ten-pair path.

    ``orbital_preconditioner='shooting'`` uses a two-sided positive radial
    Green solve for update directions. The Hamiltonian, no-pair space,
    orthogonality constraints, and spectral convergence diagnostic are
    unchanged. A failed search or less than twofold residual reduction over
    20 iterations switches back to the spectral inverse and clears history.

    Robustness extensions over the plain Riemannian L-BFGS:

    * Line-search trials that raise ``ValueError`` (rank-deficient polar
      retraction) or ``FloatingPointError`` (non-finite trial CI) are
      treated as rejections and the step is halved, instead of aborting
      the whole solve.
    * When the backtracking budget is exhausted the search falls back to
      a preconditioned steepest-descent step (secant history reset); the
      solve only stops when the fallback also fails.
    Each search starts at unit step. The factory/refresh arguments remain
    accepted for call compatibility, but this implementation fixes its
    initial spectral metric and does not perform automatic refreshes.

    ``projector_source`` selects the positive-energy (no-pair) subspace the
    orbital update is constrained to.  ``"nuclear"`` (default) fixes the
    projector to the nuclear Dirac spectrum — exact for hydrogenic
    shells, but a deliberately crude positive-energy boundary for a
    screened heavy-atom valence shell: valence spinors see a much
    weaker potential than Z/r, so their small components and continuum
    mixing differ from the nuclear eigenbasis, and the SCF can carry
    small negative-energy leakage the nuclear projector never removes
    (critical review F3).  ``"fock"`` rebuilds the projector from the
    occupation average of complete seed Fock matrices
    (``preconditioner_matrices``, including the nuclear one-body term).
    This screened boundary stays fixed throughout the optimization;
    it requires ``preconditioner_matrices``.
    Either way the returned diagnostics carry
    ``no_pair_projector_leakage``, the worst S-norm weight any active
    orbital carries outside the *alternative* projector (nuclear vs
    Fock); a value near 1 means the two no-pair boundaries disagree on
    that shell and the projector choice matters.

    A resume_state restores its original projector, coordinates and inverse
    metric without rebuilding seed spectra or population scaling. Import
    archives with OrbitalOptimizationState.from_arrays. Every resume clears
    secant history and evaluates the full supplied objective and gradient;
    a changed caller-declared objective_identity requires explicit opt-in.
    Tolerances remain explicit arguments of this call. Optional checkpoint
    callbacks receive (iteration, state, orbitals, ci, diagnostics) only at
    accepted points, including zero. Budget exits return the last such point
    without declaring convergence. Large sealed geometry is shared between
    snapshots; return_state adds a typed state only when explicitly requested.
    """
    from atomx.numerics.bspline.dirac_assembly import solve_dirac
    from atomx.numerics.bspline.orbital_subspace import radial_projector_start

    if max_wall_time is not None and (isinstance(max_wall_time, (bool, np.bool_))
            or not np.isfinite(max_wall_time) or max_wall_time < 0):
        raise ValueError("max_wall_time must be finite and nonnegative")
    started = time.monotonic() if max_wall_time is not None else None
    if (resume_state is not None or return_state or checkpoint_callback is not None) and (
            not isinstance(objective_identity, str) or not objective_identity):
        raise ValueError("state/checkpoint requires explicit nonempty objective_identity")
    if orbital_preconditioner not in ("spectral", "shooting"):
        raise ValueError("orbital_preconditioner must be 'spectral' or 'shooting'")
    if (isinstance(history_size, (bool, np.bool_))
            or not isinstance(history_size, (int, np.integer)) or history_size < 1):
        raise ValueError("history_size must be a positive integer")
    if not np.isfinite(occupation_floor) or occupation_floor <= 0:
        raise ValueError("occupation_floor must be finite and positive")
    if projector_source not in ("nuclear", "fock"):
        raise ValueError(
            f"projector_source must be 'nuclear' or 'fock', got {projector_source!r}")
    if projector_source == "fock" and preconditioner_matrices is None and resume_state is None:
        raise ValueError(
            "projector_source='fock' needs the SCF Fock fields as "
            "preconditioner_matrices to build the no-pair space from")

    resumed = resume_state is not None
    if resumed:
        if not isinstance(resume_state, OrbitalOptimizationState):
            raise TypeError("resume_state must be OrbitalOptimizationState")
        if preconditioner_matrices is not None or preconditioner_factory is not None or orbital_preconditioner != "spectral":
            raise ValueError("resume restores original metric; matrices/factory/shooting are incompatible")
        if projector_source != resume_state.projector_source or occupation_floor != resume_state.occupation_floor:
            raise ValueError("resume projector/occupation policy differs from original state")
        if objective_identity is None:
            raise ValueError("resume requires explicit objective_identity")
        if objective_identity != resume_state.objective_identity and not allow_objective_change:
            raise ValueError("changed objective_identity requires allow_objective_change=True")
        resume_state.validate(functional)
        layout = resume_state.layout
        inverse_blocks = resume_state.inverse_blocks
        initial, inverse_diagonal = [resume_state.coordinates], [resume_state.inverse_diagonal]
        no_pair_leakage = {}
        projector_operators = dict(resume_state._geometry.projector_operators)
    else:
        if orbitals is None:
            raise ValueError("fresh optimization requires orbital seeds")
        projector_operators = {}
        groups = {}
        for key in functional.keys:
            groups.setdefault(key[1], []).append(key)
        one_electron = sum(sh.occupation for sh in functional.csfs[0].subshells) == 1
        layout = []
        inverse_blocks = []
        initial, inverse_diagonal = [], []
        cursor = 0
        # Occupation weights for averaging per-kappa Fock fields (projector
        # diagnostics and the 'fock' no-pair space).  Subshell occupations from
        # the leading CSF; missing entries default to equal sharing.
        occ_weights = {}
        for csf0 in functional.csfs[:1]:
            for sh in csf0.subshells:
                occ_weights[(sh.n, sh.kappa)] = max(float(sh.occupation), 1.0)

        no_pair_leakage = {}
        for kappa, keys in sorted(groups.items()):
            metric = functional.metric[kappa]
            result = solve_dirac(functional.one_body[kappa], metric, functional.ns, kappa,
                                 nsp=functional.data.get("nsp"), nsq=functional.data.get("nsq"))
            # Preserve an explicitly excited active shell when lower shells of
            # this symmetry are absent (e.g. a hydrogenic 2s/2p calculation).
            # Otherwise unrestricted minimization would silently replace 2s by
            # 1s. Missing lower reference roots are frozen outside the active
            # space, within the same stated no-pair projector model.
            first = radial_projector_start(kappa, min(key[0] for key in keys), functional.data)
            proj = result.coefficients[:, first:]
            eps_source = result.eigenvalues[first:]
            if projector_source == "fock" or preconditioner_matrices is not None:
                # The alternative no-pair boundary: a screened Fock field
                # (occupation-averaged over this symmetry's active shells)
                # instead of the bare nuclear one.  Diagonalising it against
                # the same metric and keeping the branch above the sea edge is
                # the GRASP-style construction.  Used either as THE projector
                # (projector_source="fock") or as the leakage yardstick for the
                # nuclear projector.
                fock_k = np.zeros_like(functional.one_body[kappa])
                total = 0.0
                for key in keys:
                    if preconditioner_matrices is not None and key in preconditioner_matrices:
                        w = occ_weights.get(key, 1.0)
                        fock_k += w * preconditioner_matrices[key]
                        total += w
                if total > 0.0:
                    # These are complete Fock matrices, also used unchanged
                    # by build_blocks below. Adding one_body again would
                    # double the nuclear Dirac term in the no-pair model.
                    fock_k = fock_k / total
                    alt = solve_dirac(fock_k, metric, functional.ns, kappa,
                                      nsp=functional.data.get("nsp"),
                                      nsq=functional.data.get("nsq"))
                    alt_projector = alt.coefficients[:, first:]
                    if projector_source == "fock":
                        proj = alt_projector
                        eps_source = alt.eigenvalues[first:]
                    # Worst S-norm weight the active orbitals carry outside the
                    # no-pair space actually in use: for the nuclear projector
                    # the yardstick is the screened Fock space (how much of the
                    # shell a GRASP-style boundary would reclassify), and vice
                    # versa.  Far from 0: the two no-pair models disagree on
                    # this symmetry's shell and the projector choice (F3) is a
                    # physical decision, not a numerical nicety.
                    yardstick = (result.coefficients[:, first:]
                                 if projector_source == "fock" else alt_projector)
                    coefficients = np.column_stack(
                        [np.concatenate(orbitals[key]) for key in keys])
                    overlap = yardstick.T @ metric @ coefficients
                    norms = np.einsum("ij,ij->j", overlap, overlap)
                    seed_norms = np.einsum("ij,ij->j", coefficients,
                                           metric @ coefficients)
                    with np.errstate(invalid="ignore", divide="ignore"):
                        leak = 1.0 - norms / np.maximum(seed_norms, 1e-30)
                    no_pair_leakage[kappa] = float(np.clip(np.max(leak), 0.0, 1.0))
            projector_operators[kappa] = (functional.one_body[kappa] if projector_source == "nuclear" else fock_k)
            coefficients = np.column_stack([np.concatenate(orbitals[key]) for key in keys])
            y = proj.T @ metric @ coefficients
            if one_electron:
                # Complete one-electron CI is invariant under active-orbital
                # rotations. Fix that gauge to actual nuclear eigenfunctions,
                # so principal labels and unoccupied orbital energies remain
                # meaningful instead of retaining an arbitrary SCF rotation.
                y[:] = 0.
                for column, (n, _) in enumerate(keys):
                    root = radial_projector_start(kappa, n, functional.data) - first
                    y[root, column] = 1.
            gram_values, gram_vectors = np.linalg.eigh(y.T @ y)
            if gram_values.min() < 1e-10:
                raise ValueError("orbital seed has no independent positive-energy projection")
            y = y @ ((gram_vectors / np.sqrt(gram_values)) @ gram_vectors.T)
            extent = y.size
            layout.append((keys, proj, y.shape, slice(cursor, cursor + extent)))
            cursor += extent
            initial.append(y.ravel())
            eps = eps_source
            orbital_eps = np.sum(y * (eps[:, None] * y), axis=0)
            inverse_diagonal.append((1.0 / (2 * np.maximum(abs(eps[:, None] - orbital_eps), 0.5))).ravel())
    def build_blocks(mats, point):
        """Spectral inverse-Hessian blocks from full-space Fock matrices.

        The reference for each column's spectral gap is its Rayleigh
        quotient at the supplied coordinate point. Population scaling is
        applied separately during fresh initialization. Resume directly
        restores the already scaled original blocks.
        """
        blocks_out = []
        for keys, projector, shape, where in layout:
            y = point[where].reshape(shape)
            blocks = []
            for column, key in enumerate(keys):
                fock = projector.T @ mats[key] @ projector
                fock = 0.5 * (fock + fock.T)
                values, eigenvectors = np.linalg.eigh(fock)
                reference = float(y[:, column] @ fock @ y[:, column])
                inverse = 1. / (2 * np.maximum(abs(values - reference), 0.5))
                blocks.append((eigenvectors * inverse) @ eigenvectors.T)
            blocks_out.append(blocks)
        return blocks_out

    x = np.concatenate(initial)
    inverse_diagonal = np.concatenate(inverse_diagonal)

    def scale_blocks(blocks_out):
        """Apply the occupation-weight scaling shared by init and refresh."""
        for group, (keys, _, _shape, _where) in enumerate(layout):
            populations = np.array([max(functional.last_occupations.get(key, occupation_floor), occupation_floor)
                                    for key in keys])
            blocks_out[group] = [block / population for block, population
                                 in zip(blocks_out[group], populations, strict=True)]
        return blocks_out

    def retract(point):
        result = point.copy()
        for _, _, shape, where in layout:
            y = point[where].reshape(shape)
            result[where] = _polar_retraction(y).ravel()
        return result

    def tangent(point, vector):
        result = vector.copy()
        for _, _, shape, where in layout:
            y = point[where].reshape(shape)
            g = vector[where].reshape(shape)
            multiplier = y.T @ g
            result[where] = (g - y @ (0.5 * (multiplier + multiplier.T))).ravel()
        return result

    def unpack(point):
        orbitals = {}
        for keys, projector, shape, where in layout:
            coefficients = projector @ point[where].reshape(shape)
            for i, key in enumerate(keys):
                orbitals[key] = (coefficients[:functional.ns, i].copy(),
                                 coefficients[functional.ns:, i].copy())
        return orbitals

    def evaluate(point, gradient=True, ci_result=None):
        if not gradient:
            return functional.evaluate(unpack(point), gradient=False)
        energy, radial_gradient, ci = functional.evaluate(unpack(point), ci_result=ci_result)
        g = np.zeros_like(point)
        for keys, projector, _shape, where in layout:
            derivative = np.column_stack([radial_gradient[key] for key in keys])
            g[where] = (projector.T @ derivative).ravel()
        return energy, tangent(point, g), ci

    if resumed and orbitals is not None:
        expected_orbitals = unpack(x)
        if (set(orbitals) != set(expected_orbitals) or any(
                not np.array_equal(given, expected) for key in expected_orbitals
                for given, expected in zip(orbitals[key], expected_orbitals[key], strict=True))):
            raise ValueError("resume input orbitals disagree with authoritative coordinates")
    energy, g, ci = evaluate(x)
    if not resumed:
        # The orbital Hessian scales with its occupation. Omitting this factor
        # makes a nearly empty correlation orbital converge orders of magnitude
        # more slowly than the closed core. Keep a finite floor for null-density
        # directions, whose radial form is not defined by a ground-state target.
        for _group, (keys, _, shape, where) in enumerate(layout):
            populations = np.array([max(functional.last_occupations.get(key, occupation_floor), occupation_floor) for key in keys])
            inverse_diagonal[where] /= np.broadcast_to(populations, shape).ravel()
        if preconditioner_matrices is not None:
            inverse_blocks = scale_blocks(build_blocks(preconditioner_matrices, x))

    # These spectral gaps/occupations are fixed for this run. Restarting
    # at its endpoint creates a different residual metric even when the
    # physical gradient is identical; fingerprint the actual representation
    # so comparisons cannot silently assume those numbers are the same.
    residual_metric_sha256 = _residual_metric_sha256(layout, inverse_diagonal, inverse_blocks)
    geometry = None
    if resumed:
        geometry = resume_state._geometry
    elif return_state or checkpoint_callback is not None:
        geometry = _seal_geometry(functional, layout, inverse_diagonal, inverse_blocks,
                                  projector_source, projector_operators, occupation_floor,
                                  verify_positive=False)


    def spectral_precondition(vector):
        if not inverse_blocks:
            return inverse_diagonal * vector
        result = np.zeros_like(vector)
        for (_, _, shape, where), blocks in zip(layout, inverse_blocks, strict=True):
            matrix = vector[where].reshape(shape)
            result[where] = np.column_stack([block @ matrix[:, i]
                                            for i, block in enumerate(blocks)]).ravel()
        return result
    precondition = spectral_precondition
    if orbital_preconditioner == "shooting":
        from atomx.numerics.shooting.preconditioner import ShootingPreconditioner
        precondition = ShootingPreconditioner(functional, layout, occupation_floor,
                                              x, preconditioner_matrices)
    energies = [energy]
    history = []
    converged = False
    termination_reason = "maximum_iterations"
    line_search_fallbacks = 0
    stable_energy_comparisons = 0
    shooting_active = orbital_preconditioner == "shooting"
    shooting_iterations = 0
    shooting_fallback_reason = None
    checkpoint_residual = None
    for iteration in range(max_iter + 1):
        # Convergence must not depend on how small the proposed update is.
        # Keep exactly the same residual yardstick for both numerical paths.
        preconditioned_gradient = tangent(x, spectral_precondition(g))
        residual = float(np.linalg.norm(preconditioned_gradient))
        energy_error = 0.5 * abs(float(g @ preconditioned_gradient))
        if progress_callback is not None:
            progress_callback(iteration, unpack(x), ci, {"energy_ha": energy,
                "orbital_residual": residual, "estimated_energy_error": energy_error})
        if checkpoint_callback is not None:
            point_diagnostics = {"energy_ha": energy, "orbital_residual": residual,
                "estimated_energy_error": energy_error, "residual_metric_sha256": residual_metric_sha256,
                "objective_identity": objective_identity, "resumed": resumed,
                "objective_changed": resumed and objective_identity != resume_state.objective_identity,
                "initial_history_size": 0, "accepted_point": True,
                "orbital_tolerance": tolerance, "energy_tolerance": energy_tolerance,
                "geometry_sha256": geometry.payload_sha256}
            state = OrbitalOptimizationState(geometry, x, objective_identity)
            checkpoint_callback(iteration, state, unpack(x), ci, point_diagnostics)
        if max_wall_time is not None and time.monotonic() - started >= max_wall_time:
            termination_reason = "wall_time_limit"
            break
        if residual < tolerance and energy_error < energy_tolerance:
            converged = True
            termination_reason = "converged"
            break
        if iteration == max_iter:
            break
        if shooting_active:
            if checkpoint_residual is None:
                checkpoint_residual = residual
            elif iteration % 20 == 0:
                if residual > .5 * checkpoint_residual:
                    precondition = spectral_precondition
                    shooting_active = False
                    shooting_fallback_reason = "spectral residual stalled over 20 iterations"
                    history.clear()
                checkpoint_residual = residual
            if shooting_active:
                shooting_iterations += 1
        q = g.copy()
        alphas = []
        for s, y, rho in reversed(history):
            alpha = rho * np.dot(s, q)
            alphas.append(alpha)
            q -= alpha * y
        direction = precondition(q)
        for (s, y, rho), alpha in zip(history, reversed(alphas), strict=True):
            direction += s * (alpha - rho * np.dot(y, direction))
        direction = -tangent(x, direction)
        derivative = float(g @ direction)
        if derivative >= 0:
            history.clear()
            direction = -preconditioned_gradient
            derivative = float(g @ direction)
        step = 1.0
        accepted = False
        budget_exhausted = False
        for attempt in range(2):
            if attempt:
                # Secant curvature can become unreliable near the energy
                # resolution limit. Retry the physical gradient before
                # declaring failure, without relaxing either tolerance.
                history.clear()
                if shooting_active:
                    shooting_active = False
                    precondition = spectral_precondition
                    shooting_fallback_reason = "shooting/secant line search exhausted"
                line_search_fallbacks += 1
                direction = -preconditioned_gradient
                derivative = float(g @ direction)
                step = 1.0
            for _ in range(28):
                if max_wall_time is not None and time.monotonic() - started >= max_wall_time:
                    budget_exhausted = True
                    break
                try:
                    candidate = retract(x + step * direction)
                    trial_energy, trial_ci = evaluate(candidate, gradient=False)
                except (ValueError, FloatingPointError):
                    step *= 0.5
                    continue
                if max_wall_time is not None and time.monotonic() - started >= max_wall_time:
                    budget_exhausted = True
                    break
                change = trial_energy - energy
                if (energy_difference is not None and np.isfinite(trial_energy)
                        and abs(change) <= 128 * np.spacing(max(abs(energy), 1.0))):
                    # A declared finite-polynomial or constraint-centered
                    # difference can resolve a physical decrease hidden by
                    # the large core constant and roundoff normal motion.
                    # The callback owns its scope/constraint guards; an
                    # unsupported density response returns None explicitly.
                    difference_kwargs = ({"increments": unpack(candidate - x)}
                                         if linear_energy_increments else {})
                    stable_change = energy_difference(unpack(x), unpack(candidate), ci, trial_ci,
                                                      **difference_kwargs)
                    if stable_change is not None:
                        change = stable_change
                        stable_energy_comparisons += 1
                if np.isfinite(change) and change <= 1e-4 * step * derivative:
                    accepted = True
                    break
                step *= 0.5
            if accepted or budget_exhausted:
                break
        if budget_exhausted:
            termination_reason = "wall_time_limit"
            break
        if not accepted:
            termination_reason = "line_search_failed"
            break
        old_x, old_g = x, g
        x = candidate
        # The accepted line-search trial already solved this exact CI
        # Hamiltonian. Reuse its amplitudes in the Hellmann–Feynman gradient.
        energy, g, ci = evaluate(x, ci_result=trial_ci)
        energies.append(energy)
        # Vector transport keeps all secant pairs in the current tangent
        # space. Ambient gradient differences otherwise contain changing
        # constraint normals and corrupt the approximate inverse Hessian.
        transported = []
        for old_s, old_y, _ in history:
            st, yt = tangent(x, old_s), tangent(x, old_y)
            sy = float(st @ yt)
            if sy > 1e-12 * np.linalg.norm(st) * np.linalg.norm(yt):
                transported.append((st, yt, 1. / sy))
        history = transported
        s = tangent(x, x - old_x)
        y = g - tangent(x, old_g)
        curvature = float(s @ y)
        if curvature > 1e-12 * np.linalg.norm(s) * np.linalg.norm(y):
            history.append((s, y, 1.0 / curvature))
            history = history[-history_size:]
    diagnostics = {
        "converged": converged, "iterations": iteration,
        "termination_reason": termination_reason,
        "line_search_fallbacks": line_search_fallbacks,
        "stable_energy_comparisons": stable_energy_comparisons,
        "occupation_floor": occupation_floor,
        "gradient_norm": float(np.linalg.norm(g)),
        "orbital_residual": residual, "orbital_tolerance": tolerance,
        "residual_metric_sha256": residual_metric_sha256,
        "residual_metric_scope": "run-initial spectral gaps and occupation scaling; endpoint restarts generally differ",
        "estimated_energy_error": energy_error, "energy_tolerance": energy_tolerance,
        "energy_history": energies,
        "functional": "correlated Dirac-Coulomb CI",
        "target": functional.target_description,
        "radial_labels": "correlation-orbital labels; not guaranteed spectroscopic excited states",
        "projector": (
            "fixed positive-energy nuclear Dirac B-spline subspace"
            if projector_source == "nuclear"
            else "fixed positive-energy Fock-field (screened) B-spline subspace"
        ),
        "no_pair_projector_leakage": no_pair_leakage,
        "excited_shell_constraints": "lower reference roots below each symmetry's first active shell are excluded",
        "preconditioner": "screened Fock spectral gaps" if inverse_blocks else "nuclear Dirac spectral gaps",
        "update_preconditioner": orbital_preconditioner,
        "history_size": int(history_size),
        "shooting_iterations": shooting_iterations,
        "shooting_fallback_reason": shooting_fallback_reason,
        "shooting_operator": (
            "positive radial kinetic resolvent; full correlated CI gradient source"
            if orbital_preconditioner == "shooting" else None),
        "breit_qed": "subsequent RCI / perturbative corrections",
    }
    if resumed or return_state or checkpoint_callback is not None or max_wall_time is not None:
        diagnostics.update(resumed=resumed, objective_identity=objective_identity,
                           objective_changed=resumed and objective_identity != resume_state.objective_identity,
                           initial_history_size=0, restored_original_metric=resumed)
        if geometry is not None:
            diagnostics["geometry_sha256"] = geometry.payload_sha256
        if resumed:
            diagnostics["residual_metric_scope"] = "restored original projector/inverse metric; no occupation or spectral rebuild"
            diagnostics["no_pair_projector_leakage_scope"] = "alternative yardstick not recomputed on resume"
    if return_state:
        diagnostics["optimizer_state"] = OrbitalOptimizationState(geometry, x, objective_identity)
    return unpack(x), ci, diagnostics

