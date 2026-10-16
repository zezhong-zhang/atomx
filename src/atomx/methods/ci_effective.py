"""Exact finite-CSF P/Q elimination in a shared Dirac-Coulomb(-Breit) space.

The full Q-Q Hamiltonian propagates the retained core-hole/configuration
states. This is all orders *within the specified finite CI space*, not
CI+all-order SD, a complete MBPT Sigma2 operator, or Fock-space CC. No
precomputed self-energy is added: P and Q are disjoint parts of one native
CI Hamiltonian. The common orbital metric, nucleus, signed amplitudes,
CSF coupling trees and absolute energy zero are captured in the snapshot.
"""
from dataclasses import dataclass

import numpy as np

from atomx.methods.mbpt_matrix import MatrixSelfEnergy

__all__ = ["FiniteCIModel", "FiniteCIRoot", "FiniteBlochReduction", "FiniteCITransitionOperator", "StaticBreitControl",
           "static_breit_second_order_control", "build_finite_ci_effective_hamiltonian"]


def _owned(a):
    a = np.asarray(a, dtype=float)
    return np.frombuffer(a.tobytes(), dtype=float).reshape(a.shape)


def _transition_storage_guard(bra, ket, itemsize, memory_bytes):
    """Bound matrix copies, contractions, solves and state-coordinate work."""
    if (isinstance(memory_bytes, bool) or not isinstance(memory_bytes, (int, np.integer))
            or memory_bytes < 1):
        raise ValueError("positive integer transition memory cap required")
    nb, nk = len(bra.hamiltonian_Ha), len(ket.hamiltonian_Ha)
    pb, pk = len(bra.p_indices), len(ket.p_indices)
    required = itemsize * (3*nb*nk + 3*pb*nk + 12*pb*pk + 6*(pb*pb+pk*pk))
    if required > memory_bytes:
        raise MemoryError(f"complete finite transition work {required} bytes exceeds cap {memory_bytes}")
    return required


@dataclass(frozen=True, eq=False, init=False)
class FiniteCITransitionOperator:
    """Owned rectangular signed operator bound to two exact parent snapshots.

    Rows belong to ``bra`` and columns to ``ket`` in their full CSF order.
    An explicit matrix is a mathematical control; a native producer must
    additionally validate spinors, nuclear model and operator conventions.
    Identity strings never authorize using a different parent snapshot.
    """
    matrix: np.ndarray
    bra: object
    ket: object
    operator_identity: str
    unit: str

    def __init__(self, matrix, bra, ket, *, operator_identity, unit,
                 memory_bytes=64*1024**2):
        if not isinstance(bra, FiniteCIModel) or not isinstance(ket, FiniteCIModel):
            raise TypeError("two exact finite CI parent snapshots required")
        if (not isinstance(operator_identity, str) or not operator_identity.strip()
                or not isinstance(unit, str) or not unit.strip()):
            raise ValueError("explicit transition operator identity and unit required")
        a = np.asarray(matrix)
        if (a.shape != (len(bra.hamiltonian_Ha), len(ket.hamiltonian_Ha))
                or a.dtype.kind not in "iufc" or a.dtype.itemsize > 16
                or (a.dtype.kind == "f" and a.dtype.itemsize > 8)):
            raise ValueError("finite numeric rectangular operator in both parent CSF orders required")
        dtype = np.complex128 if np.iscomplexobj(a) else np.float64
        _transition_storage_guard(bra, ket, np.dtype(dtype).itemsize, memory_bytes)
        if not np.all(np.isfinite(a)):
            raise ValueError("finite numeric rectangular operator in both parent CSF orders required")
        a = np.asarray(a, dtype=dtype)
        owned = np.frombuffer(a.tobytes(), dtype=dtype).reshape(a.shape)
        for name, value in dict(matrix=owned, bra=bra, ket=ket,
                                operator_identity=operator_identity, unit=unit).items():
            object.__setattr__(self, name, value)


@dataclass(frozen=True, eq=False)
class FiniteCIRoot:
    energy_Ha: float
    shift_Ha: float
    p_weight: float
    coefficients: np.ndarray
    residual_Ha: float
    reference_index: int
    reference_identity: str

    def __post_init__(self):
        object.__setattr__(self, "coefficients", _owned(self.coefficients))


@dataclass(frozen=True, eq=False)
class FiniteBlochReduction:
    """Exact finite invariant subspace, including its physical model-space metric.

    This result comes from the SAME Hermitian full Hamiltonian as its parent
    ``FiniteCIModel``. It cannot infer a physical metric from arbitrary SD/CC
    effective matrices. ``right_states`` are P projections of orthonormal full
    states; ``left_states`` are rows biorthogonal to those columns. Both carry
    the full chosen subspace, including signs and off-diagonal mixing.
    """
    root_indices: tuple
    energies_Ha: np.ndarray
    wave_operator: np.ndarray
    metric: np.ndarray
    directed_Ha: np.ndarray
    orthonormal_wave_operator: np.ndarray
    orthonormal_Ha: np.ndarray
    right_states: np.ndarray
    left_states: np.ndarray
    projection_condition: float
    intertwining_residual_Ha: float
    metric_adjoint_residual_Ha: float
    parent: object

    def __post_init__(self):
        for name in ("energies_Ha", "wave_operator", "metric", "directed_Ha",
                     "orthonormal_wave_operator", "orthonormal_Ha", "right_states", "left_states"):
            object.__setattr__(self, name, _owned(getattr(self, name)))

    def effective_operator(self, full_operator, *, representation):
        """Transform a full signed property in the parent CSF order.

        ``bloch`` returns G^-1 Omega^T O Omega, with biorthogonal model
        states. ``orthonormal`` uses the physical isometry Omega G^-1/2.
        Neither mode substitutes bare P O P or guesses a normalization.
        The operator may be complex/non-Hermitian, e.g. a spherical component.
        """
        operator = np.asarray(full_operator)
        if (operator.shape != self.parent.hamiltonian_Ha.shape
                or not np.issubdtype(operator.dtype, np.number)
                or not np.all(np.isfinite(operator))):
            raise ValueError("full finite signed property in the parent CSF order required")
        if representation == "bloch":
            result = np.linalg.solve(self.metric, self.wave_operator.T @ operator @ self.wave_operator)
        elif representation == "orthonormal":
            w = self.orthonormal_wave_operator
            result = w.T @ operator @ w
        else:
            raise ValueError("explicit bloch or orthonormal operator representation required")
        if not np.all(np.isfinite(result)):
            raise RuntimeError("nonfinite physical effective property")
        return np.frombuffer(result.tobytes(), dtype=result.dtype).reshape(result.shape)

    def effective_transition(self, operator, ket, *, representation,
                             memory_bytes=64*1024**2):
        """Transform a bound rectangular operator between two invariant spaces.

        Bloch: G_b^-1 Omega_b^dagger O_bk Omega_k. Orthonormal:
        W_b^dagger O_bk W_k. Each endpoint retains its own physical metric;
        a Euclidean adjoint of a directed Bloch operator is not its reverse.
        """
        if (not isinstance(ket, FiniteBlochReduction)
                or not isinstance(operator, FiniteCITransitionOperator)):
            raise TypeError("bound finite transition and ket Bloch reduction required")
        if operator.bra is not self.parent or operator.ket is not ket.parent:
            raise ValueError("transition operator parent snapshot mismatch")
        _transition_storage_guard(self.parent, ket.parent, operator.matrix.dtype.itemsize,
                                  memory_bytes)
        if representation == "bloch":
            result = np.linalg.solve(self.metric,
                                     self.wave_operator.T @ operator.matrix @ ket.wave_operator)
        elif representation == "orthonormal":
            result = (self.orthonormal_wave_operator.T @ operator.matrix
                      @ ket.orthonormal_wave_operator)
        else:
            raise ValueError("explicit bloch or orthonormal operator representation required")
        if not np.all(np.isfinite(result)):
            raise RuntimeError("nonfinite physical effective transition")
        return np.frombuffer(result.tobytes(), dtype=result.dtype).reshape(result.shape)

    def transition_amplitudes(self, operator, ket, *, representation,
                              memory_bytes=64*1024**2):
        """Signed amplitudes between normalized full-root states, at fixed O.

        Rows/columns follow the selected bra/ket root orders. For a photon
        operator, its frequency is fixed by the producer; this matrix is
        not a collection of rates at different transition frequencies.
        """
        effective = self.effective_transition(operator, ket, representation=representation,
                                              memory_bytes=memory_bytes)
        if representation == "bloch":
            result = self.left_states @ effective @ ket.right_states
        else:
            bra_coordinates = (self.orthonormal_wave_operator.T @ self.wave_operator
                               @ self.right_states)
            ket_coordinates = (ket.orthonormal_wave_operator.T @ ket.wave_operator
                               @ ket.right_states)
            result = bra_coordinates.T @ effective @ ket_coordinates
        return np.frombuffer(result.tobytes(), dtype=result.dtype).reshape(result.shape)


@dataclass(frozen=True)
class StaticBreitControl:
    """Restricted H0-based perturbation controls, never additive to full DCB.

    The supplied H0 contains the frozen reference mean field. These terms
    cover only the declared Q configurations, not full atomic MBPT B2.
    ``coulomb_breit_Ha`` is the coefficient of g*b in the eigenenergy of
    H0 + g*(H_DC-H0) + b*B. No denominator or diagram is silently dropped.
    """
    first_order_breit_Ha: float
    coulomb_second_order_Ha: float
    coulomb_breit_Ha: float
    breit_second_order_Ha: float
    h0_diagonal_Ha: np.ndarray
    minimum_denominator_Ha: float

    def __post_init__(self):
        object.__setattr__(self, "h0_diagonal_Ha", _owned(self.h0_diagonal_Ha))


def static_breit_second_order_control(dc, dcb, h0_diagonal_Ha):
    """Match two native snapshots, then evaluate strict finite-Q C-B interference.

    Requires one P CSF, the same ordered CSFs/P split/radial/nuclear binding,
    and a zeroth-order P energy matching H_DC. The native DCB matrix replaces
    the DC matrix; this control is a diagnostic alternative, not a correction
    to add to a DCB eigenenergy. Exactly uncoupled zero denominators give zero;
    any coupled near-zero denominator rejects the entire control.
    """
    if not isinstance(dc, FiniteCIModel) or not isinstance(dcb, FiniteCIModel):
        raise TypeError("native finite DC and DCB snapshots required")
    if (dc.hamiltonian_terms != ("dirac-coulomb",)
            or dcb.hamiltonian_terms != ("dirac-coulomb", "static-breit")
            or dc.csf_identities != dcb.csf_identities or dc.p_indices != dcb.p_indices
            or dc.radial_binding != dcb.radial_binding or len(dc.p_indices) != 1):
        raise ValueError("matched native DC/DCB basis, partition and operator coverage required")
    h0 = np.asarray(h0_diagonal_Ha)
    if (h0.shape != (len(dc.hamiltonian_Ha),) or not np.isrealobj(h0)
            or not np.all(np.isfinite(h0))):
        raise ValueError("finite real H0 diagonal in the full CSF order required")
    p = dc.p_indices[0]
    if abs(float(h0[p])-dc.hamiltonian_Ha[p, p]) > 1e-10:
        raise ValueError("H0 P energy does not match the common CI zero")
    q = list(dc.q_indices)
    b = dcb.hamiltonian_Ha-dc.hamiltonian_Ha
    c_pq = dc.hamiltonian_Ha[p, q]
    b_pq = b[p, q]
    den = h0[p]-h0[q]
    coupled = (c_pq != 0) | (b_pq != 0)
    if np.any(abs(den[coupled]) < dc.self_energy.denominator_floor_Ha):
        raise ValueError("coupled resonant H0 denominator in Breit control")
    c, bq, d = c_pq[coupled], b_pq[coupled], den[coupled]
    return StaticBreitControl(float(b[p, p]), float(np.sum(c*c/d)),
                              float(np.sum(2*c*bq/d)), float(np.sum(bq*bq/d)),
                              h0, float(np.min(abs(d))) if len(d) else float('inf'))


@dataclass(frozen=True, eq=False, init=False)
class FiniteCIModel:
    """Owned native CI snapshot with a finite, explicitly disjoint P/Q split.

    Energies are total N-electron energies, including the common core zero;
    they are not the orbital attachment arguments of ``MatrixSelfEnergy``.
    ``solve`` returns only a P-connected ordered branch in a pole-free
    bracket; Q-dominated states or degenerate P branches need another state
    matching contract. Nonzero coupling poles cannot be crossed or skipped.
    """

    def __init__(self, hamiltonian_Ha, p_indices, *, reference_identity,
                 csf_identities, radial_binding, hamiltonian_terms=("explicit-matrix",)):
        h = np.asarray(hamiltonian_Ha)
        p = tuple(p_indices)
        if (h.ndim != 2 or h.shape[0] != h.shape[1] or h.shape[0] == 0
                or not np.isrealobj(h) or not np.all(np.isfinite(h))
                or not np.array_equal(h, h.T)):
            raise ValueError("finite CI requires a finite real symmetric Hamiltonian")
        if (not p or len(set(p)) != len(p)
                or any(isinstance(i, bool) or not isinstance(i, (int, np.integer))
                       or not 0 <= i < len(h) for i in p)):
            raise ValueError("distinct valid nonempty P indices required")
        if (not isinstance(reference_identity, str) or not reference_identity.strip()
                or len(csf_identities) != len(h)
                or len(set(csf_identities)) != len(h) or radial_binding is None):
            raise ValueError("finite CI requires distinct CSF and physical reference identities")
        if (not isinstance(hamiltonian_terms, tuple) or not hamiltonian_terms
                or any(not isinstance(t, str) or not t.strip() for t in hamiltonian_terms)
                or len(set(hamiltonian_terms)) != len(hamiltonian_terms)):
            raise ValueError("distinct Hamiltonian term identities required")
        q = tuple(i for i in range(len(h)) if i not in p)
        eps, rotation = np.linalg.eigh(h[np.ix_(p, p)])
        hqq = h[np.ix_(q, q)]
        poles, q_rotation = np.linalg.eigh(hqq)
        coupling = rotation.T @ h[np.ix_(p, q)] @ q_rotation
        residues = np.einsum("it,jt->tij", coupling, coupling)
        sigma = MatrixSelfEnergy(eps, poles, residues, reference_identity)
        for name, value in dict(hamiltonian_Ha=_owned(h), p_indices=p, q_indices=q,
                                p_rotation=_owned(rotation), q_rotation=_owned(q_rotation),
                                q_poles_Ha=_owned(poles), coupling_Ha=_owned(coupling),
                                self_energy=sigma, reference_identity=reference_identity,
                                csf_identities=tuple(csf_identities), radial_binding=radial_binding,
                                hamiltonian_terms=hamiltonian_terms).items():
            object.__setattr__(self, name, value)

    def effective_hamiltonian(self, energy_Ha):
        """H_PP + H_PQ(E-H_QQ)^-1 H_QP in the original P CSF basis."""
        r = self.p_rotation
        return self.hamiltonian_Ha[np.ix_(self.p_indices, self.p_indices)] + r @ self.self_energy.evaluate(energy_Ha) @ r.T

    def derivative(self, energy_Ha):
        r = self.p_rotation
        return r @ self.self_energy.derivative(energy_Ha) @ r.T

    def bloch_reduction(self, root_indices, *, dimension_cap=256,
                        memory_bytes=64*1024**2, projection_condition_max=1e8,
                        residual_tolerance_Ha=1e-10, degeneracy_tolerance_Ha=1e-10):
        """Exact finite Bloch wave operator and metric, with explicit root selection.

        For selected full eigenstates Psi, C=P Psi must be square/invertible.
        Omega=Psi C^-1, G=Omega^T Omega and H_B=P H Omega obey H Omega=
        Omega H_B and H_B^T G=G H_B. The orthonormal operator is formed by
        the physical isometry Omega G^-1/2; matrix entries are never averaged.
        Partial degenerate subspaces, singular projections, excess resources
        or failed full residuals reject the request. No maximum-overlap root
        substitution is made. This dense finite anchor is capped explicitly,
        and supplies no metric/accuracy certificate for approximate Sigma1/2.
        """
        roots = tuple(root_indices)
        n, p = len(self.hamiltonian_Ha), len(self.p_indices)
        if (len(roots) != p or len(set(roots)) != p
                or any(isinstance(i, bool) or not isinstance(i, (int, np.integer))
                       or not 0 <= i < n for i in roots)):
            raise ValueError("one distinct full root per model-space dimension required")
        if (isinstance(dimension_cap, bool) or not isinstance(dimension_cap, (int, np.integer))
                or dimension_cap < n or isinstance(memory_bytes, bool)
                or not isinstance(memory_bytes, (int, np.integer)) or memory_bytes < 1):
            raise ValueError("positive finite dense dimension/resource caps required")
        # Full eigensystem/workspace, transformations and all audit residuals.
        if 8 * (10*n*n + 8*n*p + 20*p*p) > memory_bytes:
            raise MemoryError("complete finite wave-operator/metric work exceeds cap")
        controls = (projection_condition_max, residual_tolerance_Ha, degeneracy_tolerance_Ha)
        if any(isinstance(x, bool) or np.ndim(x) != 0 or not np.isrealobj(x)
               or not np.isfinite(x) or x <= 0 for x in controls):
            raise ValueError("positive real condition, residual and degeneracy controls required")
        if projection_condition_max < 1:
            raise ValueError("projection condition bound must be at least one")
        energy, vectors = np.linalg.eigh(self.hamiltonian_Ha)
        selected = set(roots)
        for i in roots:
            cluster = np.flatnonzero(abs(energy-energy[i]) <= degeneracy_tolerance_Ha)
            if any(int(j) not in selected for j in cluster):
                raise ValueError("selected roots split a degenerate invariant subspace")
        states = vectors[:, roots]
        right = states[list(self.p_indices)]
        singular = np.linalg.svd(right, compute_uv=False)
        condition = float(singular[0]/singular[-1]) if singular[-1] > 0 else float('inf')
        if not np.isfinite(condition) or condition > projection_condition_max:
            raise ValueError("selected invariant subspace has a singular/ill-conditioned P projection")
        # Solve C^T Omega^T=Psi^T; no eigenvector/diagonal-energy fitting.
        wave = np.linalg.solve(right.T, states.T).T
        left = np.linalg.solve(right, np.eye(p))
        metric = wave.T @ wave
        directed = self.hamiltonian_Ha[list(self.p_indices)] @ wave
        values, rotation = np.linalg.eigh(metric)
        if np.any(values <= 0) or not np.all(np.isfinite(values)):
            raise RuntimeError("physical wave-operator metric is not positive definite")
        inverse_sqrt = (rotation / np.sqrt(values)) @ rotation.T
        isometry = wave @ inverse_sqrt
        orthonormal = isometry.T @ self.hamiltonian_Ha @ isometry
        residual = float(np.max(abs(self.hamiltonian_Ha @ wave - wave @ directed)))
        adjoint = float(np.max(abs(directed.T @ metric - metric @ directed)))
        physical = max(residual, adjoint, float(np.max(abs(orthonormal-orthonormal.T))),
                       float(np.max(abs(directed @ right-right*energy[list(roots)]))),
                       float(np.max(abs(left @ directed-energy[list(roots), None]*left))))
        norm_error = max(float(np.max(abs(wave[list(self.p_indices)]-np.eye(p)))),
                         float(np.max(abs(left @ right-np.eye(p)))),
                         float(np.max(abs(right.T @ metric @ right-np.eye(p)))),
                         float(np.max(abs(isometry.T @ isometry-np.eye(p)))))
        if not np.isfinite(physical) or physical > residual_tolerance_Ha or norm_error > 1e-10:
            raise RuntimeError("full Bloch/metric/left-right/orthonormal residual gate failed")
        return FiniteBlochReduction(roots, energy[list(roots)], wave, metric, directed,
                                    isometry, orthonormal, right, left, condition,
                                    residual, adjoint, self)

    def diagonal_q_control(self, energy_Ha):
        """Diagnostic with off-diagonal Q-Q couplings removed in the CSF basis.

        This is representation dependent and is NOT complete MBPT(2).
        The retained diagonal includes Coulomb terms. Do not add its result
        to the full-Q result; both act on the same P/Q configurations.
        """
        # Validate scalar energy using the shared self-energy contract.
        self.self_energy._denominators(energy_Ha)
        hpp = self.hamiltonian_Ha[np.ix_(self.p_indices, self.p_indices)]
        hpq = self.hamiltonian_Ha[np.ix_(self.p_indices, self.q_indices)]
        diagonal = np.diag(self.hamiltonian_Ha)[list(self.q_indices)]
        den = energy_Ha-diagonal
        coupled = np.any(hpq != 0, axis=0)
        if np.any(abs(den[coupled]) < self.self_energy.denominator_floor_Ha):
            raise ValueError("resonant diagonal-Q control denominator")
        return hpp + (hpq[:, coupled]/den[coupled]) @ hpq[:, coupled].T

    def solve(self, bracket_Ha, *, reference_index=0):
        root = self.self_energy.solve_dyson(bracket_Ha, reference_index=reference_index)
        p = self.p_rotation @ root.coefficients
        rhs = self.coupling_Ha.T @ root.coefficients
        den = root.energy_Ha-self.q_poles_Ha
        # Exactly uncoupled Q coordinates contribute zero even if their
        # eigenvalue equals E. No division by zero or magnitude threshold.
        coupled = np.any(self.coupling_Ha != 0, axis=0)
        q_eigen = np.zeros(len(den))
        q_eigen[coupled] = rhs[coupled]/den[coupled]
        q = self.q_rotation @ q_eigen
        vector = np.zeros(len(self.hamiltonian_Ha))
        vector[list(self.p_indices)] = p*np.sqrt(root.residue)
        vector[list(self.q_indices)] = q*np.sqrt(root.residue)
        residual = float(np.max(abs(self.hamiltonian_Ha @ vector-root.energy_Ha*vector)))
        if abs(float(vector @ vector)-1) > 1e-10 or residual > 1e-10:
            raise RuntimeError("finite CI reconstructed state failed norm or full Hamiltonian residual")
        return FiniteCIRoot(root.energy_Ha, root.shift_Ha, root.residue, vector,
                            residual, reference_index, self.reference_identity)


def build_finite_ci_effective_hamiltonian(csfs, p_indices, orbitals_dict,
                                        bsp_data, Z, ns, *, reference_identity,
                                        nuclear_potential=None, dimension_cap=256,
                                        max_determinants=2000, include_breit=False,
                                        radial_snapshot=None):
    """Build one DC or static DCB Hamiltonian and eliminate complementary Q.

    Pass a complete declared CSF list and explicit P indices in that list.
    Core holes are allowed in Q because no separate core Sigma is injected.
    The API rejects mixed N/J/parity, duplicate coupling trees and excess
    dimensions before native matrix construction. Basis caps raise; they
    never select or shrink the physical space. ``include_breit=True`` adds
    the shared Gaunt plus static-retardation operator throughout PP, PQ and
    QQ before elimination. Re-solving includes its finite-space correlated
    response; no Breit scalar is added afterwards. This does not optimize
    orbitals or include frequency-dependent exchange, QED or external Sigma.
    ``radial_snapshot`` may reuse only matching native DC radial data.
    """
    from atomx.angular.csf_hamiltonian import csf_signature
    from atomx.config.csf import get_shared_orbital_basis
    from atomx.physics.csf.effective_one_body import CIReferenceBinding
    from atomx.physics.csf.matrix import build_ci_matrix

    if not isinstance(include_breit, bool):
        raise ValueError("include_breit must be bool")
    csfs = tuple(csfs)
    if (isinstance(dimension_cap, bool) or not isinstance(dimension_cap, (int, np.integer))
            or dimension_cap < 1 or not csfs or len(csfs) > dimension_cap):
        raise ValueError("finite CI space exceeds the positive declared dimension cap")
    identities = tuple(csf_signature(c).key for c in csfs)
    if len(set(identities)) != len(identities):
        raise ValueError("duplicate P/Q CSF: model spaces must be disjoint")
    if len({(c.n_electrons, c.total_J2, c.parity) for c in csfs}) != 1:
        raise ValueError("finite P/Q CSFs must have one N, J and parity")
    if any(c.parity != (-1)**sum((sh.kappa if sh.kappa > 0 else -sh.kappa-1)*sh.occupation
                                 for sh in c.subshells) for c in csfs):
        raise ValueError("CSF declared parity disagrees with occupations")
    p_indices = tuple(p_indices)
    if (not p_indices or len(set(p_indices)) != len(p_indices)
            or any(isinstance(i, bool) or not isinstance(i, (int, np.integer))
                   or not 0 <= i < len(csfs) for i in p_indices)):
        raise ValueError("distinct valid nonempty P indices required")
    if not isinstance(reference_identity, str) or not reference_identity.strip():
        raise ValueError("missing physical reference identity")
    keys = get_shared_orbital_basis(csfs)
    binding = CIReferenceBinding.capture([(*orbitals_dict[key], key[1]) for key in keys],
                                         {}, bsp_data, Z, ns, nuclear_potential)
    h = build_ci_matrix(csfs, orbitals_dict, keys, bsp_data, Z, ns,
                        nuclear_potential=nuclear_potential,
                        max_determinants=max_determinants,
                        include_breit=include_breit, radial_snapshot=radial_snapshot)
    return FiniteCIModel(h, p_indices, reference_identity=reference_identity,
                         csf_identities=identities, radial_binding=binding,
                         hamiltonian_terms=("dirac-coulomb", "static-breit")
                         if include_breit else ("dirac-coulomb",))
