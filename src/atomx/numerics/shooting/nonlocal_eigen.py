"""Matrix-free nonlocal bound states using a shooting resolvent.

The caller supplies exchange as a linear whole-spinor action with other
orbitals frozen. Neither exchange coefficients nor SCF policy live here.
"""
import numpy as np
from scipy.sparse.linalg import LinearOperator, eigs, gmres
from .constants import SL
from .dirac_response import DiracResponse
from .dirac_orbital import OrbitalSolveError, _count_nodes


def _direct_nonlocal_bound_state(energy, n, kappa, problem, initial, exchange,
                              *, minimum_overlap=.9, residual_tolerance=1e-7):
    """Resolve an unconstrained continuation root of h_local - exchange.

    Applies (F-sigma)^-1 = (I-G_sigma K)^-1 G_sigma. All inverses use
    the same full-box shooting quadrature. This is not a no-pair projector
    or a solver for occupied same-kappa constraints.
    """
    weights = problem.weights
    initial = np.asarray(initial, dtype=float)
    if (initial.shape != (2, len(weights)) or not np.all(np.isfinite(initial))
            or np.any(weights < 0) or not 0 < minimum_overlap <= 1
            or not np.isfinite(residual_tolerance) or residual_tolerance <= 0):
        raise ValueError('invalid nonlocal continuation inputs')
    if kappa == 0 or n <= (kappa if kappa > 0 else -kappa-1):
        raise ValueError('invalid bound-state quantum numbers')
    mask = weights > 0
    scale = np.sqrt(weights[mask])
    size = 2*len(scale)
    if size < 8:
        raise ValueError('nonlocal continuation grid is too small')

    def pack(value):
        return (value[:, mask]*scale).ravel()

    def unpack(value):
        result = np.zeros_like(initial)
        result[:, mask] = value.reshape(2, -1)/scale
        return result

    seed = pack(initial)
    norm = np.linalg.norm(seed)
    if not np.isfinite(norm) or norm <= 1e-15:
        raise ValueError('zero-norm continuation seed')
    seed /= norm
    # At SCF stationarity the incoming multiplier is an eigenvalue of F.
    # Offset only the inverse's shift; verify roots at their returned energy.
    sigma = float(energy)-max(abs(float(energy))*1e-3, 1e-8)
    response = DiracResponse(sigma, int(kappa), problem, boundary='box')
    calls = [0, 0]

    def source(vector):
        calls[0] += 1
        value = np.asarray(exchange(unpack(vector)), dtype=float)
        if value.shape != initial.shape or not np.all(np.isfinite(value)):
            raise ValueError('exchange action must return a finite radial spinor')
        return value

    def left(vector):
        return vector-pack(response.apply(source(vector)))

    operator = LinearOperator((size, size), matvec=left, dtype=float)

    def inverse(vector):
        calls[1] += 1
        rhs = pack(response.apply(unpack(vector)))
        result, info = gmres(operator, rhs, rtol=1e-10, atol=0.,
                             restart=40, maxiter=100)
        if info:
            raise OrbitalSolveError(f'nonlocal resolvent GMRES failed: status={info}')
        return result

    resolvent = LinearOperator((size, size), matvec=inverse, dtype=float)
    values, vectors = eigs(resolvent, k=4, which='LM', v0=seed,
                           tol=1e-10, maxiter=200)
    required_nodes = n-(kappa if kappa > 0 else -kappa-1)-1
    candidates = []
    diagnostics = []
    for j, value in enumerate(values):
        if abs(value) < 1e-20:
            continue
        candidate_energy = sigma+1/value
        if abs(candidate_energy.imag) > 1e-10 or not -SL**2 < candidate_energy.real < 0:
            continue
        if np.linalg.norm(vectors[:, j].imag) > 1e-8:
            continue
        vector = vectors[:, j].real
        vector /= np.linalg.norm(vector)
        overlap = float(np.dot(seed, vector))
        if overlap < 0:
            vector = -vector
            overlap = -overlap
        psi = unpack(vector)
        nodes = _count_nodes(psi[0], len(weights))
        diagnostic = dict(energy=float(candidate_energy.real), overlap=overlap,
                          dominant_nodes=int(nodes))
        diagnostics.append(diagnostic)
        if overlap < minimum_overlap or nodes != required_nodes:
            continue
        e = float(candidate_energy.real)
        # Verify (h-sigma) psi = K psi + (e-sigma) psi with the
        # off-pole inverse. Inverting h-e instead loses relative accuracy
        # when exchange is weak, and is undefined when exchange vanishes.
        rhs = source(vector)+(e-sigma)*psi
        residual = float(np.linalg.norm(vector-pack(response.apply(rhs))))
        diagnostic['residual'] = residual
        if residual <= residual_tolerance:
            candidates.append((overlap, psi, e, residual))
    if len(candidates) != 1:
        error = OrbitalSolveError(
            f'nonlocal continuation requires one accepted root; found {len(candidates)}; '
            f'n={n}, kappa={kappa}, seed_energy={energy}, candidates={diagnostics}')
        error.candidate_diagnostics = diagnostics
        error.exchange_applications = calls[0]
        error.inverse_applications = calls[1]
        raise error
    overlap, psi, e, residual = candidates[0]
    return psi[0], psi[1], e, dict(overlap=overlap, inverse_shift=sigma,
        green_equation_residual=residual, residual_form="shifted_green_identity",
        exchange_applications=calls[0],
        inverse_applications=calls[1], dominant_nodes=required_nodes)


def solve_nonlocal_bound_state(energy, n, kappa, problem, initial, exchange,
                              *, minimum_overlap=.9, residual_tolerance=1e-7):
    """Track a requested radial root, retaining every overlap/residual gate.

    An unconverged SCF orbital can contain neighboring-state admixtures. When
    direct continuation rejects it, anchor to the local n,kappa eigenstate and
    introduce exchange in four steps. Each step must pass the same root,
    overlap and shifted-residual checks. Incorrect incoming node labels still
    fail. This is an unconstrained-channel numerical continuation, not an
    approximation to exchange or a change of the final Hamiltonian.
    """
    try:
        return _direct_nonlocal_bound_state(
            energy, n, kappa, problem, initial, exchange,
            minimum_overlap=minimum_overlap, residual_tolerance=residual_tolerance)
    except OrbitalSolveError as error:
        if not hasattr(error, 'candidate_diagnostics'):
            raise
        required = n - (kappa if kappa > 0 else -kappa-1) - 1
        if _count_nodes(initial[0], len(problem.r)) != required:
            raise
        exchange_calls = error.exchange_applications
        inverse_calls = error.inverse_applications
    from .dirac_bound import dbound
    count = len(problem.r)
    e, ier, p, q, _ = dbound(float(energy), 1e-12, int(n), int(kappa),
        problem.r, problem.rv, *problem.spline, count, problem.r, count)
    if ier:
        raise OrbitalSolveError(f'continuation local anchor failed: {ier}')
    psi = np.array([p, q])
    psi /= np.sqrt(np.sum(psi*psi*problem.weights))
    steps = []
    for coupling in (.25, .5, .75, 1.):
        # Hellmann–Feynman predictor for dF/dlambda = -K.
        predicted = e - .25*float(np.sum(psi*exchange(psi)*problem.weights))
        exchange_calls += 1
        p, q, e, report = _direct_nonlocal_bound_state(
            predicted, n, kappa, problem, psi,
            lambda x: coupling*exchange(x), minimum_overlap=minimum_overlap,
            residual_tolerance=residual_tolerance)
        psi = np.array([p, q])
        steps.append(dict(coupling=coupling, energy=e, **report))
    report = dict(report)
    report['exchange_continuation'] = steps
    report['exchange_applications'] = exchange_calls + sum(step['exchange_applications'] for step in steps)
    report['inverse_applications'] = inverse_calls + sum(step['inverse_applications'] for step in steps)
    incoming = np.asarray(initial)
    report['overlap_with_incoming_orbital'] = float(
        abs(np.sum(psi*incoming*problem.weights)) /
        np.sqrt(np.sum(incoming*incoming*problem.weights)))
    return p, q, e, report
