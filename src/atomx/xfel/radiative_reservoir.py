"""Signed radiative clustering and a finite-time dissipator error bound.

The bound compares geometric coefficients to arithmetic coefficients inside
retained frequency clusters, with the same Hamiltonian and other channels.
It does not bound secularization, omitted Lamb shifts or microscopic bath error.
No n²-by-n² superoperator is built to certify the production bound.
"""

import numpy as np


def arithmetic_difference_bound(cluster):
    """Upper bound eta/s on the induced trace norm of D_arith - D_geom.

    C_a=sqrt(gamma_a) A_a; gamma(E) proportional to E³. For each ordered
    frequency pair delta=(gamma_a+gamma_b)/(2 sqrt(gamma_a gamma_b))-1.
    ||C_a X C_b† - {C_b†C_a,X}/2||_1 <= 2||C_a||_2||C_b||_2||X||_1.
    Use ||C||_2 <= sqrt(||C||_1 ||C||_infinity), avoiding expensive SVDs.
    Summing both orders and spherical q gives eta. For a CPTP geometric
    evolution, Gronwall yields ||rho_arith-rho_geom||_1 <= exp(eta*T)-1.
    The arithmetic reference itself need not be completely positive.
    """
    norms = [tuple(np.sqrt(np.linalg.norm(op, 1) * np.linalg.norm(op, np.inf))
                   for op in group) for _, group, _ in cluster]
    eta = 0.0
    for a, (ea, _, _) in enumerate(cluster):
        for b in range(a + 1, len(cluster)):
            eb = cluster[b][0]
            # cosh(log(sqrt(gamma_b/gamma_a)))-1, stable near degeneracy.
            delta = 2 * np.sinh(0.75 * np.log(eb / ea)) ** 2
            eta += 4 * delta * sum(x * y for x, y in zip(norms[a], norms[b], strict=True))
    if not np.isfinite(eta):
        raise ValueError("nonfinite radiative generator error bound")
    return float(eta)


def finite_time_error_bound(eta_per_s, duration_fs):
    """Conservative trace norm (not half-trace-distance) error bound."""
    if not np.isfinite([eta_per_s, duration_fs]).all() or eta_per_s < 0 or duration_fs < 0:
        raise ValueError("finite nonnegative generator bound and duration required")
    exponent = eta_per_s * duration_fs * 1e-15
    return float(np.expm1(exponent)) if exponent < 700 else np.inf


def cluster_radiative_baths(baths, mode):
    """Reuse signed q operators; return clustered baths, rate factors, records.

    Legacy spectral-coefficient gates are unchanged. geometric_bounded instead
    records an absolute generator bound, gated against the actual integration
    duration by CoherentSystem.solve. The tolerance is fixed at 1e-3 in both
    contracts, but they bound different explicitly named quantities.
    """
    geometric = mode in ("geometric_clustered", "geometric_bounded")
    if mode not in ("clustered", "geometric_clustered", "geometric_bounded"):
        raise ValueError("explicit radiative cluster mode required")
    clusters = []
    for bath in sorted(baths, key=lambda entry: entry[0]):
        if not np.isfinite(bath[0]) or bath[0] <= 0:
            raise ValueError("positive finite radiative frequency required")
        if (clusters and bath[0] - clusters[-1][-1][0]
                < 10 * max(bath[2], clusters[-1][-1][2])):
            clusters[-1].append(bath)
        else:
            clusters.append([bath])
    merged, factors, records = [], {}, []
    for cluster in clusters:
        center = 0.5 * (cluster[0][0] + cluster[-1][0])
        error = max(abs((center / entry[0]) ** 3 - 1) for entry in cluster)
        ratio = (cluster[-1][0] / cluster[0][0]) ** 3
        geometric_error = 1 - 2 * np.sqrt(ratio) / (1 + ratio)
        checked_error = geometric_error if geometric else error
        if mode != "geometric_bounded" and checked_error > 1e-3:
            raise ValueError(
                f"radiative cluster {cluster[0][0]:.8g}..{cluster[-1][0]:.8g} eV "
                f"varies too much in vacuum spectral approximation: {checked_error:.6g}"
            )
        operators = [np.zeros_like(cluster[0][1][0]) for _ in range(3)]
        for energy, group, _ in cluster:
            factors[energy] = 1.0 if geometric else (center / energy) ** 3
            for target, op in zip(operators, group, strict=True):
                target += np.sqrt(factors[energy]) * op
        merged.append([center, operators, max(entry[2] for entry in cluster)])
        record = {
            "frequencies_eV": tuple(entry[0] for entry in cluster),
            "reference_eV": center,
            "max_rate_relative_change": 0.0 if geometric else error,
            "geometric_vs_arithmetic_spectral_error": geometric_error,
        }
        if mode == "geometric_bounded":
            record["arithmetic_generator_difference_bound_per_s"] = arithmetic_difference_bound(cluster)
        records.append(record)
    return merged, factors, records
