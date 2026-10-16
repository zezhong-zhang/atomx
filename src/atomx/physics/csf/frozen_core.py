"""Normal ordering of the Dirac–Coulomb Hamiltonian over closed subshells.

The radial callback uses the same keys as the full CSF Hamiltonian. Keeping
this contraction independent of the radial representation lets frozen-core
and full CI use identical integrals, including off-diagonal one-body terms.
"""
from atomx.angular.coefficients import reduced_C_jj


def closed_core_one_body(active, core, radial, *, bare_one_body=None):
    """Return ``h_ab + sum_c (J_c - K_c)_ab`` for active radial orbitals.

    ``active`` contains ``(n, kappa)`` keys; ``core`` maps disjoint keys to
    their occupations, which must be exactly ``2j+1``. ``radial`` evaluates
    ``('h', n_a, kappa, n_b)`` and ``('r', rank, a, c, b, d)`` (pair-density
    order). An optional bare one-body map can supply an eigenstate identity
    for a common local potential, avoiding numerical differentiation.

    A closed subshell has only a monopole direct field. Wigner–Eckart
    orthogonality sums its exchange over all magnetic substates analytically:
    the coefficient is ``|<kappa_a||C^k||kappa_c>|² / (2j_a+1)``.
    The constant core energy is omitted; it cancels between states sharing
    the same frozen core.
    """
    active = tuple(active)
    if set(active) & core.keys():
        raise ValueError('active and frozen-core orbitals must be disjoint')
    return closed_shell_one_body(active, core, radial, bare_one_body=bare_one_body)


def closed_shell_one_body(active, core, radial, *, bare_one_body=None):
    """Full scalar closed-shell Fock contraction, including occupied targets.

    The same analytic magnetic-substate sum used for frozen-core normal
    ordering applies to the full core Fock operator. ``active`` may therefore
    overlap ``core`` here: self-interaction cancels between direct and
    exchange contributions, rather than removing a subshell. The frozen-core
    interface above retains its strict disjoint-space contract.
    """
    active = sorted(set(active))
    for c, occupation in core.items():
        if occupation != 2 * abs(c[1]):
            raise ValueError(f'frozen core orbital {c} must be a closed subshell')
    matrix = {}
    for a in active:
        for b in active:
            if a[1] != b[1]:
                continue
            value = (radial(('h', a[0], a[1], b[0])) if bare_one_body is None
                     else bare_one_body[a, b])
            for c, occupation in core.items():
                value += occupation * radial(('r', 0, a, b, c, c))
                for rank in range(abs(abs(a[1]) - abs(c[1])),
                                  abs(a[1]) + abs(c[1])):
                    angular = reduced_C_jj(a[1], c[1], rank)**2 / (2 * abs(a[1]))
                    if angular:
                        value -= angular * radial(('r', rank, a, c, c, b))
            matrix[a, b] = float(value)
    return matrix
