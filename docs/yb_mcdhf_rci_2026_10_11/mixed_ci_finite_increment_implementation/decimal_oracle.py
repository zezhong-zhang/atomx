"""Independent Decimal interpretation of complete signed native float64 assets.

No Atomx import or numerical kernel call. The caller captures immutable angular,
cell-bank, one-body and metric assets. Every binary64 asset is converted exactly;
this is an arithmetic oracle for that finite operator, not another backend.
"""
from decimal import Decimal, localcontext
from functools import lru_cache, wraps

ZERO, TWO = Decimal(0), Decimal(2)


def arithmetic80(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        with localcontext() as ctx:
            ctx.prec = 80
            return function(*args, **kwargs)
    return wrapped


def D(value):
    return Decimal.from_float(float(value))


def dot(a, b):
    return sum((x * y for x, y in zip(a, b, strict=True)), ZERO)


def matvec(matrix, vector):
    return tuple(dot(row, vector) for row in matrix)


def quotient(matrix, vector):
    return dot(vector, matvec(matrix, vector)) / dot(vector, vector)


def response_oracle(H0, H1, c0, c1, lambda0, lambda1, phase):
    """Direct residual/finite quotient identities with arbitrary positive norms."""
    with localcontext() as ctx:
        ctx.prec = 80
        h0, h1 = tuple(tuple(map(D, r)) for r in H0), tuple(tuple(map(D, r)) for r in H1)
        a, b = tuple(map(D, c0)), tuple(map(D, c1))
        n0, n1 = dot(a, a), dot(b, b)
        l0, l1 = D(lambda0), D(lambda1)
        x = tuple(Decimal(phase) * v for v in a)
        change = tuple(y - z for y, z in zip(b, x, strict=True))
        delta = tuple(tuple(y - x for x, y in zip(r0, r1, strict=True))
                      for r0, r1 in zip(h0, h1, strict=True))
        radial = dot(b, matvec(delta, b)) / n1
        q0, q1 = quotient(h0, a), quotient(h1, b)
        def centered(h, origin):
            return tuple(tuple(v - (origin if i == j else ZERO) for j, v in enumerate(row))
                         for i, row in enumerate(h))
        A0, A1 = centered(h0, l0), centered(h1, l1)
        r0, r1 = matvec(A0, x), matvec(A1, b)
        reported = (dot(change, matvec(A0, change)) / n1,
                    TWO * dot(change, r0) / n1, dot(x, r0) / n1, -dot(b, r1) / n1)
        Aq = centered(h0, q0)
        rq = matvec(Aq, x)
        rayleigh = (dot(change, matvec(Aq, change)) / n1,
                    TWO * dot(change, rq) / n1, dot(x, rq) / n1)
        return dict(radial=radial, reported_terms=reported, rayleigh_terms=rayleigh,
                    reported_response=sum(reported, ZERO), rayleigh_response=sum(rayleigh, ZERO),
                    reported_delta=l1-l0, quotient_delta=q1-q0,
                    quotient_minus_reported=(q0-l0, q1-l1), norm0=n0, norm1=n1)


class DecimalDCRadialOperator:
    """All P/Q cells, ordered ket norm and signed CSF projection, precision80.

    Banks have shape (left_band,right_band,left_row,right_row), exactly the
    captured public rkbT orientation. Density and source-potential caches retain
    all exact terms. They accelerate reuse without filtering small amplitudes.
    """

    def __init__(self, *, ns, orders, one_body, banks, blocks, dimension):
        self.ns, self.orders, self.dimension = ns, tuple(orders), dimension
        self.one_body = {k: (tuple(tuple(map(D, r)) for r in h),
                             tuple(tuple(map(D, r)) for r in s))
                         for k, (h, s) in one_body.items()}
        self.banks = banks
        self.blocks = blocks
        self.points = {}
        # Only this finite oracle instance retains captured assets.
        for name in ("density", "decimal_bank", "potential", "radial"):
            setattr(self, name, lru_cache(maxsize=None)(getattr(self, name)))

    @arithmetic80
    def add_point(self, name, reference, increments=None):
        # Decimal sums express the ideal reference+explicit increment domain;
        # otherwise preserve the exact stored binary64 endpoint domain.
        self.points[name] = {key: tuple(tuple(D(v) + (D(d) if increments is not None else ZERO)
                                        for v, d in zip(component,
                                            increments[key][side] if increments is not None
                                            else (0 for _ in component), strict=True))
                                        for side, component in enumerate(pair))
                             for key, pair in reference.items()}

    @arithmetic80
    def density(self, point, a, b, side):
        order = self.orders[side]
        x, y = self.points[point][a][side], self.points[point][b][side]
        return tuple(tuple(x[i]*y[i] if offset == 0 else
                           x[i]*y[i+offset] + x[i+offset]*y[i]
                           for offset in range(min(order, self.ns-i)))
                     for i in range(self.ns))

    def decimal_bank(self, rank, left, right):
        bank = self.banks[rank, left, right]
        return tuple(tuple(tuple(tuple(D(bank[ip, jp, i, j]) for j in range(self.ns-jp))
                                  for i in range(self.ns-ip))
                            for jp in range(self.orders[right]))
                      for ip in range(self.orders[left]))

    @arithmetic80
    def potential(self, point, rank, a, b, left):
        output = [[ZERO for _ in range(min(self.orders[left], self.ns-i))]
                  for i in range(self.ns)]
        for right in range(2):
            bank = self.decimal_bank(rank, left, right)
            density = self.density(point, a, b, right)
            for ip in range(self.orders[left]):
                for i in range(self.ns-ip):
                    value = ZERO
                    for jp in range(self.orders[right]):
                        # Complete native cell bank: only out-of-extent cells
                        # are structurally absent, no magnitude cutoff.
                        value += sum((v*density[j][jp] for j, v in enumerate(bank[ip][jp][i])), ZERO)
                    output[i][ip] += value
        return tuple(tuple(row) for row in output)

    @arithmetic80
    def radial(self, point, key):
        if key[0] == 'h':
            _, na, k, nb = key
            a = sum(self.points[point][na, k], ())
            b = sum(self.points[point][nb, k], ())
            h, s = self.one_body[k]
            return dot(a, matvec(h, b)) / dot(b, matvec(s, b))
        _, rank, a, c, b, d = key
        source, target = tuple(sorted((b, d))), tuple(sorted((a, c)))
        total = ZERO
        for side in range(2):
            potential = self.potential(point, rank, *source, side)
            density = self.density(point, *target, side)
            total += sum((dot(row, drow) for row, drow in zip(potential, density, strict=True)), ZERO)
        return total

    def matrix(self, point):
        """Independently project complete determinant table, then symmetrize."""
        with localcontext() as ctx:
            ctx.prec = 80
            matrix = [[ZERO for _ in range(self.dimension)] for _ in range(self.dimension)]
            for block in self.blocks:
                keys, projection, pairs, rows, columns, symmetric = block
                p = tuple(tuple(map(D, row)) for row in projection)
                for pair, entries in zip(pairs, rows, strict=True):
                    i, j = pair
                    scalar = sum((D(value)*self.radial(point, keys[index])
                                  for index, value in entries), ZERO)
                    for a, global_a in enumerate(columns):
                        for b, global_b in enumerate(columns):
                            matrix[global_a][global_b] += p[i][a]*scalar*p[j][b]
                            if symmetric and i != j:
                                matrix[global_a][global_b] += p[j][a]*scalar*p[i][b]
            return tuple(tuple((matrix[i][j]+matrix[j][i])/TWO
                               for j in range(self.dimension)) for i in range(self.dimension))

    def contracted_difference(self, left, right, targets):
        """Complete signed trial-vector contraction, not diagonal occupations."""
        with localcontext() as ctx:
            ctx.prec = 80
            return sum((D(weight)*dot(tuple(map(D, vector)),
                tuple(dot(tuple(x-y for x, y in zip(new, old, strict=True)), tuple(map(D, vector)))
                      for old, new in zip(left, right, strict=True)))
                / dot(tuple(map(D, vector)), tuple(map(D, vector)))
                for weight, vector in targets), ZERO)
