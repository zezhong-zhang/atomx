"""Exact stored-array full-Gram centering, with no physical producer.

Fixed baseline multipliers qualify two declared finite controls only. This is
consumer bookkeeping, never a new energy/gradient callback or global bound.
"""
from fractions import Fraction


class ConstraintBookkeeping:
    def __init__(self, keys, waves, gradient, full_weights, operators, logical,
                 layout, *, max_products, gram_gate=1e-10):
        self.products = 0
        self.maximum = max_products
        self.gate = self.f(gram_gate)
        self.keys = tuple(keys)
        self.layout = layout
        self.c = {k: self.vector((*waves[k][0], *waves[k][1])) for k in keys}
        self.g = {k: self.vector(gradient[k]) for k in keys}
        self.operators = {k: tuple(self.sparse(m) for m in pair) for k, pair in operators.items()}
        logical = {k: self.sparse(m) for k, m in logical.items()}
        self.sc = {k: self.action(self.operators[k[1]][1], self.c[k]) for k in keys}
        # Complete actual raw h/ket-norm quotient correction; signed h terms.
        for term, coefficient in full_weights.items():
            if term[0] != 'h':
                continue
            _, na, kappa, nb = term
            a, b = (na, kappa), (nb, kappa)
            weight = self.f(coefficient)
            raw = self.operators[kappa][0]
            transpose = self.sparse(operators[kappa][0].T)
            hb, hta = self.action(raw, self.c[b]), self.action(transpose, self.c[a])
            lb, la = self.action(logical[kappa], self.c[b]), self.action(logical[kappa], self.c[a])
            norm, numerator = self.dot(self.c[b], self.sc[b]), self.dot(self.c[a], hb)
            if norm <= 0:
                raise ValueError('Nonpositive exact radial ket norm')
            for j in range(len(hb)):
                self.g[a][j] += self.mul(weight, hb[j] / norm - lb[j])
                self.g[b][j] += self.mul(weight, hta[j] / norm - la[j]
                                       - self.mul(2 * numerator / norm**2, self.sc[b][j]))
        self.blocks = []
        for group, _, _, _ in layout:
            cg = [[self.dot(self.c[a], self.g[b]) for b in group] for a in group]
            lam = [[(cg[i][j] + cg[j][i]) / 4 for j in range(len(group))] for i in range(len(group))]
            gram = [[self.dot(self.c[a], self.sc[b]) for b in group] for a in group]
            if max(abs(v - (i == j)) for i, row in enumerate(gram) for j, v in enumerate(row)) > self.gate:
                raise ValueError('Exact baseline actual Gram gate failed')
            self.blocks.append((group, lam, gram))

    @staticmethod
    def f(value):
        return Fraction.from_float(float(value))

    def mul(self, a, b):
        self.products += 1
        if self.products > self.maximum:
            raise RuntimeError('Exact full-Gram scalar-product budget')
        return a * b

    def dot(self, a, b):
        return sum((self.mul(x, y) for x, y in zip(a, b, strict=True)), Fraction(0))

    def vector(self, values):
        return [self.f(v) for v in values]

    def sparse(self, matrix):
        return [[(j, self.f(v)) for j, v in enumerate(row) if v != 0] for row in matrix]

    def action(self, rows, vector):
        return [sum((self.mul(v, vector[j]) for j, v in row), Fraction(0)) for row in rows]

    def ideal(self, direction):
        waves = {}
        for group, projector, shape, where in self.layout:
            sparse_p = self.sparse(projector)
            section = self.vector(direction[where])
            for j, k in enumerate(group):
                waves[k] = self.action(sparse_p, section[j::shape[1]])
        ambient = sum((self.dot(self.g[k], waves[k]) for k in self.keys), Fraction(0))
        normal = Fraction(0)
        for group, lam, _ in self.blocks:
            for i, a in enumerate(group):
                for j, b in enumerate(group):
                    dg = self.dot(waves[a], self.sc[b]) + self.dot(waves[b], self.sc[a])
                    normal += self.mul(lam[i][j], dg)
        return {'ambient': ambient, 'normal': normal, 'centered': ambient - normal}

    def trial(self, actual, increments):
        point = {k: self.vector((*actual[k][0], *actual[k][1])) for k in self.keys}
        total = {k: [t - c for t, c in zip(point[k], self.c[k], strict=True)] for k in self.keys}
        explicit = {k: self.vector((*increments[k][0], *increments[k][1])) for k in self.keys}
        reconstruction = {k: [d - e for d, e in zip(total[k], explicit[k], strict=True)] for k in self.keys}
        normal, normal_explicit = Fraction(0), Fraction(0)
        blocks = []
        for group, lam, base_gram in self.blocks:
            metric = self.operators[group[0][1]][1]
            st = {k: self.action(metric, total[k]) for k in group}
            se = {k: self.action(metric, explicit[k]) for k in group}
            sr = {k: self.action(metric, reconstruction[k]) for k in group}
            delta_gram, split_gram, trial_gram = [], [], []
            for i, a in enumerate(group):
                row, split_row, gram_row = [], [], []
                for j, b in enumerate(group):
                    dg = self.dot(total[a], self.sc[b]) + self.dot(total[b], self.sc[a]) + self.dot(total[a], st[b])
                    de = self.dot(explicit[a], self.sc[b]) + self.dot(explicit[b], self.sc[a]) + self.dot(explicit[a], se[b])
                    dr = (self.dot(reconstruction[a], self.sc[b]) + self.dot(reconstruction[b], self.sc[a])
                          + self.dot(explicit[a], sr[b]) + self.dot(reconstruction[a], se[b])
                          + self.dot(reconstruction[a], sr[b]))
                    tg = self.dot(point[a], [c + d for c, d in zip(self.sc[b], st[b], strict=True)])
                    if dg != de + dr or dg != tg - base_gram[i][j]:
                        raise ValueError('Exact actual full-Gram quadratic/reconstruction identity failed')
                    row.append(dg)
                    split_row.append(de)
                    gram_row.append(tg)
                    normal += self.mul(lam[i][j], dg)
                    normal_explicit += self.mul(lam[i][j], de)
                delta_gram.append(row)
                split_gram.append(split_row)
                trial_gram.append(gram_row)
            if any(delta_gram[i][j] != delta_gram[j][i] for i in range(len(group)) for j in range(len(group))):
                raise ValueError('Exact signed Gram symmetry identity failed')
            defect = max(abs(v - (i == j)) for i, row in enumerate(trial_gram) for j, v in enumerate(row))
            if defect > self.gate:
                raise ValueError('Exact actual trial Gram gate failed')
            blocks.append({'kappa': group[0][1], 'keys': group,
                           'Lambda': [[float(v) for v in row] for row in lam],
                           'actual_Gram_defect': float(defect),
                           'DeltaGram': [[float(v) for v in row] for row in delta_gram],
                           'explicit_DeltaGram': [[float(v) for v in row] for row in split_gram],
                           'all_signed_offdiagonal_quadratic_reconstruction_identities_passed': True})
        return {'normal': normal, 'normal_explicit': normal_explicit, 'blocks': blocks}
