"""Independent signed angular ladders in complete declared excitation ranks."""

import math

import numpy as np


def act(mask, operations):
    occupied = [i for i in range(mask.bit_length()) if mask & (1 << i)]
    sign = 1
    for orbital, create in reversed(operations):
        if (orbital in occupied) == create:
            return None, 0
        sign *= (-1) ** sum(i < orbital for i in occupied)
        if create:
            occupied.append(orbital)
        else:
            occupied.remove(orbital)
    return sum(1 << i for i in occupied), sign


class AngularCoordinates:
    def __init__(self, op, attachment):
        labels = op.labels
        self.size = (op.attachment if attachment else op.core).size
        self.masks = []
        self.phases = []
        self.m2 = []
        self.parity = []
        reference = (1 << op.o) - 1
        phase_cache = {}
        for _section, particles, holes, _shape in (
            op.attachment if attachment else op.core
        ).blocks:
            for particle, hole in zip(particles, holes, strict=True):
                ops = tuple((op.o + int(i), True) for i in particle) + tuple(
                    (int(i), False) for i in reversed(hole)
                )
                key = tuple(hole)
                if key not in phase_cache:
                    _, phase_cache[key] = act(reference, ops)
                phase = phase_cache[key]
                assert phase in (-1, 1)
                mask = (reference ^ sum(1 << int(i) for i in hole)) | sum(
                    1 << (op.o + int(i)) for i in particle
                )
                picked = [labels[op.o + int(i)] for i in particle]
                removed = [labels[int(i)] for i in hole]
                self.masks.append(mask)
                self.phases.append(phase)
                self.m2.append(sum(x[2] for x in picked) - sum(x[2] for x in removed))
                self.parity.append(
                    sum(x[1] if x[1] > 0 else -x[1] - 1 for x in picked + removed) % 2
                )
        self.phases = np.asarray(self.phases)
        self.m2 = np.asarray(self.m2)
        self.parity = np.asarray(self.parity)
        self.lookup = {mask: i for i, mask in enumerate(self.masks)}
        where = {tuple(label): i for i, label in enumerate(labels)}
        self.raising = []
        for q, (n, k, m) in enumerate(labels):
            j = abs(k) - 0.5
            if m < 2 * j:
                self.raising.append(
                    (where[n, k, m + 2], q, math.sqrt((j - m / 2) * (j + m / 2 + 1)))
                )

    def ladder(self, vector, direction):
        assert direction in (-1, 1)
        result = np.zeros(self.size, dtype=np.asarray(vector).dtype)
        for col in np.flatnonzero(np.asarray(vector) != 0):
            mask = self.masks[col]
            for p, q, factor in self.raising:
                if direction < 0:
                    p, q = q, p
                if not mask & (1 << q) or mask & (1 << p):
                    continue
                end, phase = act(mask, ((p, True), (q, False)))
                assert phase and end in self.lookup
                row = self.lookup[end]
                result[row] += (
                    factor * phase * self.phases[row] * self.phases[col] * vector[col]
                )
        return result

    def j_squared(self, vector, m):
        return self.ladder(self.ladder(vector, 1), -1) + m * (m + 1) * vector

    def sector(self, m2, parity=0):
        return np.flatnonzero((self.m2 == m2) & (self.parity == parity))
