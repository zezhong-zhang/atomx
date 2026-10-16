"""Independent occupation-list Hamiltonian, cluster and angular diagnostics."""
import itertools
import math

import numpy as np


def act(mask, operations):
    occupied = [p for p in range(mask.bit_length()) if mask & (1 << p)]
    sign = 1
    for p, create in reversed(operations):
        if (p in occupied) == create:
            return None, 0
        sign *= (-1)**sum(q < p for q in occupied)
        if create:
            occupied.append(p)
        else:
            occupied.remove(p)
    return sum(1 << p for p in occupied), sign


def quantum(labels, mask):
    picked = [x for i, x in enumerate(labels) if mask & (1 << i)]
    return (sum(x[2] for x in picked),
            sum(x[1] if x[1] > 0 else -x[1]-1 for x in picked) % 2)


def sector(labels, electrons, m2):
    return tuple(mask for row in itertools.combinations(range(len(labels)), electrons)
                 if quantum(labels, mask := sum(1 << i for i in row)) == (m2, 0))


def coordinates(op, attachment):
    reference = (1 << op.o)-1
    masks, phases, operations = [], [], []
    for _section, particles, holes, _shape in (op.attachment if attachment else op.core).blocks:
        for particle, hole in zip(particles, holes, strict=True):
            ops = tuple((op.o+int(p), True) for p in particle)+tuple((int(h), False) for h in reversed(hole))
            end, phase = act(reference, ops)
            assert phase in (-1, 1)
            masks.append(end)
            phases.append(phase)
            operations.append(ops)
    return np.array(masks), np.array(phases), tuple(operations)


def cluster(amplitudes, operations, masks):
    lookup = {mask: i for i, mask in enumerate(masks)}
    matrix = np.zeros((len(masks), len(masks)))
    outside_max = 0.
    for value, ops in zip(amplitudes, operations, strict=True):
        if value == 0:
            continue
        for col, mask in enumerate(masks):
            end, sign = act(mask, ops)
            if sign:
                if end in lookup:
                    matrix[lookup[end], col] += sign*value
                else:
                    outside_max = max(outside_max, abs(value))
    return matrix, outside_max


def hamiltonian(h, g, frozen, masks):
    lookup = {mask: i for i, mask in enumerate(masks)}
    matrix = np.eye(len(masks))*frozen
    terms = [(h[p, q], ((p, True), (q, False)))
             for p, q in itertools.product(range(len(h)), repeat=2) if h[p, q] != 0]
    # Full four-index tensor and1/4, independent of native pair reduction.
    terms += [(.25*g[p, q, r, s], ((p, True), (q, True), (s, False), (r, False)))
              for p, q, r, s in itertools.product(range(len(h)), repeat=4) if g[p, q, r, s] != 0]
    outside_max = 0.
    for col, mask in enumerate(masks):
        for value, ops in terms:
            end, sign = act(mask, ops)
            if sign:
                if end in lookup:
                    matrix[lookup[end], col] += value*sign
                else:
                    outside_max = max(outside_max, abs(value))
    return matrix, outside_max


def j_squared(labels, masks):
    lookup = {mask: i for i, mask in enumerate(masks)}
    where = {tuple(label): i for i, label in enumerate(labels)}
    raising = []
    for i, (n, k, m) in enumerate(labels):
        j = abs(k)-.5
        if m < 2*j:
            raising.append((where[n, k, m+2], i, math.sqrt((j-m/2)*(j+m/2+1))))
    z = np.array([quantum(labels, mask)[0]/2 for mask in masks])
    matrix = np.diag(z*(z+1))
    for col, mask in enumerate(masks):
        for p, q, first in raising:
            mid, sign1 = act(mask, ((p, True), (q, False)))
            if not sign1:
                continue
            for r, s, second in raising:
                end, sign2 = act(mid, ((s, True), (r, False)))
                if sign2:
                    assert end in lookup
                    matrix[lookup[end], col] += first*second*sign1*sign2
    return matrix


def select(energies, physical, masks, labels, j2, m2, occupied):
    output = {}
    reference = (1 << occupied)-1
    lookup = {mask: i for i, mask in enumerate(masks)}
    for name, key, angular in [('S', (6, -1), .5), ('D3', (5, 2), 1.5), ('D5', (5, -3), 2.5)]:
        if abs(m2) > 2*angular:
            continue
        model = [lookup[reference | (1 << i)] for i, label in enumerate(labels)
                 if tuple(label[:2]) == key and label[2] == m2]
        assert len(model) == 1
        weights = np.sum(abs(physical[model])**2, axis=0)
        chosen = np.flatnonzero((weights >= .5) & (abs(np.asarray(energies).imag) < 1e-10))
        assert len(chosen) == 1, (name, m2, weights.max())
        k = int(chosen[0])
        wave = physical[:, k]
        output[name] = dict(root=k, energy_Ha=float(energies[k].real),
                            bare_weight=float(weights[k]),
                            J2_defect=float(np.linalg.norm(j2@wave-angular*(angular+1)*wave)))
    return output
