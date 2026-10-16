"""Complete finite DC radial changes in the public native CI cell operator."""
from __future__ import annotations

import hashlib
import math
from functools import lru_cache

import numpy as np

from atomx.numerics.bspline.convol import ConvolCache, _density_sym, _sum_amb_banded
from atomx.numerics.bspline.dirac_assembly import build_dirac_matrices
from atomx.physics.csf.matrix import GridCIWorkspace, _bspline_pair_potential


def _owned_waves(waves, ns):
    result = {}
    for key, pair in waves.items():
        if len(pair) != 2:
            raise ValueError("DC radial increment requires P/Q pairs")
        values = []
        for value in pair:
            array = np.asarray(value)
            if (array.shape != (ns,) or array.dtype != np.dtype(np.float64)
                    or np.iscomplexobj(array)
                    or not np.isfinite(array).all()):
                raise ValueError("DC radial increment requires finite real ns-component waves")
            values.append(np.frombuffer(array.tobytes(), dtype=np.float64))
        result[key] = tuple(values)
    return result


class _DCRadialIncrement:
    """Call-scoped radial-key polynomial and separately retained reconstruction.

    No CI is built. The actual trial may differ from reference plus the
    explicit increment because unpacked endpoints round separately. Retain
    the complete additional quadratic/quartic contribution, rather than
    claiming that a small waveform difference gives an exact energy identity.
    Both contractions use the ordinary CI bank/banded cell operation.
    """

    def __init__(self, reference, increments, workspace, Z, ns, *,
                 nuclear_potential, cache, trial):
        if isinstance(workspace, GridCIWorkspace) or workspace.get("ns") != ns:
            raise ValueError("DC radial increments require the same B-spline workspace")
        if (set(reference) != set(increments) or not reference
                or trial is not None and set(trial) != set(reference)):
            raise ValueError("DC radial increment keys must match complete endpoints")
        self.ns = ns
        # Instance-owned caches release with this call-scoped operator.
        self._density = lru_cache(maxsize=512)(self._density)
        self._potential = lru_cache(maxsize=1024)(self._potential)
        self.components = lru_cache(maxsize=8192)(self.components)
        self.reference = _owned_waves(reference, ns)
        self.increments = _owned_waves(increments, ns)
        if trial is None:
            # No reconstruction contribution is claimed without an actual
            # separately stored trial; callers use the explicit endpoint.
            self.reconstruction = {key: tuple(np.zeros(ns) for _ in range(2))
                                   for key in reference}
        else:
            actual = _owned_waves(trial, ns)
            self.reconstruction = {
                key: tuple(np.asarray((t.astype(np.longdouble) - r.astype(np.longdouble))
                                      - d.astype(np.longdouble), dtype=np.float64)
                           for r, d, t in zip(self.reference[key], self.increments[key],
                                              actual[key], strict=True))
                for key in reference
            }
        self.orders = (workspace.get("ksp", workspace["ks"]),
                       workspace.get("ksq", workspace["ks"]))
        self.cache = ConvolCache(workspace) if cache is None else cache
        if not self.cache.same_grid(workspace):
            raise ValueError("DC radial increment cache does not match the workspace")
        # Assemble from the actual sampled nuclear operator, independently of
        # a mutable process memo. The unchanged public builder uses this same
        # numerical routine and its original one-body, not a fitted operator.
        self.one_body = {k: build_dirac_matrices(ns, k, Z, workspace, nuclear_potential)
                         for k in sorted({key[1] for key in reference})}
        framed = b"".join(repr(key).encode() + b"".join(v.tobytes() for v in self.increments[key])
                          for key in sorted(self.increments))
        self.increment_sha256 = hashlib.sha256(framed).hexdigest()
        self.reconstruction_max = max(float(np.max(np.abs(v)))
                                      for pair in self.reconstruction.values() for v in pair)

    def _density(self, mode, a, b):
        output = []
        for side, order in enumerate(self.orders):
            ra, rb = self.reference[a][side], self.reference[b][side]
            da, db = self.increments[a][side], self.increments[b][side]
            if mode == "reference":
                value = _density_sym(ra, rb, self.ns, order)
            elif mode == "increment":
                value = (_density_sym(da, rb, self.ns, order)
                         + _density_sym(ra, db, self.ns, order)
                         + _density_sym(da, db, self.ns, order))
            else:
                ea, eb = self.reconstruction[a][side], self.reconstruction[b][side]
                value = (_density_sym(ea, rb, self.ns, order)
                         + _density_sym(ra, eb, self.ns, order)
                         + _density_sym(ea, db, self.ns, order)
                         + _density_sym(da, eb, self.ns, order)
                         + _density_sym(ea, eb, self.ns, order))
            output.append(value)
        return tuple(output)

    def _potential(self, mode, rank, a, b):
        return _bspline_pair_potential(self.cache, self.ns, self.orders, rank,
                                      self._density(mode, a, b))

    def _one_body(self, key):
        _, na, kappa, nb = key
        a, b = (na, kappa), (nb, kappa)
        va, vb = np.concatenate(self.reference[a]), np.concatenate(self.reference[b])
        da, db = np.concatenate(self.increments[a]), np.concatenate(self.increments[b])
        ea, eb = np.concatenate(self.reconstruction[a]), np.concatenate(self.reconstruction[b])
        h, metric = self.one_body[kappa]
        hb, hdb = h @ vb, h @ db
        norm, numerator = float(vb @ metric @ vb), float(va @ hb)
        dn = math.fsum((float(db @ metric @ vb), float(vb @ metric @ db),
                        float(db @ metric @ db)))
        delta_n = math.fsum((float(da @ hb), float(va @ hdb), float(da @ hdb)))
        if not np.isfinite(norm + dn) or norm <= 0 or norm + dn <= 0:
            raise ValueError("DC radial increment requires positive endpoint ket norms")
        original = numerator / norm
        delta = (delta_n - original * dn) / (norm + dn)
        h_eb = h @ eb
        reconstruction_n = math.fsum((float(ea @ hb), float(va @ h_eb),
                                     float(ea @ hdb), float(da @ h_eb), float(ea @ h_eb)))
        reconstruction_s = math.fsum((float(eb @ metric @ vb), float(vb @ metric @ eb),
                                     float(eb @ metric @ db), float(db @ metric @ eb),
                                     float(eb @ metric @ eb)))
        final_norm = math.fsum((norm, dn, reconstruction_s))
        if not np.isfinite(final_norm) or final_norm <= 0:
            raise ValueError("DC radial reconstruction requires a positive trial ket norm")
        correction = (reconstruction_n - (original + delta) * reconstruction_s) / final_norm
        return delta, correction

    def components(self, key):
        """Return explicit polynomial and full rounded-endpoint contribution."""
        if key[0] == "h" and len(key) == 4:
            output = self._one_body(key)
        elif key[0] == "r" and len(key) == 6:
            _, rank, a, c, b, d = key
            # Match public B-spline source/target orientation exactly. Only
            # ordering within each symmetric pair is canonicalized.
            source, target = tuple(sorted((b, d))), tuple(sorted((a, c)))
            original = self._potential("reference", rank, *source)
            delta = self._potential("increment", rank, *source)
            extra = self._potential("reconstruction", rank, *source)
            rho = self._density("reference", *target)
            drho = self._density("increment", *target)
            erho = self._density("reconstruction", *target)
            pieces, corrections = [], []
            for side, order in enumerate(self.orders):
                def dot(potential, density, side=side, order=order):
                    return float(_sum_amb_banded(potential[side], density[side], self.ns, order))

                pieces.extend((dot(original, drho), dot(delta, rho), dot(delta, drho)))
                corrections.extend((dot(original, erho), dot(delta, erho), dot(extra, rho),
                                    dot(extra, drho), dot(extra, erho)))
            output = math.fsum(pieces), math.fsum(corrections)
        else:
            raise ValueError("DC radial increments support only complete h/r radial keys")
        if not np.isfinite(output).all():
            raise FloatingPointError("non-finite DC radial finite increment")
        return output

    def __call__(self, key):
        return self.components(key)[0]

    def diagnostics(self):
        return dict(representation="native-CI-cell-bank-banded",
                    increment_sha256=self.increment_sha256,
                    reconstruction_wave_max=self.reconstruction_max,
                    reconstruction_dtype=np.longdouble.__name__,
                    reconstruction_mantissa_fraction_bits=int(np.finfo(np.longdouble).nmant),
                    polynomial_dtype="float64", certified_error_bound=False)


def bspline_dc_radial_increment(reference, increments, workspace, Z, ns, *,
                                nuclear_potential=None, cache=None, trial=None):
    """Capture one complete finite-cell DC increment without a CI solve.

    ``components(key)`` separates the complete explicit polynomial from the
    contribution to the actual separately rounded ``trial`` when provided.
    The factory does not validate physical state matching or CI ownership;
    that belongs to the consuming method. No Breit/grid/effective terms exist.
    """
    return _DCRadialIncrement(reference, increments, workspace, Z, ns,
                             nuclear_potential=nuclear_potential, cache=cache, trial=trial)
