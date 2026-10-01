"""Read Cowan RCN energy ledgers without conflating correlation switches.

The seven printed ETOT columns are defined in RCN OUTPT, format 50.
RCN always exports the last (corrected) column to RCN2, even for CORRF=0.
"""
import math
import re

# RCN2 G5INP's conversion, preserved for reproducibility of this code version.
RCN_RYDBERG_CM1 = 109737.31
ENERGY_COLUMNS = ('uncorrected', 'relativistic', 'old_correlation',
                  'old_correlation_relativistic', 'new_correlation',
                  'new_correlation_relativistic', 'exported')


def summarize_centroid_comparison(rows):
    """Score excited centroids on a common cohort for each CORRF value.

    Ground references and missing/nonfinite predictions never count as zero
    error. The shared cohort prevents missing Atomx states from improving its
    apparent ranking against Cowan. These metrics do not score resolved lines.
    """
    models = ('atomx_cm1', 'hfr_before_correlation_cm1', 'cowan_exported_cm1')

    def finite(value):
        return isinstance(value, (int, float)) and math.isfinite(value)

    summaries = []
    for corrf in sorted({r['corrf'] for r in rows}):
        eligible = [r for r in rows if r['corrf'] == corrf
                    and finite(r.get('nist_excitation_cm1'))
                    and r['nist_excitation_cm1'] > 0]
        common = [r for r in eligible if all(finite(r.get(k)) for k in models)]
        metrics = {}
        for model in models:
            residuals = [r[model] - r['nist_excitation_cm1'] for r in common]
            metrics[model] = dict(
                count=len(residuals),
                mean_error_cm1=sum(residuals)/len(residuals) if residuals else None,
                mae_cm1=sum(map(abs, residuals))/len(residuals) if residuals else None,
                rms_cm1=math.sqrt(sum(x*x for x in residuals)/len(residuals)) if residuals else None,
            )
        summaries.append(dict(corrf=corrf, eligible_count=len(eligible),
            common_count=len(common),
            common_configurations=[dict(case=r['case'], configuration=r['configuration']) for r in common],
            excluded_configurations=[dict(case=r['case'], configuration=r['configuration'],
                missing_models=[k for k in models if not finite(r.get(k))])
                for r in eligible if not all(finite(r.get(k)) for k in models)],
            metrics=metrics))
    return summaries


def parse_rcn_energies(text):
    """Return each configuration's printed totals in Ry; no model inference."""
    result = []
    for line in text.splitlines():
        if not line.lstrip().startswith('etot='):
            continue
        values = [float(x.replace('D', 'E')) for x in
                  re.findall(r'[-+]?\d+\.\d+(?:[DE][-+]?\d+)?', line)]
        if len(values) != len(ENERGY_COLUMNS):
            raise ValueError('incomplete Cowan ETOT ledger')
        result.append(dict(zip(ENERGY_COLUMNS, values, strict=False)))
    if not result:
        raise ValueError('Cowan output contains no ETOT ledger')
    return result


def rcn2_centroid_energies(text, labels):
    """Read unscaled relative Eav from RCN2's high precision listing (cm^-1)."""
    result = {}
    for label in labels:
        rows = re.findall(r'^0' + re.escape(label) + r'\s+([-+\d.]+).*?fact=([^\n]+)',
                          text, re.M)
        if len(rows) != 1:
            raise ValueError(f'expected one RCN2 centroid for {label}')
        energy, scales = rows[0]
        if [float(x) for x in scales.split()] != [1.] * 5:
            raise ValueError('Cowan benchmark requires all five radial scales = 1')
        result[label] = 1000. * float(energy)
    return result


def read_rcn_radial_records(path):
    """Read this LANL RCN build's little-endian ``tape2n`` radial records.

    The layout follows RCN OUTPT/RCN2 INPUT: 20 shell slots, 1801 radial
    slots, REAL*8, INTEGER*4, and 4-byte unformatted record markers.
    Other compiler layouts must be handled explicitly, never guessed.
    Only the stored orbitals and physical mesh entries are returned.
    """
    import numpy as np
    from scipy.io import FortranEOFError, FortranFile
    records = []
    with FortranFile(path, 'r', header_dtype=np.dtype('<u4')) as stream:
        while True:
            try:
                raw = stream.read_record(np.uint8)
            except FortranEOFError:
                break
            def integer(offset, *, raw=raw):
                return int(raw[offset:offset+4].view('<i4')[0])
            if len(raw) < 1306:
                raise ValueError('truncated LANL RCN radial header')
            shells, mesh, stored = integer(320), integer(774), integer(1302)
            if not (1 <= stored <= shells <= 20 and 2 <= mesh <= 1801):
                raise ValueError('unsupported LANL RCN radial layout')
            if len(raw) != 1306+(stored+2)*1801*8+4:
                raise ValueError('LANL RCN radial record length disagrees with layout')
            principal = raw[:80].view('<i4')[:shells]
            angular = raw[80:160].view('<i4')[:shells]
            occupation = raw[160:320].view('<f8')[:shells]
            arrays = raw[1306:-4].view('<f8').reshape(stored+2, 1801)[:, :mesh].copy()
            if (not np.all(np.isfinite(arrays)) or arrays[0,0] < 0
                    or not np.all(np.diff(arrays[0]) > 0)):
                raise ValueError('invalid LANL RCN radial grid or orbital data')
            records.append(dict(configuration=bytes(raw[724:742]).decode('ascii').strip(),
                z=integer(746), charge=integer(758),
                n=principal[-stored:].copy(), l=angular[-stored:].copy(),
                occupations=occupation[-stored:].copy(),
                radius=arrays[0], potential=arrays[1], orbitals=arrays[2:]))
    if not records:
        raise ValueError('empty LANL RCN radial archive')
    return records


def radial_overlap_summary(record):
    """Audit normalized same-l overlaps; different-l angular states are orthogonal."""
    import numpy as np
    from scipy.integrate import simpson
    radius, orbitals = record['radius'], record['orbitals']
    norms = np.array([simpson(p*p, x=radius) for p in orbitals])
    if not np.all(np.isfinite(norms)) or np.any(norms <= 0):
        raise ValueError('invalid Cowan orbital norm')
    pairs = []
    for i in range(len(orbitals)):
        for j in range(i):
            if record['l'][i] != record['l'][j]:
                continue
            product = orbitals[i]*orbitals[j]/np.sqrt(norms[i]*norms[j])
            pairs.append(dict(n=[int(record['n'][j]), int(record['n'][i])],
                l=int(record['l'][i]), overlap=float(simpson(product, x=radius)),
                trapezoid_check=float(np.trapezoid(product, x=radius))))
    return dict(norms=norms.tolist(), same_l_pairs=pairs,
        max_absolute_overlap=max((abs(p['overlap']) for p in pairs), default=0.),
        scope='Stored RCN radial functions; Simpson quadrature with trapezoidal cross-check. Not an energy correction.')
