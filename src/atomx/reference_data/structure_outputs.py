"""Strict energy/orbital identities in external atomic-structure output.

Printed rounding bounds describe serialization precision, not experimental
uncertainty or a solver's numerical error estimate.
"""
import re
import math
from decimal import Decimal

NUMBER = r'[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[DdEe][-+]?\d+)?'
LETTERS = 'spdfghiklmnoqrtuvwxyz'


def number(token):
    return float(token.replace('D', 'E').replace('d', 'e'))


def rounding_bound(token):
    value = Decimal(token.replace('D', 'E').replace('d', 'e'))
    return float(Decimal(5).scaleb(value.as_tuple().exponent-1))


def dbsr_orbital_label(label):
    """DBSR '-' means j=l-1/2; unsigned (or '+') means j=l+1/2."""
    match = re.fullmatch(r'(\d+)(['+LETTERS+r'])([-+]?)', label.strip())
    if not match:
        raise ValueError(f'unsupported DBSR orbital label: {label!r}')
    n, letter, sign = match.groups()
    ell = LETTERS.index(letter)
    twoj = 2*ell-1 if sign == '-' else 2*ell+1
    if int(n) <= ell or twoj <= 0:
        raise ValueError(f'invalid DBSR orbital label: {label!r}')
    return f'{int(n)}{letter}{twoj}/2'


def parse_dbsr_summary(text):
    energy = re.findall(r'Coulomb energy:\s*('+NUMBER+r')\s+au', text)
    if not energy:
        raise ValueError('DBSR final Coulomb summary is missing')
    headers = list(re.finditer(r'\bnl\s+e\(nl\)\s+dmp\s+ns\s+<r>', text))
    if not headers:
        raise ValueError('DBSR final orbital summary is missing')
    tail = text[headers[-1].end():]
    orbitals, radii, radius_rounding = {}, {}, {}
    for match in re.finditer(r'^\s*(\d+['+LETTERS+r'][-+]?)\s+('+NUMBER+r')\s+'
                            +NUMBER+r'\s+\d+\s+('+NUMBER+r')', tail, re.M):
        label = dbsr_orbital_label(match[1])
        orbitals[label], radii[label] = number(match[2]), number(match[3])
        radius_rounding[label] = rounding_bound(match[3])
    if not orbitals:
        raise ValueError('DBSR orbital summary is empty')
    fock_expectations = {}
    marker = 'Final orbital Fock expectations (au):'
    if marker in tail:
        block = tail.split(marker, 1)[1]
        for match in re.finditer(r'^\s*(\d+['+LETTERS+r'][-+]?)\s+('+NUMBER+r')\s*$', block, re.M):
            fock_expectations[dbsr_orbital_label(match[1])] = number(match[2])
        if fock_expectations.keys() != orbitals.keys():
            raise ValueError('DBSR final Fock diagnostic does not cover every orbital')
    residuals = {}
    for label in ('Orbital', 'Energy'):
        values = re.findall(label+r'\s+convergence:\s*('+NUMBER+r')', tail)
        if not values:
            raise ValueError(f'DBSR {label} convergence is missing')
        residuals[label.lower()] = number(values[-1])
    return dict(total_energy_Ha=number(energy[-1]),
                printed_energy_rounding_Ha=rounding_bound(energy[-1]),
                orbital_energies_Ha=orbitals, orbital_mean_radius_bohr=radii,
                printed_radius_rounding_bohr=radius_rounding,
                orbital_fock_expectations_Ha=fock_expectations,
                residuals=residuals)


def parse_radial_summary(text):
    energies = re.findall(r'Total binding energy\s*=\s*('+NUMBER+r')\s+Eh', text)
    if not energies:
        raise ValueError('RADIAL total binding energy in Hartree is missing')
    orbitals = {}
    for match in re.finditer(r'Shell:\s*(\d+)\s*(['+LETTERS+r'])\s*(\d+)/2'
                            r'[^\n]*?E\s*=\s*('+NUMBER+r')\s+Eh', text):
        n, letter, twoj, energy = match.groups()
        ell = LETTERS.index(letter)
        if int(twoj) not in (2*ell-1, 2*ell+1) or int(twoj) <= 0:
            raise ValueError('invalid RADIAL angular momentum label')
        orbitals[f'{int(n)}{letter}{int(twoj)}/2'] = number(energy)
    if not orbitals:
        raise ValueError('RADIAL orbital eigenvalues are missing')
    return dict(total_energy_Ha=number(energies[-1]),
                printed_energy_rounding_Ha=rounding_bound(energies[-1]),
                orbital_energies_Ha=orbitals)


def orbital_differences(calculated, reference):
    """Compare identical spectroscopic labels and report incomplete coverage."""
    common = calculated.keys() & reference.keys()
    differences = {key: calculated[key]-reference[key] for key in sorted(common)}
    return dict(differences_Ha=differences,
                missing_calculated=sorted(reference.keys()-calculated.keys()),
                missing_reference=sorted(calculated.keys()-reference.keys()),
                max_abs_Ha=max(map(abs, differences.values()), default=None))


def parse_dbsr_configurations(text):
    """Read the actual integer-jj ensemble from DBSR's generated .conf file."""
    from atomx.config.subshell import Subshell
    from atomx.config.jj_coupling import JJConfig
    try:
        core = text.split('Core subshells:',1)[1].split('Peel subshells:',1)[0]
        entries = text.split('CSF(s):',1)[1]
    except IndexError as error:
        raise ValueError('DBSR configuration headers are missing') from error
    def shell(label, occupation=None):
        canonical = dbsr_orbital_label(label)
        match = re.fullmatch(r'(\d+)(['+LETTERS+r'])(\d+)/2',canonical)
        n, letter, twoj = int(match[1]), match[2], int(match[3])
        ell = LETTERS.index(letter)
        kappa = -(ell+1) if twoj == 2*ell+1 else ell
        q = float(twoj+1 if occupation is None else occupation)
        if not math.isfinite(q) or q != int(q) or not 0 <= q <= twoj+1:
            raise ValueError('DBSR configuration needs integer occupations within each jj capacity')
        return Subshell(n,kappa,q)
    core_labels = re.findall(r'\d+['+LETTERS+r'][-+]?',core)
    configurations = []
    pattern = r'(\d+['+LETTERS+r'][-+]?)\s*\(\s*('+NUMBER+r')\s*\)'
    for line in entries.splitlines():
        matches = list(re.finditer(pattern,line))
        if not matches:
            continue
        remainder = line[matches[-1].end():].strip()
        if not re.fullmatch(NUMBER,remainder):
            raise ValueError('DBSR ensemble weight is missing or malformed')
        shells = [shell(label) for label in core_labels]
        shells += [shell(m[1],number(m[2])) for m in matches]
        if len({(s.n,s.kappa) for s in shells}) != len(shells):
            raise ValueError('DBSR repeats an occupied shell')
        weight = number(remainder)
        if not math.isfinite(weight) or weight <= 0:
            raise ValueError('DBSR configuration weight must be positive')
        configurations.append(JJConfig(shells,weight))
    if not configurations:
        raise ValueError('DBSR configuration ensemble is empty')
    if len({sum(s.occupation for s in c.subshells) for c in configurations}) != 1:
        raise ValueError('DBSR configurations have different electron counts')
    normalization = sum(c.weight for c in configurations)
    keys = sorted({(s.n,s.kappa) for c in configurations for s in c.subshells},
                  key=lambda k:(k[0],abs(k[1]),k[1]))
    for config in configurations:
        occupations = {(s.n,s.kappa): s.occupation for s in config.subshells}
        config.subshells = [Subshell(n,k,occupations.get((n,k),0.)) for n,k in keys]
        config.weight /= normalization
    shells = [Subshell(n,k,sum(c.weight*c.subshells[i].occupation for c in configurations))
              for i,(n,k) in enumerate(keys)]
    return shells, configurations
