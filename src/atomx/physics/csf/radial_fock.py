"""Coherent monoconfiguration CSF Coulomb fields for radial DF equations.

The full c†Hc angular contraction supplies the radial coefficients. Diagonal
occupations alone never replace two-body interference. Amplitudes are held
fixed during a radial sweep; the method driver updates the Hamiltonian state.
"""
import numpy as np
from atomx.physics.hartree_fock import RadialHartreeFock
from atomx.physics.csf.radial_density import CIRadialDensity
from atomx.numerics.shooting.coulomb import _yk_over_r
from atomx.numerics.shooting.quadrature import slag6_weights


class CSFRadialFock(RadialHartreeFock):
    # Fixed coherent amplitudes need not be invariant under a sign change
    # of one orbital. Cubic/linear interference adds odd harmonics to the
    # quartic two-orbital energy; configuration averages have only 2 and 4.
    rotation_harmonics = (1, 2, 3, 4)

    def __init__(self, shells, csfs, workspace, reference):
        super().__init__(shells)
        self.csfs = csfs
        self.workspace = workspace
        self.reference = reference
        self.keys = tuple((s.n, s.kappa) for s in self.subshells)
        self.lookup = {key: i for i, key in enumerate(self.keys)}
        self.radial_density = CIRadialDensity(csfs)
        self.weights = slag6_weights(workspace.n)*workspace.difr
        self.coefficients = {}
        self.terms = {}

    def set_state(self, ci):
        self.coefficients = self.radial_density.contract(ci)
        self.terms = {key: [] for key in self.keys}
        for term, value in self.coefficients.items():
            if term[0] == 'h':
                _, n, k, nb = term
                if n != nb:
                    raise ValueError('monoconfiguration DF requires diagonal one-body density')
                continue
            _, rank, a, b, c, d = term
            # R^k(ab,cd): every occurrence is differentiated, including
            # coincident labels. The factor 1/(2q) is applied only below.
            for target, other, left, right in ((a,b,c,d),(b,a,c,d),(c,d,a,b),(d,c,a,b)):
                self.terms[target].append((rank, left, right, other, value))
        for shell in self.subshells:
            value = self.coefficients.get(('h', shell.n, shell.kappa, shell.n), 0.)
            capacity = 2*abs(shell.kappa)
            shell.occupation = (float(capacity) if abs(value-capacity) < 1e-12
                                else max(0., float(value)))

    def angular_tables(self):
        return None, None, None

    def coulomb_energy_terms(self):
        """Full coherent contraction in the same radial key convention as HF."""
        for term, value in self.coefficients.items():
            if term[0] == 'r':
                _, rank, a, b, c, d = term
                yield False, value, rank, *(self.lookup[key] for key in (a,b,c,d))

    def _two_body_action(self, a, pa, qa, trial=None):
        ws = self.workspace
        key = self.keys[a]
        out = np.zeros((2, ws.n))
        potentials = {}
        for rank, left, right, other, coefficient in self.terms[key]:
            i, j, b = self.lookup[left], self.lookup[right], self.lookup[other]
            label = (rank, *sorted((i, j)))
            if label not in potentials:
                pi, qi, pj, qj = pa[i], qa[i], pa[j], qa[j]
                if trial is not None and other != key:
                    if (left == key)+(right == key) != 1:
                        raise NotImplementedError('this CSF source is not a homogeneous one-orbital operator')
                    if left == key: pi, qi = trial
                    else: pj, qj = trial
                potentials[label] = _yk_over_r(pi*pj+qi*qj, rank, ws.r, ws.difr, ws.n)
            orbital = trial if trial is not None and other == key else np.array([pa[b], qa[b]])
            out += coefficient*potentials[label]*orbital
        return out/(2*self.subshells[a].occupation)

    def orbital_field(self, a, pa, qa, r, difr, rvnuc, angular):
        if self.subshells[a].occupation < 1e-12:
            # A zero-occupation orbital has no ground-state Euler equation.
            # Retain its explicitly documented reference field, without
            # inventing a population in the Hamiltonian or density.
            return self.reference.orbital_field(a, pa, qa, r, difr, rvnuc,
                                                self.reference.angular_tables())
        local = np.zeros(len(r))
        for b, shell in enumerate(self.subshells):
            local += (shell.occupation-(a == b))*_yk_over_r(
                pa[b]**2+qa[b]**2, 0, r, difr, len(r))
        source = local*np.array([pa[a], qa[a]])-self._two_body_action(a, pa, qa)
        return rvnuc+r*local, source[0], source[1], local

    def exchange_action(self, a, trial, pa, qa, r, difr, exch):
        if self.subshells[a].occupation < 1e-12:
            return self.reference.exchange_action(a, trial, pa, qa, r, difr,
                                                  self.reference.angular_tables()[1])
        local = np.zeros(len(r))
        for b, shell in enumerate(self.subshells):
            local += (shell.occupation-(a == b))*_yk_over_r(
                pa[b]**2+qa[b]**2, 0, r, difr, len(r))
        return local*trial-self._two_body_action(a, pa, qa, trial=trial)

    def two_electron_energy(self, pa, qa, r, difr, np_):
        densities, potentials = {}, {}
        energy = 0.
        for term, value in self.coefficients.items():
            if term[0] != 'r': continue
            _, rank, a, b, c, d = term
            ab, cd = tuple(sorted((a,b))), tuple(sorted((c,d)))
            for pair in (ab, cd):
                if pair not in densities:
                    i, j = map(self.lookup.__getitem__, pair)
                    densities[pair] = pa[i]*pa[j]+qa[i]*qa[j]
            if (rank, ab) not in potentials:
                potentials[rank, ab] = _yk_over_r(densities[ab], rank, r, difr, np_)
            energy += value*float((potentials[rank, ab]*densities[cd])@self.weights)
        # This is the complete coherent Coulomb contraction. A split into
        # classical J and exchange K is not a unique decomposition here.
        return energy, 0.
