"""jj-Average configuration generation (DBSR_HF convention).

For a given non-relativistic configuration (e.g., "1s2 2s2 2p2"),
generates all possible jj-coupled configurations where electrons are
distributed among the j- (kappa=+l) and j+ (kappa=-(l+1)) subshells,
then computes the weighted average occupations.

With equal averaging: each integer distribution has weight 1.
With statistical averaging: weight = product of binomial coefficients C(2j+1, q_j).

The WW function (generalized product of occupations) is also computed
for use in the av_energy_coef formula.

Reference: DBSR_HF hf_def_conf_LS.f90, hf_MOD_energy.f90
"""

from dataclasses import dataclass
from math import comb
from typing import List, Tuple

from atomx.methods.dirac_fock import Subshell


@dataclass
class JJConfig:
    """A single jj-coupled configuration with specific integer occupations."""
    subshells: List[Subshell]  # all subshells with integer occupations
    weight: float              # statistical weight


def generate_jj_average(config_str: str,
                        averaging: str = "statistical") -> Tuple[List[Subshell], List[JJConfig]]:
    """Generate jj-average subshells from a non-relativistic configuration string.

    Parameters
    ----------
    config_str : str
        e.g., "1s2 2s2 2p2"
    averaging : str
        "equal" or "statistical" (default: "statistical").
        - "equal": each jj-configuration has equal weight.
        - "statistical": weight by (2j+1) degeneracy (recommended).

    Returns
    -------
    avg_subshells : list of Subshell
        All j-subshells with fractional (averaged) occupations.
        Includes subshells with occupation > 0.
    configs : list of JJConfig
        All individual jj-configurations with their weights.
    """
    # Map averaging parameter to the internal eal flag (FAC convention:
    # 1 = equal, 5 = statistical) used by the enumeration below.
    eal = {"equal": 1, "statistical": 5}[averaging]

    l_map = {'s': 0, 'p': 1, 'd': 2, 'f': 3, 'g': 4, 'h': 5}

    # Parse into non-relativistic shells
    nr_shells = []
    for token in config_str.strip().split():
        n = int(token[0])
        l_char = token[1]
        l = l_map[l_char]
        q = int(token[2:])
        nr_shells.append((n, l, q))

    # For each shell, determine the j-subshell occupations
    # s-orbitals (l=0): only j=1/2, no splitting needed
    # l>0: j- = l-1/2 (kappa=+l, max_occ=2l) and j+ = l+1/2 (kappa=-(l+1), max_occ=2l+2)

    # Build list of j-subshell specs: (n, kappa, max_occ)
    jshell_specs = []
    # For each NR shell, list the possible integer distributions
    shell_distributions = []  # list of lists of (q_minus, q_plus) or (q,) for s-shells

    for n, l, q in nr_shells:
        if l == 0:
            # s-shell: only j=1/2
            kappa = -(l + 1)  # = -1
            jshell_specs.append((n, kappa, 2))  # max_occ = 2
            shell_distributions.append([(q,)])
        else:
            # l > 0: two j-subshells
            kappa_minus = l              # j = l - 1/2
            kappa_plus = -(l + 1)        # j = l + 1/2
            max_minus = 2 * l            # 2j- + 1
            max_plus = 2 * (l + 1)       # 2j+ + 1
            jshell_specs.append((n, kappa_minus, max_minus))
            jshell_specs.append((n, kappa_plus, max_plus))

            # Enumerate all valid distributions of q electrons
            dists = []
            for q_minus in range(max(0, q - max_plus), min(q, max_minus) + 1):
                q_plus = q - q_minus
                if 0 <= q_plus <= max_plus:
                    dists.append((q_minus, q_plus))
            shell_distributions.append(dists)

    # Generate all configurations as Cartesian product of distributions
    def cartesian_product(lists):
        if not lists:
            return [()]
        result = []
        for item in lists[0]:
            for rest in cartesian_product(lists[1:]):
                result.append((item,) + rest)
        return result

    all_combos = cartesian_product(shell_distributions)

    # Build JJConfig objects with weights
    configs = []
    for combo in all_combos:
        subshells = []
        weight = 1.0
        idx = 0
        for shell_dist in combo:
            if len(shell_dist) == 1:
                # s-shell
                n, kappa, max_occ = jshell_specs[idx]
                q = shell_dist[0]
                subshells.append(Subshell(n=n, kappa=kappa, occupation=q))
                if eal == 5:
                    weight *= comb(max_occ, q)
                idx += 1
            else:
                # l>0: two j-subshells
                q_minus, q_plus = shell_dist
                n_m, kappa_m, max_m = jshell_specs[idx]
                n_p, kappa_p, max_p = jshell_specs[idx + 1]
                subshells.append(Subshell(n=n_m, kappa=kappa_m, occupation=q_minus))
                subshells.append(Subshell(n=n_p, kappa=kappa_p, occupation=q_plus))
                if eal == 5:
                    weight *= comb(max_m, q_minus) * comb(max_p, q_plus)
                idx += 2

        configs.append(JJConfig(subshells=subshells, weight=weight))

    # Normalize weights
    total_weight = sum(c.weight for c in configs)
    for c in configs:
        c.weight /= total_weight

    # Compute average occupations
    avg_occ = {}  # (n, kappa) -> float
    for c in configs:
        for sh in c.subshells:
            key = (sh.n, sh.kappa)
            avg_occ[key] = avg_occ.get(key, 0.0) + c.weight * sh.occupation

    # Build average subshell list (include all j-subshells, even if occ=0)
    avg_subshells = []
    for n, kappa, _max_occ in jshell_specs:
        occ = avg_occ.get((n, kappa), 0.0)
        # Clean IEEE float noise from the weighted average: the weight
        # normalisation above (c.weight /= total_weight) leaves Σ weights at
        # ~1-1e-16, so occupations come out as e.g. 1.9999999999999998 (closed
        # 2) or 3.5999999999999996 (statistical 3.6). Statistical occupations
        # are exact rationals (multiples of 1/(2l+2)), well within 12 decimals,
        # so rounding there removes the noise without touching physics.
        occ = round(occ, 12)
        avg_subshells.append(Subshell(n=n, kappa=kappa, occupation=occ))

    return avg_subshells, configs


def compute_WW(i: int, j: int, configs: List[JJConfig]) -> float:
    """Compute WW(i,j) — the generalized occupation product for the energy formula.

    For i == j: WW = Σ_c weight(c) * q_c(i) * (q_c(i) - 1)
    For i != j: WW = Σ_c weight(c) * q_c(i) * q_c(j)

    Parameters
    ----------
    i, j : int
        Orbital indices (0-based) in the subshell list.
    configs : list of JJConfig
    """
    ww = 0.0
    for c in configs:
        qi = c.subshells[i].occupation
        if i == j:
            ww += c.weight * qi * (qi - 1.0)
        else:
            qj = c.subshells[j].occupation
            ww += c.weight * qi * qj
    return ww


