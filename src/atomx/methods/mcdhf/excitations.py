"""Generate correlation CSFs via single and double excitations.

For MCDHF correlation recovery, we generate excitations from the
reference (ground) configuration into virtual orbitals.

The active space approach:
- Core orbitals: frozen (not excited)
- Valence orbitals: can be excited from
- Virtual orbitals: targets for excitations (3d, 4s, 4p, 4d, ...)

Standard SD (singles-doubles) excitations capture the bulk of
correlation energy (~95% for atoms).
"""

import re

from atomx.methods.mcdhf.config import _parse_config_string

# Orbital-label letters, index == l (shared with config parsing).
_L_SYMBOLS = "spdfghiklmno"


def _split_nl(key):
    """Split an orbital key like ``"10h"`` into (n, l_char); n may be >= 10."""
    m = re.fullmatch(r"(\d+)([spdfghiklmno])", key)
    if m is None:
        raise ValueError(f"cannot parse orbital label {key!r}")
    return int(m.group(1)), m.group(2)


def generate_sd_excitations(ref_config_str, virtual_orbitals,
                             core_orbitals=None, max_excitations=2):
    """Generate CSFs from single and double excitations.

    Parameters
    ----------
    ref_config_str : str
        Reference configuration (e.g., "1s2 2s2 2p6").
    virtual_orbitals : list of str
        Virtual orbital labels to excite into (e.g., ["3d", "4s", "4p"]).
    core_orbitals : list of str, optional
        Orbitals to freeze (not excite from). Default: none (all active).
    max_excitations : int
        1 = singles only, 2 = singles + doubles (default).

    Returns
    -------
    csf_strings : list of str
        Configuration strings including the reference + all SD excitations.
    """
    ref_shells = _parse_config_string(ref_config_str)

    # Build occupation dict for reference: (n, l_char) -> occupation
    # Cover high-l virtuals (l up to 'o') so DR/continuum drivers that request
    # n up to ~10 with l_max >= 5 don't KeyError on 'h', 'i', ...
    l_map_inv = {i: ch for i, ch in enumerate(_L_SYMBOLS)}
    ref_occ = {}
    for sh in ref_shells:
        l = sh.kappa if sh.kappa > 0 else -sh.kappa - 1
        key = f"{sh.n}{l_map_inv[l]}"
        ref_occ[key] = ref_occ.get(key, 0) + sh.occupation

    # Parse core
    frozen = set()
    if core_orbitals:
        frozen = set(core_orbitals)

    # Active (excitable) orbitals
    active_from = [(k, v) for k, v in ref_occ.items() if k not in frozen and v > 0]

    # Generate all virtual orbital slots
    l_map = {ch: i for i, ch in enumerate(_L_SYMBOLS)}
    virtual_slots = []
    for virt in virtual_orbitals:
        m = re.fullmatch(r"(\d+)([spdfghiklmno])", virt)
        if m is None:
            raise ValueError(f"cannot parse virtual orbital label {virt!r}")
        l = l_map[m.group(2)]
        max_q = 2 * (2 * l + 1)  # total capacity of nl shell
        virtual_slots.append((virt, max_q))

    configs = [ref_config_str]  # always include reference

    # Singles: move 1 electron from active → virtual
    if max_excitations >= 1:
        for src_key, src_occ in active_from:
            if src_occ < 1:
                continue
            for virt_key, virt_max in virtual_slots:
                if virt_key == src_key:
                    continue
                # Build new config string
                new_occ = dict(ref_occ)
                new_occ[src_key] = new_occ[src_key] - 1
                new_occ[virt_key] = new_occ.get(virt_key, 0) + 1
                if new_occ[virt_key] > virt_max:
                    continue
                cfg = _occ_to_string(new_occ)
                if cfg not in configs:
                    configs.append(cfg)

    # Doubles: move 2 electrons
    if max_excitations >= 2:
        for i, (src1, occ1) in enumerate(active_from):
            for j, (src2, occ2) in enumerate(active_from):
                if j < i:
                    continue
                # Check enough electrons
                if i == j and occ1 < 2:
                    continue
                if i != j and (occ1 < 1 or occ2 < 1):
                    continue

                for vi, (virt1, _vmax1) in enumerate(virtual_slots):
                    for vj, (virt2, _vmax2) in enumerate(virtual_slots):
                        if vj < vi:
                            continue
                        if virt1 in (src1, src2):
                            continue
                        if virt2 in (src1, src2):
                            continue

                        new_occ = dict(ref_occ)
                        # Remove exactly TWO source electrons.  ``i == j``
                        # means both come from the same subshell (a² → b²
                        # or a² → b¹c¹): decrementing src1 twice was
                        # previously done *on top of* the unconditional
                        # decrements of src1 and src2 (== src1), taking
                        # three electrons out of a doubly-occupied shell,
                        # so the negative occupation was rejected by the
                        # validation below and every same-subshell double
                        # was silently dropped — ``generate_sd_excitations
                        # ("1s2", ...)`` returned ZERO doubles and
                        # ``correlation="sd"`` under-recovered.
                        new_occ[src1] = new_occ[src1] - 1
                        if i == j:
                            new_occ[src1] -= 1  # 2nd electron from same shell
                        else:
                            new_occ[src2] = new_occ[src2] - 1
                        new_occ[virt1] = new_occ.get(virt1, 0) + 1
                        new_occ[virt2] = new_occ.get(virt2, 0) + 1

                        # Validate
                        if any(v < 0 for v in new_occ.values()):
                            raise ValueError(
                                f"double excitation {ref_config_str!r} → "
                                f"({src1},{src2})→({virt1},{virt2}) removed "
                                f"more electrons than the reference holds: "
                                f"{new_occ}")
                        l_check = {k: l_map[_split_nl(k)[1]] for k in new_occ}
                        if any(new_occ[k] > 2 * (2 * l_check[k] + 1) for k in new_occ if new_occ[k] > 0):
                            continue

                        cfg = _occ_to_string(new_occ)
                        if cfg not in configs:
                            configs.append(cfg)

    return configs


def _occ_to_string(occ_dict):
    """Convert occupation dict to config string."""
    l_order = {ch: i for i, ch in enumerate(_L_SYMBOLS)}
    parts = []
    for key in sorted(occ_dict.keys(), key=lambda k: (_split_nl(k)[0], l_order[_split_nl(k)[1]])):
        q = occ_dict[key]
        if q > 0:
            parts.append(f"{key}{int(q)}")
    return " ".join(parts)


def suggest_virtual_orbitals(Z, n_virtual=2):
    """Suggest virtual orbitals for correlation based on Z.

    Returns virtual orbital labels (e.g., ["3d", "4s", "4p"]).
    """
    # Find highest occupied n
    from atomx.config import ground_state_config
    try:
        config = ground_state_config(Z)
    except ValueError:
        config = ""
    l_map = {'s': 0, 'p': 1, 'd': 2, 'f': 3}

    max_n = 1
    max_l = 0
    for token in config.split():
        n = int(token[0])
        l = l_map.get(token[1], 0)
        max_n = max(max_n, n)
        max_l = max(max_l, l)

    virtuals = []
    # Add same-n higher-l (only if n > l, i.e., l < n)
    for l in range(max_l + 1, min(max_n, 5)):  # l < n required
        label = f"{max_n}{'spdfg'[l]}"
        virtuals.append(label)

    # Add next-n orbitals (n > l always satisfied for l < nn)
    for extra_n in range(1, n_virtual + 1):
        nn = max_n + extra_n
        for l in range(min(nn, 4)):
            label = f"{nn}{'spdfg'[l]}"
            if label not in virtuals:
                virtuals.append(label)

    return virtuals[:6]
