"""Electron configuration parsing and ground state data.

Ground-state configurations for Z=1..118 using noble gas prefix notation
(adapted from zespec).  Configurations are expanded on first access.

Usage::

    from atomx.config import ground_state_config
    config = ground_state_config(10)   # "1s2 2s2 2p6"
    config = ground_state_config(26)   # "1s2 2s2 2p6 3s2 3p6 3d6 4s2"
"""

from atomx.constants import chemical_symbols

# Noble gas prefix notation (from zespec electron_configurations.txt)
_CONFIGS_PREFIX = {
    1: "1s1", 2: "1s2",
    3: "[He] 2s1", 4: "[He] 2s2",
    5: "[He] 2s2 2p1", 6: "[He] 2s2 2p2", 7: "[He] 2s2 2p3",
    8: "[He] 2s2 2p4", 9: "[He] 2s2 2p5", 10: "[He] 2s2 2p6",
    11: "[Ne] 3s1", 12: "[Ne] 3s2",
    13: "[Ne] 3s2 3p1", 14: "[Ne] 3s2 3p2", 15: "[Ne] 3s2 3p3",
    16: "[Ne] 3s2 3p4", 17: "[Ne] 3s2 3p5", 18: "[Ne] 3s2 3p6",
    19: "[Ar] 4s1", 20: "[Ar] 4s2",
    21: "[Ar] 3d1 4s2", 22: "[Ar] 3d2 4s2", 23: "[Ar] 3d3 4s2",
    24: "[Ar] 3d5 4s1", 25: "[Ar] 3d5 4s2",
    26: "[Ar] 3d6 4s2", 27: "[Ar] 3d7 4s2", 28: "[Ar] 3d8 4s2",
    29: "[Ar] 3d10 4s1", 30: "[Ar] 3d10 4s2",
    31: "[Ar] 3d10 4s2 4p1", 32: "[Ar] 3d10 4s2 4p2",
    33: "[Ar] 3d10 4s2 4p3", 34: "[Ar] 3d10 4s2 4p4",
    35: "[Ar] 3d10 4s2 4p5", 36: "[Ar] 3d10 4s2 4p6",
    37: "[Kr] 5s1", 38: "[Kr] 5s2",
    39: "[Kr] 4d1 5s2", 40: "[Kr] 4d2 5s2",
    41: "[Kr] 4d4 5s1", 42: "[Kr] 4d5 5s1", 43: "[Kr] 4d5 5s2",
    44: "[Kr] 4d7 5s1", 45: "[Kr] 4d8 5s1", 46: "[Kr] 4d10",
    47: "[Kr] 4d10 5s1", 48: "[Kr] 4d10 5s2",
    49: "[Kr] 4d10 5s2 5p1", 50: "[Kr] 4d10 5s2 5p2",
    51: "[Kr] 4d10 5s2 5p3", 52: "[Kr] 4d10 5s2 5p4",
    53: "[Kr] 4d10 5s2 5p5", 54: "[Kr] 4d10 5s2 5p6",
    55: "[Xe] 6s1", 56: "[Xe] 6s2",
    57: "[Xe] 5d1 6s2", 58: "[Xe] 4f1 5d1 6s2",
    59: "[Xe] 4f3 6s2", 60: "[Xe] 4f4 6s2",
    61: "[Xe] 4f5 6s2", 62: "[Xe] 4f6 6s2",
    63: "[Xe] 4f7 6s2", 64: "[Xe] 4f7 5d1 6s2",
    65: "[Xe] 4f9 6s2", 66: "[Xe] 4f10 6s2",
    67: "[Xe] 4f11 6s2", 68: "[Xe] 4f12 6s2",
    69: "[Xe] 4f13 6s2", 70: "[Xe] 4f14 6s2",
    71: "[Xe] 4f14 5d1 6s2", 72: "[Xe] 4f14 5d2 6s2",
    73: "[Xe] 4f14 5d3 6s2", 74: "[Xe] 4f14 5d4 6s2",
    75: "[Xe] 4f14 5d5 6s2", 76: "[Xe] 4f14 5d6 6s2",
    77: "[Xe] 4f14 5d7 6s2", 78: "[Xe] 4f14 5d9 6s1",
    79: "[Xe] 4f14 5d10 6s1", 80: "[Xe] 4f14 5d10 6s2",
    81: "[Xe] 4f14 5d10 6s2 6p1", 82: "[Xe] 4f14 5d10 6s2 6p2",
    83: "[Xe] 4f14 5d10 6s2 6p3", 84: "[Xe] 4f14 5d10 6s2 6p4",
    85: "[Xe] 4f14 5d10 6s2 6p5", 86: "[Xe] 4f14 5d10 6s2 6p6",
    87: "[Rn] 7s1", 88: "[Rn] 7s2",
    89: "[Rn] 6d1 7s2", 90: "[Rn] 6d2 7s2",
    91: "[Rn] 5f2 6d1 7s2", 92: "[Rn] 5f3 6d1 7s2",
    93: "[Rn] 5f4 6d1 7s2", 94: "[Rn] 5f6 7s2",
    95: "[Rn] 5f7 7s2", 96: "[Rn] 5f7 6d1 7s2",
    97: "[Rn] 5f9 7s2", 98: "[Rn] 5f10 7s2",
    99: "[Rn] 5f11 7s2", 100: "[Rn] 5f12 7s2",
    101: "[Rn] 5f13 7s2", 102: "[Rn] 5f14 7s2",
    103: "[Rn] 5f14 7s2 7p1", 104: "[Rn] 5f14 6d2 7s2",
    105: "[Rn] 5f14 6d3 7s2", 106: "[Rn] 5f14 6d4 7s2",
    107: "[Rn] 5f14 6d5 7s2", 108: "[Rn] 5f14 6d6 7s2",
    109: "[Rn] 5f14 6d7 7s2", 110: "[Rn] 5f14 6d8 7s2",
    111: "[Rn] 5f14 6d9 7s2", 112: "[Rn] 5f14 6d10 7s2",
    113: "[Rn] 5f14 6d10 7s2 7p1", 114: "[Rn] 5f14 6d10 7s2 7p2",
    115: "[Rn] 5f14 6d10 7s2 7p3", 116: "[Rn] 5f14 6d10 7s2 7p4",
    117: "[Rn] 5f14 6d10 7s2 7p5", 118: "[Rn] 5f14 6d10 7s2 7p6",
}

# Expanded configs cache (built lazily)
_GROUND_CONFIGS = None


def _expand_configs():
    """Expand noble gas prefixes into full configuration strings."""
    global _GROUND_CONFIGS
    if _GROUND_CONFIGS is not None:
        return _GROUND_CONFIGS

    expanded = {}
    for Z in sorted(_CONFIGS_PREFIX.keys()):
        config = _CONFIGS_PREFIX[Z].strip()
        # Expand [He], [Ne], [Ar], [Kr], [Xe], [Rn] prefixes
        while "[" in config:
            start = config.index("[")
            end = config.index("]")
            noble_symbol = config[start + 1:end]
            from atomx.constants import atomic_numbers
            noble_Z = atomic_numbers[noble_symbol]
            noble_config = expanded[noble_Z]
            config = noble_config + config[end + 1:]
        expanded[Z] = config.strip()

    _GROUND_CONFIGS = expanded
    return _GROUND_CONFIGS


def ground_state_config(Z: int) -> str:
    """Get the ground-state electron configuration string for atomic number Z.

    Parameters
    ----------
    Z : int
        Atomic number (1-118).

    Returns
    -------
    str
        Configuration string, e.g. ``"1s2 2s2 2p6"`` for Z=10 (Neon).

    Raises
    ------
    ValueError
        If Z is not in the database.
    """
    configs = _expand_configs()
    if Z not in configs:
        raise ValueError(f"No ground-state config for Z={Z}. "
                         f"Provide config= explicitly.")
    return configs[Z]
