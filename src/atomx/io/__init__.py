"""I/O utilities for atomx.

Currently exposes a writer / reader pair for FAC's ASCII tables
(LEV, TR, CE) so atomx output can be dropped into pipelines that
already consume FAC's ``PrintTable(..., 1)`` files (CHIANTI, FLYCHK,
SPECT3D, MEPHIE, ...).
"""

from atomx.io.fac_writer import (
    FAC_TYPE_CE,
    FAC_TYPE_LEV,
    FAC_TYPE_TR,
    FAC_VERSION,
    read_ce,
    read_lev,
    read_tra,
    write_ce,
    write_lev,
    write_tra,
)

__all__ = [
    "FAC_TYPE_CE",
    "FAC_TYPE_LEV",
    "FAC_TYPE_TR",
    "FAC_VERSION",
    "read_ce",
    "read_lev",
    "read_tra",
    "write_ce",
    "write_lev",
    "write_tra",
]
