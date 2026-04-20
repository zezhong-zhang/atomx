"""Pytest root conftest for the worktree.

The atomx venv has its editable install pinned to the main repo's
``src/`` directory (via a ``.pth`` file), which means running tests
from a git worktree would otherwise import the main tree rather than
the worktree tree.  Prepend this worktree's ``src`` to ``sys.path`` so
tests exercise the code here.
"""

from __future__ import annotations

import sys
from pathlib import Path

_WORKTREE_SRC = Path(__file__).resolve().parent / "src"
if str(_WORKTREE_SRC) not in sys.path:
    sys.path.insert(0, str(_WORKTREE_SRC))
