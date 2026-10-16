"""Revalidate the previous finite physical-state/E1 panel at corrected sources."""

import runpy
from pathlib import Path

OWNER = Path(__file__).resolve().parent
helpers = runpy.run_path(str(OWNER / "probe_monovalent_sd.py"))
helpers["main"].__globals__["OUT"] = OWNER / "monovalent_sd008"
helpers["main"]()
