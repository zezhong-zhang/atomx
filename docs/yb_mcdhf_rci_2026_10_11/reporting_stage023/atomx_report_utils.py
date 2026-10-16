"""Reuse the existing standard-library reporting helpers without running its main."""
import importlib.util
from pathlib import Path

source = Path(__file__).resolve().parents[1] / 'reporting_stage022/integrate.py'
spec = importlib.util.spec_from_file_location('closed_reporting_stage022', source)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
bound, load, once, sha = module.bound, module.load, module.once, module.sha
