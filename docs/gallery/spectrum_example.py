"""Example Manim scene: synthetic spectrum build-up.

Run with:  manim -pqh docs/gallery/spectrum_example.py NeSpectrum
"""

from atomx.spectrum import Spectrum
from atomx.viz.manim import SpectrumScene


def build_spectrum():
    # Three "Ne-like" emission lines (fake for illustration)
    lines = [(580.0, 1.0), (585.2, 0.6), (592.7, 0.8)]
    frames = []
    s = Spectrum.linear(570.0, 600.0, 5001)
    for i, (lam, I) in enumerate(lines, start=1):
        s.add_line(lam, intensity=I, sigma_nm=0.1)
        # Snapshot a copy for each step
        from copy import deepcopy
        frames.append((f"After line {i}: {lam:.1f} nm", deepcopy(s)))
    return frames


class NeSpectrum(SpectrumScene):
    spectrum_factory = staticmethod(build_spectrum)
