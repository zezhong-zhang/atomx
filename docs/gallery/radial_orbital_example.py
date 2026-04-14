"""Example Manim scene: radial wavefunction of Ne 2p3/2.

Run with:

    manim -pqh docs/gallery/radial_orbital_example.py NeRadial2p

Requires ``manim-community`` (install via  pip install manim).
"""

from atomx import Atom
from atomx.viz.manim import RadialOrbitalScene


ne = Atom(Z=10).solve()


class NeRadial2p(RadialOrbitalScene):
    wavefunctions = ne.wavefunctions
    orbital_label = "2p3/2"
    r_max_bohr = 4.0
    show_small_component = True
