"""Example Manim scene: Grotrian diagram with transitions.

Run with:  manim -pqh docs/gallery/grotrian_example.py HydrogenGrotrian
"""

from atomx.viz.manim import GrotrianScene


class HydrogenGrotrian(GrotrianScene):
    # Energies in eV, L-labels for horizontal grouping
    levels = [
        {"label": "1s", "energy": -13.6, "L": 0},
        {"label": "2s", "energy": -3.4,  "L": 0},
        {"label": "2p", "energy": -3.4,  "L": 1},
        {"label": "3s", "energy": -1.51, "L": 0},
        {"label": "3p", "energy": -1.51, "L": 1},
        {"label": "3d", "energy": -1.51, "L": 2},
    ]
    transitions = [
        # Lyman series: 2p → 1s, 3p → 1s
        {"upper": 2, "lower": 0, "wavelength_nm": 121.6},
        {"upper": 4, "lower": 0, "wavelength_nm": 102.6},
        # Balmer: 3p, 3s, 3d → 2s/2p
        {"upper": 4, "lower": 1, "wavelength_nm": 656.3},
        {"upper": 5, "lower": 2, "wavelength_nm": 656.3},
    ]
    energy_unit = "eV"
